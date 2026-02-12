"""
ST-GCN 실시간 추론 모듈
- 60 프레임 버퍼링을 통한 시계열 낙상 감지
- 기존 YOLO Pose keypoints 호환
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

# 상대 경로 import (GUI 폴더 내 stgcn 패키지)
from stgcn.st_gcn import Model
from stgcn.graph import Graph
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


class STGCNInference:
    """
    ST-GCN 실시간 추론 클래스
    
    사용법:
        model = STGCNInference(model_path='path/to/best_model_binary.pth')
        
        # 매 프레임마다 호출
        keypoints = yolo_pose.get_keypoints()  # (17, 3)
        result = model.update(keypoints)
        
        if result is not None:
            label, confidence = result
            print(f"Detection: {label} ({confidence:.2f})")
    """
    
    # 클래스 상수
    LABELS = ['Normal', 'Fall']  # Binary classification
    SEQUENCE_LENGTH = 60         # 60 frames (~3초 @20fps)
    NUM_KEYPOINTS = 17           # COCO keypoints
    NUM_CHANNELS = 3             # x, y, confidence
    
    def __init__(self, model_path: str, device: str = 'auto', num_class: int = 2):
        """
        ST-GCN 추론 모듈 초기화
        
        Args:
            model_path: best_model_binary.pth 경로
            device: 'cuda', 'cpu', 또는 'auto' (자동 감지)
            num_class: 클래스 수 (기본 2: Normal, Fall)
        """
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 모델 초기화
        self.model = Model(
            num_class=num_class,
            num_point=self.NUM_KEYPOINTS,
            num_person=1,
            in_channels=self.NUM_CHANNELS,
            graph_args={}
        )
        
        # 가중치 로드
        self._load_weights(model_path)
        
        # 모델을 디바이스로 이동 및 평가 모드
        self.model.to(self.device)
        self.model.eval()
        
        # 키포인트 버퍼 (슬라이딩 윈도우)
        self.keypoints_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        
        # 정규화 파라미터 (필요시 조정)
        self.normalize_coords = True
        self.frame_width = 640   # 기본값, set_frame_size()로 변경 가능
        self.frame_height = 480
        
        print(f"[ST-GCN] Model loaded on {self.device}")
        print(f"[ST-GCN] Sequence length: {self.SEQUENCE_LENGTH} frames")
    
    def _load_weights(self, model_path: str):
        """모델 가중치 로드"""
        try:
            # PyTorch 2.6+ 호환성: weights_only=False 필요
            # (신뢰할 수 있는 로컬 모델 파일에만 사용)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 체크포인트 형식에 따라 처리
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"[ST-GCN] Weights loaded from: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
    
    def set_frame_size(self, width: int, height: int):
        """
        프레임 크기 설정 (좌표 정규화용)
        
        Args:
            width: 프레임 너비
            height: 프레임 높이
        """
        self.frame_width = width
        self.frame_height = height
        print(f"[ST-GCN] Frame size set to {width}x{height}")
    
    def reset_buffer(self):
        """버퍼 초기화 (새 시퀀스 시작 시)"""
        self.keypoints_buffer.clear()
        print("[ST-GCN] Buffer reset")
    
    def preprocess(self, keypoints_list: List[np.ndarray]) -> torch.Tensor:
        """
        키포인트 리스트를 모델 입력 텐서로 변환
        
        Args:
            keypoints_list: list of (17, 3) arrays, length=60
            
        Returns:
            tensor: (1, 3, 60, 17, 1) - (N, C, T, V, M)
        """
        # (T, V, C) numpy array
        data = np.array(keypoints_list)  # (60, 17, 3)
        
        # 좌표 정규화 (0~1 범위)
        if self.normalize_coords:
            data_normalized = data.copy()
            data_normalized[:, :, 0] = data[:, :, 0] / self.frame_width   # x
            data_normalized[:, :, 1] = data[:, :, 1] / self.frame_height  # y
            # confidence는 이미 0~1 범위
            data = data_normalized
        
        # (T, V, C) → (C, T, V)
        data = data.transpose(2, 0, 1)  # (3, 60, 17)
        
        # (C, T, V) → (1, C, T, V, 1)
        data = data[np.newaxis, :, :, :, np.newaxis]  # (1, 3, 60, 17, 1)
        
        # Tensor 변환
        tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        return tensor
    
    @torch.no_grad()
    def predict(self, keypoints_list: List[np.ndarray]) -> Tuple[str, float]:
        """
        직접 추론 (버퍼 미사용)
        
        Args:
            keypoints_list: list of (17, 3) arrays, length >= 60
            
        Returns:
            (label_name: str, confidence: float)
        """
        # 최근 60 프레임만 사용
        buffer = list(keypoints_list)[-self.SEQUENCE_LENGTH:]
        
        if len(buffer) < self.SEQUENCE_LENGTH:
            raise ValueError(f"Need at least {self.SEQUENCE_LENGTH} frames, got {len(buffer)}")
        
        # 전처리
        input_tensor = self.preprocess(buffer)
        
        # 추론
        output = self.model(input_tensor)
        
        # Softmax → 확률
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        label_idx = predicted.item()
        label_name = self.LABELS[label_idx]
        conf_value = confidence.item()
        
        return label_name, conf_value
    
    def update(self, keypoints: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        새 프레임 키포인트 추가 및 추론 (버퍼 사용)
        
        Args:
            keypoints: (17, 3) array - [x, y, confidence] for each joint
            
        Returns:
            None: 버퍼가 아직 안 찼음
            (label_name, confidence): 추론 결과
        """
        # 키포인트 유효성 검사
        if keypoints is None or keypoints.shape != (self.NUM_KEYPOINTS, self.NUM_CHANNELS):
            # 감지 실패 시 zero 프레임 추가 (또는 이전 프레임 유지)
            keypoints = np.zeros((self.NUM_KEYPOINTS, self.NUM_CHANNELS))
        
        # 버퍼에 추가
        self.keypoints_buffer.append(keypoints.copy())
        
        # 버퍼가 충분히 찼을 때만 추론
        if len(self.keypoints_buffer) >= self.SEQUENCE_LENGTH:
            return self.predict(list(self.keypoints_buffer))
        
        return None
    
    def get_buffer_status(self) -> Tuple[int, int, str]:
        """
        버퍼 상태 반환
        
        Returns:
            (current_length, required_length, status_string)
        """
        current = len(self.keypoints_buffer)
        required = self.SEQUENCE_LENGTH
        
        if current >= required:
            status = "Ready"
        else:
            status = f"Buffering... {current}/{required}"
        
        return current, required, status
    
    def is_ready(self) -> bool:
        """추론 준비 완료 여부"""
        return len(self.keypoints_buffer) >= self.SEQUENCE_LENGTH
    
    def get_buffer_progress(self) -> float:
        """버퍼 진행률 (0.0 ~ 1.0)"""
        return min(1.0, len(self.keypoints_buffer) / self.SEQUENCE_LENGTH)


class STGCNInferenceWithSmoothing(STGCNInference):
    """
    결과 스무딩이 적용된 ST-GCN 추론
    - 연속된 추론 결과를 평활화하여 떨림 방지
    """
    
    def __init__(self, model_path: str, device: str = 'auto', 
                 num_class: int = 2, smoothing_window: int = 5):
        """
        Args:
            smoothing_window: 결과 평활화 윈도우 크기
        """
        super().__init__(model_path, device, num_class)
        
        self.smoothing_window = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)
    
    def update(self, keypoints: np.ndarray) -> Optional[Tuple[str, float]]:
        """스무딩이 적용된 추론"""
        result = super().update(keypoints)
        
        if result is not None:
            label, confidence = result
            label_idx = self.LABELS.index(label)
            
            # 히스토리에 추가
            self.prediction_history.append((label_idx, confidence))
            
            # 다수결 투표
            if len(self.prediction_history) >= self.smoothing_window:
                labels = [p[0] for p in self.prediction_history]
                avg_conf = np.mean([p[1] for p in self.prediction_history])
                
                # 가장 많이 나온 레이블
                from collections import Counter
                most_common = Counter(labels).most_common(1)[0][0]
                
                return self.LABELS[most_common], avg_conf
        
        return result
    
    def reset_buffer(self):
        """버퍼 및 히스토리 초기화"""
        super().reset_buffer()
        self.prediction_history.clear()


# ========== 유틸리티 함수 ==========

def create_stgcn_inference(model_path: str, use_smoothing: bool = False, 
                           smoothing_window: int = 5) -> STGCNInference:
    """
    ST-GCN 추론 객체 생성 헬퍼 함수
    
    Args:
        model_path: 모델 파일 경로
        use_smoothing: 결과 스무딩 사용 여부
        smoothing_window: 스무딩 윈도우 크기
        
    Returns:
        STGCNInference 또는 STGCNInferenceWithSmoothing 객체
    """
    if use_smoothing:
        return STGCNInferenceWithSmoothing(
            model_path=model_path,
            smoothing_window=smoothing_window
        )
    else:
        return STGCNInference(model_path=model_path)


# ========== 테스트 코드 ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ST-GCN Inference Test')
    parser.add_argument('--model', type=str, 
                        # default='/home/gjkong/dev_ws/st_gcn/checkpoints/best_model_binary.pth',
                        default=str(PATHS.STGCN_ORIGINAL) if PATHS.STGCN_ORIGINAL else str(PATHS.STGCN_V2),
                        help='Model path')
    parser.add_argument('--test', action='store_true', help='Run simple test')
    args = parser.parse_args()
    
    if args.test:
        print("=" * 50)
        print("ST-GCN Inference Module Test")
        print("=" * 50)
        
        # 모델 로드 테스트
        try:
            model = STGCNInference(model_path=args.model)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            exit(1)
        
        # 더미 데이터로 추론 테스트
        print("\n[Test] Dummy inference...")
        dummy_keypoints = np.random.rand(17, 3).astype(np.float32)
        dummy_keypoints[:, 0] *= 640  # x
        dummy_keypoints[:, 1] *= 480  # y
        
        # 버퍼 채우기
        for i in range(60):
            result = model.update(dummy_keypoints)
            if i < 59:
                assert result is None, "Should not predict before buffer full"
        
        print(f"✅ Buffer status: {model.get_buffer_status()}")
        
        # 추론 결과
        if result is not None:
            label, conf = result
            print(f"✅ Prediction: {label} (confidence: {conf:.4f})")
        
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
