#!/usr/bin/env python3
"""
ST-GCN 실시간 추론 모듈 (Fine-tuned 모델 지원)
YOLO Pose 키포인트를 받아 낙상 여부를 판단
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Tuple, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


# ============================================================================
# 그래프 정의 (COCO 17 keypoints)
# ============================================================================

class Graph:
    """COCO 17 keypoints 스켈레톤 그래프"""
    
    def __init__(self, layout='coco'):
        self.layout = layout
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        # COCO 17 keypoints 연결
        self.inward = [
            (15, 13), (13, 11), (16, 14), (14, 12),  # legs
            (11, 12),  # hip connection
            (5, 11), (6, 12),  # torso-leg connection
            (5, 6),  # shoulder connection
            (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (1, 3), (2, 4),  # ears-eyes
            (0, 1), (0, 2),  # nose-eyes
            (1, 2),  # eye connection
            (0, 5), (0, 6),  # nose-shoulders
        ]
        
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        
        self.A = self.get_adjacency_matrix()
    
    def get_adjacency_matrix(self):
        """인접 행렬 생성"""
        A = np.zeros((3, self.num_node, self.num_node))
        
        # Self-loop
        for i, j in self.self_link:
            A[0, i, j] = 1
        
        # Inward
        for i, j in self.inward:
            A[1, i, j] = 1
        
        # Outward
        for i, j in self.outward:
            A[2, i, j] = 1
        
        return A


# ============================================================================
# ST-GCN 모델 (Fine-tuned 구조)
# ============================================================================

class STGCNBlock(nn.Module):
    """ST-GCN 기본 블록"""
    
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        
        self.gcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        # 인접 행렬 등록
        self.register_buffer('A', torch.FloatTensor(A))
    
    def forward(self, x):
        res = self.residual(x)
        
        # GCN
        N, C, T, V = x.size()
        x = x.view(N, C * T, V)
        x = torch.matmul(x, self.A.sum(0))
        x = x.view(N, C, T, V)
        x = self.gcn(x)
        
        # TCN
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNFineTuned(nn.Module):
    """ST-GCN 모델 (Fine-tuned 구조)"""
    
    def __init__(self, num_class=2, num_point=17, num_person=1, 
                 in_channels=3, graph_layout='coco'):
        super().__init__()
        
        self.graph = Graph(graph_layout)
        A = self.graph.A
        
        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # ST-GCN layers
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A),
        ])
        
        # Classification head
        self.fc = nn.Linear(256, num_class)
        
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
    
    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        
        # (N, M, V, C, T) -> (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        
        # (N, C, T, V*M)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(N, C, T, V * M)
        
        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=(2, 3))  # (N, C)
        
        # Classification
        x = self.fc(x)
        
        return x


# ============================================================================
# 추론 클래스
# ============================================================================

class STGCNInference:
    """ST-GCN 실시간 추론 래퍼 (Fine-tuned 모델 지원)"""
    
    LABELS = ['Normal', 'Fall']
    SEQUENCE_LENGTH = 60      # 60 프레임 시퀀스
    NUM_KEYPOINTS = 17        # COCO keypoints
    NUM_CHANNELS = 3          # x, y, confidence
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'auto',
                 sequence_length: int = 60):
        """
        Args:
            model_path: 모델 체크포인트 경로 (None이면 path_config 기본값)
            device: 'cuda', 'cpu', 또는 'auto'
            sequence_length: 시퀀스 길이 (기본 60)
        """
        if model_path is None:
            model_path = str(PATHS.STGCN_V2)
        
        self.sequence_length = sequence_length
        self.keypoints_buffer = deque(maxlen=sequence_length)
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        print(f"[ST-GCN] Model loaded on {self.device}")
        print(f"[ST-GCN] Sequence length: {self.sequence_length} frames")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """모델 로드"""
        model = STGCNFineTuned(
            num_class=2,
            num_point=self.NUM_KEYPOINTS,
            num_person=1,
            in_channels=self.NUM_CHANNELS
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # state_dict 추출
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print(f"[ST-GCN] Weights loaded from: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, keypoints_list: List[np.ndarray]) -> torch.Tensor:
        """
        키포인트 시퀀스를 모델 입력 형태로 변환
        
        Args:
            keypoints_list: 키포인트 리스트 [(17, 3), (17, 3), ...]
        
        Returns:
            텐서 (1, 3, T, 17, 1)
        """
        T = len(keypoints_list)
        
        # (T, 17, 3) array
        sequence = np.array(keypoints_list)  # (T, 17, 3)
        
        # (C, T, V) 형태로 변환
        # sequence: (T, V, C) -> (C, T, V)
        sequence = sequence.transpose(2, 0, 1)  # (3, T, 17)
        
        # ⭐ Hip center 정규화 (학습 데이터와 동일)
        # left_hip=11, right_hip=12
        hip_center = (sequence[:2, :, 11] + sequence[:2, :, 12]) / 2  # (2, T)
        sequence[:2, :, :] -= hip_center[:, :, np.newaxis]  # center
        max_dist = np.abs(sequence[:2, :, :]).max()
        if max_dist > 0:
            sequence[:2, :, :] /= max_dist  # scale to -1~1
        
        # (N, C, T, V, M) 형태로 확장
        sequence = sequence[np.newaxis, :, :, :, np.newaxis]  # (1, 3, T, 17, 1)
        
        return torch.FloatTensor(sequence).to(self.device)
    
    def predict(self, keypoints_list: List[np.ndarray]) -> Tuple[str, float]:
        """
        키포인트 시퀀스로 낙상 예측
        
        Args:
            keypoints_list: 키포인트 리스트 (최소 sequence_length 이상)
        
        Returns:
            (예측 레이블, 신뢰도)
        """
        if len(keypoints_list) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} frames, got {len(keypoints_list)}")
        
        # 마지막 sequence_length 프레임 사용
        keypoints_list = list(keypoints_list)[-self.sequence_length:]
        
        # 전처리
        x = self.preprocess(keypoints_list)
        
        # 추론
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        label = self.LABELS[pred_idx]
        normal_prob = probs[0, 0].item()
        fall_prob = probs[0, 1].item()
        
        return label, confidence, normal_prob, fall_prob
    
    def update(self, keypoints: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        새 프레임 추가 및 추론 (버퍼가 충분하면)
        
        Args:
            keypoints: 단일 프레임 키포인트 (17, 3)
        
        Returns:
            버퍼가 충분하면 (레이블, 신뢰도), 아니면 None
        """
        # 버퍼에 추가
        self.keypoints_buffer.append(keypoints)
        
        # 버퍼가 충분하면 예측
        if len(self.keypoints_buffer) >= self.sequence_length:
            return self.predict(list(self.keypoints_buffer))
        
        return None
    
    def get_buffer_status(self) -> Tuple[int, int, str]:
        """
        버퍼 상태 반환
        
        Returns:
            (현재 프레임 수, 필요 프레임 수, 상태 문자열)
        """
        current = len(self.keypoints_buffer)
        required = self.sequence_length
        
        if current >= required:
            status = "Ready"
        else:
            status = f"Buffering... {current}/{required}"
        
        return current, required, status
    
    def reset_buffer(self):
        """버퍼 초기화"""
        self.keypoints_buffer.clear()


class STGCNInferenceWithSmoothing(STGCNInference):
    """결과 스무딩이 적용된 ST-GCN 추론"""
    
    def __init__(self, *args, smoothing_window: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing_window = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)
    
    def update(self, keypoints: np.ndarray) -> Optional[Tuple[str, float]]:
        """스무딩이 적용된 업데이트"""
        result = super().update(keypoints)
        
        if result is not None:
            label, confidence = result
            pred_idx = self.LABELS.index(label)
            self.prediction_history.append((pred_idx, confidence))
            
            # 스무딩: 최근 예측들의 가중 평균
            if len(self.prediction_history) >= 3:
                fall_votes = sum(1 for p, c in self.prediction_history if p == 1)
                total = len(self.prediction_history)
                
                if fall_votes > total // 2:
                    # Fall이 과반수
                    avg_conf = np.mean([c for p, c in self.prediction_history if p == 1])
                    return 'Fall', avg_conf
                else:
                    avg_conf = np.mean([c for p, c in self.prediction_history if p == 0])
                    return 'Normal', avg_conf
        
        return result


# ============================================================================
# 테스트
# ============================================================================

def test_basic():
    """기본 테스트"""
    print("\n" + "="*50)
    print("ST-GCN Fine-tuned Inference Module Test")
    print("="*50)
    
    # 모델 로드
    model_path = str(PATHS.STGCN_V2)
    
    try:
        inference = STGCNInference(model_path=model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return False
    
    # 더미 데이터 테스트
    print("\n[Test] Dummy inference...")
    dummy_keypoints = [np.random.rand(17, 3).astype(np.float32) for _ in range(60)]
    
    label, confidence = inference.predict(dummy_keypoints)
    print(f"✅ Prediction: {label} (confidence: {confidence:.4f})")
    
    # 버퍼 테스트
    inference.reset_buffer()
    for i in range(60):
        kp = np.random.rand(17, 3).astype(np.float32)
        result = inference.update(kp)
        if result:
            print(f"✅ Buffer ready, prediction: {result}")
            break
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    return True


if __name__ == '__main__':
    test_basic()
