"""
낙상 감지 유틸리티
Feature 추출 및 Random Forest 예측
"""

import numpy as np
import pickle
from collections import deque


class FallDetector:
    """낙상 감지기"""
    
    def __init__(self, model_path: str, feature_path: str):
        """
        초기화
        
        Args:
            model_path: Random Forest 모델 경로
            feature_path: Feature 컬럼 순서 파일 경로
        """
        # 모델 로드
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Feature 컬럼 순서 로드
        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        # 프레임 버퍼 (동적 feature용)
        self.frame_buffer = deque(maxlen=30)
        
        # 클래스 이름
        self.class_names = ['Normal', 'Falling', 'Fallen']
        
        print(f"✅ FallDetector 초기화 완료")
        print(f"   Feature 수: {len(self.feature_columns)}")
        print(f"   클래스: {self.class_names}")
    
    def extract_features(self, keypoints):
        """
        Keypoints에서 Feature 추출
        
        Args:
            keypoints: YOLO Pose keypoints (17 x 3) [x, y, confidence]
        
        Returns:
            features: Feature 딕셔너리
        """
        if len(keypoints) == 0:
            return None
        
        # 첫 번째 사람만 사용
        kps = keypoints[0]
        
        # 신뢰도 낮은 keypoints 제거
        valid_kps = kps[kps[:, 2] > 0.5]
        if len(valid_kps) < 5:
            return None
        
        features = {}
        
        # 1. 기본 위치 정보
        for i in range(17):
            features[f'x_{i}'] = kps[i, 0]
            features[f'y_{i}'] = kps[i, 1]
            features[f'confidence_{i}'] = kps[i, 2]
        
        # 2. Hip Height (골반 높이)
        left_hip = kps[11]  # 왼쪽 골반
        right_hip = kps[12]  # 오른쪽 골반
        
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            features['hip_height'] = hip_center_y
        else:
            features['hip_height'] = 0
        
        # 3. Spine Angle (척추 각도)
        nose = kps[0]  # 코
        if nose[2] > 0.5 and left_hip[2] > 0.5:
            dx = nose[0] - left_hip[0]
            dy = nose[1] - left_hip[1]
            spine_angle = np.arctan2(dy, dx) * 180 / np.pi
            features['spine_angle'] = spine_angle
        else:
            features['spine_angle'] = 0
        
        # 4. Body Aspect Ratio
        x_coords = kps[:, 0][kps[:, 2] > 0.5]
        y_coords = kps[:, 1][kps[:, 2] > 0.5]
        
        if len(x_coords) > 0:
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            if height > 0:
                features['aspect_ratio'] = width / height
            else:
                features['aspect_ratio'] = 1.0
        else:
            features['aspect_ratio'] = 1.0
        
        # 5. Center of Mass
        if len(valid_kps) > 0:
            features['center_x'] = np.mean(valid_kps[:, 0])
            features['center_y'] = np.mean(valid_kps[:, 1])
        else:
            features['center_x'] = 0
            features['center_y'] = 0
        
        return features
    
    def add_dynamic_features(self, features_list):
        """
        동적 Feature 추가 (속도, 가속도 등)
        
        Args:
            features_list: Feature 딕셔너리 리스트
        
        Returns:
            features: 동적 feature가 추가된 딕셔너리
        """
        if len(features_list) < 2:
            return features_list[-1]
        
        current = features_list[-1]
        previous = features_list[-2]
        
        # 속도 계산
        if 'center_y' in current and 'center_y' in previous:
            current['velocity_y'] = current['center_y'] - previous['center_y']
            current['velocity_x'] = current['center_x'] - previous['center_x']
        else:
            current['velocity_y'] = 0
            current['velocity_x'] = 0
        
        # Hip height 변화
        if 'hip_height' in current and 'hip_height' in previous:
            current['hip_height_change'] = current['hip_height'] - previous['hip_height']
        else:
            current['hip_height_change'] = 0
        
        return current
    
    def predict(self, features):
        """
        낙상 예측
        
        Args:
            features: Feature 딕셔너리
        
        Returns:
            prediction: 예측 클래스 (0: Normal, 1: Falling, 2: Fallen)
            probabilities: 각 클래스별 확률
        """
        # Feature 벡터 생성 (컬럼 순서에 맞춰)
        feature_vector = []
        for col in self.feature_columns:
            if col in features:
                feature_vector.append(features[col])
            else:
                feature_vector.append(0.0)  # 없는 feature는 0으로
        
        # 예측
        feature_array = np.array(feature_vector).reshape(1, -1)
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]
        
        return prediction, probabilities
