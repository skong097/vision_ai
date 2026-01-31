"""
Feature Engineering - Skeleton 데이터에서 추가 특징 추출
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Skeleton 데이터에서 추가 feature 생성"""
    
    def __init__(self, skeleton_dir, output_dir):
        self.skeleton_dir = Path(skeleton_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Keypoint 이름
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def calculate_angle(self, p1, p2, p3):
        """3개 점으로 각도 계산 (p2가 꼭지점)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def calculate_distance(self, p1, p2):
        """두 점 사이의 유클리드 거리"""
        return np.linalg.norm(p1 - p2)
    
    def extract_static_features(self, row):
        """정적 특징 추출 (단일 프레임)"""
        features = {}
        
        # 1. 관절 각도 계산
        keypoints = {}
        for kp_name in self.keypoint_names:
            x = row[f'{kp_name}_x']
            y = row[f'{kp_name}_y']
            keypoints[kp_name] = np.array([x, y])
        
        # 왼쪽 팔꿈치 각도
        if all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            features['left_elbow_angle'] = self.calculate_angle(
                keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist']
            )
        else:
            features['left_elbow_angle'] = 0
        
        # 오른쪽 팔꿈치 각도
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            features['right_elbow_angle'] = self.calculate_angle(
                keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist']
            )
        else:
            features['right_elbow_angle'] = 0
        
        # 왼쪽 무릎 각도
        if all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            features['left_knee_angle'] = self.calculate_angle(
                keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle']
            )
        else:
            features['left_knee_angle'] = 0
        
        # 오른쪽 무릎 각도
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            features['right_knee_angle'] = self.calculate_angle(
                keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle']
            )
        else:
            features['right_knee_angle'] = 0
        
        # 척추 각도 (어깨 중심 - 골반 중심 - 수직선)
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_center = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
            hip_center = (keypoints['left_hip'] + keypoints['right_hip']) / 2
            
            # 수직선 (아래로 향하는 벡터)
            vertical = np.array([0, 1])
            spine_vector = hip_center - shoulder_center
            
            # 각도 계산
            cos_angle = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            features['spine_angle'] = np.degrees(np.arccos(cos_angle))
        else:
            features['spine_angle'] = 0
        
        # 2. 신체 중심점 높이
        if all(k in keypoints for k in ['left_hip', 'right_hip']):
            hip_center = (keypoints['left_hip'] + keypoints['right_hip']) / 2
            features['hip_height'] = hip_center[1]
        else:
            features['hip_height'] = 0
        
        # 3. 어깨 높이
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder']):
            shoulder_center = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
            features['shoulder_height'] = shoulder_center[1]
        else:
            features['shoulder_height'] = 0
        
        # 4. 코(머리) 높이
        features['head_height'] = keypoints['nose'][1] if 'nose' in keypoints else 0
        
        # 5. Bounding Box 계산
        x_coords = [keypoints[k][0] for k in keypoints if not np.isnan(keypoints[k][0])]
        y_coords = [keypoints[k][1] for k in keypoints if not np.isnan(keypoints[k][1])]
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            features['bbox_width'] = bbox_width
            features['bbox_height'] = bbox_height
            features['bbox_aspect_ratio'] = bbox_width / (bbox_height + 1e-6)
        else:
            features['bbox_width'] = 0
            features['bbox_height'] = 0
            features['bbox_aspect_ratio'] = 0
        
        # 6. 신체 기울기 (어깨 라인)
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder']):
            shoulder_diff = keypoints['right_shoulder'] - keypoints['left_shoulder']
            features['shoulder_tilt'] = np.degrees(np.arctan2(shoulder_diff[1], shoulder_diff[0]))
        else:
            features['shoulder_tilt'] = 0
        
        # 7. 평균 confidence
        conf_cols = [col for col in row.index if col.endswith('_conf')]
        features['avg_confidence'] = row[conf_cols].mean()
        
        return features
    
    def extract_dynamic_features(self, df, window_size=5):
        """동적 특징 추출 (시계열)"""
        
        # 속도 계산 (현재 - 이전) / dt
        for kp_name in self.keypoint_names:
            x_col = f'{kp_name}_x'
            y_col = f'{kp_name}_y'
            
            if x_col in df.columns and y_col in df.columns:
                # 속도 (픽셀/프레임)
                df[f'{kp_name}_vx'] = df[x_col].diff().fillna(0)
                df[f'{kp_name}_vy'] = df[y_col].diff().fillna(0)
                df[f'{kp_name}_speed'] = np.sqrt(df[f'{kp_name}_vx']**2 + df[f'{kp_name}_vy']**2)
                
                # 가속도 (속도 변화)
                df[f'{kp_name}_ax'] = df[f'{kp_name}_vx'].diff().fillna(0)
                df[f'{kp_name}_ay'] = df[f'{kp_name}_vy'].diff().fillna(0)
                df[f'{kp_name}_accel'] = np.sqrt(df[f'{kp_name}_ax']**2 + df[f'{kp_name}_ay']**2)
        
        # 신체 중심점 이동
        if 'hip_height' in df.columns:
            df['hip_velocity'] = df['hip_height'].diff().fillna(0)
            df['hip_acceleration'] = df['hip_velocity'].diff().fillna(0)
        
        # Rolling window 통계
        for feature in ['hip_height', 'shoulder_height', 'head_height']:
            if feature in df.columns:
                df[f'{feature}_mean_{window_size}'] = df[feature].rolling(window=window_size, min_periods=1).mean()
                df[f'{feature}_std_{window_size}'] = df[feature].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # 가속도 센서 변화율
        if 'acc_mag' in df.columns:
            df['acc_mag_diff'] = df['acc_mag'].diff().fillna(0)
            df['acc_mag_mean_{}'.format(window_size)] = df['acc_mag'].rolling(window=window_size, min_periods=1).mean()
            df['acc_mag_std_{}'.format(window_size)] = df['acc_mag'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        return df
    
    def process_file(self, skeleton_file):
        """단일 skeleton 파일 처리"""
        
        print(f"\n📊 처리 중: {skeleton_file.name}")
        
        # Skeleton 데이터 로드
        df = pd.read_csv(skeleton_file)
        
        # 정적 특징 추출
        static_features_list = []
        for idx, row in df.iterrows():
            features = self.extract_static_features(row)
            static_features_list.append(features)
        
        static_df = pd.DataFrame(static_features_list)
        
        # 원본 데이터와 병합
        df_combined = pd.concat([df, static_df], axis=1)
        
        # 동적 특징 추출
        df_final = self.extract_dynamic_features(df_combined)
        
        # 저장
        output_file = self.output_dir / skeleton_file.name.replace('-skeleton', '-features')
        df_final.to_csv(output_file, index=False)
        
        print(f"✅ 저장 완료: {output_file}")
        print(f"   총 Features: {len(df_final.columns)}개")
        
        return df_final
    
    def process_all(self):
        """모든 skeleton 파일 처리"""
        
        print("=" * 60)
        print("🔧 Feature Engineering Pipeline")
        print("=" * 60)
        
        skeleton_files = sorted(self.skeleton_dir.glob("fall-*-skeleton.csv"))
        print(f"\n📂 찾은 파일: {len(skeleton_files)}개")
        
        for skeleton_file in tqdm(skeleton_files, desc="Processing files"):
            self.process_file(skeleton_file)
        
        print("\n" + "=" * 60)
        print(f"✅ 완료! 출력 경로: {self.output_dir}")
        print("=" * 60)


def main():
    # 경로 설정
    skeleton_dir = '/home/gjkong/dev_ws/yolo/myproj/skeleton'
    output_dir = '/home/gjkong/dev_ws/yolo/myproj/features'
    
    # Feature Engineering 실행
    engineer = FeatureEngineer(skeleton_dir, output_dir)
    engineer.process_all()


if __name__ == "__main__":
    main()