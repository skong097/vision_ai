"""
실시간 낙상 감지 시스템 (웹캠 기반) - 물리 필터 강화 버전
Author: Home Safe Solution Team
Date: 2026-02-01
"""

import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')


class RealtimeFallDetector:
    """웹캠 기반 실시간 낙상 감지"""
    
    def __init__(self, model_path, model_type='3class', buffer_size=30):
        """
        Args:
            model_path: Random Forest 모델 경로
            model_type: 'binary' or '3class'
            buffer_size: Feature 계산을 위한 프레임 버퍼 크기
        """
        # 모델 로드
        print("🤖 모델 로딩 중...")
        self.rf_model = joblib.load(model_path)
        self.yolo_model = YOLO('yolo11s-pose.pt')
        # self.yolo_model = YOLO('yolov8n-pose.pt')
        self.model_type = model_type
        
        # 클래스 정의
        if model_type == 'binary':
            self.class_names = {0: 'Normal', 1: 'Fall'}
            self.class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        else:
            self.class_names = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
            self.class_colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
        
        # Feature 버퍼 (시계열 계산용)
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # 통계
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Keypoint 이름
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # COCO Skeleton 연결선
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (0, 5), (0, 6),  # 코 -> 어깨
            (5, 7), (7, 9),  # 왼팔
            (6, 8), (8, 10), # 오른팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15),  # 왼다리
            (12, 14), (14, 16)   # 오른다리
        ]
        
        # 학습 시 사용된 Feature 순서 로드 (CRITICAL!)
        self.feature_names = None
        feature_file = Path(model_path).parent / 'feature_columns.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()
                self.feature_names = []
                for line in lines[2:]:  # 헤더 2줄 스킵
                    if '. ' in line:
                        feature_name = line.strip().split('. ', 1)[1]
                        self.feature_names.append(feature_name)
            print(f"✅ Feature 순서 로드: {len(self.feature_names)}개")
        else:
            print("⚠️  경고: feature_columns.txt를 찾을 수 없습니다!")
        
        print("✅ 모델 로드 완료!")
    
    def calculate_angle(self, p1, p2, p3):
        """3개 점으로 각도 계산"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def extract_features_from_keypoints(self, keypoints):
        """Keypoint에서 정적 특징 추출"""
        features = {}
        if len(keypoints) == 0: return None
        
        kp = keypoints[0]
        for i, name in enumerate(self.keypoint_names):
            features[f'{name}_x'] = kp[i, 0]
            features[f'{name}_y'] = kp[i, 1]
            features[f'{name}_conf'] = kp[i, 2]
        
        kp_dict = {name: kp[i, :2] for i, name in enumerate(self.keypoint_names) if kp[i, 2] > 0.3}
        
        # 관절 및 척추 각도 계산
        for side in ['left', 'right']:
            for joint in ['elbow', 'knee']:
                pts = [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist'] if joint == 'elbow' else [f'{side}_hip', f'{side}_knee', f'{side}_ankle']
                features[f'{side}_{joint}_angle'] = self.calculate_angle(kp_dict[pts[0]], kp_dict[pts[1]], kp_dict[pts[2]]) if all(k in kp_dict for k in pts) else 0

        if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            spine_vector = ((kp_dict['left_hip'] + kp_dict['right_hip'])/2) - ((kp_dict['left_shoulder'] + kp_dict['right_shoulder'])/2)
            features['spine_angle'] = np.degrees(np.arccos(np.clip(np.dot(spine_vector, np.array([0, 1])) / (np.linalg.norm(spine_vector) + 1e-6), -1.0, 1.0)))
        else: features['spine_angle'] = 0
        
        # 신체 높이 및 BBox
        features['hip_height'] = (kp_dict['left_hip'][1] + kp_dict['right_hip'][1])/2 if all(k in kp_dict for k in ['left_hip', 'right_hip']) else 0
        features['shoulder_height'] = (kp_dict['left_shoulder'][1] + kp_dict['right_shoulder'][1])/2 if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder']) else 0
        features['head_height'] = kp[0, 1]
        
        x_coords = [kp[i, 0] for i in range(17) if kp[i, 2] > 0.3]
        y_coords = [kp[i, 1] for i in range(17) if kp[i, 2] > 0.3]
        if x_coords:
            bw, bh = max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)
            features.update({'bbox_width': bw, 'bbox_height': bh, 'bbox_aspect_ratio': bw/(bh+1e-6)})
        else: features.update({'bbox_width': 0, 'bbox_height': 0, 'bbox_aspect_ratio': 0})
        
        features['avg_confidence'] = np.mean(kp[:, 2])
        features.update({'acc_x': 0, 'acc_y': 0, 'acc_z': 0, 'acc_mag': 0}) # 실시간은 0 처리
        return features
    
    def add_dynamic_features(self, features_list):
        """동적 특징 계산 (속도, 가속도, 이동 평균)"""
        current = features_list[-1].copy()
        if len(features_list) < 2:
            for kn in self.keypoint_names: current.update({f'{kn}_vx':0, f'{kn}_vy':0, f'{kn}_speed':0, f'{kn}_ax':0, f'{kn}_ay':0, f'{kn}_accel':0})
            current.update({'hip_velocity':0, 'hip_acceleration':0, 'hip_height_mean_5':current['hip_height'], 'hip_height_std_5':0})
            return current
        
        prev = features_list[-2]
        for kn in self.keypoint_names:
            vx, vy = current[f'{kn}_x'] - prev[f'{kn}_x'], current[f'{kn}_y'] - prev[f'{kn}_y']
            current.update({f'{kn}_vx': vx, f'{kn}_vy': vy, f'{kn}_speed': np.sqrt(vx**2 + vy**2)})
            if len(features_list) >= 3:
                prev2 = features_list[-3]
                ax, ay = vx - (prev[f'{kn}_x'] - prev2[f'{kn}_x']), vy - (prev[f'{kn}_y'] - prev2[f'{kn}_y'])
                current.update({f'{kn}_ax': ax, f'{kn}_ay': ay, f'{kn}_accel': np.sqrt(ax**2 + ay**2)})
        
        current['hip_velocity'] = current['hip_height'] - prev['hip_height']
        window = features_list[-min(5, len(features_list)):]
        current['hip_height_mean_5'] = np.mean([f['hip_height'] for f in window])
        return current
    
    def predict(self, features):
        """RF 모델 예측"""
        df = pd.DataFrame([features]).fillna(0)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns: df[col] = 0
            df = df[self.feature_names]
        return self.rf_model.predict(df)[0], self.rf_model.predict_proba(df)[0]
    
    def draw_skeleton(self, frame, keypoints):
        """스켈레톤 시각화"""
        kp = keypoints[0]
        for i in range(17):
            if kp[i, 2] > 0.3: cv2.circle(frame, (int(kp[i,0]), int(kp[i,1])), 5, (0, 255, 0), -1)
        for s, e in self.skeleton_connections:
            if kp[s, 2] > 0.3 and kp[e, 2] > 0.3: cv2.line(frame, (int(kp[s,0]), int(kp[s,1])), (int(kp[e,0]), int(kp[e,1])), (255, 0, 0), 2)
        return frame

    def draw_info(self, frame, prediction, proba, features):
        """정보 오버레이"""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {self.class_names[prediction]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[prediction], 2)
        cv2.putText(frame, f"Spine: {features.get('spine_angle',0):.1f} | Ratio: {features.get('bbox_aspect_ratio',0):.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"V-Vel: {features.get('hip_velocity',0):.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return frame

    def run(self, camera_id=0):
        """실시간 감지 루프 (물리 필터 적용)"""
        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self.fps = 1.0 / (time.time() - self.last_time + 1e-6)
            self.last_time = time.time()
            
            results = self.yolo_model(frame, verbose=False)
            if results and results[0].keypoints is not None:
                kp_data = results[0].keypoints.data.cpu().numpy()
                if len(kp_data) > 0:
                    frame = self.draw_skeleton(frame, kp_data)
                    features = self.extract_features_from_keypoints(kp_data)
                    if features:
                        self.frame_buffer.append(features)
                        if len(self.frame_buffer) >= 2:
                            dyn_feat = self.add_dynamic_features(list(self.frame_buffer))
                            pred, prob = self.predict(dyn_feat)
                            
                            # --- [CRITICAL] 3중 물리 필터 로직 추가 ---
                            # 1. 높이 급락 (수직 속도), 2. 형태 변화 (종횡비), 3. 각도 변화 (척추 기울기)
                            #
                            h_vel = dyn_feat.get('hip_velocity', 8.0)
                            a_ratio = dyn_feat.get('bbox_aspect_ratio', 0)
                            s_angle = dyn_feat.get('spine_angle', 0)
                            hip_y = dyn_feat.get('hip_height', 0)
                            
                            # 조건을 모두 충족할 때만 낙상으로 확정 (앉기 오탐지 방지)
                            # 1. 속도 10 이상, 2. 가로비율 1.5 이상, 3. 척추각도 75도 이상, 4. 엉덩이 위치 하단(350px 이상)
# 네 가지를 동시에 만족할 때만 낙상으로 인정
                            is_physical_fall = (h_vel > 15.0) and (a_ratio > 1.2) and (s_angle > 75.0) and (hip_y > 300)
                            
                            if pred > 0 and not is_physical_fall:
                                pred = 0 # 조건 미달 시 Normal로 변경
                            
                            frame = self.draw_info(frame, pred, prob, dyn_feat)
                            if pred > 0:
                                cv2.putText(frame, "!!! FALL DETECTED !!!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            cv2.imshow('Home Safe Solution', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 경로 설정 및 실행
    m_path = Path('/home/gjkong/dev_ws/yolo/myproj/models/3class/random_forest_model.pkl')
    if m_path.exists():
        detector = RealtimeFallDetector(str(m_path))
        detector.run()
    else:
        print(f"모델을 찾을 수 없습니다: {m_path}")