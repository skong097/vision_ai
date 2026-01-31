"""
실시간 낙상 감지 시스템 (웹캠 기반)
Author: Home Safe Solution Team
Date: 2026-01-28
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
        self.yolo_model = YOLO('yolov8n-pose.pt')
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
        
        print("✅ 모델 로드 완료!")
        print(f"   모델 타입: {model_type}")
        print(f"   버퍼 크기: {buffer_size} frames")
    
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
        
        if len(keypoints) == 0:
            return None
        
        # 첫 번째 사람만 사용
        kp = keypoints[0]  # shape: (17, 3)
        
        # Keypoint 딕셔너리 생성
        kp_dict = {}
        for i, name in enumerate(self.keypoint_names):
            if kp[i, 2] > 0.3:  # confidence threshold
                kp_dict[name] = kp[i, :2]
        
        # 필수 keypoint가 없으면 None 반환
        if len(kp_dict) < 5:
            return None
        
        # 1. 관절 각도
        if all(k in kp_dict for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            features['left_elbow_angle'] = self.calculate_angle(
                kp_dict['left_shoulder'], kp_dict['left_elbow'], kp_dict['left_wrist']
            )
        else:
            features['left_elbow_angle'] = 0
        
        if all(k in kp_dict for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            features['right_elbow_angle'] = self.calculate_angle(
                kp_dict['right_shoulder'], kp_dict['right_elbow'], kp_dict['right_wrist']
            )
        else:
            features['right_elbow_angle'] = 0
        
        if all(k in kp_dict for k in ['left_hip', 'left_knee', 'left_ankle']):
            features['left_knee_angle'] = self.calculate_angle(
                kp_dict['left_hip'], kp_dict['left_knee'], kp_dict['left_ankle']
            )
        else:
            features['left_knee_angle'] = 0
        
        if all(k in kp_dict for k in ['right_hip', 'right_knee', 'right_ankle']):
            features['right_knee_angle'] = self.calculate_angle(
                kp_dict['right_hip'], kp_dict['right_knee'], kp_dict['right_ankle']
            )
        else:
            features['right_knee_angle'] = 0
        
        # 2. 척추 각도
        if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_center = (kp_dict['left_shoulder'] + kp_dict['right_shoulder']) / 2
            hip_center = (kp_dict['left_hip'] + kp_dict['right_hip']) / 2
            spine_vector = hip_center - shoulder_center
            vertical = np.array([0, 1])
            cos_angle = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            features['spine_angle'] = np.degrees(np.arccos(cos_angle))
        else:
            features['spine_angle'] = 0
        
        # 3. 신체 높이
        if all(k in kp_dict for k in ['left_hip', 'right_hip']):
            hip_center = (kp_dict['left_hip'] + kp_dict['right_hip']) / 2
            features['hip_height'] = hip_center[1]
        else:
            features['hip_height'] = 0
        
        if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder']):
            shoulder_center = (kp_dict['left_shoulder'] + kp_dict['right_shoulder']) / 2
            features['shoulder_height'] = shoulder_center[1]
        else:
            features['shoulder_height'] = 0
        
        features['head_height'] = kp_dict['nose'][1] if 'nose' in kp_dict else 0
        
        # 4. Bounding box
        x_coords = [kp_dict[k][0] for k in kp_dict]
        y_coords = [kp_dict[k][1] for k in kp_dict]
        
        if len(x_coords) > 0:
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            features['bbox_width'] = bbox_width
            features['bbox_height'] = bbox_height
            features['bbox_aspect_ratio'] = bbox_width / (bbox_height + 1e-6)
        else:
            features['bbox_width'] = 0
            features['bbox_height'] = 0
            features['bbox_aspect_ratio'] = 0
        
        # 5. 어깨 기울기
        if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder']):
            shoulder_diff = kp_dict['right_shoulder'] - kp_dict['left_shoulder']
            features['shoulder_tilt'] = np.degrees(np.arctan2(shoulder_diff[1], shoulder_diff[0]))
        else:
            features['shoulder_tilt'] = 0
        
        # 6. Confidence
        features['avg_confidence'] = np.mean([kp[i, 2] for i in range(17)])
        
        # 7. 가속도 센서 (웹캠에는 없으므로 0으로 설정)
        features['acc_x'] = 0
        features['acc_y'] = 0
        features['acc_z'] = 0
        features['acc_mag'] = 0
        
        return features
    
    def add_dynamic_features(self, features_list):
        """버퍼의 feature들로부터 동적 특징 계산"""
        if len(features_list) < 2:
            return features_list[-1]  # 동적 feature 없이 반환
        
        current = features_list[-1].copy()
        
        # 속도 계산 (현재 - 이전)
        if len(features_list) >= 2:
            prev = features_list[-2]
            current['hip_velocity'] = current['hip_height'] - prev['hip_height']
        else:
            current['hip_velocity'] = 0
        
        # 가속도 계산
        if len(features_list) >= 3:
            prev2 = features_list[-3]
            prev_velocity = prev['hip_height'] - prev2['hip_height']
            current['hip_acceleration'] = current['hip_velocity'] - prev_velocity
        else:
            current['hip_acceleration'] = 0
        
        # Rolling statistics (간단 버전)
        window = min(5, len(features_list))
        recent = features_list[-window:]
        
        hip_heights = [f['hip_height'] for f in recent]
        current['hip_height_mean_5'] = np.mean(hip_heights)
        current['hip_height_std_5'] = np.std(hip_heights) if len(hip_heights) > 1 else 0
        
        shoulder_heights = [f['shoulder_height'] for f in recent]
        current['shoulder_height_mean_5'] = np.mean(shoulder_heights)
        current['shoulder_height_std_5'] = np.std(shoulder_heights) if len(shoulder_heights) > 1 else 0
        
        head_heights = [f['head_height'] for f in recent]
        current['head_height_mean_5'] = np.mean(head_heights)
        current['head_height_std_5'] = np.std(head_heights) if len(head_heights) > 1 else 0
        
        # 가속도 센서 관련 (모두 0)
        current['acc_mag_diff'] = 0
        current['acc_mag_mean_5'] = 0
        current['acc_mag_std_5'] = 0
        
        return current
    
    def predict(self, features):
        """Feature로부터 예측"""
        # DataFrame으로 변환
        df = pd.DataFrame([features])
        
        # NaN 처리
        df = df.fillna(0)
        
        # 예측
        try:
            prediction = self.rf_model.predict(df)[0]
            proba = self.rf_model.predict_proba(df)[0]
            return prediction, proba
        except Exception as e:
            print(f"⚠️  예측 오류: {e}")
            return 0, [1.0, 0.0]
    
    def draw_skeleton(self, frame, keypoints):
        """프레임에 skeleton 그리기"""
        if len(keypoints) == 0:
            return frame
        
        kp = keypoints[0]
        
        # Keypoint 그리기
        for i in range(17):
            if kp[i, 2] > 0.3:
                x, y = int(kp[i, 0]), int(kp[i, 1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # 연결선 그리기
        for start_idx, end_idx in self.skeleton_connections:
            if kp[start_idx, 2] > 0.3 and kp[end_idx, 2] > 0.3:
                start_point = (int(kp[start_idx, 0]), int(kp[start_idx, 1]))
                end_point = (int(kp[end_idx, 0]), int(kp[end_idx, 1]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        
        return frame
    
    def draw_info(self, frame, prediction, proba, features):
        """프레임에 정보 오버레이"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 예측 결과
        class_name = self.class_names[prediction]
        color = self.class_colors[prediction]
        
        cv2.putText(frame, f"Status: {class_name}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 확률
        y_offset = 90
        for i, prob in enumerate(proba):
            class_text = self.class_names.get(i, f"Class {i}")
            cv2.putText(frame, f"{class_text}: {prob*100:.1f}%", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # 주요 Feature
        if features:
            y_offset += 10
            cv2.putText(frame, f"Hip Height: {features.get('hip_height', 0):.1f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(frame, f"Spine Angle: {features.get('spine_angle', 0):.1f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 프레임 수
        cv2.putText(frame, f"Frame: {self.frame_count}", (w - 150, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0, save_video=False, output_path='output.mp4'):
        """실시간 낙상 감지 실행"""
        
        print("\n" + "=" * 60)
        print("🎥 실시간 낙상 감지 시작")
        print("=" * 60)
        print("💡 종료: 'q' 키")
        print("💡 일시정지: 'p' 키")
        print("=" * 60 + "\n")
        
        # 웹캠 열기
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 오류: 웹캠을 열 수 없습니다!")
            return
        
        # 비디오 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 녹화 설정
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # FPS 계산
                    current_time = time.time()
                    self.fps = 1.0 / (current_time - self.last_time + 1e-6)
                    self.last_time = current_time
                    
                    # YOLO Pose 추론
                    results = self.yolo_model(frame, verbose=False)
                    
                    # Keypoint 추출
                    if len(results) > 0 and results[0].keypoints is not None:
                        keypoints = results[0].keypoints.data.cpu().numpy()
                        
                        if len(keypoints) > 0:
                            # Skeleton 그리기
                            frame = self.draw_skeleton(frame, keypoints)
                            
                            # Feature 추출
                            features = self.extract_features_from_keypoints(keypoints)
                            
                            if features is not None:
                                # 버퍼에 추가
                                self.frame_buffer.append(features)
                                
                                # 동적 feature 추가
                                if len(self.frame_buffer) >= 2:
                                    features_with_dynamic = self.add_dynamic_features(list(self.frame_buffer))
                                    
                                    # 예측
                                    prediction, proba = self.predict(features_with_dynamic)
                                    
                                    # 정보 오버레이
                                    frame = self.draw_info(frame, prediction, proba, features_with_dynamic)
                                    
                                    # 알람 (낙상 감지 시)
                                    if prediction > 0:  # Binary: 1=Fall, 3-Class: 1=Falling, 2=Fallen
                                        cv2.putText(frame, "!!! FALL DETECTED !!!", (50, height - 50),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    self.frame_count += 1
                    
                    # 화면 출력
                    cv2.imshow('Fall Detection System', frame)
                    
                    # 녹화
                    if save_video:
                        out.write(frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️  일시정지" if paused else "▶️  재생")
        
        finally:
            # 정리
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print(f"✅ 종료! 총 프레임: {self.frame_count}")
            if save_video:
                print(f"💾 저장: {output_path}")
            print("=" * 60)


def main():
    """메인 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 낙상 감지 시스템')
    parser.add_argument('--model-type', type=str, default='3class', choices=['binary', '3class'],
                       help='모델 타입 (binary or 3class)')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    parser.add_argument('--save', action='store_true',
                       help='비디오 저장 여부')
    parser.add_argument('--output', type=str, default='fall_detection_output.mp4',
                       help='출력 비디오 경로')
    
    args = parser.parse_args()
    
    # 모델 경로 설정
    base_path = Path('/home/gjkong/dev_ws/yolo/myproj/models')
    model_path = base_path / args.model_type / 'random_forest_model.pkl'
    
    if not model_path.exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 낙상 감지기 실행
    detector = RealtimeFallDetector(model_path, model_type=args.model_type)
    detector.run(camera_id=args.camera, save_video=args.save, output_path=args.output)


if __name__ == "__main__":
    main()
