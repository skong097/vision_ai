"""
실시간 낙상 감지 시스템 (웹캠 기반) - 로깅 기능 추가
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
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


class RealtimeFallDetector:
    """웹캠 기반 실시간 낙상 감지 (로깅 포함)"""
    
    def __init__(self, model_path, model_type='3class', buffer_size=30, log_dir='logs'):
        """
        Args:
            model_path: Random Forest 모델 경로
            model_type: 'binary' or '3class'
            buffer_size: Feature 계산을 위한 프레임 버퍼 크기
            log_dir: 로그 파일 저장 디렉토리
        """
        # 로그 디렉토리 생성
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.setup_logger()
        
        self.logger.info("=" * 60)
        self.logger.info("실시간 낙상 감지 시스템 초기화 시작")
        self.logger.info("=" * 60)
        
        # 모델 로드
        self.logger.info("🤖 모델 로딩 중...")
        try:
            self.rf_model = joblib.load(model_path)
            self.logger.info(f"✅ Random Forest 모델 로드 성공: {model_path}")
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise
        
        try:
            self.yolo_model = YOLO('yolov8n-pose.pt')
            self.logger.info("✅ YOLO Pose 모델 로드 성공")
        except Exception as e:
            self.logger.error(f"❌ YOLO 모델 로드 실패: {e}")
            raise
        
        self.model_type = model_type
        
        # 클래스 정의
        if model_type == 'binary':
            self.class_names = {0: 'Normal', 1: 'Fall'}
            self.class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        else:
            self.class_names = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
            self.class_colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
        
        self.logger.info(f"모델 타입: {model_type}")
        self.logger.info(f"클래스: {self.class_names}")
        
        # Feature 버퍼 (시계열 계산용)
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.logger.info(f"버퍼 크기: {buffer_size} frames")
        
        # 통계
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # 예측 통계
        self.prediction_stats = {label: 0 for label in self.class_names.keys()}
        self.fall_events = []
        
        # Keypoint 이름
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # COCO Skeleton 연결선
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 5), (0, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        
        # 학습 시 사용된 Feature 순서 로드
        self.feature_names = None
        feature_file = Path(model_path).parent / 'feature_columns.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()
                self.feature_names = []
                for line in lines[2:]:
                    if '. ' in line:
                        feature_name = line.strip().split('. ', 1)[1]
                        self.feature_names.append(feature_name)
            self.logger.info(f"✅ Feature 순서 로드: {len(self.feature_names)}개")
        else:
            self.logger.warning(f"⚠️  feature_columns.txt를 찾을 수 없습니다: {feature_file}")
        
        self.logger.info("✅ 초기화 완료!")
        self.logger.info("=" * 60)
    
    def setup_logger(self):
        """로거 설정"""
        # 타임스탬프로 로그 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"fall_detection_{timestamp}.log"
        
        # 로거 생성
        self.logger = logging.getLogger('FallDetector')
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        print(f"📝 로그 파일: {log_file}")
    
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
            self.logger.debug("Keypoint 없음")
            return None
        
        kp = keypoints[0]
        
        # 1. 원본 keypoint 데이터 (17 × 3 = 51개)
        for i, name in enumerate(self.keypoint_names):
            features[f'{name}_x'] = kp[i, 0]
            features[f'{name}_y'] = kp[i, 1]
            features[f'{name}_conf'] = kp[i, 2]
        
        # Keypoint 딕셔너리 생성
        kp_dict = {}
        for i, name in enumerate(self.keypoint_names):
            if kp[i, 2] > 0.3:
                kp_dict[name] = kp[i, :2]
        
        detected_keypoints = len(kp_dict)
        if detected_keypoints < 5:
            self.logger.debug(f"감지된 keypoint 부족: {detected_keypoints}개")
        
        # 2. 관절 각도
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
        
        # 3. 척추 각도
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
        
        # 4. 신체 높이
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
        
        features['head_height'] = kp[np.where(np.array(self.keypoint_names) == 'nose')[0][0], 1]
        
        # 5. Bounding box
        x_coords = [kp[i, 0] for i in range(17) if kp[i, 2] > 0.3]
        y_coords = [kp[i, 1] for i in range(17) if kp[i, 2] > 0.3]
        
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
        
        # 6. 어깨 기울기
        if all(k in kp_dict for k in ['left_shoulder', 'right_shoulder']):
            shoulder_diff = kp_dict['right_shoulder'] - kp_dict['left_shoulder']
            features['shoulder_tilt'] = np.degrees(np.arctan2(shoulder_diff[1], shoulder_diff[0]))
        else:
            features['shoulder_tilt'] = 0
        
        # 7. Confidence
        features['avg_confidence'] = np.mean([kp[i, 2] for i in range(17)])
        
        # 8. 가속도 센서 (웹캠에는 없음)
        features['acc_x'] = 0
        features['acc_y'] = 0
        features['acc_z'] = 0
        features['acc_mag'] = 0
        
        # 🔥 CRITICAL: Keypoint 감지 실패 체크
        # 핵심 feature가 모두 0이면 예측 건너뛰기
        if (features['hip_height'] == 0 and 
            features['spine_angle'] == 0 and 
            features['shoulder_height'] == 0):
            self.logger.warning("⚠️  핵심 Keypoint 감지 실패 - 예측 건너뜀")
            return None
        
        return features
    
    def add_dynamic_features(self, features_list):
        """버퍼의 feature들로부터 동적 특징 계산"""
        if len(features_list) < 2:
            current = features_list[-1].copy()
            
            # 동적 feature 초기화
            for kp_name in self.keypoint_names:
                current[f'{kp_name}_vx'] = 0
                current[f'{kp_name}_vy'] = 0
                current[f'{kp_name}_speed'] = 0
                current[f'{kp_name}_ax'] = 0
                current[f'{kp_name}_ay'] = 0
                current[f'{kp_name}_accel'] = 0
            
            current['hip_velocity'] = 0
            current['hip_acceleration'] = 0
            current['hip_height_mean_5'] = current['hip_height']
            current['hip_height_std_5'] = 0
            current['shoulder_height_mean_5'] = current['shoulder_height']
            current['shoulder_height_std_5'] = 0
            current['head_height_mean_5'] = current['head_height']
            current['head_height_std_5'] = 0
            current['acc_mag_diff'] = 0
            current['acc_mag_mean_5'] = 0
            current['acc_mag_std_5'] = 0
            
            return current
        
        current = features_list[-1].copy()
        prev = features_list[-2]
        
        # 1. 각 keypoint의 속도 및 가속도
        for kp_name in self.keypoint_names:
            x_col = f'{kp_name}_x'
            y_col = f'{kp_name}_y'
            
            vx = current[x_col] - prev[x_col]
            vy = current[y_col] - prev[y_col]
            speed = np.sqrt(vx**2 + vy**2)
            
            current[f'{kp_name}_vx'] = vx
            current[f'{kp_name}_vy'] = vy
            current[f'{kp_name}_speed'] = speed
            
            if len(features_list) >= 3:
                prev2 = features_list[-3]
                prev_vx = prev[x_col] - prev2[x_col]
                prev_vy = prev[y_col] - prev2[y_col]
                
                ax = vx - prev_vx
                ay = vy - prev_vy
                accel = np.sqrt(ax**2 + ay**2)
                
                current[f'{kp_name}_ax'] = ax
                current[f'{kp_name}_ay'] = ay
                current[f'{kp_name}_accel'] = accel
            else:
                current[f'{kp_name}_ax'] = 0
                current[f'{kp_name}_ay'] = 0
                current[f'{kp_name}_accel'] = 0
        
        # 2. 신체 중심점 이동
        current['hip_velocity'] = current['hip_height'] - prev['hip_height']
        
        if len(features_list) >= 3:
            prev2 = features_list[-3]
            prev_velocity = prev['hip_height'] - prev2['hip_height']
            current['hip_acceleration'] = current['hip_velocity'] - prev_velocity
        else:
            current['hip_acceleration'] = 0
        
        # 3. Rolling window 통계
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
        
        # 4. 가속도 센서 변화
        current['acc_mag_diff'] = 0
        current['acc_mag_mean_5'] = 0
        current['acc_mag_std_5'] = 0
        
        return current
    
    def predict(self, features):
        """Feature로부터 예측"""
        df = pd.DataFrame([features])
        df = df.fillna(0)
        
        # 학습 시 사용한 feature 순서에 맞춰 정렬
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        try:
            prediction = self.rf_model.predict(df)[0]
            proba = self.rf_model.predict_proba(df)[0]
            
            # 로그 기록 (100 프레임마다)
            if self.frame_count % 100 == 0:
                self.logger.info(
                    f"Frame {self.frame_count}: {self.class_names[prediction]} "
                    f"(확률: {proba[prediction]*100:.1f}%)"
                )
            
            return prediction, proba
        except Exception as e:
            self.logger.error(f"예측 오류: {e}")
            if self.feature_names:
                self.logger.error(f"예상 Features: {len(self.feature_names)}")
                self.logger.error(f"실제 Features: {len(df.columns)}")
            return 0, [1.0] + [0.0] * (len(self.class_names) - 1)
    
    def log_fall_event(self, prediction, proba, features):
        """낙상 이벤트 로그"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'frame': self.frame_count,
            'prediction': self.class_names[prediction],
            'probability': float(proba[prediction]),
            'hip_height': float(features.get('hip_height', 0)),
            'spine_angle': float(features.get('spine_angle', 0)),
            'hip_velocity': float(features.get('hip_velocity', 0))
        }
        
        self.fall_events.append(event)
        self.logger.warning(f"🚨 낙상 감지! {json.dumps(event, indent=2)}")
    
    def draw_skeleton(self, frame, keypoints):
        """프레임에 skeleton 그리기"""
        if len(keypoints) == 0:
            return frame
        
        kp = keypoints[0]
        
        for i in range(17):
            if kp[i, 2] > 0.3:
                x, y = int(kp[i, 0]), int(kp[i, 1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
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
    
    def save_statistics(self):
        """통계 저장"""
        stats = {
            'total_frames': self.frame_count,
            'prediction_stats': {
                self.class_names[k]: v for k, v in self.prediction_stats.items()
            },
            'fall_events': self.fall_events,
            'avg_fps': self.fps
        }
        
        stats_file = self.log_dir / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 통계 저장: {stats_file}")
        return stats
    
    def run(self, camera_id=0, save_video=False, output_path='output.mp4'):
        """실시간 낙상 감지 실행"""
        
        self.logger.info("=" * 60)
        self.logger.info("🎥 실시간 낙상 감지 시작")
        self.logger.info(f"카메라 ID: {camera_id}")
        self.logger.info(f"비디오 저장: {save_video}")
        self.logger.info("=" * 60)
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error("❌ 웹캠을 열 수 없습니다!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"해상도: {width}x{height}")
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            self.logger.info(f"녹화 시작: {output_path}")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("프레임 읽기 실패")
                        break
                    
                    # FPS 계산
                    current_time = time.time()
                    self.fps = 1.0 / (current_time - self.last_time + 1e-6)
                    self.last_time = current_time
                    
                    # YOLO Pose 추론
                    results = self.yolo_model(frame, verbose=False)
                    
                    if len(results) > 0 and results[0].keypoints is not None:
                        keypoints = results[0].keypoints.data.cpu().numpy()
                        
                        if len(keypoints) > 0:
                            frame = self.draw_skeleton(frame, keypoints)
                            
                            features = self.extract_features_from_keypoints(keypoints)
                            
                            if features is not None:
                                self.frame_buffer.append(features)
                                
                                if len(self.frame_buffer) >= 2:
                                    features_with_dynamic = self.add_dynamic_features(list(self.frame_buffer))
                                    
                                    prediction, proba = self.predict(features_with_dynamic)
                                    
                                    # 통계 업데이트
                                    self.prediction_stats[prediction] += 1
                                    
                                    # 낙상 이벤트 로그
                                    if prediction > 0:
                                        self.log_fall_event(prediction, proba, features_with_dynamic)
                                    
                                    frame = self.draw_info(frame, prediction, proba, features_with_dynamic)
                                    
                                    if prediction > 0:
                                        cv2.putText(frame, "!!! FALL DETECTED !!!", (50, height - 50),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    self.frame_count += 1
                    cv2.imshow('Fall Detection System', frame)
                    
                    if save_video:
                        out.write(frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("사용자 종료 요청")
                    break
                elif key == ord('p'):
                    paused = not paused
                    self.logger.info("⏸️  일시정지" if paused else "▶️  재생")
        
        except Exception as e:
            self.logger.error(f"실행 중 오류: {e}", exc_info=True)
        
        finally:
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            
            # 최종 통계 저장
            stats = self.save_statistics()
            
            self.logger.info("=" * 60)
            self.logger.info(f"✅ 종료! 총 프레임: {self.frame_count}")
            self.logger.info(f"예측 통계: {stats['prediction_stats']}")
            self.logger.info(f"낙상 이벤트: {len(self.fall_events)}회")
            self.logger.info("=" * 60)


def main():
    """메인 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 낙상 감지 시스템 (로깅 포함)')
    parser.add_argument('--model-type', type=str, default='3class', choices=['binary', '3class'],
                       help='모델 타입')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID')
    parser.add_argument('--save', action='store_true',
                       help='비디오 저장')
    parser.add_argument('--output', type=str, default='fall_detection_output.mp4',
                       help='출력 비디오 경로')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='로그 디렉토리')
    
    args = parser.parse_args()
    
    base_path = Path('/home/gjkong/dev_ws/yolo/myproj/models')
    model_path = base_path / args.model_type / 'random_forest_model.pkl'
    
    if not model_path.exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    detector = RealtimeFallDetector(model_path, model_type=args.model_type, log_dir=args.log_dir)
    detector.run(camera_id=args.camera, save_video=args.save, output_path=args.output)


if __name__ == "__main__":
    main()
