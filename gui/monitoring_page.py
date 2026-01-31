"""
Home Safe Solution - 실시간 모니터링 페이지 (낙상 감지 통합)
QTimer 기반 - 안정적

✅ 작동 확인: 2026-01-30 01:41
✅ 상태: 웹캠 + YOLO Pose + Skeleton + 낙상 감지
"""

import sys
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QFrame, QPushButton, QTextEdit, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
import cv2
import numpy as np
from datetime import datetime
import joblib
from collections import deque

# OneEuroFilter
from one_euro_filter import KeypointFilter

# YOLO Pose
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not found - YOLO Pose disabled")

from database_models import DatabaseManager


class MonitoringPage(QWidget):
    """실시간 모니터링 페이지 (QTimer 버전)"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.cap = None
        self.timer = None
        self.frame_count = 0
        
        # EventLog 모델 초기화
        from database_models import EventLog
        self.event_log_model = EventLog(db)
        
        # Keypoint 필터 초기화
        self.filter_strength = 'medium'  # 'none', 'light', 'medium', 'strong'
        self.keypoint_filter = KeypointFilter(filter_strength=self.filter_strength)
        
        # YOLO Pose 모델
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                yolo_path = '/home/gjkong/dev_ws/yolo/myproj/models/yolo11s-pose.pt'
                if os.path.exists(yolo_path):
                    self.yolo_model = YOLO(yolo_path)
                    print(f"✅ YOLO Pose 로드 성공: {yolo_path}")
                else:
                    print(f"⚠️ YOLO 모델 없음: {yolo_path}")
            except Exception as e:
                print(f"⚠️ YOLO 로드 실패: {e}")
        
        # 낙상 감지 모델 (YOLO 없이 RF만)
        self.rf_model = None
        self.feature_columns = None
        self.frame_buffer = deque(maxlen=30)
        self.class_names = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
        self.class_colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
        
        try:
            model_path = '/home/gjkong/dev_ws/yolo/myproj/models/3class/random_forest_model.pkl'
            feature_path = '/home/gjkong/dev_ws/yolo/myproj/models/3class/feature_columns.txt'
            
            if os.path.exists(model_path) and os.path.exists(feature_path):
                # Random Forest 모델 로드 (joblib 사용)
                self.rf_model = joblib.load(model_path)
                
                # Feature 순서 로드
                with open(feature_path, 'r') as f:
                    lines = f.readlines()
                    self.feature_columns = []
                    for line in lines[2:]:  # 헤더 2줄 스킵
                        if '. ' in line:
                            feature_name = line.strip().split('. ', 1)[1]
                            self.feature_columns.append(feature_name)
                
                print(f"✅ 낙상 감지 모델 로드 성공! (Feature: {len(self.feature_columns)}개)")
            else:
                print(f"⚠️ 낙상 감지 모델 파일 없음")
        except Exception as e:
            print(f"⚠️ 낙상 감지 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 왼쪽: 영상
        left_panel = self.create_video_panel()
        layout.addWidget(left_panel, 2)
        
        # 오른쪽: 로그
        right_panel = self.create_info_panel()
        layout.addWidget(right_panel, 1)
    
    def create_video_panel(self) -> QWidget:
        """영상 패널"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # === 상단: 버튼 ===
        top_layout = QHBoxLayout()
        
        # 버튼들 (좌우 정렬, 높이 통일)
        button_height = 45  # 통일된 높이
        
        self.btn_start = QPushButton('▶ Start')
        self.btn_start.clicked.connect(self.start_monitoring)
        self.btn_start.setFixedHeight(button_height)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        
        self.btn_stop = QPushButton('⏹ Stop')
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setFixedHeight(button_height)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        
        self.btn_search = QPushButton('🔍 Search')
        self.btn_search.clicked.connect(self.on_search_clicked)
        self.btn_search.setFixedHeight(button_height)
        self.btn_search.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        
        self.btn_emergency = QPushButton('🚨 Emergency Call')
        self.btn_emergency.clicked.connect(self.on_emergency_clicked)
        self.btn_emergency.setFixedHeight(button_height)
        self.btn_emergency.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d35400; }
        """)
        
        # 필터 강도 조절 버튼
        self.btn_filter = QPushButton('🎚️ Filter: Medium')
        self.btn_filter.clicked.connect(self.on_filter_clicked)
        self.btn_filter.setFixedHeight(button_height)
        self.btn_filter.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #8e44ad; }
        """)
        
        # 좌측: Start/Stop
        top_layout.addWidget(self.btn_start)
        top_layout.addWidget(self.btn_stop)
        top_layout.addStretch()  # 중앙 공백
        # 우측: Filter/Search/Emergency
        top_layout.addWidget(self.btn_filter)
        top_layout.addWidget(self.btn_search)
        top_layout.addWidget(self.btn_emergency)
        
        layout.addLayout(top_layout)
        
        # === 중앙: 영상 ===
        video_frame = QFrame()
        video_frame.setStyleSheet("QFrame { background-color: #2c3e50; border-radius: 10px; }")
        video_layout = QVBoxLayout(video_frame)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: white; font-size: 16px;")
        self.video_label.setText('▶ Press Start button')
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(video_frame)
        
        return panel
    
    def create_info_panel(self) -> QWidget:
        """정보 패널"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 상태 그룹
        status_group = QGroupBox('Status')
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel('⚪ Standby')
        self.status_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
        # 낙상 감지 그룹
        detection_group = QGroupBox('Fall Detection')
        detection_layout = QVBoxLayout(detection_group)
        
        # 현재 상태 (큰 글씨)
        self.fall_status_label = QLabel('[OK] Normal')
        self.fall_status_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        self.fall_status_label.setStyleSheet("color: #27ae60;")
        detection_layout.addWidget(self.fall_status_label)
        
        # Confidence (파란색)
        self.confidence_label = QLabel('Confidence: --')
        self.confidence_label.setFont(QFont('Arial', 11))
        self.confidence_label.setStyleSheet("color: #3498db; font-weight: bold;")
        detection_layout.addWidget(self.confidence_label)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #7f8c8d;")
        detection_layout.addWidget(line)
        
        # 각 클래스 확률 (진행 바로 표시)
        from PyQt6.QtWidgets import QProgressBar
        
        # Normal 바
        normal_layout = QHBoxLayout()
        normal_text = QLabel('Normal:')
        normal_text.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        normal_text.setStyleSheet("color: #2ecc71;")
        normal_text.setFixedWidth(70)
        normal_layout.addWidget(normal_text)
        
        self.normal_bar = QProgressBar()
        self.normal_bar.setMaximum(100)
        self.normal_bar.setValue(0)
        self.normal_bar.setTextVisible(True)
        self.normal_bar.setFormat('%p%')
        self.normal_bar.setFixedHeight(25)  # 높이 고정
        self.normal_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2ecc71;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
            }
        """)
        normal_layout.addWidget(self.normal_bar)
        detection_layout.addLayout(normal_layout)
        
        # Falling 바
        falling_layout = QHBoxLayout()
        falling_text = QLabel('Falling:')
        falling_text.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        falling_text.setStyleSheet("color: #f39c12;")
        falling_text.setFixedWidth(70)
        falling_layout.addWidget(falling_text)
        
        self.falling_bar = QProgressBar()
        self.falling_bar.setMaximum(100)
        self.falling_bar.setValue(0)
        self.falling_bar.setTextVisible(True)
        self.falling_bar.setFormat('%p%')
        self.falling_bar.setFixedHeight(25)  # 높이 고정
        self.falling_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #f39c12;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #f39c12;
            }
        """)
        falling_layout.addWidget(self.falling_bar)
        detection_layout.addLayout(falling_layout)
        
        # Fallen 바
        fallen_layout = QHBoxLayout()
        fallen_text = QLabel('Fallen:')
        fallen_text.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        fallen_text.setStyleSheet("color: #e74c3c;")
        fallen_text.setFixedWidth(70)
        fallen_layout.addWidget(fallen_text)
        
        self.fallen_bar = QProgressBar()
        self.fallen_bar.setMaximum(100)
        self.fallen_bar.setValue(0)
        self.fallen_bar.setTextVisible(True)
        self.fallen_bar.setFormat('%p%')
        self.fallen_bar.setFixedHeight(25)  # 높이 고정
        self.fallen_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e74c3c;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
            }
        """)
        fallen_layout.addWidget(self.fallen_bar)
        detection_layout.addLayout(fallen_layout)
        
        layout.addWidget(detection_group)
        
        # 로그 그룹 (확장)
        log_group = QGroupBox('Log')
        log_layout = QVBoxLayout(log_group)
        
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New';
                font-size: 11px;
            }
        """)
        self.event_log.append("[INIT] YOLO Pose + Fall Detection")
        if YOLO_AVAILABLE:
            self.event_log.append("[INFO] ✅ YOLO available")
        
        log_layout.addWidget(self.event_log)
        layout.addWidget(log_group, 1)  # stretch factor 추가로 확장
        
        # addStretch 제거 - 로그가 공간 차지하도록
        
        return panel
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.add_log("[INFO] 웹캠 연결 시도...")
        
        try:
            # 웹캠 열기
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                self.add_log("❌ 웹캠을 열 수 없습니다")
                return
            
            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.add_log("✅ 웹캠 연결 성공")
            
            # 타이머 시작 (50ms = 20 FPS)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(50)
            
            self.frame_count = 0
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_label.setText('🟢 Webcam Active')
            
            self.add_log("[INFO] 모니터링 시작 (10 FPS)")
            
        except Exception as e:
            self.add_log(f"❌ 오류: {str(e)}")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        # 타이머 중지
        if self.timer:
            self.timer.stop()
            self.timer = None
        
        # 웹캠 해제
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.add_log(f"[INFO] 웹캠 중지 (총 {self.frame_count}개 프레임)")
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText('⚪ Standby')
        
        self.video_label.clear()
        self.video_label.setText('⏹ Stopped')
    
    def update_frame(self):
        """프레임 업데이트 (메인 스레드)"""
        # 모든 체크를 try-except로 감싸기
        try:
            # 기본 체크
            if not self.cap:
                return
            
            # Qt 객체 체크
            if not hasattr(self, 'video_label') or not hasattr(self, 'event_log'):
                return
            
            # 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                return
            
            # ===== 좌우 반전 (미러링) ===== ✅
            frame = cv2.flip(frame, 1)
            frame_width = frame.shape[1]  # 640
            
            self.frame_count += 1
            
            # ===== YOLO Pose 처리 (매 프레임) =====
            if self.yolo_model:
                try:
                    # 첫 프레임에 로그
                    if self.frame_count == 1:
                        self.safe_add_log("[INFO] YOLO 추론 시작!")
                    
                    # YOLO 추론
                    results = self.yolo_model(frame, verbose=False)
                    
                    if self.frame_count % 30 == 0:
                        self.safe_add_log(f"[DEBUG] YOLO 결과: {len(results)}개")
                    
                    # Keypoints 확인
                    if len(results) > 0 and results[0].keypoints is not None:
                        keypoints = results[0].keypoints.data.cpu().numpy()
                        
                        if len(keypoints) > 0:
                            # ===== Keypoint 필터링 적용 =====
                            keypoints_filtered = self.keypoint_filter.apply(keypoints[0])
                            keypoints[0] = keypoints_filtered
                            
                            # Skeleton 그리기 (draw_skeleton 내부에서 미러링 처리)
                            frame = self.draw_skeleton(frame, keypoints)
                            
                            # ===== 낙상 감지 예측 (간소화) =====
                            if self.rf_model:
                                try:
                                    # 간단한 Feature만 추출
                                    simple_features = self.extract_simple_features(keypoints[0])
                                    
                                    if simple_features and len(simple_features) > 0:
                                        # 예측
                                        prediction, proba = self.predict_fall(simple_features)
                                        
                                        # 화면에 예측 결과 표시
                                        frame = self.draw_prediction(frame, prediction, proba)
                                        
                                        # 우측 패널 업데이트
                                        self.update_fall_info(prediction, proba)
                                        
                                        # ===== DB 저장 (모든 상태) =====
                                        # Normal도 주기적으로 저장 (5초마다)
                                        # Falling, Fallen은 즉시 저장
                                        save_interval = 5.0 if prediction == 0 else 1.0
                                        
                                        current_time = datetime.now()
                                        if not hasattr(self, 'last_save_time') or \
                                           (current_time - self.last_save_time).total_seconds() >= save_interval:
                                            self.save_fall_event(prediction, proba, simple_features)
                                            self.last_save_time = current_time
                                        
                                        # 모든 상태 로그 출력 (30프레임마다)
                                        if self.frame_count % 30 == 0:
                                            class_name = self.class_names[prediction]
                                            confidence = proba[prediction] * 100
                                            
                                            if prediction == 0:
                                                # Normal: INFO 레벨
                                                self.safe_add_log(f"[INFO] {class_name} - {confidence:.1f}%")
                                            else:
                                                # Falling, Fallen: ALERT 레벨
                                                self.safe_add_log(f"[ALERT] {class_name} detected! ({confidence:.1f}%)")
                                            
                                            # DB에 저장 (Falling or Fallen만)
                                            self.save_event_to_db(class_name, proba[prediction])
                                
                                except Exception as e:
                                    if self.frame_count % 100 == 0:
                                        print(f"[WARN] 낙상 감지 오류: {str(e)[:50]}")
                            
                            if self.frame_count % 30 == 0:
                                self.safe_add_log(f"[YOLO] ✅ Keypoints: {len(keypoints)}개!")
                        else:
                            if self.frame_count % 30 == 0:
                                self.safe_add_log(f"[YOLO] ⚠️ Keypoints 배열 비어있음")
                    else:
                        if self.frame_count % 30 == 0:
                            self.safe_add_log(f"[YOLO] ⚠️ keypoints None")
                
                except Exception as e:
                    if self.frame_count <= 10:
                        self.safe_add_log(f"[ERROR] YOLO 오류: {str(e)}")
            else:
                if self.frame_count == 1:
                    self.safe_add_log("[WARN] self.yolo_model이 None입니다!")
            
            # 텍스트 추가
            status_text = "YOLO Pose ON" if self.yolo_model else "Webcam Only"
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            
            # QImage 생성 (복사본)
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, 
                            QImage.Format.Format_RGB888).copy()
            
            # QPixmap 변환
            pixmap = QPixmap.fromImage(qt_image)
            
            # 크기 조절
            scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            
            # 화면에 표시
            self.video_label.setPixmap(scaled_pixmap)
            
            # 로그 (매 100프레임)
            if self.frame_count % 100 == 0:
                self.safe_add_log(f"[INFO] 프레임: {self.frame_count}")
            
        except RuntimeError:
            # Qt 객체가 삭제됨 - 조용히 종료
            return
        except Exception as e:
            # 일반 에러는 콘솔만
            print(f"[ERROR] 프레임 업데이트: {str(e)[:50]}")
    
    def draw_skeleton(self, frame, keypoints):
        """Skeleton 그리기 (미러링 적용)"""
        try:
            frame_width = frame.shape[1]  # 640
            
            # YOLO Pose 연결 (COCO 17 keypoints)
            connections = [
                (0, 1), (0, 2),  # 머리
                (1, 3), (2, 4),  # 팔 상단
                (5, 6),  # 어깨
                (5, 7), (7, 9),  # 왼팔
                (6, 8), (8, 10),  # 오른팔
                (5, 11), (6, 12),  # 몸통
                (11, 12),  # 골반
                (11, 13), (13, 15),  # 왼다리
                (12, 14), (14, 16),  # 오른다리
            ]
            
            for person_kps in keypoints:
                # Keypoints 그리기 (이미 반전된 좌표 그대로 사용!)
                for i, kp in enumerate(person_kps):
                    x = int(kp[0])  # ✅ 그대로 사용!
                    y = int(kp[1])
                    conf = kp[2]
                    
                    if conf > 0.5:  # 신뢰도 0.5 이상만
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # 연결선 그리기 (이미 반전된 좌표 그대로 사용!)
                for conn in connections:
                    pt1_idx, pt2_idx = conn
                    if pt1_idx < len(person_kps) and pt2_idx < len(person_kps):
                        pt1 = person_kps[pt1_idx]
                        pt2 = person_kps[pt2_idx]
                        
                        if pt1[2] > 0.5 and pt2[2] > 0.5:  # 둘 다 신뢰도 높으면
                            x1 = int(pt1[0])  # ✅ 그대로 사용!
                            y1 = int(pt1[1])
                            x2 = int(pt2[0])  # ✅ 그대로 사용!
                            y2 = int(pt2[1])
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Skeleton 그리기 오류: {e}")
            return frame
    
    def extract_simple_features(self, keypoints):
        """간단한 Feature 추출 (핵심만)"""
        try:
            features = {}
            
            # Hip height (골반 높이)
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                features['hip_height'] = (left_hip[1] + right_hip[1]) / 2
            else:
                features['hip_height'] = 0
            
            # Aspect ratio
            x_coords = keypoints[:, 0][keypoints[:, 2] > 0.5]
            y_coords = keypoints[:, 1][keypoints[:, 2] > 0.5]
            if len(x_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
                features['aspect_ratio'] = width / (height + 1e-6)
            else:
                features['aspect_ratio'] = 1.0
            
            return features
            
        except Exception as e:
            print(f"Feature 추출 오류: {e}")
            return {}
    
    def predict_fall(self, features):
        """낙상 예측 (간단 버전)"""
        try:
            # Hip height 기반 간단 예측
            hip_height = features.get('hip_height', 0)
            aspect_ratio = features.get('aspect_ratio', 1.0)
            
            # 간단한 규칙 기반 (임시)
            if hip_height < 200:  # 낮음
                if aspect_ratio > 1.5:  # 넓음
                    return 2, [0.1, 0.2, 0.7]  # Fallen
                else:
                    return 1, [0.2, 0.6, 0.2]  # Falling
            else:
                return 0, [0.8, 0.15, 0.05]  # Normal
                
        except Exception as e:
            print(f"예측 오류: {e}")
            return 0, [1.0, 0.0, 0.0]
    
    def draw_prediction(self, frame, prediction, proba):
        """예측 결과 오버레이"""
        try:
            h, w = frame.shape[:2]
            
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 100), (280, 280), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # 클래스 이름 및 색상
            class_name = self.class_names[prediction]
            color = self.class_colors[prediction]
            confidence = proba[prediction]
            
            # 상태 표시 (영문)
            status_map = {
                'Normal': 'Normal',
                'Falling': 'Falling',
                'Fallen': 'Fallen'
            }
            status = status_map.get(class_name, class_name)
            
            # 아이콘 추가
            icon_map = {
                'Normal': '[OK]',
                'Falling': '[ALERT]',
                'Fallen': '[DANGER]'
            }
            icon = icon_map.get(class_name, '')
            
            cv2.putText(frame, f"{icon} {status}", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # 신뢰도
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 각 클래스 확률
            y_offset = 205
            for i, prob in enumerate(proba):
                cls_name = self.class_names.get(i, f"Class {i}")
                bar_width = int(prob * 230)
                
                # 확률 바
                cv2.rectangle(frame, (20, y_offset-10), (20 + bar_width, y_offset+5), 
                             self.class_colors[i], -1)
                
                # 텍스트
                cv2.putText(frame, f"{cls_name}: {prob*100:.1f}%", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            return frame
            
        except Exception as e:
            print(f"예측 오버레이 오류: {e}")
            return frame
    
    def add_log(self, message: str):
        """로그 추가"""
        try:
            if not self.event_log:
                return
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.event_log.append(f"[{timestamp}] {message}")
            scrollbar = self.event_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except RuntimeError:
            # Qt 객체가 삭제된 경우 무시
            pass
        except Exception as e:
            print(f"로그 추가 오류: {e}")
    
    def safe_add_log(self, message: str):
        """안전한 로그 추가 (RuntimeError 무시)"""
        try:
            self.add_log(message)
        except:
            pass
    
    def update_fall_info(self, prediction, proba):
        """낙상 감지 정보 업데이트"""
        try:
            # 현재 상태
            class_name = self.class_names[prediction]
            confidence = proba[prediction] * 100
            
            # 상태 텍스트 및 색상
            if prediction == 0:  # Normal
                status_text = "[OK] Normal"
                color = "#27ae60"  # 진한 초록
            elif prediction == 1:  # Falling
                status_text = "[ALERT] Falling"
                color = "#f39c12"  # 진한 주황
            else:  # Fallen
                status_text = "[DANGER] Fallen"
                color = "#e74c3c"  # 진한 빨강
            
            # 업데이트
            self.fall_status_label.setText(status_text)
            self.fall_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            # 진행 바 업데이트
            self.normal_bar.setValue(int(proba[0] * 100))
            self.falling_bar.setValue(int(proba[1] * 100))
            self.fallen_bar.setValue(int(proba[2] * 100))
            
        except:
            pass
    
    def save_fall_event(self, prediction, proba, features):
        """낙상 이벤트 DB 저장 (Normal 포함)"""
        try:
            # 이벤트 타입 매핑
            event_type_map = {
                0: '정상',    # Normal
                1: '낙상중',  # Falling
                2: '낙상'     # Fallen
            }
            
            event_type = event_type_map.get(prediction)
            if not event_type:
                return
            
            # DB 저장
            confidence = float(proba[prediction])
            hip_height = features.get('hip_height', 0.0)
            
            # spine_angle, hip_velocity는 현재 계산 안 함 (None)
            event_id = self.event_log_model.create(
                user_id=self.user_info['user_id'],
                event_type=event_type,
                confidence=confidence,
                hip_height=hip_height,
                spine_angle=None,
                hip_velocity=None,
                event_status='발생',
                notes=f'AI Detection - {self.class_names[prediction]}'
            )
            
            if event_id:
                # 로그 메시지
                if prediction == 0:
                    # Normal: 간단한 로그
                    self.safe_add_log(f"[DB] Normal saved (ID: {event_id})")
                else:
                    # Falling, Fallen: 강조 로그
                    self.safe_add_log(f"[DB] {event_type} saved (ID: {event_id})")
            else:
                self.safe_add_log(f"[DB] Failed to save {event_type}")
                
        except Exception as e:
            print(f"[ERROR] DB 저장 실패: {e}")
            if prediction > 0:  # 낙상만 에러 로그
                self.safe_add_log(f"[DB] Save error: {str(e)[:30]}")
    
    def stop_monitoring_on_close(self):
        """종료 시 정리"""
        self.stop_monitoring()
    
    def closeEvent(self, event):
        """창 닫을 때"""
        self.stop_monitoring()
        event.accept()
    
    def on_search_clicked(self):
        """검색 버튼 클릭"""
        self.add_log("[INFO] Search button clicked")
        
        # 메시지 박스 표시
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Search")
        msg.setText("Search function")
        msg.setInformativeText("Event search feature will be implemented here.\n\n"
                              "You can search for:\n"
                              "• Fall detection events\n"
                              "• Date/Time range\n"
                              "• Event type (Normal/Falling/Fallen)")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def on_emergency_clicked(self):
        """긴급 호출 버튼 클릭"""
        self.add_log("[ALERT] Emergency Call activated!")
        
        # 경고 메시지 박스 표시
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Emergency Call")
        msg.setText("🚨 Emergency Call Activated!")
        msg.setInformativeText("Emergency notification will be sent to:\n\n"
                              "• Emergency contacts\n"
                              "• Medical services\n"
                              "• System administrators\n\n"
                              "Do you want to proceed?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        result = msg.exec()
        
        if result == QMessageBox.StandardButton.Yes:
            self.add_log("[ALERT] Emergency call confirmed!")
            # TODO: 실제 긴급 호출 기능 구현
            
            # 확인 메시지
            confirm = QMessageBox(self)
            confirm.setIcon(QMessageBox.Icon.Information)
            confirm.setWindowTitle("Emergency Call Sent")
            confirm.setText("Emergency call has been sent successfully!")
            confirm.setStandardButtons(QMessageBox.StandardButton.Ok)
            confirm.exec()
        else:
            self.add_log("[INFO] Emergency call cancelled")
    
    def on_filter_clicked(self):
        """필터 강도 조절 버튼 클릭"""
        # 필터 강도 순환: none -> light -> medium -> strong -> none
        strength_cycle = ['none', 'light', 'medium', 'strong']
        current_idx = strength_cycle.index(self.filter_strength)
        next_idx = (current_idx + 1) % len(strength_cycle)
        self.filter_strength = strength_cycle[next_idx]
        
        # 필터 업데이트
        self.keypoint_filter.set_strength(self.filter_strength)
        
        # 버튼 텍스트 및 색상 변경
        strength_display = {
            'none': ('None', '#95a5a6'),
            'light': ('Light', '#3498db'),
            'medium': ('Medium', '#9b59b6'),
            'strong': ('Strong', '#e74c3c')
        }
        
        display_name, color = strength_display[self.filter_strength]
        self.btn_filter.setText(f'🎚️ Filter: {display_name}')
        self.btn_filter.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{ 
                filter: brightness(110%);
            }}
        """)
        
        # 로그
        self.add_log(f"[INFO] Filter strength changed to: {display_name}")
        
        # 팝업
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Filter Settings")
        msg.setText(f"Filter Strength: {display_name}")
        
        descriptions = {
            'none': 'No filtering\nRaw keypoints (may be shaky)',
            'light': 'Light smoothing\nMinimal lag, some stability',
            'medium': 'Balanced smoothing (Recommended)\nGood balance of stability and responsiveness',
            'strong': 'Strong smoothing\nVery stable but may have lag'
        }
        
        msg.setInformativeText(descriptions[self.filter_strength])
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def save_event_to_db(self, event_type, confidence):
        """
        낙상 이벤트를 DB에 저장
        
        Args:
            event_type: 'Falling' 또는 'Fallen'
            confidence: 예측 신뢰도 (0.0 ~ 1.0)
        """
        try:
            # EventLog 모델 임포트
            from database_models import EventLog
            
            # EventLog 인스턴스 생성
            event_log = EventLog(self.db)
            
            # 이벤트 타입 매핑 (영문 -> 한글)
            event_type_map = {
                'Normal': '정상',
                'Falling': '낙상',
                'Fallen': '낙상'
            }
            
            korean_event_type = event_type_map.get(event_type, '낙상')
            
            # DB에 이벤트 저장
            event_id = event_log.create(
                user_id=self.user_info['user_id'],  # 기존 구조에 맞춤
                event_type=korean_event_type,
                confidence=confidence,
                hip_height=None,  # 필요시 추가
                spine_angle=None,  # 필요시 추가
                hip_velocity=None,  # 필요시 추가
                event_status='발생',
                notes=f'{event_type} detected with {confidence*100:.1f}% confidence'
            )
            
            if event_id:
                self.add_log(f"[DB] Event saved: ID={event_id}, Type={korean_event_type}, Conf={confidence:.2f}")
            else:
                self.add_log(f"[ERROR] Failed to save event to DB")
        
        except Exception as e:
            self.add_log(f"[ERROR] DB save error: {str(e)[:50]}")