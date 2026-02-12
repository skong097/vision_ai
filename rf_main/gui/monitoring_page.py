"""
Home Safe Solution - 실시간 모니터링 페이지 (낙상 감지 통합)
QTimer 기반 - 안정적

✅ 작동 확인: 2026-01-30 01:41
✅ 상태: 웹캠 + YOLO Pose + Skeleton + 낙상 감지
✅ 업데이트: 2026-01-31 - 새 모델 적용 (93.19% 정확도)
✅ 업데이트: 2026-02-05 - ST-GCN 모델 통합 (84.21% 정확도)
   - 모델 선택 다이얼로그 추가 (Random Forest / ST-GCN)
   - 60프레임 버퍼링 기반 시계열 분석
   - 실시간 추론 및 상태 표시
✅ 업데이트: 2026-02-05 - 단일 대상자 추적 (select_target_person)
   - 다중 객체 감지 → 가장 큰 Bounding Box 1명만 추적
   - Skeleton 및 낙상 감지 모두 대상자 1명에 집중
"""

import sys
import os
import csv
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QFrame, QPushButton, QTextEdit, QGroupBox, QMessageBox,
                             QRadioButton, QButtonGroup, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
import cv2
import numpy as np
from datetime import datetime
import joblib
from collections import deque
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # rf_main/gui 내 파일
from path_config import PATHS

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
from video_control_panel import VideoControlPanel

# ========== ST-GCN 모델 통합 ==========
try:
    # from stgcn_inference import STGCNInference
    from stgcn_inference_finetuned import STGCNInference
    STGCN_AVAILABLE = True
except ImportError:
    STGCN_AVAILABLE = False
    print("⚠️ ST-GCN module not available")

from model_selection_dialog import show_model_selection_dialog



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⭐ 정확도 트래커 클래스 ⭐
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AccuracyTracker:
    """5분 정확도 추적"""
    
    def __init__(self, window_seconds=300):
        self.window_seconds = window_seconds
        self.predictions = deque()
        self.ground_truth = 'Normal'
    
    def set_ground_truth(self, state):
        if state in ['Normal', 'Falling', 'Fallen']:
            self.ground_truth = state
    
    def record_prediction(self, predicted_state):
        current_time = time.time()
        is_correct = (self.ground_truth == predicted_state)
        
        self.predictions.append({
            'timestamp': current_time,
            'ground_truth': self.ground_truth,
            'predicted': predicted_state,
            'correct': is_correct
        })
        
        cutoff_time = current_time - self.window_seconds
        while self.predictions and self.predictions[0]['timestamp'] < cutoff_time:
            self.predictions.popleft()
    
    def get_accuracy(self):
        if len(self.predictions) == 0:
            return 0.0
        correct_count = sum(1 for p in self.predictions if p['correct'])
        total_count = len(self.predictions)
        return (correct_count / total_count) * 100
    
    def get_sample_count(self):
        return len(self.predictions)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MonitoringPage(QWidget):
    """실시간 모니터링 페이지 (QTimer 버전)"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.cap = None
        self.timer = None
        self.frame_count = 0
        
        # ⭐ 입력 소스 선택 (카메라 또는 파일)
        from input_selection_dialog import show_input_selection_dialog
        input_config = show_input_selection_dialog(self)
        
        if input_config is None:
            # 취소 시 기본 카메라
            self.input_type = 'camera'
            self.camera_index = 0
            self.video_path = None
            print("[WARNING] 입력 선택 취소됨. 기본 카메라(0번) 사용")
        else:
            self.input_type = input_config['type']  # 'camera' or 'file'
            
            if self.input_type == 'camera':
                self.camera_index = input_config['camera_index']
                self.video_path = None
                print(f"[INFO] 카메라 {self.camera_index}번 선택됨")
            else:
                self.camera_index = None
                self.video_path = input_config['filepath']
                print(f"[INFO] 동영상 파일: {os.path.basename(self.video_path)}")
        
        # ========== 모델 선택 (ST-GCN 통합) ==========
        model_config = show_model_selection_dialog(self)
        self.model_type = model_config['type']  # 'random_forest' or 'stgcn'
        self.model_name = model_config['name']
        print(f"[INFO] 선택된 모델: {self.model_name}")
        
        # ST-GCN 관련 변수 초기화
        self.stgcn_model = None
        self.keypoints_buffer = []
        self.stgcn_buffer_size = 60  # 60 frames (~3초)
        self.stgcn_ready = False
        
        # 동영상 재생 제어 변수
        self.is_paused = False
        self.playback_speed = 1.0
        self.total_frames = 0
        self.current_frame_num = 0
        self.original_fps = 20
        self.loop_playback = False
        self.current_display_frame = None
        
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
                # yolo_path = '/home/gjkong/dev_ws/yolo/myproj/models/yolo11s-pose.pt'
                yolo_path = str(PATHS.YOLO_MODEL)
                if os.path.exists(yolo_path):
                    self.yolo_model = YOLO(yolo_path)
                    print(f"✅ YOLO Pose 로드 성공: {yolo_path}")
                else:
                    print(f"⚠️ YOLO 모델 없음: {yolo_path}")
            except Exception as e:
                print(f"⚠️ YOLO 로드 실패: {e}")
        
        # 낙상 감지 모델 (새 모델 경로) ⭐⭐⭐ 변경됨!
        self.rf_model = None
        self.feature_columns = None
        self.frame_buffer = deque(maxlen=30)
        self.class_names = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
        self.class_colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
        
        try:
            # ⭐ Binary v3 모델 경로 (2026-02-07 정규화 재학습)
            # model_path = '/home/gjkong/dev_ws/yolo/myproj/models_integrated/binary_v3/random_forest_model.pkl'
            model_path = str(PATHS.RF_MODEL)

            if os.path.exists(model_path):
                # Random Forest 모델 로드
                self.rf_model = joblib.load(model_path)
                
                # ⭐ n_jobs=1 강제 (QTimer 스레드 충돌/메모리 누수 방지)
                self.rf_model.n_jobs = 1
                self.rf_model.verbose = 0
                
                # Feature 순서: 모델에 저장된 feature_names 사용
                if hasattr(self.rf_model, 'feature_names_in_'):
                    self.feature_columns = list(self.rf_model.feature_names_in_)
                
                print(f"✅ 낙상 감지 모델 로드 성공! (Binary RF, n_jobs=1)")
                print(f"   Feature: {len(self.feature_columns) if self.feature_columns else '?'}개")
                print(f"   Classes: {self.rf_model.classes_}")
                print(f"   경로: {model_path}")
            else:
                print(f"⚠️ 낙상 감지 모델 파일 없음")
                print(f"   모델: {model_path}")
                print(f"   피처: {feature_path}")
        except Exception as e:
            print(f"⚠️ 낙상 감지 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # ⭐ 정확도 트래커 초기화
        self.accuracy_tracker = AccuracyTracker(window_seconds=300)  # 5분
        print(f"✅ 정확도 트래커 활성화! (5분 윈도우)")
        
        self.init_ui()
    
    # ... (나머지 코드는 동일) ...
    
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
        
        # ⭐ Switch 버튼 (입력 소스 전환)
        self.btn_switch = QPushButton('🔄 Switch')
        self.btn_switch.clicked.connect(self.on_switch_input)
        self.btn_switch.setFixedHeight(button_height)
        self.btn_switch.setStyleSheet("""
            QPushButton {
                background-color: #16a085;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1abc9c; }
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
        
        # 좌측: Start/Stop/Switch
        top_layout.addWidget(self.btn_start)
        top_layout.addWidget(self.btn_stop)
        top_layout.addWidget(self.btn_switch)
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
        
        # ⭐ 재생 제어 패널 (동영상 파일 재생 시만 표시)
        self.video_control_panel = VideoControlPanel()
        self.video_control_panel.setVisible(False)  # 기본 숨김
        video_layout.addWidget(self.video_control_panel)
        
        # 시그널 연결
        self.video_control_panel.play_pause_clicked.connect(self.toggle_play_pause)
        self.video_control_panel.seek_first_clicked.connect(self.seek_first)
        self.video_control_panel.seek_last_clicked.connect(self.seek_last)
        self.video_control_panel.seek_forward_clicked.connect(self.seek_forward)
        self.video_control_panel.seek_backward_clicked.connect(self.seek_backward)
        self.video_control_panel.loop_toggled.connect(self.on_loop_toggled)
        self.video_control_panel.speed_changed.connect(self.on_speed_changed)
        self.video_control_panel.slider_pressed.connect(self.on_slider_pressed)
        self.video_control_panel.slider_released.connect(self.on_slider_released)
        self.video_control_panel.save_results_clicked.connect(self.save_results_to_csv)
        
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
        # ⭐ Ground Truth 기본값 설정 (UI 없음)
        self.accuracy_tracker.set_ground_truth('Fallen')
        
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
        """모니터링 시작 (카메라 + 파일 지원)"""
        
        try:
            if self.input_type == 'camera':
                # === 카메라 모드 ===
                self.add_log(f"[INFO] 카메라 {self.camera_index}번 연결 시도...")
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    self.add_log(f"❌ 카메라를 열 수 없습니다")
                    return
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.add_log(f"✅ 카메라 연결 성공")
                
                timer_interval = 50  # 20 FPS
                self.video_control_panel.setVisible(False)
            
            elif self.input_type == 'file':
                # === 파일 모드 ===
                self.add_log(f"[FILE] 동영상 로드 중...")
                self.add_log(f"[FILE] {os.path.basename(self.video_path)}")
                
                self.cap = cv2.VideoCapture(self.video_path)
                
                if not self.cap.isOpened():
                    self.add_log(f"❌ 동영상 파일을 열 수 없습니다")
                    return
                
                # 파일 정보
                self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
                
                self.add_log(f"✅ 동영상 로드 성공")
                self.add_log(f"[FILE] FPS: {self.original_fps:.2f}, Frames: {self.total_frames}")
                self.add_log(f"[FILE] Duration: {int(duration//60):02d}:{int(duration%60):02d}")
                
                timer_interval = int(1000 / (self.original_fps * self.playback_speed))
                
                # 재생 제어 패널 표시 및 초기화
                self.video_control_panel.setVisible(True)
                self.video_control_panel.set_time(0, duration)
                self.video_control_panel.set_progress(0, self.total_frames)
            
            # 타이머 시작
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            
            # ========== ST-GCN 모델 초기화 ==========
            if self.model_type == 'stgcn':
                if not self.init_stgcn_model():
                    self.add_log("[WARNING] ST-GCN 로드 실패, Random Forest로 전환")
                    self.model_type = 'random_forest'
            
            self.timer.start(timer_interval)
            
            self.frame_count = 0
            self.current_frame_num = 0
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            
            if self.input_type == 'camera':
                self.status_label.setText('🟢 Webcam Active')
            else:
                self.status_label.setText('🎬 Video Playing')
            
            self.add_log("[INFO] 모니터링 시작")
            
        except Exception as e:
            self.add_log(f"❌ 오류: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
    
    def on_gt_changed(self, state):
        """Ground Truth 변경 이벤트"""
        self.accuracy_tracker.set_ground_truth(state)
        self.add_log(f"[GT] Ground Truth: {state}")
    
    def update_frame(self):
        """프레임 업데이트 (메인 스레드)"""
        # 모든 체크를 try-except로 감싸기
        try:
            # ⭐ 일시정지 상태면 리턴
            if self.is_paused:
                return
            
            # 기본 체크
            if not self.cap:
                return
            
            # Qt 객체 체크
            if not hasattr(self, 'video_label') or not hasattr(self, 'event_log'):
                return
            
            # 프레임 읽기
            ret, frame = self.cap.read()
            
            # ⭐ 프레임 읽기 실패 처리
            if not ret:
                if self.input_type == 'camera':
                    if self.frame_count % 100 == 0:
                        self.safe_add_log("[WARN] 프레임 읽기 실패 (카메라)")
                    return
                
                elif self.input_type == 'file':
                    # 파일 끝 도달
                    if self.loop_playback:
                        # 반복 재생
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_num = 0
                        self.safe_add_log("[VIDEO] 반복 재생 시작")
                        return
                    else:
                        # 재생 종료
                        self.safe_add_log("[VIDEO] 동영상 재생 완료")
                        self.on_video_end()
                        return
            
            # ⭐ 현재 프레임 저장 (캡처용)
            self.current_display_frame = frame.copy()
            
            # ⭐ 파일인 경우 진행 상황 업데이트
            if self.input_type == 'file':
                self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_seconds = self.current_frame_num / self.original_fps if self.original_fps > 0 else 0
                total_seconds = self.total_frames / self.original_fps if self.original_fps > 0 else 0
                
                # 재생 제어 패널 업데이트
                self.video_control_panel.set_time(current_seconds, total_seconds)
                self.video_control_panel.set_progress(self.current_frame_num, self.total_frames)
            
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
                        
                        # ⭐ 다중 객체 중 모니터링 대상자 1명 선택
                        target_idx = self.select_target_person(results, method='largest')
                        keypoints_all = results[0].keypoints.data.cpu().numpy()
                        
                        if target_idx is not None and len(keypoints_all) > 0:
                            # 대상자 키포인트만 추출
                            target_keypoints = keypoints_all[target_idx]
                            
                            # ===== Keypoint 필터링 적용 (대상자만) =====
                            keypoints_filtered = self.keypoint_filter.apply(target_keypoints)
                            
                            # ⭐ 대상자 1명만 skeleton 그리기
                            frame = self.draw_skeleton(frame, keypoints_filtered.reshape(1, -1, 3))
                            
                            # ========== 모델별 추론 분기 ==========
                            if self.model_type == 'stgcn':
                                # ST-GCN 추론
                                self.process_stgcn_inference(keypoints_filtered, frame)
                            
                            elif self.model_type == 'random_forest':
                                # ===== 기존 Random Forest 낙상 감지 =====
                                if self.rf_model:
                                    try:
                                        # 간단한 Feature만 추출
                                        simple_features = self.extract_simple_features(keypoints_filtered)
                                        
                                        if simple_features and len(simple_features) > 0:
                                            # ⭐ 3프레임마다 RF 추론 (부하 경감)
                                            if self.frame_count % 3 == 0:
                                                prediction, proba = self.predict_fall(simple_features)
                                                self._last_prediction = prediction
                                                self._last_proba = proba
                                            else:
                                                prediction = getattr(self, '_last_prediction', 0)
                                                proba = getattr(self, '_last_proba', [1.0, 0.0, 0.0])
                                            
                                            # ⭐ 정확도 트래커에 기록
                                            class_name = self.class_names[prediction]
                                            self.accuracy_tracker.record_prediction(class_name)
                                            
                                            # 우측 패널 업데이트
                                            self.update_fall_info(prediction, proba)
                                            
                                            # ===== DB 저장 (모든 상태) =====
                                            save_interval = 10.0 if prediction == 0 else 3.0
                                            
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
                                                    self.safe_add_log(f"[INFO] {class_name} - {confidence:.1f}%")
                                                else:
                                                    self.safe_add_log(f"[ALERT] {class_name} detected! ({confidence:.1f}%)")
                                    
                                    except Exception as e:
                                        if self.frame_count % 100 == 0:
                                            print(f"[WARN] 낙상 감지 오류: {str(e)[:50]}")
                            
                            if self.frame_count % 30 == 0:
                                num_detected = len(keypoints_all)
                                if num_detected > 1:
                                    self.safe_add_log(f"[YOLO] ✅ {num_detected}명 감지 → 대상자 #{target_idx} 추적 중")
                                else:
                                    self.safe_add_log(f"[YOLO] ✅ 1명 감지 (대상자 추적 중)")
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
            
            # ⭐ 정확도 오버레이 추가
            frame = self.draw_accuracy_overlay(frame)
            
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
        """Skeleton 그리기"""
        try:
            h, w = frame.shape[:2]
            
            # COCO Keypoint 연결 정의
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
                (5, 11), (6, 12), (11, 12),  # 몸통
                (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
            ]
            
            # Keypoint 그리기
            for i, kpt in enumerate(keypoints):
                x, y, conf = int(kpt[0]), int(kpt[1]), kpt[2]
                if conf > 0.5:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Skeleton 연결선 그리기
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            return frame
        except Exception as e:
            print(f"[ERROR] draw_skeleton: {e}")
            return frame
    
    def select_target_person(self, results, method='largest'):
        """
        여러 사람 중 모니터링 대상자 선택
        
        Args:
            results: YOLO 결과 객체
            method: 선택 방법
                - 'largest': 가장 큰 Bounding Box (기본, 추천)
                - 'center': 화면 중앙에 가장 가까운 사람
                - 'combined': 크기 + 중앙 거리 조합 (60% + 40%)
        
        Returns:
            int: 선택된 사람의 인덱스 (없으면 None)
        """
        if len(results) == 0:
            return None
        
        # Keypoints와 Boxes 가져오기
        keypoints = results[0].keypoints
        boxes = results[0].boxes
        
        if keypoints is None or boxes is None:
            return None
        
        keypoints_data = keypoints.data.cpu().numpy()
        boxes_data = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        
        if len(keypoints_data) == 0 or len(boxes_data) == 0:
            return None
        
        num_people = len(keypoints_data)
        
        # 한 명만 있으면 바로 반환
        if num_people == 1:
            return 0
        
        # ===== 방법 1: 가장 큰 Bounding Box (면적) =====
        if method == 'largest':
            areas = []
            for box in boxes_data:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                areas.append(area)
            
            # 가장 큰 면적의 인덱스 반환
            best_idx = areas.index(max(areas))
            
            # 로그 (30프레임마다)
            if self.frame_count % 30 == 0:
                self.safe_add_log(f"[INFO] {num_people}명 감지 → 가장 큰 사람 선택 (#{best_idx})")
            
            return best_idx
        
        # ===== 방법 2: 화면 중앙 거리 =====
        elif method == 'center':
            # 프레임 중앙
            h, w = frame.shape[:2] if hasattr(self, 'current_frame') else (480, 640)
            frame_center_x = w / 2
            frame_center_y = h / 2
            
            distances = []
            for box in boxes_data:
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                
                distance = ((box_center_x - frame_center_x)**2 + 
                           (box_center_y - frame_center_y)**2)**0.5
                distances.append(distance)
            
            best_idx = distances.index(min(distances))
            
            if self.frame_count % 30 == 0:
                self.safe_add_log(f"[INFO] {num_people}명 감지 → 중앙 사람 선택 (#{best_idx})")
            
            return best_idx
        
        # ===== 방법 3: 크기 + 중앙 거리 조합 =====
        elif method == 'combined':
            h, w = frame.shape[:2] if hasattr(self, 'current_frame') else (480, 640)
            frame_center_x = w / 2
            frame_center_y = h / 2
            
            scores = []
            areas = []
            distances = []
            
            # 면적과 거리 계산
            for box in boxes_data:
                x1, y1, x2, y2 = box
                
                # 면적
                width = x2 - x1
                height = y2 - y1
                area = width * height
                areas.append(area)
                
                # 중앙 거리
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = ((box_center_x - frame_center_x)**2 + 
                           (box_center_y - frame_center_y)**2)**0.5
                distances.append(distance)
            
            # 정규화
            max_area = max(areas)
            max_distance = max(distances)
            
            # 가중치 계산
            for i in range(len(boxes_data)):
                area_normalized = areas[i] / max_area
                distance_normalized = distances[i] / max_distance
                
                # 면적 60% + 중앙 거리 40%
                score = (area_normalized * 0.6) + ((1 - distance_normalized) * 0.4)
                scores.append(score)
            
            best_idx = scores.index(max(scores))
            
            if self.frame_count % 30 == 0:
                self.safe_add_log(f"[INFO] {num_people}명 감지 → 조합 방식 선택 (#{best_idx})")
            
            return best_idx
        
        # 기본: 첫 번째 사람
        return 0
    
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
        """⭐ 정규화된 181개 Feature 추출 (v3b: 2026-02-07)"""
        try:
            features = {}
            CONF_THRESHOLD = 0.3
            
            kp_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # ===== bbox 계산 =====
            valid = keypoints[:, 2] > CONF_THRESHOLD
            if np.any(valid):
                xs = keypoints[valid, 0]
                ys = keypoints[valid, 1]
                bbox_x_min = float(np.min(xs))
                bbox_y_min = float(np.min(ys))
                bbox_w = float(np.max(xs) - bbox_x_min)
                bbox_h = float(np.max(ys) - bbox_y_min)
                if bbox_w < 1: bbox_w = 1.0
                if bbox_h < 1: bbox_h = 1.0
            else:
                bbox_x_min, bbox_y_min = 0.0, 0.0
                bbox_w, bbox_h = 1.0, 1.0
            
            # ===== 정규화 (0~1 클램핑) =====
            kp_norm = np.zeros((17, 3))
            for i in range(17):
                if keypoints[i][2] > CONF_THRESHOLD:
                    kp_norm[i][0] = np.clip((keypoints[i][0] - bbox_x_min) / bbox_w, 0, 1)
                    kp_norm[i][1] = np.clip((keypoints[i][1] - bbox_y_min) / bbox_h, 0, 1)
                else:
                    kp_norm[i][0] = 0.0
                    kp_norm[i][1] = 0.0
                kp_norm[i][2] = float(keypoints[i][2])
            
            # prev 정규화
            def norm_prev(kp):
                if kp is None:
                    return None
                normed = np.zeros((17, 3))
                for i in range(17):
                    if kp[i][2] > CONF_THRESHOLD:
                        normed[i][0] = np.clip((kp[i][0] - bbox_x_min) / bbox_w, 0, 1)
                        normed[i][1] = np.clip((kp[i][1] - bbox_y_min) / bbox_h, 0, 1)
                    normed[i][2] = float(kp[i][2])
                return normed
            
            if not hasattr(self, '_prev_keypoints'):
                self._prev_keypoints = None
                self._prev2_keypoints = None
            
            prev_norm = norm_prev(self._prev_keypoints)
            prev2_norm = norm_prev(self._prev2_keypoints)
            
            # ===== 1~51: 정규화된 keypoint =====
            for i, name in enumerate(kp_names):
                features[f'{name}_x'] = float(kp_norm[i][0])
                features[f'{name}_y'] = float(kp_norm[i][1])
                features[f'{name}_conf'] = float(kp_norm[i][2])
            
            # ===== 52~55: 가속도 =====
            features['acc_x'] = 0.0
            features['acc_y'] = 0.0
            features['acc_z'] = 0.0
            features['acc_mag'] = 0.0
            
            # ===== 56~60: 각도 (원본 좌표 — 스케일 불변) =====
            def calc_angle(a, b, c):
                ba = np.array([a[0]-b[0], a[1]-b[1]])
                bc = np.array([c[0]-b[0], c[1]-b[1]])
                cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            
            features['left_elbow_angle'] = calc_angle(keypoints[5], keypoints[7], keypoints[9])
            features['right_elbow_angle'] = calc_angle(keypoints[6], keypoints[8], keypoints[10])
            features['left_knee_angle'] = calc_angle(keypoints[11], keypoints[13], keypoints[15])
            features['right_knee_angle'] = calc_angle(keypoints[12], keypoints[14], keypoints[16])
            
            shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
            hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
            vertical = np.array([hip_mid[0], hip_mid[1] - 100])
            features['spine_angle'] = calc_angle(shoulder_mid, hip_mid, vertical)
            
            # ===== 61~68: 정규화된 높이/비율 =====
            hip_mid_n = (kp_norm[11][:2] + kp_norm[12][:2]) / 2
            shoulder_mid_n = (kp_norm[5][:2] + kp_norm[6][:2]) / 2
            
            features['hip_height'] = float(hip_mid_n[1])
            features['shoulder_height'] = float(shoulder_mid_n[1])
            features['head_height'] = float(kp_norm[0][1])
            
            features['bbox_width'] = float(bbox_w / (bbox_w + bbox_h))
            features['bbox_height'] = float(bbox_h / (bbox_w + bbox_h))
            features['bbox_aspect_ratio'] = float(bbox_w / bbox_h)
            
            features['shoulder_tilt'] = float(abs(kp_norm[5][1] - kp_norm[6][1]))
            features['avg_confidence'] = float(np.mean(keypoints[:, 2]))
            
            # ===== 69~170: 정규화된 속도/가속도 =====
            for i, name in enumerate(kp_names):
                if prev_norm is not None and kp_norm[i][2] > CONF_THRESHOLD and prev_norm[i][2] > CONF_THRESHOLD:
                    vx = float(kp_norm[i][0] - prev_norm[i][0])
                    vy = float(kp_norm[i][1] - prev_norm[i][1])
                else:
                    vx, vy = 0.0, 0.0
                
                speed = float(np.sqrt(vx**2 + vy**2))
                features[f'{name}_vx'] = vx
                features[f'{name}_vy'] = vy
                features[f'{name}_speed'] = speed
                
                if (prev2_norm is not None and prev_norm is not None and
                    kp_norm[i][2] > CONF_THRESHOLD and prev_norm[i][2] > CONF_THRESHOLD and prev2_norm[i][2] > CONF_THRESHOLD):
                    prev_vx = float(prev_norm[i][0] - prev2_norm[i][0])
                    prev_vy = float(prev_norm[i][1] - prev2_norm[i][1])
                    ax = vx - prev_vx
                    ay = vy - prev_vy
                else:
                    ax, ay = 0.0, 0.0
                
                features[f'{name}_ax'] = ax
                features[f'{name}_ay'] = ay
                features[f'{name}_accel'] = float(np.sqrt(ax**2 + ay**2))
            
            # ===== 171~172 =====
            features['hip_velocity'] = (features.get('left_hip_speed', 0) + features.get('right_hip_speed', 0)) / 2
            features['hip_acceleration'] = (features.get('left_hip_accel', 0) + features.get('right_hip_accel', 0)) / 2
            
            # ===== 173~181: 시계열 =====
            if not hasattr(self, '_feature_history'):
                self._feature_history = []
            
            self._feature_history.append({
                'hip_height': features['hip_height'],
                'shoulder_height': features['shoulder_height'],
                'head_height': features['head_height'],
                'acc_mag': features['acc_mag'],
            })
            if len(self._feature_history) > 5:
                self._feature_history = self._feature_history[-5:]
            
            hist = self._feature_history
            for key in ['hip_height', 'shoulder_height', 'head_height']:
                vals = [h[key] for h in hist]
                features[f'{key}_mean_5'] = float(np.mean(vals))
                features[f'{key}_std_5'] = float(np.std(vals))
            
            features['acc_mag_diff'] = 0.0
            vals = [h['acc_mag'] for h in hist]
            features['acc_mag_mean_5'] = float(np.mean(vals))
            features['acc_mag_std_5'] = float(np.std(vals))
            
            # 이전 프레임 저장 (원본 좌표)
            self._prev2_keypoints = self._prev_keypoints.copy() if self._prev_keypoints is not None else None
            self._prev_keypoints = keypoints.copy()
            
            return features
            
        except Exception as e:
            print(f"Feature 추출 오류: {e}")
            return {}

    def predict_fall(self, features):
        """낙상 예측 (RF 모델 사용) ⭐ 2026-02-07"""
        try:
            import pandas as pd
            
            if self.rf_model and self.feature_columns:
                # DataFrame 사용 (feature name 경고 방지)
                row = {col: features.get(col, 0) for col in self.feature_columns}
                df = pd.DataFrame([row])
                proba = self.rf_model.predict_proba(df)[0]
                
                # Binary(2class) → 3class 변환 (기존 UI 호환)
                if len(proba) == 2:
                    prediction = 0 if proba[0] > proba[1] else 2
                    return prediction, [float(proba[0]), 0.0, float(proba[1])]
                else:
                    prediction = int(np.argmax(proba))
                    return prediction, [float(p) for p in proba]
            
            # RF 모델 없으면 기존 규칙 기반 fallback
            hip_height = features.get('hip_height', 0)
            aspect_ratio = features.get('aspect_ratio', 1.0)
            if hip_height < 200:
                if aspect_ratio > 1.5:
                    return 2, [0.1, 0.2, 0.7]
                else:
                    return 1, [0.2, 0.6, 0.2]
            else:
                return 0, [0.8, 0.15, 0.05]
                
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
        """낙상 이벤트 DB 저장 (Normal 포함) ⭐ 2026-02-07 수정"""
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
            
            # ⭐ numpy float → python float 변환
            confidence = float(proba[prediction])
            hip_height = float(features.get('hip_height', 0.0))
            spine_angle = float(features.get('spine_angle', 0.0)) if features.get('spine_angle') else None
            hip_velocity = float(features.get('hip_velocity', 0.0)) if features.get('hip_velocity') else None
            
            # 정확도 가져오기 (최근 5분 평균)
            accuracy = float(self.accuracy_tracker.get_accuracy())
            
            event_id = self.event_log_model.create(
                user_id=self.user_info['user_id'],
                event_type=event_type,
                confidence=confidence,
                hip_height=hip_height,
                spine_angle=spine_angle,
                hip_velocity=hip_velocity,
                accuracy=accuracy,
                event_status='발생',
                notes=f'AI Detection - {self.class_names[prediction]}'
            )
            
            if event_id:
                if prediction == 0:
                    self.safe_add_log(f"[DB] Normal saved (ID: {event_id}, Acc: {accuracy:.1f}%)")
                else:
                    self.safe_add_log(f"[DB] {event_type} saved (ID: {event_id}, Acc: {accuracy:.1f}%)")
            else:
                self.safe_add_log(f"[DB] Failed to save {event_type}")
                
        except Exception as e:
            print(f"[ERROR] DB 저장 실패: {e}")
            if prediction > 0:
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
        
        # 가장 최근 낙상 이벤트 조회
        recent_fall = self.event_log_model.get_recent_fall_event(user_id=self.user_info['user_id'])
        
        if not recent_fall:
            # 낙상 이벤트가 없으면 경고
            no_event_msg = QMessageBox(self)
            no_event_msg.setIcon(QMessageBox.Icon.Warning)
            no_event_msg.setWindowTitle("No Fall Event")
            no_event_msg.setText("⚠️ No recent fall event detected!")
            no_event_msg.setInformativeText("Emergency call can only be made when a fall is detected.")
            no_event_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            no_event_msg.exec()
            self.add_log("[WARNING] No recent fall event found")
            return
        
        # 경고 메시지 박스 표시
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Emergency Call")
        msg.setText("🚨 Emergency Call Activated!")
        msg.setInformativeText(f"Emergency notification will be sent to:\n\n"
                              f"• Emergency contacts\n"
                              f"• Medical services\n"
                              f"• System administrators\n\n"
                              f"Recent fall event: ID={recent_fall['event_id']}\n"
                              f"Occurred at: {recent_fall['occurred_at']}\n\n"
                              f"Do you want to proceed?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        result = msg.exec()
        
        if result == QMessageBox.StandardButton.Yes:
            self.add_log("[ALERT] Emergency call confirmed!")
            
            # ⭐ DB 업데이트: action_taken을 '2차_긴급호출'로 변경
            call_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            action_result = f"긴급 호출 발송 완료 ({call_time})"
            
            success = self.event_log_model.update_action(
                event_id=recent_fall['event_id'],
                action_taken='2차_긴급호출',
                action_result=action_result
            )
            
            if success:
                self.add_log(f"[DB] Emergency call logged: Event ID={recent_fall['event_id']}")
            else:
                self.add_log(f"[ERROR] Failed to log emergency call")
            
            # 확인 메시지
            confirm = QMessageBox(self)
            confirm.setIcon(QMessageBox.Icon.Information)
            confirm.setWindowTitle("Emergency Call Sent")
            confirm.setText(f"🚨 Emergency call has been sent successfully!\n\n"
                          f"Event ID: {recent_fall['event_id']}\n"
                          f"Time: {call_time}")
            confirm.setStandardButtons(QMessageBox.StandardButton.Ok)
            confirm.exec()
        else:
            self.add_log("[INFO] Emergency call cancelled")
    
    def on_switch_input(self):
        """입력 소스 전환 (카메라 ↔ 동영상)"""
        self.add_log("[INFO] Switch input source requested")
        
        # 1. 현재 모니터링 중이면 중지
        is_monitoring = self.timer and self.timer.isActive()
        
        if is_monitoring:
            # 사용자 확인
            reply = QMessageBox.question(
                self,
                '입력 소스 전환',
                '현재 모니터링 중입니다.\n입력 소스를 변경하시겠습니까?\n\n(현재 모니터링이 중지됩니다)',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                self.add_log("[INFO] Switch cancelled by user")
                return
            
            # 모니터링 중지
            self.add_log("[INFO] Stopping current monitoring...")
            self.stop_monitoring()
        
        # 2. 현재 입력 소스 백업 (취소 대비)
        backup_type = self.input_type
        backup_camera = self.camera_index
        backup_path = self.video_path
        
        current_source = f"Camera {backup_camera}" if backup_type == 'camera' else f"File: {os.path.basename(backup_path)}"
        self.add_log(f"[INFO] Current source: {current_source}")
        
        # 3. input_selection_dialog 표시
        try:
            from input_selection_dialog import show_input_selection_dialog
            input_config = show_input_selection_dialog(self)
        except Exception as e:
            self.add_log(f"[ERROR] Failed to show dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open input selection dialog:\n{str(e)}")
            return
        
        # 4. 결과 처리
        if input_config is None:
            # 취소: 이전 소스로 복원
            self.input_type = backup_type
            self.camera_index = backup_camera
            self.video_path = backup_path
            self.add_log("[INFO] Input source change cancelled")
            return
        
        # 5. 새 입력 소스로 업데이트
        self.input_type = input_config['type']
        
        if self.input_type == 'camera':
            self.camera_index = input_config['camera_index']
            self.video_path = None
            self.add_log(f"[SUCCESS] Input changed to: Camera {self.camera_index}")
        else:  # file
            self.camera_index = None
            self.video_path = input_config['filepath']
            filename = os.path.basename(self.video_path)
            self.add_log(f"[SUCCESS] Input changed to: {filename}")
        
        # 6. 재생 관련 변수 초기화
        self.is_paused = False
        self.playback_speed = 1.0
        self.current_frame_num = 0
        self.frame_count = 0
        if hasattr(self, 'last_save_time'):
            delattr(self, 'last_save_time')
        
        # 7. 자동 시작 확인
        reply = QMessageBox.question(
            self,
            '자동 시작',
            f'새 입력 소스로 변경되었습니다.\n\n바로 모니터링을 시작하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.add_log("[INFO] Starting monitoring with new input source...")
            self.start_monitoring()
        else:
            self.add_log("[INFO] Ready to start with new input source")
    
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
    
    def draw_accuracy_overlay(self, frame):
        """영상 우측 상단에 정확도 오버레이 (그래프 형식)"""
        try:
            h, w = frame.shape[:2]
            accuracy = self.accuracy_tracker.get_accuracy()
            
            # 배경 박스
            overlay = frame.copy()
            box_x = w - 250
            box_y = 10
            box_w = 240
            box_h = 100
            
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
            
            # 타이틀
            cv2.putText(frame, "Recent 5 min", (box_x + 10, box_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.line(frame, (box_x + 10, box_y + 40), (box_x + box_w - 10, box_y + 40), (255, 255, 255), 1)
            
            # FN Detection Acc: XX% 형식으로 표시
            cv2.putText(frame, f"FN Detection Acc: {accuracy:.1f}%", (box_x + 10, box_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            
            # 그래프 바 (진행 바 형식)
            bar_x = box_x + 10
            bar_y = box_y + 70
            bar_w = box_w - 20  # 220px
            bar_h = 15
            
            # 배경 (회색)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            
            # 정확도에 따른 색상
            if accuracy >= 90:
                bar_color = (0, 255, 0)  # 녹색
            elif accuracy >= 70:
                bar_color = (0, 255, 255)  # 노란색
            else:
                bar_color = (0, 0, 255)  # 빨간색
            
            # 채워진 바 (정확도 비율)
            filled_w = int(bar_w * (accuracy / 100.0))
            if filled_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
            
            # 테두리
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            
            return frame
        except Exception as e:
            print(f"[ERROR] Accuracy overlay: {e}")
            return frame
    
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
            
            accuracy = self.accuracy_tracker.get_accuracy()
            # DB에 이벤트 저장
            event_id = event_log.create(
                user_id=self.user_info['user_id'],  # 기존 구조에 맞춤
                event_type=korean_event_type,
                confidence=confidence,
                hip_height=None,  # 필요시 추가
                spine_angle=None,  # 필요시 추가
                hip_velocity=None,  # 필요시 추가
                accuracy=accuracy,  
                event_status='발생',
                notes=f'{event_type} detected with {confidence*100:.1f}% confidence'
            )
            
            if event_id:
                self.add_log(f"[DB] Event saved: ID={event_id}, Type={korean_event_type}, Conf={confidence:.2f}, Acc={accuracy:.1f}%")
            else:
                self.add_log(f"[ERROR] Failed to save event to DB")
        
        except Exception as e:
            self.add_log(f"[ERROR] DB save error: {str(e)[:50]}")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 동영상 재생 제어 메소드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def toggle_play_pause(self):
        """재생/일시정지 토글"""
        if self.input_type != "file":
            return
        
        self.is_paused = not self.is_paused
        self.video_control_panel.set_play_pause_icon(not self.is_paused)
        
        if self.is_paused:
            self.safe_add_log("[VIDEO] 일시정지")
        else:
            self.safe_add_log("[VIDEO] 재생 재개")
    
    def seek_first(self):
        """처음으로 이동"""
        if self.input_type != "file" or not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_num = 0
        self.safe_add_log("[VIDEO] 처음으로 이동")
    
    def seek_last(self):
        """마지막으로 이동"""
        if self.input_type != "file" or not self.cap:
            return
        
        last_frame = max(0, self.total_frames - 10)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        self.current_frame_num = last_frame
        self.safe_add_log("[VIDEO] 마지막으로 이동")
    
    def seek_backward(self):
        """10초 뒤로"""
        if self.input_type != "file" or not self.cap:
            return
        
        skip_frames = int(self.original_fps * 10)
        new_frame = max(0, self.current_frame_num - skip_frames)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_num = new_frame
        self.safe_add_log(f"[VIDEO] 10초 뒤로 (Frame: {new_frame})")
    
    def seek_forward(self):
        """10초 앞으로"""
        if self.input_type != "file" or not self.cap:
            return
        
        skip_frames = int(self.original_fps * 10)
        new_frame = min(self.total_frames - 1, self.current_frame_num + skip_frames)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_num = new_frame
        self.safe_add_log(f"[VIDEO] 10초 앞으로 (Frame: {new_frame})")
    
    def on_slider_pressed(self):
        """슬라이더 드래그 시작 - 일시정지"""
        if self.input_type == "file":
            self.was_playing = not self.is_paused
            self.is_paused = True
    
    def on_slider_released(self):
        """슬라이더 드래그 종료 - 프레임 이동"""
        if self.input_type != "file" or not self.cap:
            return
        
        new_frame = self.video_control_panel.progress_slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_num = new_frame
        
        if hasattr(self, "was_playing") and self.was_playing:
            self.is_paused = False
        
        self.safe_add_log(f"[VIDEO] 프레임 이동: {new_frame}")
    
    def on_speed_changed(self, speed):
        """재생 속도 변경"""
        if self.input_type != "file":
            return
        
        self.playback_speed = speed
        
        if self.timer and self.original_fps > 0:
            new_interval = int(1000 / (self.original_fps * self.playback_speed))
            self.timer.setInterval(new_interval)
        
        self.safe_add_log(f"[VIDEO] 재생 속도: {speed}x")
    
    def on_loop_toggled(self, enabled):
        """반복 재생 토글"""
        if self.input_type != "file":
            return
        
        self.loop_playback = enabled
        
        if enabled:
            self.safe_add_log("[VIDEO] 반복 재생 ON")
        else:
            self.safe_add_log("[VIDEO] 반복 재생 OFF")
    
    def save_results_to_csv(self):
        """분석 결과 CSV로 저장"""
        if self.input_type != "file":
            self.safe_add_log("[WARN] 동영상 파일 모드에서만 사용 가능")
            return
        
        try:
            # 저장 경로 선택
            default_filename = f"fall_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "결과 저장",
                os.path.expanduser(f"~/Downloads/{default_filename}"),
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not filepath:
                return
            
            # CSV 저장
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                writer.writerow([
                    "Frame Number",
                    "Time (seconds)",
                    "Event Type",
                    "Confidence",
                    "Detection Accuracy (%)",
                    "Timestamp"
                ])
                
                try:
                    from database_models import EventLog
                    event_log = EventLog(self.db)
                    
                    query = """
                    SELECT event_id, occurred_at, et.type_name, confidence, accuracy
                    FROM event_logs el
                    JOIN event_types et ON el.event_type_id = et.event_type_id
                    WHERE el.user_id = %s
                    ORDER BY occurred_at DESC
                    LIMIT 1000
                    """
                    
                    results = self.db.execute_query(query, (self.user_info["user_id"],))
                    
                    for row in results:
                        writer.writerow([
                            row.get("event_id", "N/A"),
                            "N/A",
                            row.get("type_name", "Unknown"),
                            f"{row.get('confidence', 0):.2f}",
                            f"{row.get('accuracy', 0):.1f}",
                            row.get("occurred_at", "N/A")
                        ])
                
                except Exception as e:
                    print(f"[WARN] DB 조회 실패: {e}")
                    writer.writerow([
                        self.frame_count,
                        f"{self.current_frame_num / self.original_fps:.2f}",
                        "Analysis Complete",
                        "N/A",
                        "N/A",
                        datetime.now().isoformat()
                    ])
            
            self.safe_add_log(f"[SAVE] 결과 저장: {os.path.basename(filepath)}")
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("저장 완료")
            msg.setText(f"분석 결과가 저장되었습니다.")
            msg.setInformativeText(filepath)
            msg.exec()
            
        except Exception as e:
            self.safe_add_log(f"[ERROR] 저장 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_video_end(self):
        """동영상 재생 종료"""
        self.stop_monitoring()
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("재생 완료")
        msg.setText("동영상 재생이 완료되었습니다.")
        msg.setInformativeText(
            f"총 {self.frame_count}개 프레임 처리 완료\n"
            f"파일: {os.path.basename(self.video_path)}"
        )
        msg.exec()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⭐ ST-GCN 관련 메소드 ⭐
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def init_stgcn_model(self):
        """ST-GCN 모델 초기화"""
        if not STGCN_AVAILABLE:
            self.add_log("[ERROR] ST-GCN 모듈을 찾을 수 없습니다.")
            return False
        
        try:
            self.stgcn_model = STGCNInference(
                model_path=str(PATHS.STGCN_V2)
                # model_path='/home/gjkong/dev_ws/st_gcn/checkpoints_v2/best_model.pth'
            )
            
            # 프레임 크기 설정
            if self.cap:
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # self.stgcn_model.set_frame_size(frame_width, frame_height)  # v2 불필요
            
            self.keypoints_buffer = []
            self.stgcn_ready = False
            self.add_log(f"[ST-GCN] 모델 로드 완료 (버퍼: {self.stgcn_buffer_size}프레임)")
            return True
            
        except Exception as e:
            self.add_log(f"[ERROR] ST-GCN 모델 로드 실패: {e}")
            return False
    
    def process_stgcn_inference(self, keypoints, frame):
        """
        ST-GCN 모델로 낙상 감지 추론
        
        Args:
            keypoints: 필터링된 키포인트 (17, 3)
            frame: 현재 프레임 (시각화용)
        """
        if self.stgcn_model is None:
            return
        
        # 버퍼에 키포인트 추가
        self.keypoints_buffer.append(keypoints.copy())
        
        # 버퍼 크기 유지 (슬라이딩 윈도우)
        if len(self.keypoints_buffer) > self.stgcn_buffer_size:
            self.keypoints_buffer.pop(0)
        
        # 버퍼 진행률
        buffer_progress = len(self.keypoints_buffer) / self.stgcn_buffer_size
        buffer_percent = int(buffer_progress * 100)
        
        # 추론 수행
        if len(self.keypoints_buffer) >= self.stgcn_buffer_size:
            self.stgcn_ready = True
            
            try:
                label, confidence, normal_prob, fall_prob = self.stgcn_model.predict(self.keypoints_buffer)
                
                # 결과 처리
                if label == 'Fall':
                    # 낙상 감지
                    # 정확도 트래커에 기록
                    self.accuracy_tracker.record_prediction('Fallen')
                    
                    # 로그 (30프레임마다)
                    if self.frame_count % 30 == 0:
                        self.safe_add_log(f"[ST-GCN] 🚨 낙상 감지! (신뢰도: {confidence:.1%})")
                    
                    # UI 업데이트
                    self.update_stgcn_fall_info('Fall', confidence, normal_prob, fall_prob)
                    
                    # DB 저장 (10프레임마다)
                    if self.frame_count % 10 == 0:
                        self.save_event_to_db('Falling', confidence)
                    
                else:
                    # 정상
                    self.accuracy_tracker.record_prediction('Normal')
                    
                    # UI 업데이트
                    self.update_stgcn_fall_info('Normal', confidence, normal_prob, fall_prob)
                    
                    # DB 저장 (10프레임마다)
                    if self.frame_count % 10 == 0:
                        self.save_event_to_db('Normal', confidence)
                
                # 상태 라벨 업데이트
                self.update_stgcn_status_label(label, confidence, buffer_percent)
                
            except Exception as e:
                if self.frame_count % 60 == 0:
                    self.safe_add_log(f"[ST-GCN] 추론 오류: {e}")
        else:
            # 버퍼링 중
            self.stgcn_ready = False
            self.update_stgcn_status_label('버퍼링', 0.0, buffer_percent)
    
    def update_stgcn_fall_info(self, label: str, confidence: float, normal_prob: float = 0.0, fall_prob: float = 0.0):
        """ST-GCN 낙상 감지 결과를 UI에 업데이트"""
        if label == 'Fall':
            self.fall_status_label.setText('🚨 [FALL] 낙상 감지!')
            self.fall_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.confidence_label.setText(f'Confidence: {confidence:.1%}')
            self.confidence_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            # 확률 바 업데이트 (Fall → Fallen 매핑)
            self.normal_bar.setValue(int((1 - confidence) * 100))
            self.falling_bar.setValue(0)
            self.fallen_bar.setValue(int(confidence * 100))
        else:
            self.fall_status_label.setText('[OK] Normal')
            self.fall_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            self.confidence_label.setText(f'Confidence: {confidence:.1%}')
            self.confidence_label.setStyleSheet("color: #3498db; font-weight: bold;")
        # 확률 바 업데이트 (실제 softmax 확률)
        self.normal_bar.setValue(int(normal_prob * 100))
        self.falling_bar.setValue(0)
        self.fallen_bar.setValue(int(fall_prob * 100))
    
    def update_stgcn_status_label(self, status: str, confidence: float, buffer_percent: int):
        """ST-GCN 상태 표시 업데이트"""
        if status == '낙상' or status == 'Fall':
            color = '#e74c3c'  # Red
            status_text = f"🚨 ST-GCN: 낙상 ({confidence:.1%})"
        elif status == '정상' or status == 'Normal':
            color = '#27ae60'  # Green
            status_text = f"✅ ST-GCN: 정상 ({confidence:.1%})"
        else:  # 버퍼링
            color = '#f39c12'  # Orange
            status_text = f"⏳ ST-GCN 버퍼링... {buffer_percent}%"
        
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
    
    def reset_stgcn_buffer(self):
        """ST-GCN 버퍼 초기화"""
        self.keypoints_buffer = []
        self.stgcn_ready = False
        if self.stgcn_model:
            self.stgcn_model.reset_buffer()
        self.safe_add_log("[ST-GCN] 버퍼 초기화됨")