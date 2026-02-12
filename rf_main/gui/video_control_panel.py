"""
동영상 재생 제어 패널 위젯
monitoring_page.py에서 사용
"""

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal


class VideoControlPanel(QFrame):
    """동영상 재생 제어 패널"""
    
    # 시그널 정의
    play_pause_clicked = pyqtSignal()
    seek_first_clicked = pyqtSignal()
    seek_last_clicked = pyqtSignal()
    seek_forward_clicked = pyqtSignal()
    seek_backward_clicked = pyqtSignal()
    loop_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    slider_pressed = pyqtSignal()
    slider_released = pyqtSignal()
    slider_value_changed = pyqtSignal(int)
    save_results_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # === 1. 진행 바 영역 ===
        progress_layout = QHBoxLayout()
        
        # 현재 시간
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        progress_layout.addWidget(self.time_label)
        
        # 슬라이더
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2c3e50;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
        """)
        self.progress_slider.sliderPressed.connect(self.slider_pressed.emit)
        self.progress_slider.sliderReleased.connect(self.slider_released.emit)
        self.progress_slider.valueChanged.connect(self.slider_value_changed.emit)
        progress_layout.addWidget(self.progress_slider)
        
        # 총 시간
        self.duration_label = QLabel("00:00:00")
        self.duration_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        progress_layout.addWidget(self.duration_label)
        
        layout.addLayout(progress_layout)
        
        # === 2. 컨트롤 버튼 영역 ===
        button_layout = QHBoxLayout()
        
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                min-width: 45px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1c5a85; }
        """
        
        # 처음으로
        self.btn_first = QPushButton("⏮")
        self.btn_first.setStyleSheet(button_style)
        self.btn_first.clicked.connect(self.seek_first_clicked.emit)
        button_layout.addWidget(self.btn_first)
        
        # 뒤로 (10초)
        self.btn_backward = QPushButton("⏪")
        self.btn_backward.setStyleSheet(button_style)
        self.btn_backward.clicked.connect(self.seek_backward_clicked.emit)
        button_layout.addWidget(self.btn_backward)
        
        # 재생/일시정지
        self.btn_play_pause = QPushButton("▶")
        self.btn_play_pause.setStyleSheet(button_style)
        self.btn_play_pause.clicked.connect(self.play_pause_clicked.emit)
        button_layout.addWidget(self.btn_play_pause)
        
        # 앞으로 (10초)
        self.btn_forward = QPushButton("⏩")
        self.btn_forward.setStyleSheet(button_style)
        self.btn_forward.clicked.connect(self.seek_forward_clicked.emit)
        button_layout.addWidget(self.btn_forward)
        
        # 마지막으로
        self.btn_last = QPushButton("⏭")
        self.btn_last.setStyleSheet(button_style)
        self.btn_last.clicked.connect(self.seek_last_clicked.emit)
        button_layout.addWidget(self.btn_last)
        
        button_layout.addSpacing(20)
        
        # 반복 재생
        self.btn_loop = QPushButton("🔁")
        self.btn_loop.setStyleSheet(button_style)
        self.btn_loop.setCheckable(True)
        self.btn_loop.clicked.connect(lambda: self.loop_toggled.emit(self.btn_loop.isChecked()))
        button_layout.addWidget(self.btn_loop)
        
        # 결과 저장
        self.btn_save = QPushButton("💾")
        self.btn_save.setStyleSheet(button_style)
        self.btn_save.clicked.connect(self.save_results_clicked.emit)
        button_layout.addWidget(self.btn_save)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # === 3. 속도 조절 영역 ===
        speed_layout = QHBoxLayout()
        
        speed_label = QLabel("재생 속도:")
        speed_label.setStyleSheet("color: white; font-weight: bold;")
        speed_layout.addWidget(speed_label)
        
        speed_style = """
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
                min-width: 50px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
            QPushButton:checked {
                background-color: #3498db;
                font-weight: bold;
            }
        """
        
        self.speed_button_group = QButtonGroup()
        
        for speed in [0.25, 0.5, 1.0, 2.0]:
            btn = QPushButton(f"{speed}x")
            btn.setStyleSheet(speed_style)
            btn.setCheckable(True)
            btn.setProperty("speed", speed)
            btn.clicked.connect(lambda checked, s=speed: self.speed_changed.emit(s))
            
            if speed == 1.0:
                btn.setChecked(True)
            
            self.speed_button_group.addButton(btn)
            speed_layout.addWidget(btn)
        
        # 프레임 정보
        self.frame_info_label = QLabel("Frame: 0/0")
        self.frame_info_label.setStyleSheet("color: #ecf0f1; margin-left: 20px;")
        speed_layout.addWidget(self.frame_info_label)
        
        speed_layout.addStretch()
        
        layout.addLayout(speed_layout)
    
    def set_play_pause_icon(self, is_playing):
        """재생/일시정지 아이콘 변경"""
        if is_playing:
            self.btn_play_pause.setText("⏸")
        else:
            self.btn_play_pause.setText("▶")
    
    def set_time(self, current, duration):
        """시간 표시 업데이트"""
        self.time_label.setText(self.format_time(current))
        self.duration_label.setText(self.format_time(duration))
    
    def set_progress(self, current_frame, total_frames):
        """진행 바 업데이트"""
        if total_frames > 0:
            self.progress_slider.blockSignals(True)
            self.progress_slider.setMaximum(total_frames)
            self.progress_slider.setValue(current_frame)
            self.progress_slider.blockSignals(False)
        
        self.frame_info_label.setText(f"Frame: {current_frame}/{total_frames}")
    
    def format_time(self, seconds):
        """초를 HH:MM:SS 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
