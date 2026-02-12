"""
입력 소스 선택 다이얼로그 (카메라 + 동영상 파일)
"""

import cv2
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QRadioButton, QButtonGroup, QPushButton, 
                             QFileDialog, QGroupBox, QLineEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class InputSourceDialog(QDialog):
    """입력 소스 선택 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_config = None
        self.available_cameras = []
        self.selected_file = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("입력 소스 선택")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 타이틀
        title = QLabel("📹 입력 소스를 선택하세요")
        title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 설명
        desc = QLabel("실시간 카메라 또는 동영상 파일을 선택하세요.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # ===== 입력 타입 선택 =====
        type_group = QButtonGroup(self)
        
        # 카메라 라디오 버튼
        self.radio_camera = QRadioButton("📹 실시간 카메라")
        self.radio_camera.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.radio_camera.setChecked(True)
        self.radio_camera.setStyleSheet("padding: 5px;")
        type_group.addButton(self.radio_camera)
        layout.addWidget(self.radio_camera)
        
        # 카메라 선택 패널
        self.camera_panel = self.create_camera_panel()
        layout.addWidget(self.camera_panel)
        
        # 파일 라디오 버튼
        self.radio_file = QRadioButton("🎬 동영상 파일")
        self.radio_file.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.radio_file.setStyleSheet("padding: 5px;")
        type_group.addButton(self.radio_file)
        layout.addWidget(self.radio_file)
        
        # 파일 선택 패널
        self.file_panel = self.create_file_panel()
        layout.addWidget(self.file_panel)
        
        # 라디오 버튼 이벤트
        self.radio_camera.toggled.connect(self.on_type_changed)
        self.radio_file.toggled.connect(self.on_type_changed)
        
        # 초기 상태
        self.on_type_changed()
        
        # 버튼들
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 확인 버튼
        ok_btn = QPushButton("✓ 확인")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #229954; }
        """)
        ok_btn.clicked.connect(self.on_accept)
        button_layout.addWidget(ok_btn)
        
        # 취소 버튼
        cancel_btn = QPushButton("✗ 취소")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #7f8c8d; }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def create_camera_panel(self):
        """카메라 선택 패널 생성"""
        panel = QGroupBox("사용 가능한 카메라")
        panel_layout = QVBoxLayout()
        
        # 카메라 감지
        self.detect_cameras()
        
        self.camera_button_group = QButtonGroup()
        
        if not self.available_cameras:
            no_camera_label = QLabel("⚠️ 사용 가능한 카메라를 찾을 수 없습니다.")
            no_camera_label.setStyleSheet("color: #e74c3c; padding: 10px;")
            panel_layout.addWidget(no_camera_label)
        else:
            for camera_id, camera_name in self.available_cameras:
                radio = QRadioButton(f"카메라 {camera_id}: {camera_name}")
                radio.setProperty("camera_id", camera_id)
                radio.setStyleSheet("padding: 3px;")
                
                if camera_id == self.available_cameras[0][0]:
                    radio.setChecked(True)
                
                self.camera_button_group.addButton(radio)
                panel_layout.addWidget(radio)
        
        panel.setLayout(panel_layout)
        return panel
    
    def create_file_panel(self):
        """파일 선택 패널 생성"""
        panel = QGroupBox("동영상 파일")
        panel_layout = QVBoxLayout()
        
        # 파일 경로 입력
        file_layout = QHBoxLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("동영상 파일 경로...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)
        
        # 파일 선택 버튼
        browse_btn = QPushButton("📁 파일 선택")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        panel_layout.addLayout(file_layout)
        
        # 파일 정보 레이블
        self.file_info_label = QLabel("선택된 파일 없음")
        self.file_info_label.setStyleSheet("color: #7f8c8d; padding: 10px; font-size: 12px;")
        self.file_info_label.setWordWrap(True)
        panel_layout.addWidget(self.file_info_label)
        
        panel.setLayout(panel_layout)
        return panel
    
    def detect_cameras(self):
        """사용 가능한 카메라 감지"""
        self.available_cameras = []
        
        print("[INFO] 카메라 감지 중...")
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    camera_name = f"{width}x{height}"
                    
                    if i == 0:
                        camera_name += " (내장 웹캠)"
                    elif i == 2:
                        camera_name += " (USB 카메라)"
                    
                    self.available_cameras.append((i, camera_name))
                    print(f"[INFO] 카메라 {i}번 감지: {camera_name}")
                
                cap.release()
        
        print(f"[INFO] 총 {len(self.available_cameras)}개 카메라 감지 완료")
    
    def browse_file(self):
        """파일 선택 다이얼로그"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "동영상 파일 선택",
            os.path.expanduser("~"),
            "동영상 파일 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;모든 파일 (*.*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_path_edit.setText(file_path)
            
            # 파일 정보 표시
            self.show_file_info(file_path)
    
    def show_file_info(self, file_path):
        """파일 정보 표시"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # 파일 크기
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                info_text = (
                    f"✓ {os.path.basename(file_path)}\n"
                    f"• 해상도: {width}x{height}\n"
                    f"• FPS: {fps:.2f}\n"
                    f"• 프레임: {frame_count}\n"
                    f"• 길이: {int(duration//60):02d}:{int(duration%60):02d}\n"
                    f"• 크기: {size_mb:.1f} MB"
                )
                
                self.file_info_label.setText(info_text)
                self.file_info_label.setStyleSheet("color: #27ae60; padding: 10px; font-size: 12px;")
                
                cap.release()
            else:
                self.file_info_label.setText("⚠️ 파일을 열 수 없습니다.")
                self.file_info_label.setStyleSheet("color: #e74c3c; padding: 10px; font-size: 12px;")
        
        except Exception as e:
            self.file_info_label.setText(f"⚠️ 오류: {str(e)}")
            self.file_info_label.setStyleSheet("color: #e74c3c; padding: 10px; font-size: 12px;")
    
    def on_type_changed(self):
        """입력 타입 변경 시"""
        is_camera = self.radio_camera.isChecked()
        
        self.camera_panel.setEnabled(is_camera)
        self.file_panel.setEnabled(not is_camera)
    
    def on_accept(self):
        """확인 버튼 클릭"""
        if self.radio_camera.isChecked():
            # 카메라 선택
            if not self.available_cameras:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "카메라 없음", 
                                   "사용 가능한 카메라가 없습니다.")
                return
            
            # 선택된 카메라 찾기
            selected_camera = self.available_cameras[0][0]
            for button in self.camera_button_group.buttons():
                if button.isChecked():
                    selected_camera = button.property("camera_id")
                    break
            
            self.input_config = {
                'type': 'camera',
                'camera_index': selected_camera
            }
            
            print(f"[INFO] 카메라 {selected_camera}번 선택됨")
        
        elif self.radio_file.isChecked():
            # 파일 선택
            if not self.selected_file:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "파일 없음", 
                                   "동영상 파일을 선택해주세요.")
                return
            
            self.input_config = {
                'type': 'file',
                'filepath': self.selected_file
            }
            
            print(f"[INFO] 파일 선택됨: {self.selected_file}")
        
        self.accept()
    
    def get_input_config(self):
        """입력 설정 반환"""
        return self.input_config


def show_input_selection_dialog(parent=None):
    """
    입력 소스 선택 다이얼로그 표시
    
    Returns:
        dict: {
            'type': 'camera' or 'file',
            'camera_index': int (카메라인 경우),
            'filepath': str (파일인 경우)
        }
        또는 None (취소 시)
    """
    dialog = InputSourceDialog(parent)
    result = dialog.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_input_config()
    else:
        return None
