"""
카메라 선택 다이얼로그
"""

import cv2
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QRadioButton, QButtonGroup, QPushButton, 
                             QMessageBox, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class CameraSelectionDialog(QDialog):
    """카메라 선택 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_camera = 0
        self.available_cameras = []
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("카메라 선택")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 타이틀
        title = QLabel("📹 카메라를 선택하세요")
        title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 설명
        desc = QLabel("사용 가능한 카메라 목록입니다.\n원하는 카메라를 선택해주세요.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # 카메라 감지
        self.detect_cameras()
        
        # 카메라 목록 그룹박스
        camera_group = QGroupBox("사용 가능한 카메라")
        camera_layout = QVBoxLayout()
        
        self.button_group = QButtonGroup()
        
        if not self.available_cameras:
            # 카메라가 없을 때
            no_camera_label = QLabel("⚠️ 사용 가능한 카메라를 찾을 수 없습니다.")
            no_camera_label.setStyleSheet("color: #e74c3c; padding: 20px;")
            no_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            camera_layout.addWidget(no_camera_label)
        else:
            # 카메라 라디오 버튼 생성
            for camera_id, camera_name in self.available_cameras:
                radio = QRadioButton(f"카메라 {camera_id}: {camera_name}")
                radio.setProperty("camera_id", camera_id)
                radio.setStyleSheet("padding: 8px; font-size: 14px;")
                
                # 첫 번째 카메라 기본 선택
                if camera_id == self.available_cameras[0][0]:
                    radio.setChecked(True)
                    self.selected_camera = camera_id
                
                self.button_group.addButton(radio)
                camera_layout.addWidget(radio)
            
            # 버튼 클릭 이벤트
            self.button_group.buttonClicked.connect(self.on_camera_selected)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 버튼들
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 새로고침 버튼
        refresh_btn = QPushButton("🔄 새로고침")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        refresh_btn.clicked.connect(self.refresh_cameras)
        button_layout.addWidget(refresh_btn)
        
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
        ok_btn.clicked.connect(self.accept)
        ok_btn.setEnabled(len(self.available_cameras) > 0)
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
    
    def detect_cameras(self):
        """사용 가능한 카메라 감지"""
        self.available_cameras = []
        
        print("[INFO] 카메라 감지 중...")
        
        # 카메라 0-9번까지 테스트
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 카메라 이름 가져오기 (가능한 경우)
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    camera_name = f"{width}x{height}"
                    
                    # 특정 카메라 식별
                    if i == 0:
                        camera_name += " (내장 웹캠)"
                    elif i == 2:
                        camera_name += " (USB 카메라)"
                    
                    self.available_cameras.append((i, camera_name))
                    print(f"[INFO] 카메라 {i}번 감지: {camera_name}")
                
                cap.release()
        
        if not self.available_cameras:
            print("[WARNING] 사용 가능한 카메라를 찾지 못했습니다.")
        else:
            print(f"[INFO] 총 {len(self.available_cameras)}개 카메라 감지 완료")
    
    def on_camera_selected(self, button):
        """카메라 선택 시"""
        self.selected_camera = button.property("camera_id")
        print(f"[INFO] 카메라 {self.selected_camera}번 선택됨")
    
    def refresh_cameras(self):
        """카메라 목록 새로고침"""
        # 현재 다이얼로그 닫고 새로 열기
        self.close()
        new_dialog = CameraSelectionDialog(self.parent())
        result = new_dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self.selected_camera = new_dialog.selected_camera
            self.accept()
        else:
            self.reject()
    
    def get_selected_camera(self):
        """선택된 카메라 번호 반환"""
        return self.selected_camera


def show_camera_selection_dialog(parent=None):
    """
    카메라 선택 다이얼로그 표시
    
    Returns:
        int: 선택된 카메라 번호 (취소 시 None)
    """
    dialog = CameraSelectionDialog(parent)
    result = dialog.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_selected_camera()
    else:
        return None
