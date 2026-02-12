#!/usr/bin/env python3
"""
모델 정보 표시 위젯
모니터링 화면에서 현재 로드된 모델 정보를 상시 표시
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


class ModelInfoWidget(QWidget):
    """현재 로드된 모델 정보 표시 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_info = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # 그룹박스
        group = QGroupBox("🤖 현재 모델")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(5)
        
        # 모델 이름
        self.name_label = QLabel("선택되지 않음")
        self.name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        group_layout.addWidget(self.name_label)
        
        # 모델 타입 (아이콘 + 텍스트)
        type_layout = QHBoxLayout()
        self.type_icon = QLabel("")
        self.type_icon.setFont(QFont("Arial", 14))
        type_layout.addWidget(self.type_icon)
        
        self.type_label = QLabel("")
        self.type_label.setStyleSheet("color: #666;")
        type_layout.addWidget(self.type_label)
        type_layout.addStretch()
        group_layout.addLayout(type_layout)
        
        # 정확도
        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet("color: #333;")
        group_layout.addWidget(self.accuracy_label)
        
        # 모델 버전/경로 (짧게)
        self.version_label = QLabel("")
        self.version_label.setStyleSheet("color: #888; font-size: 9px;")
        self.version_label.setWordWrap(True)
        group_layout.addWidget(self.version_label)
        
        # 상태 표시
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #e8f5e9;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_icon = QLabel("✅")
        status_layout.addWidget(self.status_icon)
        
        self.status_label = QLabel("모델 로드됨")
        self.status_label.setStyleSheet("color: #2e7d32; font-size: 10px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        group_layout.addWidget(self.status_frame)
        
        layout.addWidget(group)
    
    def set_model_info(self, model_info: dict):
        """
        모델 정보 설정
        
        Args:
            model_info: 모델 정보 딕셔너리
                - name: 모델 이름
                - type: 'random_forest' 또는 'stgcn'
                - accuracy: 정확도 (%)
                - model_path: 모델 파일 경로 (선택)
                - model_version: 'original' 또는 'finetuned' (ST-GCN만)
                - inference_type: 'frame' 또는 'sequence'
        """
        self.model_info = model_info
        
        if not model_info:
            self.name_label.setText("선택되지 않음")
            self.type_icon.setText("")
            self.type_label.setText("")
            self.accuracy_label.setText("")
            self.version_label.setText("")
            self._set_status("warning", "모델 없음")
            return
        
        # 이름
        name = model_info.get('name', 'Unknown')
        self.name_label.setText(name)
        
        # 타입 아이콘
        model_type = model_info.get('type', '')
        if model_type == 'random_forest':
            self.type_icon.setText("🌲")
            self.type_label.setText("프레임 단위 추론")
        elif model_type == 'stgcn':
            version = model_info.get('model_version', '')
            if version == 'finetuned':
                self.type_icon.setText("🚀")
                self.type_label.setText("ST-GCN Fine-tuned")
            else:
                self.type_icon.setText("📊")
                self.type_label.setText("ST-GCN Original")
        else:
            self.type_icon.setText("❓")
            self.type_label.setText(model_type)
        
        # 정확도
        accuracy = model_info.get('accuracy', 0)
        self.accuracy_label.setText(f"정확도: {accuracy:.2f}%")
        
        # 버전/경로
        model_path = model_info.get('model_path', '')
        if model_path:
            # 파일명만 표시
            import os
            filename = os.path.basename(model_path)
            self.version_label.setText(f"📁 {filename}")
        else:
            self.version_label.setText("")
        
        # 상태
        self._set_status("success", "모델 로드됨")
    
    def _set_status(self, status_type: str, message: str):
        """상태 표시 업데이트"""
        if status_type == "success":
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #e8f5e9;
                    border-radius: 3px;
                }
            """)
            self.status_icon.setText("✅")
            self.status_label.setStyleSheet("color: #2e7d32; font-size: 10px;")
        elif status_type == "warning":
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #fff3e0;
                    border-radius: 3px;
                }
            """)
            self.status_icon.setText("⚠️")
            self.status_label.setStyleSheet("color: #e65100; font-size: 10px;")
        elif status_type == "error":
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #ffebee;
                    border-radius: 3px;
                }
            """)
            self.status_icon.setText("❌")
            self.status_label.setStyleSheet("color: #c62828; font-size: 10px;")
        elif status_type == "loading":
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #e3f2fd;
                    border-radius: 3px;
                }
            """)
            self.status_icon.setText("⏳")
            self.status_label.setStyleSheet("color: #1565c0; font-size: 10px;")
        
        self.status_label.setText(message)
    
    def set_loading(self):
        """로딩 상태로 변경"""
        self._set_status("loading", "모델 로딩 중...")
    
    def set_error(self, message: str = "모델 로드 실패"):
        """에러 상태로 변경"""
        self._set_status("error", message)
    
    def set_inference_active(self, is_active: bool = True):
        """추론 활성 상태 표시"""
        if is_active:
            self._set_status("success", "추론 중...")
        else:
            self._set_status("success", "모델 로드됨")


class ModelInfoBar(QWidget):
    """컴팩트한 모델 정보 바 (상단 표시용)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
        """)
        
        # 모델 아이콘
        self.icon_label = QLabel("🤖")
        self.icon_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.icon_label)
        
        # 모델 이름
        self.name_label = QLabel("모델: 선택되지 않음")
        self.name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.name_label)
        
        # 구분선
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #ccc;")
        layout.addWidget(sep)
        
        # 정확도
        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet("color: #666;")
        layout.addWidget(self.accuracy_label)
        
        # 구분선
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #ccc;")
        layout.addWidget(sep2)
        
        # 상태
        self.status_label = QLabel("⏳ 대기 중")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 모델 파일명
        self.file_label = QLabel("")
        self.file_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(self.file_label)
    
    def set_model_info(self, model_info: dict):
        """모델 정보 설정"""
        if not model_info:
            self.name_label.setText("모델: 선택되지 않음")
            self.accuracy_label.setText("")
            self.file_label.setText("")
            return
        
        # 아이콘
        model_type = model_info.get('type', '')
        version = model_info.get('model_version', '')
        
        if model_type == 'random_forest':
            self.icon_label.setText("🌲")
        elif model_type == 'stgcn':
            if version == 'finetuned':
                self.icon_label.setText("🚀")
            else:
                self.icon_label.setText("📊")
        
        # 이름
        name = model_info.get('name', 'Unknown')
        self.name_label.setText(f"모델: {name}")
        
        # 정확도
        accuracy = model_info.get('accuracy', 0)
        self.accuracy_label.setText(f"정확도: {accuracy:.2f}%")
        
        # 파일명
        model_path = model_info.get('model_path', '')
        if model_path:
            import os
            self.file_label.setText(f"📁 {os.path.basename(model_path)}")
        else:
            self.file_label.setText("")
        
        self.status_label.setText("✅ 로드됨")
    
    def set_status(self, status: str):
        """상태 텍스트 업데이트"""
        self.status_label.setText(status)


# 테스트
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    app = QApplication(sys.argv)
    
    # 테스트 윈도우
    window = QMainWindow()
    window.setWindowTitle("Model Info Widget Test")
    window.setMinimumSize(400, 500)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    
    # 상단 바
    bar = ModelInfoBar()
    layout.addWidget(bar)
    
    # 상세 위젯
    info = ModelInfoWidget()
    layout.addWidget(info)
    
    layout.addStretch()
    
    window.setCentralWidget(central)
    
    # 테스트 데이터
    test_models = [
        {
            'name': 'Random Forest',
            'type': 'random_forest',
            'accuracy': 93.19,
            'model_path': None,
            'inference_type': 'frame'
        },
        {
            'name': 'ST-GCN (Original)',
            'type': 'stgcn',
            'accuracy': 84.21,
            # 'model_path': '/home/gjkong/dev_ws/st_gcn/checkpoints/best_model_binary.pth',
            'model_path': str(PATHS.STGCN_ORIGINAL) if PATHS.STGCN_ORIGINAL else '',
            'model_version': 'original',
            'inference_type': 'sequence'
        },
        {
            'name': 'ST-GCN (Fine-tuned)',
            'type': 'stgcn',
            'accuracy': 91.89,
            'model_path': str(PATHS.STGCN_FINETUNED),
            # 'model_path': '/home/gjkong/dev_ws/st_gcn/checkpoints_finetuned/best_model_finetuned.pth',
            'model_version': 'finetuned',
            'inference_type': 'sequence'
        }
    ]
    
    # Fine-tuned 모델로 설정
    bar.set_model_info(test_models[2])
    info.set_model_info(test_models[2])
    
    window.show()
    sys.exit(app.exec())
