#!/usr/bin/env python3
"""
모델 선택 다이얼로그 (개선 버전)
- Random Forest / ST-GCN 선택
- ST-GCN: Original / Fine-tuned 서브 옵션
- 모델 정보 상세 표시
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QFrame, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


class ModelSelectionDialog(QDialog):
    """낙상 감지 모델 선택 다이얼로그"""
    
    # 모델 정보 정의
    MODELS = {
        'random_forest': {
            'name': 'Random Forest',
            'accuracy': 94.50,
            'description': '프레임 단위 즉시 추론\n빠른 응답 속도 (0.01ms)',
            'icon': '🌲',
            'type': 'random_forest',
            'model_path': None,
            'inference_type': 'frame'
        },
        'stgcn_finetuned': {
            'name': 'ST-GCN (Fine-tuned v2)',
            'accuracy': 99.63,
            'description': '60프레임 시퀀스 분석\nPYSKL Pre-trained + 대규모 데이터',
            'icon': '🚀',
            'type': 'stgcn',
            # 'model_path': '/home/gjkong/dev_ws/st_gcn/checkpoints_v2/best_model.pth',
            'model_path': str(PATHS.STGCN_V2),
            'inference_type': 'sequence',
            'model_version': 'finetuned'
        }
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_model = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("🤖 낙상 감지 모델 선택")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # 제목
        title = QLabel("낙상 감지 모델을 선택하세요")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        
        # 라디오 버튼 그룹
        self.button_group = QButtonGroup(self)
        
        # Random Forest 옵션
        rf_widget = self._create_model_option('random_forest')
        layout.addWidget(rf_widget)
        
        # ST-GCN 그룹
        stgcn_group = QGroupBox("ST-GCN 모델")
        stgcn_layout = QVBoxLayout(stgcn_group)
        
        # ST-GCN Fine-tuned (권장 표시)
        stgcn_ft = self._create_model_option('stgcn_finetuned', recommended=True)
        stgcn_layout.addWidget(stgcn_ft)
        
        layout.addWidget(stgcn_group)
        
        # 모델 정보 표시 영역
        self.info_frame = QFrame()
        self.info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        info_layout = QVBoxLayout(self.info_frame)
        
        self.info_label = QLabel("모델을 선택하세요")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(self.info_frame)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("확인")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                padding: 8px 20px;
                border-radius: 5px;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        
        # 기본 선택: Random Forest
        self.button_group.buttons()[0].setChecked(True)
        self._on_model_selected()
    
    def _create_model_option(self, model_key: str, recommended: bool = False) -> QWidget:
        """모델 옵션 위젯 생성"""
        model = self.MODELS[model_key]
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 라디오 버튼
        radio = QRadioButton()
        radio.setProperty('model_key', model_key)
        radio.toggled.connect(self._on_model_selected)
        self.button_group.addButton(radio)
        layout.addWidget(radio)
        
        # 아이콘
        icon_label = QLabel(model['icon'])
        icon_label.setFont(QFont("Arial", 20))
        layout.addWidget(icon_label)
        
        # 정보
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        
        # 이름 + 권장 배지
        name_layout = QHBoxLayout()
        name_label = QLabel(model['name'])
        name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        name_layout.addWidget(name_label)
        
        if recommended:
            badge = QLabel("⭐ 권장")
            badge.setStyleSheet("""
                QLabel {
                    background-color: #FFD700;
                    color: #333;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
            name_layout.addWidget(badge)
        
        name_layout.addStretch()
        info_layout.addLayout(name_layout)
        
        # 정확도
        acc_label = QLabel(f"정확도: {model['accuracy']:.2f}%")
        acc_label.setStyleSheet("color: #666;")
        info_layout.addWidget(acc_label)
        
        layout.addWidget(info_widget)
        layout.addStretch()
        
        # 모델 파일 존재 여부 표시
        if model['model_path']:
            exists = os.path.exists(model['model_path'])
            status = "✅" if exists else "❌ 파일 없음"
            status_label = QLabel(status)
            status_label.setStyleSheet("color: green;" if exists else "color: red;")
            layout.addWidget(status_label)
            
            # 파일이 없으면 라디오 버튼 비활성화
            if not exists:
                radio.setEnabled(False)
        
        return widget
    
    def _on_model_selected(self):
        """모델 선택 시 정보 업데이트"""
        checked_button = self.button_group.checkedButton()
        if not checked_button:
            return
        
        model_key = checked_button.property('model_key')
        model = self.MODELS[model_key]
        self.selected_model = model_key
        
        # 정보 업데이트
        info_text = f"""
<b>{model['icon']} {model['name']}</b><br><br>
<b>정확도:</b> {model['accuracy']:.2f}%<br>
<b>추론 방식:</b> {'프레임 단위' if model['inference_type'] == 'frame' else '60프레임 시퀀스'}<br>
<b>설명:</b> {model['description'].replace(chr(10), '<br>')}
"""
        
        if model['model_path']:
            # 파일 경로 표시 (짧게)
            path = model['model_path']
            short_path = '...' + path[-50:] if len(path) > 50 else path
            info_text += f"<br><b>모델 파일:</b><br><code style='font-size:9px;'>{short_path}</code>"
        
        self.info_label.setText(info_text)
        self.ok_button.setEnabled(True)
    
    def get_selected_model(self) -> dict:
        """선택된 모델 정보 반환"""
        if self.selected_model:
            model = self.MODELS[self.selected_model].copy()
            model['key'] = self.selected_model
            return model
        return None


def show_model_selection_dialog(parent=None) -> dict:
    """
    모델 선택 다이얼로그 표시
    
    Returns:
        선택된 모델 정보 dict 또는 기본값 (취소 시)
    """
    dialog = ModelSelectionDialog(parent)
    result = dialog.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_selected_model()
    else:
        # 기본값: Random Forest
        return {
            'key': 'random_forest',
            'name': 'Random Forest',
            'accuracy': 94.50,
            'type': 'random_forest',
            'model_path': None,
            'inference_type': 'frame'
        }


# 테스트
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    result = show_model_selection_dialog()
    print("\n선택된 모델:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    sys.exit(0)
