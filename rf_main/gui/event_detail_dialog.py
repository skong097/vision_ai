"""
Home Safe Solution - 이벤트 상세 조회 다이얼로그
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTextEdit, QGroupBox, QMessageBox,
                             QGridLayout, QFrame, QScrollArea, QWidget,
                             QComboBox, QLineEdit)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QFont, QColor, QPixmap
from datetime import datetime
from database_models import DatabaseManager, EventLog
import os


class EventDetailDialog(QDialog):
    """이벤트 상세 조회 다이얼로그"""
    
    def __init__(self, event_data: dict, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.event_data = event_data
        self.db = db
        self.event_log_model = EventLog(db)
        self.is_modified = False
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(f'이벤트 상세 정보 - ID: {self.event_data["event_id"]}')
        self.setMinimumSize(800, 700)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                min-height: 30px;
            }
            QLineEdit, QTextEdit, QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        # ===== 타이틀 =====
        title_layout = QHBoxLayout()
        
        title_label = QLabel(f'📋 이벤트 상세 정보')
        title_label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title_label.setStyleSheet('color: #2c3e50;')
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # 이벤트 ID 배지
        event_id_badge = QLabel(f'ID: {self.event_data["event_id"]}')
        event_id_badge.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        event_id_badge.setStyleSheet("""
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 15px;
        """)
        title_layout.addWidget(event_id_badge)
        
        scroll_layout.addLayout(title_layout)
        
        # ===== 기본 정보 =====
        basic_group = QGroupBox('기본 정보')
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(10)
        
        row = 0
        
        # 발생 시간
        basic_layout.addWidget(QLabel('📅 발생 시간:'), row, 0)
        occurred_at = self.event_data['occurred_at']
        if isinstance(occurred_at, str):
            time_str = occurred_at
        else:
            time_str = occurred_at.strftime('%Y-%m-%d %H:%M:%S')
        time_label = QLabel(time_str)
        time_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        basic_layout.addWidget(time_label, row, 1)
        row += 1
        
        # 이벤트 타입 (편집 가능)
        basic_layout.addWidget(QLabel('🏷️ 이벤트 타입:'), row, 0)
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(['정상', '낙상중', '낙상'])
        self.event_type_combo.setCurrentText(self.event_data['event_type'])
        self.event_type_combo.currentTextChanged.connect(self.on_data_changed)
        basic_layout.addWidget(self.event_type_combo, row, 1)
        row += 1
        
        # 신뢰도
        basic_layout.addWidget(QLabel('📊 신뢰도:'), row, 0)
        confidence = self.event_data.get('confidence', 0) * 100
        confidence_label = QLabel(f'{confidence:.1f}%')
        confidence_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        
        # 신뢰도 색상
        if confidence >= 80:
            confidence_label.setStyleSheet('color: #27ae60;')
        elif confidence >= 60:
            confidence_label.setStyleSheet('color: #f39c12;')
        else:
            confidence_label.setStyleSheet('color: #e74c3c;')
        
        basic_layout.addWidget(confidence_label, row, 1)
        row += 1
        
        # 이벤트 상태 (편집 가능)
        basic_layout.addWidget(QLabel('✅ 이벤트 상태:'), row, 0)
        self.event_status_combo = QComboBox()
        self.event_status_combo.addItems(['발생', '확인', '처리중', '완료', '무시'])
        self.event_status_combo.setCurrentText(self.event_data.get('event_status', '발생'))
        self.event_status_combo.currentTextChanged.connect(self.on_data_changed)
        basic_layout.addWidget(self.event_status_combo, row, 1)
        
        scroll_layout.addWidget(basic_group)
        
        # ===== 센서 데이터 =====
        sensor_group = QGroupBox('센서 데이터')
        sensor_layout = QGridLayout(sensor_group)
        sensor_layout.setSpacing(10)
        
        row = 0
        
        # Hip Height
        hip_height = self.event_data.get('hip_height')
        if hip_height is not None:
            sensor_layout.addWidget(QLabel('📏 Hip Height:'), row, 0)
            hip_label = QLabel(f'{hip_height:.1f} px')
            hip_label.setFont(QFont('Courier New', 11))
            sensor_layout.addWidget(hip_label, row, 1)
            
            # 상태 표시
            if hip_height < 200:
                status_label = QLabel('⚠️ 낮음 (위험)')
                status_label.setStyleSheet('color: #e74c3c; font-weight: bold;')
            elif hip_height < 300:
                status_label = QLabel('⚠️ 중간 (주의)')
                status_label.setStyleSheet('color: #f39c12; font-weight: bold;')
            else:
                status_label = QLabel('✅ 정상')
                status_label.setStyleSheet('color: #27ae60; font-weight: bold;')
            sensor_layout.addWidget(status_label, row, 2)
            row += 1
        
        # Spine Angle
        spine_angle = self.event_data.get('spine_angle')
        if spine_angle is not None:
            sensor_layout.addWidget(QLabel('📐 Spine Angle:'), row, 0)
            spine_label = QLabel(f'{spine_angle:.1f}°')
            spine_label.setFont(QFont('Courier New', 11))
            sensor_layout.addWidget(spine_label, row, 1)
            
            # 상태 표시
            if spine_angle < 90:
                status_label = QLabel('⚠️ 굽힘 (위험)')
                status_label.setStyleSheet('color: #e74c3c; font-weight: bold;')
            elif spine_angle < 150:
                status_label = QLabel('⚠️ 약간 굽힘 (주의)')
                status_label.setStyleSheet('color: #f39c12; font-weight: bold;')
            else:
                status_label = QLabel('✅ 직립')
                status_label.setStyleSheet('color: #27ae60; font-weight: bold;')
            sensor_layout.addWidget(status_label, row, 2)
            row += 1
        
        # Hip Velocity
        hip_velocity = self.event_data.get('hip_velocity')
        if hip_velocity is not None:
            sensor_layout.addWidget(QLabel('⚡ Hip Velocity:'), row, 0)
            velocity_label = QLabel(f'{hip_velocity:.1f} px/frame')
            velocity_label.setFont(QFont('Courier New', 11))
            sensor_layout.addWidget(velocity_label, row, 1)
            
            # 상태 표시
            if abs(hip_velocity) > 50:
                status_label = QLabel('⚠️ 빠른 움직임 (위험)')
                status_label.setStyleSheet('color: #e74c3c; font-weight: bold;')
            elif abs(hip_velocity) > 20:
                status_label = QLabel('⚠️ 보통 속도 (주의)')
                status_label.setStyleSheet('color: #f39c12; font-weight: bold;')
            else:
                status_label = QLabel('✅ 느린 움직임')
                status_label.setStyleSheet('color: #27ae60; font-weight: bold;')
            sensor_layout.addWidget(status_label, row, 2)
            row += 1
        
        if row == 0:
            no_data_label = QLabel('센서 데이터 없음')
            no_data_label.setStyleSheet('color: #95a5a6; font-style: italic;')
            sensor_layout.addWidget(no_data_label, 0, 0, 1, 3)
        
        scroll_layout.addWidget(sensor_group)
        
        # ===== 비고 (편집 가능) =====
        notes_group = QGroupBox('비고')
        notes_layout = QVBoxLayout(notes_group)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.event_data.get('notes', ''))
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.textChanged.connect(self.on_data_changed)
        notes_layout.addWidget(self.notes_edit)
        
        scroll_layout.addWidget(notes_group)
        
        # ===== 영상/썸네일 (미래 기능) =====
        media_group = QGroupBox('영상 & 썸네일')
        media_layout = QVBoxLayout(media_group)
        
        # 썸네일 (구현 예정)
        thumbnail_label = QLabel('📷 썸네일: 구현 예정')
        thumbnail_label.setStyleSheet('color: #95a5a6; font-style: italic; padding: 20px;')
        thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        media_layout.addWidget(thumbnail_label)
        
        # 비디오 재생 버튼 (구현 예정)
        btn_play = QPushButton('▶️ 비디오 재생 (구현 예정)')
        btn_play.setEnabled(False)
        btn_play.setStyleSheet('background-color: #95a5a6; color: white;')
        media_layout.addWidget(btn_play)
        
        scroll_layout.addWidget(media_group)
        
        # ===== 관련 이벤트 =====
        related_group = QGroupBox('관련 이벤트')
        related_layout = QVBoxLayout(related_group)
        
        related_label = QLabel('이 이벤트 전후 5분 이내의 이벤트를 조회할 수 있습니다.')
        related_label.setStyleSheet('color: #7f8c8d; font-style: italic;')
        related_layout.addWidget(related_label)
        
        btn_show_related = QPushButton('🔍 관련 이벤트 보기')
        btn_show_related.clicked.connect(self.show_related_events)
        btn_show_related.setStyleSheet('background-color: #3498db; color: white;')
        related_layout.addWidget(btn_show_related)
        
        scroll_layout.addWidget(related_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # ===== 하단 버튼 =====
        button_layout = QHBoxLayout()
        
        # 삭제 버튼 (왼쪽)
        self.btn_delete = QPushButton('🗑️ 삭제')
        self.btn_delete.clicked.connect(self.delete_event)
        self.btn_delete.setStyleSheet('background-color: #e74c3c; color: white;')
        button_layout.addWidget(self.btn_delete)
        
        button_layout.addStretch()
        
        # 저장 버튼
        self.btn_save = QPushButton('💾 저장')
        self.btn_save.clicked.connect(self.save_changes)
        self.btn_save.setStyleSheet('background-color: #27ae60; color: white;')
        self.btn_save.setEnabled(False)
        button_layout.addWidget(self.btn_save)
        
        # 닫기 버튼
        btn_close = QPushButton('닫기')
        btn_close.clicked.connect(self.close_dialog)
        btn_close.setStyleSheet('background-color: #7f8c8d; color: white;')
        button_layout.addWidget(btn_close)
        
        layout.addLayout(button_layout)
    
    def on_data_changed(self):
        """데이터 변경 시"""
        self.is_modified = True
        self.btn_save.setEnabled(True)
    
    def save_changes(self):
        """변경사항 저장"""
        try:
            # 업데이트할 데이터
            event_type = self.event_type_combo.currentText()
            event_status = self.event_status_combo.currentText()
            notes = self.notes_edit.toPlainText()
            
            # DB 업데이트
            success = self.event_log_model.update(
                event_id=self.event_data['event_id'],
                event_type=event_type,
                event_status=event_status,
                notes=notes
            )
            
            if success:
                QMessageBox.information(self, '성공', '변경사항이 저장되었습니다.')
                self.is_modified = False
                self.btn_save.setEnabled(False)
                
                # 데이터 업데이트
                self.event_data['event_type'] = event_type
                self.event_data['event_status'] = event_status
                self.event_data['notes'] = notes
            else:
                QMessageBox.warning(self, '실패', '저장에 실패했습니다.')
                
        except Exception as e:
            QMessageBox.critical(self, '오류', f'저장 중 오류 발생:\n{str(e)}')
    
    def delete_event(self):
        """이벤트 삭제"""
        reply = QMessageBox.question(
            self,
            '삭제 확인',
            f'이벤트 ID {self.event_data["event_id"]}를 정말 삭제하시겠습니까?\n\n'
            '이 작업은 되돌릴 수 없습니다.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                success = self.event_log_model.delete(self.event_data['event_id'])
                
                if success:
                    QMessageBox.information(self, '성공', '이벤트가 삭제되었습니다.')
                    self.accept()  # 다이얼로그 닫기 (성공)
                else:
                    QMessageBox.warning(self, '실패', '삭제에 실패했습니다.')
                    
            except Exception as e:
                QMessageBox.critical(self, '오류', f'삭제 중 오류 발생:\n{str(e)}')
    
    def show_related_events(self):
        """관련 이벤트 표시"""
        try:
            occurred_at = self.event_data['occurred_at']
            if isinstance(occurred_at, str):
                occurred_at = datetime.strptime(occurred_at, '%Y-%m-%d %H:%M:%S')
            
            # 전후 5분
            from datetime import timedelta
            start_time = (occurred_at - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            end_time = (occurred_at + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            
            # 관련 이벤트 조회
            events = self.event_log_model.search(
                user_id=None,  # 동일 사용자만 (구현 필요)
                start_date=start_time,
                end_date=end_time
            )
            
            # 현재 이벤트 제외
            events = [e for e in events if e['event_id'] != self.event_data['event_id']]
            
            if events:
                msg = f'전후 5분 이내에 {len(events)}개의 이벤트가 있습니다:\n\n'
                for e in events[:10]:  # 최대 10개만
                    event_time = e['occurred_at']
                    if isinstance(event_time, str):
                        time_str = event_time
                    else:
                        time_str = event_time.strftime('%H:%M:%S')
                    msg += f'• [{time_str}] {e["event_type"]} (신뢰도: {e["confidence"]*100:.0f}%)\n'
                
                if len(events) > 10:
                    msg += f'\n... 외 {len(events)-10}개'
                
                QMessageBox.information(self, '관련 이벤트', msg)
            else:
                QMessageBox.information(self, '관련 이벤트', '전후 5분 이내에 다른 이벤트가 없습니다.')
                
        except Exception as e:
            QMessageBox.critical(self, '오류', f'관련 이벤트 조회 중 오류:\n{str(e)}')
    
    def close_dialog(self):
        """다이얼로그 닫기"""
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                '저장 안 됨',
                '변경사항이 저장되지 않았습니다.\n\n정말 닫으시겠습니까?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.reject()
        else:
            self.reject()
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                '저장 안 됨',
                '변경사항이 저장되지 않았습니다.\n\n정말 닫으시겠습니까?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 테스트용
    db = DatabaseManager()
    
    test_event = {
        'event_id': 1,
        'occurred_at': datetime.now(),
        'event_type': '낙상',
        'confidence': 0.87,
        'hip_height': 150.5,
        'spine_angle': 75.2,
        'hip_velocity': 80.3,
        'event_status': '발생',
        'notes': '테스트 이벤트입니다.'
    }
    
    dialog = EventDetailDialog(test_event, db)
    dialog.show()
    
    sys.exit(app.exec())
