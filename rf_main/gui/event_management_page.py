"""
Home Safe Solution - 이벤트 관리 페이지
메뉴바에서 접근하는 독립적인 페이지
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGroupBox, QDateEdit, QTimeEdit,
                             QComboBox, QSpinBox, QMessageBox, QFrame,
                             QAbstractItemView)
from PyQt6.QtCore import Qt, QDate, QTime, QDateTime
from PyQt6.QtGui import QFont, QColor
from datetime import datetime, timedelta
from database_models import DatabaseManager, EventLog


class EventManagementPage(QWidget):
    """이벤트 관리 페이지"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.event_log_model = EventLog(db)
        self.events_data = []  # 전체 이벤트 데이터 저장
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # ===== 타이틀 =====
        title_layout = QHBoxLayout()
        
        title_label = QLabel('📋 이벤트 관리')
        title_label.setFont(QFont('Arial', 20, QFont.Weight.Bold))
        title_label.setStyleSheet('color: #2c3e50;')
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # 사용자 정보
        user_label = QLabel(f'👤 {self.user_info["name"]}')
        user_label.setFont(QFont('Arial', 12))
        user_label.setStyleSheet('color: #7f8c8d;')
        title_layout.addWidget(user_label)
        
        layout.addLayout(title_layout)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #bdc3c7;")
        layout.addWidget(line)
        
        # ===== 검색 조건 그룹 =====
        search_group = QGroupBox('🔍 검색 조건')
        search_group.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        search_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
            }
        """)
        search_layout = QVBoxLayout(search_group)
        search_layout.setSpacing(15)
        
        # 날짜/시간 범위
        datetime_layout = QHBoxLayout()
        
        # 시작 날짜/시간
        start_label = QLabel('📅 시작:')
        start_label.setFixedWidth(60)
        start_label.setFont(QFont('Arial', 11))
        datetime_layout.addWidget(start_label)
        
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.setDisplayFormat('yyyy-MM-dd')
        self.start_date.setFixedWidth(140)
        self.start_date.setStyleSheet("""
            QDateEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        datetime_layout.addWidget(self.start_date)
        
        self.start_time = QTimeEdit()
        self.start_time.setTime(QTime(0, 0))
        self.start_time.setDisplayFormat('HH:mm')
        self.start_time.setFixedWidth(80)
        self.start_time.setStyleSheet("""
            QTimeEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        datetime_layout.addWidget(self.start_time)
        
        datetime_layout.addSpacing(30)
        
        # 종료 날짜/시간
        end_label = QLabel('📅 종료:')
        end_label.setFixedWidth(60)
        end_label.setFont(QFont('Arial', 11))
        datetime_layout.addWidget(end_label)
        
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setDisplayFormat('yyyy-MM-dd')
        self.end_date.setFixedWidth(140)
        self.end_date.setStyleSheet("""
            QDateEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        datetime_layout.addWidget(self.end_date)
        
        self.end_time = QTimeEdit()
        self.end_time.setTime(QTime(23, 59))
        self.end_time.setDisplayFormat('HH:mm')
        self.end_time.setFixedWidth(80)
        self.end_time.setStyleSheet("""
            QTimeEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        datetime_layout.addWidget(self.end_time)
        
        datetime_layout.addStretch()
        search_layout.addLayout(datetime_layout)
        
        # 필터 조건
        filter_layout = QHBoxLayout()
        
        # 이벤트 타입
        type_label = QLabel('🏷️ 이벤트 타입:')
        type_label.setFixedWidth(110)
        type_label.setFont(QFont('Arial', 11))
        filter_layout.addWidget(type_label)
        
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(['전체', '정상', '낙상중', '낙상'])
        self.event_type_combo.setFixedWidth(150)
        self.event_type_combo.setStyleSheet("""
            QComboBox {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        filter_layout.addWidget(self.event_type_combo)
        
        filter_layout.addSpacing(30)
        
        # 최소 신뢰도
        conf_label = QLabel('📊 최소 신뢰도:')
        conf_label.setFixedWidth(110)
        conf_label.setFont(QFont('Arial', 11))
        filter_layout.addWidget(conf_label)
        
        self.min_confidence = QSpinBox()
        self.min_confidence.setRange(0, 100)
        self.min_confidence.setValue(0)
        self.min_confidence.setSuffix('%')
        self.min_confidence.setFixedWidth(100)
        self.min_confidence.setStyleSheet("""
            QSpinBox {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        filter_layout.addWidget(self.min_confidence)
        
        filter_layout.addStretch()
        search_layout.addLayout(filter_layout)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 빠른 검색 버튼들
        quick_label = QLabel('⚡ 빠른 검색:')
        quick_label.setFont(QFont('Arial', 11))
        button_layout.addWidget(quick_label)
        
        btn_today = QPushButton('오늘')
        btn_today.clicked.connect(lambda: self.set_quick_range(0))
        btn_today.setFixedHeight(35)
        btn_today.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        button_layout.addWidget(btn_today)
        
        btn_week = QPushButton('7일')
        btn_week.clicked.connect(lambda: self.set_quick_range(7))
        btn_week.setFixedHeight(35)
        btn_week.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        button_layout.addWidget(btn_week)
        
        btn_month = QPushButton('30일')
        btn_month.clicked.connect(lambda: self.set_quick_range(30))
        btn_month.setFixedHeight(35)
        btn_month.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        button_layout.addWidget(btn_month)
        
        button_layout.addStretch()
        
        # 검색 버튼
        btn_search = QPushButton('🔍 검색')
        btn_search.clicked.connect(self.search_events)
        btn_search.setFixedHeight(35)
        btn_search.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px 30px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #229954; }
        """)
        button_layout.addWidget(btn_search)
        
        # 초기화 버튼
        btn_reset = QPushButton('🔄 초기화')
        btn_reset.clicked.connect(self.reset_search)
        btn_reset.setFixedHeight(35)
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 8px 30px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #7f8c8d; }
        """)
        button_layout.addWidget(btn_reset)
        
        search_layout.addLayout(button_layout)
        
        layout.addWidget(search_group)
        
        # ===== 결과 그룹 =====
        result_group = QGroupBox('📊 검색 결과')
        result_group.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        result_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #27ae60;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
            }
        """)
        result_layout = QVBoxLayout(result_group)
        
        # 통계 레이블
        self.stats_label = QLabel('검색 결과: 0건')
        self.stats_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.stats_label.setStyleSheet('color: #2c3e50; padding: 5px;')
        result_layout.addWidget(self.stats_label)
        
        # 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            'ID', '발생 시간', '이벤트 타입', '신뢰도', 
            'Hip Height', 'Spine Angle', 'Status', '비고'
        ])
        
        # 테이블 설정
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #3498db;
                gridline-color: #dee2e6;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
        """)
        
        # 더블클릭 이벤트
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)
        
        # 컬럼 너비 조정
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        
        result_layout.addWidget(self.table)
        
        # 하단 버튼
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        btn_export = QPushButton('📊 Excel 내보내기')
        btn_export.clicked.connect(self.export_to_excel)
        btn_export.setFixedHeight(35)
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #16a085;
                color: white;
                padding: 8px 25px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #138d75; }
        """)
        bottom_layout.addWidget(btn_export)
        
        result_layout.addLayout(bottom_layout)
        
        layout.addWidget(result_group)
        
        # 초기 검색 (최근 7일)
        self.search_events()
    
    def set_quick_range(self, days: int):
        """빠른 날짜 범위 설정"""
        if days == 0:
            self.start_date.setDate(QDate.currentDate())
            self.start_time.setTime(QTime(0, 0))
            self.end_date.setDate(QDate.currentDate())
            self.end_time.setTime(QTime(23, 59))
        else:
            self.start_date.setDate(QDate.currentDate().addDays(-days))
            self.start_time.setTime(QTime(0, 0))
            self.end_date.setDate(QDate.currentDate())
            self.end_time.setTime(QTime(23, 59))
    
    def reset_search(self):
        """검색 조건 초기화"""
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_time.setTime(QTime(0, 0))
        self.end_date.setDate(QDate.currentDate())
        self.end_time.setTime(QTime(23, 59))
        self.event_type_combo.setCurrentIndex(0)
        self.min_confidence.setValue(0)
        self.search_events()
    
    def search_events(self):
        """이벤트 검색"""
        try:
            # 날짜/시간 조합
            start_datetime = QDateTime(self.start_date.date(), self.start_time.time())
            end_datetime = QDateTime(self.end_date.date(), self.end_time.time())
            
            start_str = start_datetime.toString('yyyy-MM-dd HH:mm:ss')
            end_str = end_datetime.toString('yyyy-MM-dd HH:mm:ss')
            
            # 이벤트 타입
            event_type = self.event_type_combo.currentText()
            if event_type == '전체':
                event_type = None
            
            # 최소 신뢰도
            min_conf = self.min_confidence.value() / 100.0
            
            # DB 검색
            events = self.event_log_model.search(
                user_id=self.user_info['user_id'],
                event_type=event_type,
                start_date=start_str,
                end_date=end_str
            )
            
            # 클라이언트 측 신뢰도 필터링
            if min_conf > 0:
                events = [e for e in events if e.get('confidence', 0) >= min_conf]
            
            # 테이블 업데이트
            self.update_table(events)
            
            # 통계 업데이트
            self.update_statistics(events)
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'검색 중 오류 발생:\n{str(e)}')
            print(f"[ERROR] 검색 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def update_table(self, events: list):
        """테이블 업데이트"""
        self.table.setRowCount(0)
        self.events_data = []
        
        for event in events:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # 전체 이벤트 데이터 저장
            self.events_data.append(event)
            
            # ID
            self.table.setItem(row, 0, QTableWidgetItem(str(event['event_id'])))
            
            # 발생 시간
            occurred_at = event['occurred_at']
            if isinstance(occurred_at, str):
                time_str = occurred_at
            else:
                time_str = occurred_at.strftime('%Y-%m-%d %H:%M:%S')
            self.table.setItem(row, 1, QTableWidgetItem(time_str))
            
            # 이벤트 타입 (색상)
            event_type = event['event_type']
            type_item = QTableWidgetItem(event_type)
            if event_type == '정상':
                type_item.setBackground(QColor(46, 204, 113, 50))
            elif event_type == '낙상중':
                type_item.setBackground(QColor(243, 156, 18, 50))
            elif event_type == '낙상':
                type_item.setBackground(QColor(231, 76, 60, 50))
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, type_item)
            
            # 신뢰도
            confidence = event.get('confidence', 0)
            conf_item = QTableWidgetItem(f'{confidence*100:.1f}%')
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, conf_item)
            
            # Hip Height
            hip_height = event.get('hip_height')
            hip_str = f'{hip_height:.1f}' if hip_height else '-'
            hip_item = QTableWidgetItem(hip_str)
            hip_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, hip_item)
            
            # Spine Angle
            spine_angle = event.get('spine_angle')
            spine_str = f'{spine_angle:.1f}°' if spine_angle else '-'
            spine_item = QTableWidgetItem(spine_str)
            spine_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, spine_item)
            
            # Status
            status = event.get('event_status', '발생')
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, status_item)
            
            # 비고
            notes = event.get('notes', '')
            self.table.setItem(row, 7, QTableWidgetItem(notes))
    
    def update_statistics(self, events: list):
        """통계 업데이트"""
        total = len(events)
        normal_count = sum(1 for e in events if e['event_type'] == '정상')
        falling_count = sum(1 for e in events if e['event_type'] == '낙상중')
        fallen_count = sum(1 for e in events if e['event_type'] == '낙상')
        
        stats_text = (
            f'검색 결과: {total}건  '
            f'(정상: {normal_count}건, 낙상중: {falling_count}건, 낙상: {fallen_count}건)'
        )
        
        self.stats_label.setText(stats_text)
    
    def on_row_double_clicked(self, row: int, column: int):
        """테이블 행 더블클릭 시 상세 조회"""
        try:
            if row >= len(self.events_data):
                QMessageBox.warning(self, '오류', '이벤트 데이터를 찾을 수 없습니다.')
                return
            
            event_data = self.events_data[row]
            
            # 상세 조회 다이얼로그
            from event_detail_dialog import EventDetailDialog
            from PyQt6.QtWidgets import QDialog
            
            detail_dialog = EventDetailDialog(event_data, self.db, self)
            result = detail_dialog.exec()
            
            # 삭제 또는 수정 시 목록 갱신
            if result == QDialog.DialogCode.Accepted or detail_dialog.is_modified:
                self.search_events()
                
        except Exception as e:
            QMessageBox.critical(self, '오류', f'상세 조회 중 오류:\n{str(e)}')
            print(f"[ERROR] 상세 조회 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def export_to_excel(self):
        """Excel 내보내기"""
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, '경고', '내보낼 데이터가 없습니다.')
            return
        
        try:
            import csv
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'event_search_{timestamp}.csv'
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 헤더
                headers = []
                for col in range(self.table.columnCount()):
                    headers.append(self.table.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                
                # 데이터
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else '')
                    writer.writerow(row_data)
            
            QMessageBox.information(
                self, 
                '성공', 
                f'검색 결과를 내보냈습니다!\n\n파일: {filename}'
            )
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'내보내기 실패:\n{str(e)}')


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 테스트용
    db = DatabaseManager()
    user_info = {'user_id': 1, 'username': 'test', 'name': '테스트'}
    
    page = EventManagementPage(user_info, db)
    page.show()
    
    sys.exit(app.exec())
