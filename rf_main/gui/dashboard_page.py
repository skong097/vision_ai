"""
Home Safe Solution - 대시보드 페이지
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
                             QPushButton, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCharts import QChart, QChartView, QPieSeries, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from database_models import DatabaseManager, EventLog
from datetime import datetime, timedelta


class DashboardPage(QWidget):
    """대시보드 페이지"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.event_model = EventLog(db)
        self.init_ui()
        
        # 자동 업데이트 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(5000)  # 5초마다 업데이트
        
        # 초기 데이터 로드
        self.update_dashboard()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 상단: 통계 카드들
        stats_layout = QHBoxLayout()
        
        self.card_total = self.create_stat_card('총 이벤트', '0', '#3498db')
        self.card_today = self.create_stat_card('오늘', '0', '#2ecc71')
        self.card_fall = self.create_stat_card('낙상/쓰러짐', '0', '#e74c3c')
        self.card_normal = self.create_stat_card('정상', '0', '#95a5a6')
        
        stats_layout.addWidget(self.card_total)
        stats_layout.addWidget(self.card_today)
        stats_layout.addWidget(self.card_fall)
        stats_layout.addWidget(self.card_normal)
        
        layout.addLayout(stats_layout)
        
        # 중간: 차트 영역
        charts_layout = QHBoxLayout()
        
        # 이벤트 타입별 차트
        self.pie_chart_view = self.create_pie_chart()
        charts_layout.addWidget(self.pie_chart_view, 1)
        
        # 시간대별 차트
        self.bar_chart_view = self.create_bar_chart()
        charts_layout.addWidget(self.bar_chart_view, 1)
        
        layout.addLayout(charts_layout)
        
        # 하단: 최근 이벤트 테이블
        recent_label = QLabel('최근 이벤트')
        recent_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        layout.addWidget(recent_label)
        
        self.recent_table = self.create_recent_events_table()
        layout.addWidget(self.recent_table)
    
    def create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        """통계 카드 생성"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 10px;
                border-left: 5px solid {color};
            }}
        """)
        card.setFixedHeight(120)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QLabel(title)
        title_label.setStyleSheet('color: #7f8c8d; font-size: 12px;')
        
        value_label = QLabel(value)
        value_label.setFont(QFont('Arial', 28, QFont.Weight.Bold))
        value_label.setStyleSheet(f'color: {color};')
        
        # 동적 업데이트를 위해 참조 저장
        card.value_label = value_label
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        
        return card
    
    def create_pie_chart(self) -> QChartView:
        """파이 차트 생성"""
        series = QPieSeries()
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle('이벤트 타입별 분포 (최근 30일)')
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(chart_view.renderHints())
        chart_view.setStyleSheet("background-color: white; border-radius: 10px;")
        
        # 차트 참조 저장
        chart_view.series = series
        chart_view.chart = chart
        
        return chart_view
    
    def create_bar_chart(self) -> QChartView:
        """막대 차트 생성"""
        series = QBarSeries()
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle('시간대별 이벤트 발생 (최근 7일)')
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(chart_view.renderHints())
        chart_view.setStyleSheet("background-color: white; border-radius: 10px;")
        
        # 차트 참조 저장
        chart_view.series = series
        chart_view.chart = chart
        
        return chart_view
    
    def create_recent_events_table(self) -> QTableWidget:
        """최근 이벤트 테이블 생성"""
        table = QTableWidget()
        table.setColumnCount(7)  # ⭐ 6개 → 7개
        table.setHorizontalHeaderLabels([
            '발생시간', '사용자', '이벤트 유형', '상태', '신뢰도', '낙상 탐지율', '조치'  # ⭐ '낙상 탐지율' 추가
        ])
        
        # 스타일
        table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border-radius: 10px;
                gridline-color: #ecf0f1;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # 컬럼 크기 조정
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # 발생시간
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # 사용자
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # 이벤트 유형
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # 상태
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # 신뢰도
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # ⭐ 낙상 탐지율
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)           # 조치
        
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        
        return table
    
    def update_dashboard(self):
        """대시보드 데이터 업데이트"""
        try:
            # 통계 데이터
            self.update_statistics()
            
            # 차트 데이터
            self.update_pie_chart()
            self.update_bar_chart()
            
            # 최근 이벤트
            self.update_recent_events()
            
        except Exception as e:
            print(f"대시보드 업데이트 오류: {e}")
    
    def update_statistics(self):
        """통계 카드 업데이트"""
        user_id = None if self.user_info['user_type'] == '관리자' else self.user_info['user_id']
        
        # 총 이벤트
        total_query = "SELECT COUNT(*) as count FROM event_logs"
        params = []
        if user_id:
            total_query += " WHERE user_id = %s"
            params.append(user_id)
        
        total_result = self.db.execute_query(total_query, tuple(params) if params else None)
        total_count = total_result[0]['count'] if total_result else 0
        self.card_total.value_label.setText(str(total_count))
        
        # 오늘 이벤트
        today_query = f"{total_query} {'AND' if user_id else 'WHERE'} DATE(occurred_at) = CURDATE()"
        today_result = self.db.execute_query(today_query, tuple(params) if params else None)
        today_count = today_result[0]['count'] if today_result else 0
        self.card_today.value_label.setText(str(today_count))
        
        # 낙상/쓰러짐 (최근 30일)
        fall_query = """
        SELECT COUNT(*) as count 
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE et.type_name IN ('낙상', '쓰러짐')
        AND el.occurred_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """
        if user_id:
            fall_query += " AND el.user_id = %s"
        
        fall_result = self.db.execute_query(fall_query, (user_id,) if user_id else None)
        fall_count = fall_result[0]['count'] if fall_result else 0
        self.card_fall.value_label.setText(str(fall_count))
        
        # 정상 (최근 30일)
        normal_query = """
        SELECT COUNT(*) as count 
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE et.type_name = '정상'
        AND el.occurred_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """
        if user_id:
            normal_query += " AND el.user_id = %s"
        
        normal_result = self.db.execute_query(normal_query, (user_id,) if user_id else None)
        normal_count = normal_result[0]['count'] if normal_result else 0
        self.card_normal.value_label.setText(str(normal_count))
    
    def update_pie_chart(self):
        """파이 차트 업데이트"""
        user_id = None if self.user_info['user_type'] == '관리자' else self.user_info['user_id']
        
        query = """
        SELECT et.type_name, COUNT(*) as count
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE el.occurred_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """
        if user_id:
            query += " AND el.user_id = %s"
        query += " GROUP BY et.type_name"
        
        results = self.db.execute_query(query, (user_id,) if user_id else None)
        
        # 시리즈 업데이트
        series = self.pie_chart_view.series
        series.clear()
        
        for row in results:
            series.append(row['type_name'], row['count'])
    
    def update_bar_chart(self):
        """막대 차트 업데이트"""
        user_id = None if self.user_info['user_type'] == '관리자' else self.user_info['user_id']
        
        # 최근 7일 데이터
        query = """
        SELECT 
            DATE(occurred_at) as date,
            et.type_name,
            COUNT(*) as count
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE el.occurred_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        """
        if user_id:
            query += " AND el.user_id = %s"
        query += " GROUP BY DATE(occurred_at), et.type_name ORDER BY date"
        
        results = self.db.execute_query(query, (user_id,) if user_id else None)
        
        # 데이터 구조화
        data_by_type = {}
        dates = []
        
        for row in results:
            date_str = row['date'].strftime('%m/%d')
            if date_str not in dates:
                dates.append(date_str)
            
            type_name = row['type_name']
            if type_name not in data_by_type:
                data_by_type[type_name] = {}
            
            data_by_type[type_name][date_str] = row['count']
        
        # 차트 업데이트
        chart = self.bar_chart_view.chart
        chart.removeAllSeries()
        
        # 축 먼저 제거
        for axis in chart.axes():
            chart.removeAxis(axis)
        
        series = QBarSeries()
        
        for type_name, data in data_by_type.items():
            bar_set = QBarSet(type_name)
            for date in dates:
                bar_set.append(data.get(date, 0))
            series.append(bar_set)
        
        chart.addSeries(series)
        
        # 축 설정
        axis_x = QBarCategoryAxis()
        axis_x.append(dates)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
    
    def update_recent_events(self):
        """최근 이벤트 테이블 업데이트"""
        user_id = None if self.user_info['user_type'] == '관리자' else self.user_info['user_id']
        
        events = self.event_model.get_recent(user_id=user_id, limit=10)
        
        self.recent_table.setRowCount(len(events))
        
        for i, event in enumerate(events):
            # 발생시간
            occurred_at = event['occurred_at'].strftime('%Y-%m-%d %H:%M:%S')
            self.recent_table.setItem(i, 0, QTableWidgetItem(occurred_at))
            
            # 사용자
            self.recent_table.setItem(i, 1, QTableWidgetItem(event['user_name']))
            
            # 이벤트 유형
            event_type_item = QTableWidgetItem(event['event_type'])
            if event['severity'] == '위험':
                event_type_item.setForeground(QColor('#e74c3c'))
            elif event['severity'] == '경고':
                event_type_item.setForeground(QColor('#f39c12'))
            self.recent_table.setItem(i, 2, event_type_item)
            
            # 상태
            status_item = QTableWidgetItem(event['event_status'])
            if event['event_status'] == '완료':
                status_item.setForeground(QColor('#27ae60'))
            self.recent_table.setItem(i, 3, status_item)
            
            # 신뢰도
            confidence = f"{event['confidence']*100:.1f}%" if event['confidence'] else 'N/A'
            self.recent_table.setItem(i, 4, QTableWidgetItem(confidence))
            
            # ⭐ 낙상 탐지율
            accuracy = event.get('accuracy')
            if accuracy is not None:
                accuracy_text = f"{accuracy:.1f}%"
                accuracy_item = QTableWidgetItem(accuracy_text)
                
                # 색상 구분
                if accuracy >= 90:
                    accuracy_item.setForeground(QColor('#27ae60'))  # 녹색
                elif accuracy >= 70:
                    accuracy_item.setForeground(QColor('#f39c12'))  # 주황
                else:
                    accuracy_item.setForeground(QColor('#e74c3c'))  # 빨강
                
                self.recent_table.setItem(i, 5, accuracy_item)
            else:
                self.recent_table.setItem(i, 5, QTableWidgetItem('N/A'))
            
            # 조치
            action = event['action_taken'] if event['action_taken'] else '없음'
            self.recent_table.setItem(i, 6, QTableWidgetItem(action))  # ⭐ 5 → 6