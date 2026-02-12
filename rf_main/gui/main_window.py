"""
Home Safe Solution - 메인 윈도우
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QStackedWidget, QPushButton, QLabel, QFrame,
                             QApplication, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from database_models import DatabaseManager
from training_page import TrainingPage


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self, user_info: dict):
        super().__init__()
        self.user_info = user_info
        self.db = DatabaseManager()
        self.current_page = None
        self.init_ui()
        
        # 상태 업데이트 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 1초마다 업데이트
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('비전 홈 케어 도우미')
        self.setMinimumSize(1400, 900)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 메인 레이아웃 (수평)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 왼쪽 사이드바
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)
        
        # 오른쪽 콘텐츠 영역
        content_area = self.create_content_area()
        main_layout.addWidget(content_area, 1)
        
        # 스타일시트
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
        """)
        
        # 중앙 배치
        self.center()
    
    def create_sidebar(self) -> QFrame:
        """사이드바 생성"""
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-right: 1px solid #34495e;
            }
            QPushButton {
                text-align: left;
                padding: 15px 20px;
                border: none;
                background-color: transparent;
                color: #ecf0f1;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:checked {
                background-color: #3498db;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 로고/사용자 정보
        header = QFrame()
        header.setStyleSheet("background-color: #1a252f; padding: 20px;")
        header_layout = QVBoxLayout(header)
        
        title = QLabel('🏠 비전 홈 케어')
        title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title.setStyleSheet('color: white;')
        
        user_name = QLabel(f'{self.user_info["name"]}님')
        user_name.setStyleSheet('color: #95a5a6; font-size: 12px;')
        
        user_type = QLabel(f'({self.user_info["user_type"]})')
        user_type.setStyleSheet('color: #7f8c8d; font-size: 11px;')
        
        header_layout.addWidget(title)
        header_layout.addWidget(user_name)
        header_layout.addWidget(user_type)
        
        layout.addWidget(header)
        
        # 메뉴 버튼들
        self.menu_buttons = []
        
        # 대시보드
        btn_dashboard = QPushButton('📊  대시보드')
        btn_dashboard.setCheckable(True)
        btn_dashboard.setChecked(True)
        btn_dashboard.clicked.connect(lambda: self.change_page('dashboard'))
        self.menu_buttons.append(btn_dashboard)
        
        # 실시간 모니터링
        btn_monitoring = QPushButton('🎥  실시간 모니터링')
        btn_monitoring.setCheckable(True)
        btn_monitoring.clicked.connect(lambda: self.change_page('monitoring'))
        self.menu_buttons.append(btn_monitoring)
        
        # 이벤트 로그
        btn_events = QPushButton('📋  이벤트 로그')
        btn_events.setCheckable(True)
        btn_events.clicked.connect(lambda: self.change_page('events'))
        self.menu_buttons.append(btn_events)
        
        # 사용자 관리 (관리자만)
        if self.user_info['user_type'] == '관리자':
            btn_users = QPushButton('👥  사용자 관리')
            btn_users.setCheckable(True)
            btn_users.clicked.connect(lambda: self.change_page('users'))
            self.menu_buttons.append(btn_users)
        
        # 설정
        btn_settings = QPushButton('⚙️  설정')
        btn_settings.setCheckable(True)
        btn_settings.clicked.connect(lambda: self.change_page('settings'))
        self.menu_buttons.append(btn_settings)
        
        # ⭐ 학습 파이프라인 (관리자만)
        if self.user_info['user_type'] == '관리자':
            btn_training = QPushButton('🎓  모델 학습')
            btn_training.setCheckable(True)
            btn_training.clicked.connect(lambda: self.change_page('training'))
            self.menu_buttons.append(btn_training)
            self.btn_training = btn_training  # 참조 저장
        
        # 버튼 추가
        for btn in self.menu_buttons:
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # 로그아웃 버튼
        btn_logout = QPushButton('🚪  로그아웃')
        btn_logout.clicked.connect(self.logout)
        btn_logout.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                margin: 10px;
                border-radius: 5px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        layout.addWidget(btn_logout)
        
        return sidebar
    
    def create_content_area(self) -> QWidget:
        """콘텐츠 영역 생성"""
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 상단 바
        top_bar = self.create_top_bar()
        layout.addWidget(top_bar)
        
        # 페이지 스택
        self.page_stack = QStackedWidget()
        layout.addWidget(self.page_stack)
        
        # 페이지 추가
        from dashboard_page import DashboardPage
        from monitoring_page import MonitoringPage
        from events_page import EventsPage
        from users_page import UsersPage
        from settings_page import SettingsPage
        
        self.page_stack.addWidget(DashboardPage(self.user_info, self.db))  # 0: 대시보드
        self.page_stack.addWidget(MonitoringPage(self.user_info, self.db))  # 1: 모니터링
        self.page_stack.addWidget(EventsPage(self.user_info, self.db))  # 2: 이벤트
        
        if self.user_info['user_type'] == '관리자':
            self.page_stack.addWidget(UsersPage(self.user_info, self.db))  # 3: 사용자
            self.page_stack.addWidget(SettingsPage(self.user_info, self.db))  # 4: 설정
            # ⭐ TrainingPage 추가 (관리자만)
            self.training_page = TrainingPage()
            self.page_stack.addWidget(self.training_page)  # 5: 학습 파이프라인
        else:
            self.page_stack.addWidget(SettingsPage(self.user_info, self.db))  # 3: 설정
        
        return content_widget
    
    def create_top_bar(self) -> QFrame:
        """상단 바 생성"""
        top_bar = QFrame()
        top_bar.setFixedHeight(60)
        top_bar.setStyleSheet("""
            QFrame {
                background-color: white;
                border-bottom: 1px solid #ddd;
            }
        """)
        
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(30, 10, 30, 10)
        
        # 페이지 제목
        self.page_title = QLabel('📊 대시보드')
        self.page_title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        self.page_title.setStyleSheet('color: #2c3e50;')
        
        # 상태 표시
        self.status_label = QLabel('🟢 시스템 정상')
        self.status_label.setStyleSheet('color: #27ae60; font-size: 12px;')
        
        # 시간 표시
        self.time_label = QLabel()
        self.time_label.setStyleSheet('color: #7f8c8d; font-size: 12px;')
        self.update_time()
        
        layout.addWidget(self.page_title)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel('  |  '))
        layout.addWidget(self.time_label)
        
        return top_bar
    
    def change_page(self, page_name: str):
        """페이지 변경"""
        # 버튼 체크 상태 업데이트
        for btn in self.menu_buttons:
            btn.setChecked(False)
        
        sender = self.sender()
        if sender:
            sender.setChecked(True)
        
        # 페이지 전환 맵 (관리자 기준)
        if self.user_info['user_type'] == '관리자':
            page_map = {
                'dashboard': (0, '📊 대시보드'),
                'monitoring': (1, '🎥 실시간 모니터링'),
                'events': (2, '📋 이벤트 로그'),
                'users': (3, '👥 사용자 관리'),
                'settings': (4, '⚙️ 설정'),
                'training': (5, '🎓 모델 학습'),  # ⭐ 추가
            }
        else:
            page_map = {
                'dashboard': (0, '📊 대시보드'),
                'monitoring': (1, '🎥 실시간 모니터링'),
                'events': (2, '📋 이벤트 로그'),
                'settings': (3, '⚙️ 설정'),
            }
        
        if page_name in page_map:
            index, title = page_map[page_name]
            self.page_stack.setCurrentIndex(index)
            self.page_title.setText(title)
            self.current_page = page_name
    
    def update_status(self):
        """상태 업데이트"""
        # 시스템 상태 확인 (예: 카메라 연결, 모델 로드 등)
        # TODO: 실제 상태 체크 로직 구현
        pass
    
    def update_time(self):
        """시간 업데이트"""
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.setText(current_time)
        
        # 타이머로 1초마다 업데이트
        QTimer.singleShot(1000, self.update_time)
    
    def logout(self):
        """로그아웃"""
        reply = QMessageBox.question(
            self, '로그아웃', '로그아웃 하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 로그아웃 히스토리 기록
            query = """
            UPDATE login_history 
            SET logout_at = NOW()
            WHERE user_id = %s AND logout_at IS NULL
            ORDER BY login_at DESC LIMIT 1
            """
            self.db.execute_update(query, (self.user_info['user_id'],))
            
            # 로그인 화면으로
            self.close()
            from login_window import LoginWindow
            self.login_window = LoginWindow()
            self.login_window.login_success.connect(self.on_login_success)
            self.login_window.show()
    
    def on_login_success(self, user_info: dict):
        """로그인 성공 시"""
        self.user_info = user_info
        self.__init__(user_info)
        self.show()
    
    def center(self):
        """화면 중앙에 배치"""
        from PyQt6.QtGui import QScreen
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def closeEvent(self, event):
        """종료 이벤트"""
        # 모니터링 중지
        if hasattr(self, 'page_stack'):
            for i in range(self.page_stack.count()):
                page = self.page_stack.widget(i)
                if hasattr(page, 'stop_monitoring'):
                    page.stop_monitoring()
                # TrainingPage cleanup
                if hasattr(page, 'cleanup'):
                    page.cleanup()
        
        event.accept()
