"""
Home Safe Solution - 로그인 화면
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QCheckBox,
                             QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon, QScreen
from database_models import DatabaseManager, User


class LoginWindow(QWidget):
    """로그인 윈도우"""
    
    # 로그인 성공 시그널
    login_success = pyqtSignal(dict)  # 사용자 정보 전달
    
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.user_model = User(self.db)
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('Home Safe - 로그인')
        self.setFixedSize(400, 500)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            QLineEdit {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                padding: 12px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(20)
        
        # 로고/타이틀
        title_label = QLabel('🏠 Home Safe')
        title_label.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('color: #4CAF50; margin-bottom: 10px;')
        
        subtitle_label = QLabel('낙상 감지 시스템')
        subtitle_label.setFont(QFont('Arial', 12))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet('color: #666; margin-bottom: 30px;')
        
        # 아이디 입력
        id_label = QLabel('아이디')
        id_label.setFont(QFont('Arial', 10))
        id_label.setStyleSheet('color: #333;')
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText('아이디를 입력하세요')
        self.id_input.returnPressed.connect(self.login)
        
        # 비밀번호 입력
        pw_label = QLabel('비밀번호')
        pw_label.setFont(QFont('Arial', 10))
        pw_label.setStyleSheet('color: #333;')
        
        self.pw_input = QLineEdit()
        self.pw_input.setPlaceholderText('비밀번호를 입력하세요')
        self.pw_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw_input.returnPressed.connect(self.login)
        
        # 로그인 유지 체크박스
        self.remember_checkbox = QCheckBox('로그인 상태 유지')
        self.remember_checkbox.setStyleSheet('color: #666;')
        
        # 로그인 버튼
        login_btn = QPushButton('로그인')
        login_btn.clicked.connect(self.login)
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 회원가입 버튼
        register_btn = QPushButton('회원가입')
        register_btn.clicked.connect(self.show_register)
        register_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        register_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        
        # 레이아웃 구성
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addSpacing(20)
        main_layout.addWidget(id_label)
        main_layout.addWidget(self.id_input)
        main_layout.addWidget(pw_label)
        main_layout.addWidget(self.pw_input)
        main_layout.addWidget(self.remember_checkbox)
        main_layout.addSpacing(10)
        main_layout.addWidget(login_btn)
        main_layout.addWidget(register_btn)
        main_layout.addStretch()
        
        # 여백 추가
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addLayout(main_layout)
        container_layout.setContentsMargins(40, 40, 40, 40)
        container.setLayout(container_layout)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(container)
        final_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(final_layout)
        
        # 중앙 배치
        self.center()
    
    def center(self):
        """화면 중앙에 배치"""
        from PyQt6.QtGui import QScreen
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def login(self):
        """로그인 처리"""
        username = self.id_input.text().strip()
        password = self.pw_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, '입력 오류', '아이디와 비밀번호를 입력하세요.')
            return
        
        # 인증
        user = self.user_model.authenticate(username, password)
        
        if user:
            # 로그인 성공
            QMessageBox.information(self, '로그인 성공', 
                                  f'{user["name"]}님, 환영합니다!')
            
            # 로그인 히스토리 기록
            self.record_login(user['user_id'])
            
            # 시그널 발생
            self.login_success.emit(user)
            self.close()
        else:
            # 로그인 실패
            QMessageBox.warning(self, '로그인 실패', 
                              '아이디 또는 비밀번호가 일치하지 않습니다.')
            self.pw_input.clear()
            self.pw_input.setFocus()
    
    def record_login(self, user_id: int):
        """로그인 히스토리 기록"""
        query = """
        INSERT INTO login_history (user_id, login_status)
        VALUES (%s, '성공')
        """
        self.db.execute_update(query, (user_id,))
    
    def show_register(self):
        """회원가입 화면 표시"""
        QMessageBox.information(self, '회원가입', 
                              '회원가입 기능은 관리자에게 문의하세요.')


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 기본 폰트 설정
    font = QFont('맑은 고딕', 10)
    app.setFont(font)
    
    window = LoginWindow()
    window.show()
    
    sys.exit(app.exec())