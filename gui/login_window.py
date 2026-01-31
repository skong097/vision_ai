"""
Home Safe Solution - 로그인 화면
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QCheckBox,
                             QApplication, QDialog, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon, QScreen
from database_models import DatabaseManager, User


class RegisterDialog(QDialog):
    """회원가입 다이얼로그"""
    
    def __init__(self, db, user_model, parent=None):
        super().__init__(parent)
        self.db = db
        self.user_model = user_model
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('회원가입')
        self.setFixedSize(420, 620)  # 높이 증가: 550 → 620
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLineEdit, QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 13px;
                background-color: white;
                min-height: 25px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QPushButton {
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                min-height: 38px;
            }
            QLabel {
                color: #333;
                font-size: 12px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)  # 간격 축소: 10 → 8
        layout.setContentsMargins(30, 25, 30, 25)  # 상하 여백 축소
        
        # 타이틀
        title = QLabel('🏠 회원가입')
        title.setFont(QFont('Arial', 18, QFont.Weight.Bold))  # 폰트 축소: 20 → 18
        title.setStyleSheet('color: #2196F3; margin-bottom: 15px;')  # 마진 축소: 20 → 15
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 아이디
        id_label = QLabel('* 아이디')
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText('영문, 숫자 조합 (4-20자)')
        layout.addWidget(id_label)
        layout.addWidget(self.id_input)
        
        # 비밀번호
        pw_label = QLabel('* 비밀번호')
        self.pw_input = QLineEdit()
        self.pw_input.setPlaceholderText('비밀번호 입력')
        self.pw_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(pw_label)
        layout.addWidget(self.pw_input)
        
        # 비밀번호 확인
        pw_confirm_label = QLabel('* 비밀번호 확인')
        self.pw_confirm_input = QLineEdit()
        self.pw_confirm_input.setPlaceholderText('비밀번호 재입력')
        self.pw_confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(pw_confirm_label)
        layout.addWidget(self.pw_confirm_input)
        
        # 이름
        name_label = QLabel('* 이름')
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText('실명 입력')
        layout.addWidget(name_label)
        layout.addWidget(self.name_input)
        
        # 성별
        gender_label = QLabel('* 성별')
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['남성', '여성', '기타'])
        layout.addWidget(gender_label)
        layout.addWidget(self.gender_combo)
        
        layout.addSpacing(5)  # 간격 축소: 10 → 5
        
        # 버튼
        button_layout = QHBoxLayout()
        
        register_btn = QPushButton('가입하기')
        register_btn.clicked.connect(self.register)
        register_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        
        cancel_btn = QPushButton('취소')
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #999;
                color: white;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(register_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def register(self):
        """회원가입 처리"""
        username = self.id_input.text().strip()
        password = self.pw_input.text()
        password_confirm = self.pw_confirm_input.text()
        name = self.name_input.text().strip()
        gender = self.gender_combo.currentText()
        
        # 입력 검증
        if not username or not password or not name:
            QMessageBox.warning(self, '입력 오류', '모든 필수 항목(*)을 입력해주세요.')
            return
        
        if len(username) < 4 or len(username) > 20:
            QMessageBox.warning(self, '입력 오류', '아이디는 4-20자로 입력해주세요.')
            return
        
        if len(password) < 4:
            QMessageBox.warning(self, '입력 오류', '비밀번호는 최소 4자 이상이어야 합니다.')
            return
        
        if password != password_confirm:
            QMessageBox.warning(self, '입력 오류', '비밀번호가 일치하지 않습니다.')
            self.pw_confirm_input.clear()
            self.pw_confirm_input.setFocus()
            return
        
        # 중복 체크
        existing_user = self.user_model.get_by_username(username)
        if existing_user:
            QMessageBox.warning(self, '중복 오류', '이미 사용 중인 아이디입니다.')
            self.id_input.setFocus()
            return
        
        try:
            # ✅ User.create() 사용 (자동으로 bcrypt 처리됨)
            user_id = self.user_model.create(
                username=username,
                password=password,  # 평문 입력 (자동 해시됨)
                name=name,
                gender=gender,
                user_type='일반유저'
            )
            
            if user_id:
                QMessageBox.information(
                    self, 
                    '가입 완료', 
                    f'회원가입이 완료되었습니다!\n\n'
                    f'아이디: {username}\n'
                    f'이름: {name}\n\n'
                    f'로그인 화면에서 로그인해주세요.'
                )
                self.accept()  # 다이얼로그 닫기
            else:
                QMessageBox.critical(self, '가입 실패', '회원가입에 실패했습니다.')
        
        except Exception as e:
            QMessageBox.critical(self, '오류', f'오류가 발생했습니다:\n{str(e)}')


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
        self.setWindowTitle('비전 홈 케어 도우미 - 로그인')
        self.setFixedSize(500, 650)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            QLineEdit {
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 15px;
                background-color: white;
                min-height: 35px;
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
                font-size: 15px;
                font-weight: bold;
                min-height: 40px;
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
        main_layout.setSpacing(15)
        
        # 로고/타이틀
        title_label = QLabel('🏠 비전 홈 케어 도우미')
        title_label.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('color: #4CAF50; margin-bottom: 30px;')
        
        # 아이디 입력
        id_label = QLabel('아이디')
        id_label.setFont(QFont('Arial', 12))
        id_label.setStyleSheet('color: #333;')
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText('아이디를 입력하세요')
        self.id_input.returnPressed.connect(self.login)
        
        # 비밀번호 입력
        pw_label = QLabel('비밀번호')
        pw_label.setFont(QFont('Arial', 12))
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
        main_layout.addSpacing(15)
        main_layout.addWidget(id_label)
        main_layout.addWidget(self.id_input)
        main_layout.addSpacing(5)
        main_layout.addWidget(pw_label)
        main_layout.addWidget(self.pw_input)
        main_layout.addSpacing(5)
        main_layout.addWidget(self.remember_checkbox)
        main_layout.addSpacing(10)
        main_layout.addWidget(login_btn)
        main_layout.addSpacing(5)
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
            
            # 시그널 발생
            self.login_success.emit(user)
            self.close()
        else:
            # 로그인 실패
            QMessageBox.warning(self, '로그인 실패', 
                              '아이디 또는 비밀번호가 일치하지 않습니다.')
            self.pw_input.clear()
            self.pw_input.setFocus()
    
    def show_register(self):
        """회원가입 화면 표시"""
        # ✅ RegisterDialog 실행
        dialog = RegisterDialog(self.db, self.user_model, self)
        dialog.exec()


if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    # 기본 폰트 설정
    font = QFont('맑은 고딕', 10)
    app.setFont(font)
    
    window = LoginWindow()
    window.show()
    
    sys.exit(app.exec())