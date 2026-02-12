"""
Home Safe Solution - 메인 실행 파일
Author: Home Safe Solution Team
Date: 2026-01-28

실행 방법:
    python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from login_window import LoginWindow
from main_window import MainWindow


def main():
    """메인 함수"""
    # 애플리케이션 생성
    app = QApplication(sys.argv)
    
    # 기본 폰트 설정
    font = QFont('맑은 고딕', 10)
    app.setFont(font)
    
    # 애플리케이션 이름 설정
    app.setApplicationName('Home Safe Solution')
    app.setApplicationDisplayName('Home Safe - 낙상 감지 시스템')
    
    # 로그인 윈도우 생성
    login_window = LoginWindow()
    
    # 로그인 성공 시 메인 윈도우 표시
    def on_login_success(user_info):
        main_window = MainWindow(user_info)
        main_window.show()
    
    login_window.login_success.connect(on_login_success)
    login_window.show()
    
    # 이벤트 루프 시작
    sys.exit(app.exec())


if __name__ == '__main__':
    main()