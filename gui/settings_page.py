"""
Home Safe Solution - 설정 페이지
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont
from database_models import DatabaseManager


class SettingsPage(QWidget):
    """설정 페이지"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        label = QLabel('⚙️ 설정')
        label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        layout.addWidget(label)
        
        info = QLabel('시스템 설정 및 알람 설정이 여기에 표시됩니다.')
        layout.addWidget(info)
        
        layout.addStretch()
