"""
Home Safe Solution - 이벤트 로그 페이지
Author: Home Safe Solution Team
Date: 2026-01-28
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont
from database_models import DatabaseManager


class EventsPage(QWidget):
    """이벤트 로그 페이지"""
    
    def __init__(self, user_info: dict, db: DatabaseManager):
        super().__init__()
        self.user_info = user_info
        self.db = db
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        label = QLabel('📋 이벤트 로그')
        label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        layout.addWidget(label)
        
        info = QLabel('이벤트 검색 및 상세 조회 기능이 여기에 표시됩니다.')
        layout.addWidget(info)
        
        layout.addStretch()
