"""
데이터베이스 연결 및 모델 정의 (SQLite 버전)
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import sqlite3
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import os


class DatabaseManager:
    """데이터베이스 연결 및 관리 (SQLite)"""
    
    def __init__(self, db_path='home_safe.db'):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # users 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rtsp_url TEXT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            blood_type TEXT,
            address TEXT,
            birth_date TEXT,
            emergency_contact TEXT,
            user_type TEXT NOT NULL DEFAULT '일반유저',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        """)
        
        # event_types 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_types (
            event_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_name TEXT UNIQUE NOT NULL,
            severity TEXT NOT NULL DEFAULT '주의',
            description TEXT,
            is_active INTEGER DEFAULT 1
        )
        """)
        
        # event_logs 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type_id INTEGER NOT NULL,
            event_status TEXT NOT NULL DEFAULT '발생',
            occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            duration_seconds INTEGER,
            confidence REAL,
            hip_height REAL,
            spine_angle REAL,
            hip_velocity REAL,
            video_path TEXT,
            thumbnail_path TEXT,
            action_taken TEXT DEFAULT '없음',
            action_result TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (event_type_id) REFERENCES event_types(event_type_id)
        )
        """)
        
        # 기본 데이터 삽입
        # 관리자 계정 (비밀번호: admin123)
        try:
            password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
            INSERT INTO users (username, password_hash, name, gender, user_type)
            VALUES (?, ?, ?, ?, ?)
            """, ('admin', password_hash, '관리자', '남성', '관리자'))
        except sqlite3.IntegrityError:
            pass  # 이미 존재
        
        # 이벤트 타입
        event_types = [
            ('정상', '정상', '정상 상태'),
            ('낙상', '위험', '넘어지는 행동 감지'),
            ('쓰러짐', '위험', '바닥에 쓰러진 상태'),
            ('화재', '위험', '화재 감지'),
            ('침수', '경고', '침수 감지'),
            ('외부인침입', '경고', '승인되지 않은 사람 감지'),
            ('안전영역이탈', '주의', '안전 영역 이탈 감지')
        ]
        
        for type_name, severity, desc in event_types:
            try:
                cursor.execute("""
                INSERT INTO event_types (type_name, severity, description)
                VALUES (?, ?, ?)
                """, (type_name, severity, desc))
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """연결 가져오기"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # dict 형태로 반환
        return conn
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """SELECT 쿼리 실행"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            conn.close()
            return result
        except Exception as e:
            print(f"쿼리 실행 오류: {e}")
            return []
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """INSERT/UPDATE/DELETE 쿼리 실행"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            last_id = cursor.lastrowid
            affected_rows = cursor.rowcount
            conn.close()
            return last_id if last_id else affected_rows
        except Exception as e:
            print(f"업데이트 실행 오류: {e}")
            return 0


class User:
    """사용자 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    @staticmethod
    def hash_password(password: str) -> str:
        """비밀번호 해시"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """비밀번호 검증"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create(self, username: str, password: str, name: str, gender: str,
               user_type: str = '일반유저', **kwargs) -> Optional[int]:
        """사용자 생성"""
        password_hash = self.hash_password(password)
        
        query = """
        INSERT INTO users (username, password_hash, name, gender, user_type,
                          rtsp_url, blood_type, address, birth_date, emergency_contact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            username, password_hash, name, gender, user_type,
            kwargs.get('rtsp_url'), kwargs.get('blood_type'),
            kwargs.get('address'), kwargs.get('birth_date'),
            kwargs.get('emergency_contact')
        )
        
        return self.db.execute_update(query, params)
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """로그인 인증"""
        query = "SELECT * FROM users WHERE username = ? AND is_active = 1"
        users = self.db.execute_query(query, (username,))
        
        if users and self.verify_password(password, users[0]['password_hash']):
            return users[0]
        return None
    
    def get_by_id(self, user_id: int) -> Optional[Dict]:
        """사용자 ID로 조회"""
        query = "SELECT * FROM users WHERE user_id = ?"
        users = self.db.execute_query(query, (user_id,))
        return users[0] if users else None
    
    def get_all(self) -> List[Dict]:
        """모든 사용자 조회"""
        query = "SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC"
        return self.db.execute_query(query)


class EventLog:
    """이벤트 로그 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def create(self, user_id: int, event_type: str, confidence: float = None,
               hip_height: float = None, spine_angle: float = None,
               hip_velocity: float = None, **kwargs) -> Optional[int]:
        """이벤트 생성"""
        # 이벤트 타입 ID 조회
        event_type_query = "SELECT event_type_id FROM event_types WHERE type_name = ?"
        event_types = self.db.execute_query(event_type_query, (event_type,))
        
        if not event_types:
            return None
        
        event_type_id = event_types[0]['event_type_id']
        
        query = """
        INSERT INTO event_logs (user_id, event_type_id, event_status, confidence,
                               hip_height, spine_angle, hip_velocity, video_path,
                               thumbnail_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            user_id, event_type_id, kwargs.get('event_status', '발생'),
            confidence, hip_height, spine_angle, hip_velocity,
            kwargs.get('video_path'), kwargs.get('thumbnail_path'),
            kwargs.get('notes')
        )
        
        return self.db.execute_update(query, params)
    
    def get_recent(self, user_id: int = None, limit: int = 50) -> List[Dict]:
        """최근 이벤트 조회"""
        query = """
        SELECT 
            el.*,
            u.username,
            u.name as user_name,
            et.type_name as event_type,
            et.severity,
            u.address,
            u.emergency_contact,
            u.gender,
            u.blood_type
        FROM event_logs el
        JOIN users u ON el.user_id = u.user_id
        JOIN event_types et ON el.event_type_id = et.event_type_id
        """
        params = []
        
        if user_id:
            query += " WHERE el.user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY el.occurred_at DESC LIMIT ?"
        params.append(limit)
        
        return self.db.execute_query(query, tuple(params))


class AutoReport:
    """자동신고 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db


class SystemSettings:
    """시스템 설정 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db


# 사용 예제
if __name__ == "__main__":
    # 데이터베이스 연결
    db = DatabaseManager()
    
    # 사용자 관리
    user_model = User(db)
    
    # 로그인 테스트
    user = user_model.authenticate('admin', 'admin123')
    if user:
        print(f"로그인 성공: {user['name']}")
    else:
        print("로그인 실패")
