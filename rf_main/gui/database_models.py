"""
데이터베이스 연결 및 모델 정의
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import mysql.connector
from mysql.connector import Error, pooling
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json


class DatabaseManager:
    """데이터베이스 연결 및 관리"""
    
    def __init__(self, host='localhost', database='home_safe', 
                 user='homesafe', password='homesafe2026'):
        """
        Args:
            host: MySQL 호스트
            database: 데이터베이스 이름
            user: 사용자명
            password: 비밀번호
        """
        self.config = {
            'host': 'localhost',
            'database': 'home_safe',
            'user': 'homesafe',
            'password': 'homesafe2026',
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        
        # 연결 풀 생성
        self.pool = pooling.MySQLConnectionPool(
            pool_name="home_safe_pool",
            pool_size=5,
            **self.config
        )
    
    def get_connection(self):
        """연결 풀에서 연결 가져오기"""
        return self.pool.get_connection()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """SELECT 쿼리 실행"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result
        except Error as e:
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
            cursor.close()
            conn.close()
            return last_id if last_id else affected_rows
        except Error as e:
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
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        query = "SELECT * FROM users WHERE username = %s AND is_active = TRUE"
        users = self.db.execute_query(query, (username,))
        
        if users and self.verify_password(password, users[0]['password_hash']):
            return users[0]
        return None
    
    def get_by_id(self, user_id: int) -> Optional[Dict]:
        """사용자 ID로 조회"""
        query = "SELECT * FROM users WHERE user_id = %s"
        users = self.db.execute_query(query, (user_id,))
        return users[0] if users else None
    
    def get_by_username(self, username: str) -> Optional[Dict]:
        """사용자명으로 조회"""
        query = "SELECT * FROM users WHERE username = %s"
        users = self.db.execute_query(query, (username,))
        return users[0] if users else None
    
    def update(self, user_id: int, **kwargs) -> bool:
        """사용자 정보 수정 (아이디 제외)"""
        fields = []
        params = []
        
        allowed_fields = ['name', 'gender', 'blood_type', 'address', 
                         'birth_date', 'emergency_contact', 'rtsp_url', 'user_type']
        
        for field in allowed_fields:
            if field in kwargs:
                fields.append(f"{field} = %s")
                params.append(kwargs[field])
        
        # 비밀번호 변경
        if 'password' in kwargs:
            fields.append("password_hash = %s")
            params.append(self.hash_password(kwargs['password']))
        
        if not fields:
            return False
        
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"
        
        return self.db.execute_update(query, tuple(params)) > 0
    
    def delete(self, user_id: int) -> bool:
        """사용자 삭제 (soft delete)"""
        query = "UPDATE users SET is_active = FALSE WHERE user_id = %s"
        return self.db.execute_update(query, (user_id,)) > 0
    
    def get_all(self) -> List[Dict]:
        """모든 사용자 조회"""
        query = "SELECT * FROM users WHERE is_active = TRUE ORDER BY created_at DESC"
        return self.db.execute_query(query)


class EventLog:
    """이벤트 로그 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def create(self, user_id: int, event_type: str, confidence: float = None,
               hip_height: float = None, spine_angle: float = None,
               hip_velocity: float = None, accuracy: float = None, **kwargs) -> Optional[int]:
        """이벤트 생성
        
        Args:
            user_id: 사용자 ID
            event_type: 이벤트 타입 (정상, 낙상중, 낙상)
            confidence: 신뢰도 (0.0 ~ 1.0)
            hip_height: 골반 높이
            spine_angle: 척추 각도
            hip_velocity: 골반 속도
            accuracy: 정상 탐지율 (최근 5분 평균, %) ⭐ 추가
            **kwargs: 추가 파라미터
        
        Returns:
            event_id: 생성된 이벤트 ID
        """
        # 이벤트 타입 ID 조회
        event_type_query = "SELECT event_type_id FROM event_types WHERE type_name = %s"
        event_types = self.db.execute_query(event_type_query, (event_type,))
        
        if not event_types:
            return None
        
        event_type_id = event_types[0]['event_type_id']
        
        query = """
        INSERT INTO event_logs (user_id, event_type_id, event_status, confidence,
                               hip_height, spine_angle, hip_velocity, accuracy,
                               video_path, thumbnail_path, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            user_id, event_type_id, kwargs.get('event_status', '발생'),
            confidence, hip_height, spine_angle, hip_velocity, accuracy,
            kwargs.get('video_path'), kwargs.get('thumbnail_path'),
            kwargs.get('notes')
        )
        
        return self.db.execute_update(query, params)
    
    def update_status(self, event_id: int, status: str, 
                     action_taken: str = None, action_result: str = None) -> bool:
        """이벤트 상태 업데이트"""
        query = """
        UPDATE event_logs 
        SET event_status = %s, action_taken = %s, action_result = %s,
            resolved_at = CASE WHEN %s = '완료' THEN NOW() ELSE resolved_at END
        WHERE event_id = %s
        """
        params = (status, action_taken, action_result, status, event_id)
        return self.db.execute_update(query, params) > 0
    
    def update_action(self, event_id: int, action_taken: str, action_result: str = None) -> bool:
        """이벤트 조치 업데이트 (긴급 호출 전용)"""
        query = """
        UPDATE event_logs 
        SET action_taken = %s, action_result = %s
        WHERE event_id = %s
        """
        params = (action_taken, action_result, event_id)
        return self.db.execute_update(query, params) > 0
    
    def get_recent_fall_event(self, user_id: int = None) -> Optional[Dict]:
        """가장 최근 낙상 이벤트 조회 (Falling 또는 Fallen)"""
        query = """
        SELECT el.*, et.type_name
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE et.type_name IN ('낙상중', '낙상')
        """
        params = []
        
        if user_id:
            query += " AND el.user_id = %s"
            params.append(user_id)
        
        query += " ORDER BY el.occurred_at DESC LIMIT 1"
        
        result = self.db.execute_query(query, tuple(params) if params else None)
        return result[0] if result else None
    
    def update_duration(self, event_id: int) -> bool:
        """이벤트 지속 시간 계산 및 업데이트"""
        query = """
        UPDATE event_logs 
        SET duration_seconds = TIMESTAMPDIFF(SECOND, occurred_at, NOW())
        WHERE event_id = %s
        """
        return self.db.execute_update(query, (event_id,)) > 0
    
    def search(self, user_id: int = None, event_type: str = None,
               start_date: datetime = None, end_date: datetime = None,
               limit: int = 100) -> List[Dict]:
        """이벤트 검색"""
        query = "SELECT * FROM v_event_details WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = %s"
            params.append(event_type)
        
        if start_date:
            query += " AND occurred_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND occurred_at <= %s"
            params.append(end_date)
        
        query += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(limit)
        
        return self.db.execute_query(query, tuple(params))
    
    def get_recent(self, user_id: int = None, limit: int = 50) -> List[Dict]:
        """최근 이벤트 조회"""
        query = "SELECT * FROM v_event_details"
        params = []
        
        if user_id:
            query += " WHERE user_id = %s"
            params.append(user_id)
        
        query += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(limit)
        
        return self.db.execute_query(query, tuple(params))
    
    def get_statistics(self, user_id: int = None, days: int = 30) -> Dict:
        """이벤트 통계"""
        start_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            et.type_name,
            COUNT(*) as count,
            AVG(el.duration_seconds) as avg_duration
        FROM event_logs el
        JOIN event_types et ON el.event_type_id = et.event_type_id
        WHERE el.occurred_at >= %s
        """
        params = [start_date]
        
        if user_id:
            query += " AND el.user_id = %s"
            params.append(user_id)
        
        query += " GROUP BY et.type_name"
        
        results = self.db.execute_query(query, tuple(params))
        return {r['type_name']: r for r in results}


class AutoReport:
    """자동신고 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def create(self, event_id: int, report_target: str, report_type: str,
               report_content: str, recipient: str, **kwargs) -> Optional[int]:
        """자동신고 생성"""
        query = """
        INSERT INTO auto_report_logs (event_id, report_target, report_type,
                                     report_content, recipient, video_sent,
                                     delivery_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            event_id, report_target, report_type, report_content,
            recipient, kwargs.get('video_sent', False), '대기'
        )
        
        return self.db.execute_update(query, params)
    
    def update_status(self, report_id: int, status: str, result: str = None) -> bool:
        """발송 상태 업데이트"""
        query = """
        UPDATE auto_report_logs 
        SET delivery_status = %s, delivery_result = %s
        WHERE report_id = %s
        """
        return self.db.execute_update(query, (status, result, report_id)) > 0
    
    def get_by_event(self, event_id: int) -> List[Dict]:
        """이벤트별 신고 내역"""
        query = "SELECT * FROM auto_report_logs WHERE event_id = %s ORDER BY sent_at DESC"
        return self.db.execute_query(query, (event_id,))


class SystemSettings:
    """시스템 설정 모델"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def get(self, key: str) -> Any:
        """설정값 조회"""
        query = "SELECT setting_value, setting_type FROM system_settings WHERE setting_key = %s"
        results = self.db.execute_query(query, (key,))
        
        if not results:
            return None
        
        value = results[0]['setting_value']
        value_type = results[0]['setting_type']
        
        # 타입 변환
        if value_type == 'int':
            return int(value)
        elif value_type == 'float':
            return float(value)
        elif value_type == 'bool':
            return value.lower() in ('true', '1', 'yes')
        elif value_type == 'json':
            return json.loads(value)
        else:
            return value
    
    def set(self, key: str, value: Any, value_type: str = 'string') -> bool:
        """설정값 저장"""
        # 값 변환
        if value_type == 'json':
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        
        query = """
        INSERT INTO system_settings (setting_key, setting_value, setting_type)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE setting_value = %s
        """
        params = (key, value, value_type, value)
        return self.db.execute_update(query, params) > 0
    
    def get_all(self) -> Dict[str, Any]:
        """모든 설정 조회"""
        query = "SELECT setting_key, setting_value, setting_type FROM system_settings"
        results = self.db.execute_query(query)
        
        settings = {}
        for r in results:
            key = r['setting_key']
            value = r['setting_value']
            value_type = r['setting_type']
            
            if value_type == 'int':
                settings[key] = int(value)
            elif value_type == 'float':
                settings[key] = float(value)
            elif value_type == 'bool':
                settings[key] = value.lower() in ('true', '1', 'yes')
            elif value_type == 'json':
                settings[key] = json.loads(value)
            else:
                settings[key] = value
        
        return settings


# 사용 예제
if __name__ == "__main__":
    # 데이터베이스 연결
    db = DatabaseManager(
        host='localhost',
        database='home_safe',
        user='root',
        password='your_password'
    )
    
    # 사용자 관리
    user_model = User(db)
    
    # 새 사용자 생성
    user_id = user_model.create(
        username='test_user',
        password='password123',
        name='홍길동',
        gender='남성',
        user_type='일반유저',
        address='서울시 강남구',
        emergency_contact='010-1234-5678'
    )
    print(f"생성된 사용자 ID: {user_id}")
    
    # 로그인
    user = user_model.authenticate('test_user', 'password123')
    if user:
        print(f"로그인 성공: {user['name']}")
    
    # 이벤트 생성
    event_model = EventLog(db)
    event_id = event_model.create(
        user_id=user_id,
        event_type='낙상',
        confidence=0.95,
        hip_height=150.5,
        spine_angle=75.3
    )
    print(f"생성된 이벤트 ID: {event_id}")
    
    # 최근 이벤트 조회
    recent_events = event_model.get_recent(limit=10)
    print(f"최근 이벤트 {len(recent_events)}개")