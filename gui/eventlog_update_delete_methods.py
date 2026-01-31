"""
database_models.py의 EventLog 클래스에 추가할 메서드들
"""

def update(self, event_id: int, event_type: str = None, event_status: str = None, notes: str = None) -> bool:
    """이벤트 업데이트"""
    try:
        # 이벤트 타입 ID 조회 (필요한 경우)
        event_type_id = None
        if event_type:
            query = "SELECT event_type_id FROM event_types WHERE type_name = %s"
            result = self.db.fetch_one(query, (event_type,))
            if result:
                event_type_id = result['event_type_id']
        
        # 업데이트 쿼리 구성
        update_parts = []
        params = []
        
        if event_type_id is not None:
            update_parts.append("event_type_id = %s")
            params.append(event_type_id)
        
        if event_status is not None:
            update_parts.append("event_status = %s")
            params.append(event_status)
        
        if notes is not None:
            update_parts.append("notes = %s")
            params.append(notes)
        
        if not update_parts:
            return True  # 업데이트할 내용 없음
        
        # WHERE 조건 추가
        params.append(event_id)
        
        query = f"""
        UPDATE event_logs
        SET {', '.join(update_parts)}
        WHERE event_id = %s
        """
        
        self.db.execute(query, tuple(params))
        return True
        
    except Exception as e:
        print(f"[ERROR] EventLog.update: {e}")
        return False


def delete(self, event_id: int) -> bool:
    """이벤트 삭제"""
    try:
        query = "DELETE FROM event_logs WHERE event_id = %s"
        self.db.execute(query, (event_id,))
        return True
        
    except Exception as e:
        print(f"[ERROR] EventLog.delete: {e}")
        return False


# database_models.py의 EventLog 클래스에 이 두 메서드를 추가하세요!
# 위치: EventLog 클래스 내부, 다른 메서드들과 함께

"""
사용 예시:

# 업데이트
event_log = EventLog(db)
success = event_log.update(
    event_id=1,
    event_type='정상',
    event_status='확인',
    notes='확인 완료'
)

# 삭제
success = event_log.delete(event_id=1)
"""
