#!/usr/bin/env python3
"""
DB 이벤트 저장 확인 스크립트
"""

import mysql.connector
from datetime import datetime

# DB 연결
conn = mysql.connector.connect(
    host='localhost',
    database='home_safe',
    user='homesafe',
    password='homesafe2026'
)

cursor = conn.cursor(dictionary=True)

# 최근 이벤트 조회
query = """
SELECT 
    el.event_id,
    u.username,
    u.name,
    et.type_name,
    el.event_status,
    ROUND(el.confidence, 2) as confidence,
    ROUND(el.hip_height, 1) as hip_height,
    el.occurred_at,
    el.notes
FROM event_logs el
JOIN users u ON el.user_id = u.user_id
JOIN event_types et ON el.event_type_id = et.event_type_id
ORDER BY el.occurred_at DESC
LIMIT 10
"""

cursor.execute(query)
results = cursor.fetchall()

print("=" * 80)
print("최근 이벤트 10개")
print("=" * 80)

if results:
    for row in results:
        print(f"\n[이벤트 ID: {row['event_id']}]")
        print(f"  사용자: {row['name']} ({row['username']})")
        print(f"  타입: {row['type_name']}")
        print(f"  상태: {row['event_status']}")
        print(f"  신뢰도: {row['confidence']}")
        print(f"  골반 높이: {row['hip_height']}")
        print(f"  발생 시각: {row['occurred_at']}")
        print(f"  메모: {row['notes']}")
        print("-" * 80)
    
    print(f"\n✅ 총 {len(results)}개 이벤트 확인됨")
else:
    print("\n⚠️ 저장된 이벤트가 없습니다.")

# 통계
cursor.execute("""
SELECT 
    et.type_name,
    COUNT(*) as count
FROM event_logs el
JOIN event_types et ON el.event_type_id = et.event_type_id
WHERE DATE(el.occurred_at) = CURDATE()
GROUP BY et.type_name
""")

stats = cursor.fetchall()

print("\n" + "=" * 80)
print("오늘 이벤트 통계")
print("=" * 80)

if stats:
    for stat in stats:
        print(f"{stat['type_name']}: {stat['count']}개")
else:
    print("오늘 발생한 이벤트가 없습니다.")

cursor.close()
conn.close()
