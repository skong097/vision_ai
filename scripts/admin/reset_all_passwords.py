#!/usr/bin/env python3
"""
모든 사용자 비밀번호 일괄 재설정
기본 비밀번호: 1234
"""

import mysql.connector
import bcrypt

# DB 연결
conn = mysql.connector.connect(
    host='localhost',
    database='home_safe',
    user='homesafe',
    password='homesafe2026'
)

cursor = conn.cursor(dictionary=True)

# 모든 사용자 조회
cursor.execute("SELECT user_id, username FROM users")
users = cursor.fetchall()

print(f"총 {len(users)}명의 사용자 발견")
print()

default_password = "1234"  # 기본 비밀번호

for user in users:
    username = user['username']
    user_id = user['user_id']
    
    # bcrypt 해시 생성
    password_hash = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # 업데이트
    update_query = "UPDATE users SET password_hash = %s WHERE user_id = %s"
    cursor.execute(update_query, (password_hash, user_id))
    conn.commit()
    
    print(f"✅ {username} - 비밀번호: {default_password}")

print()
print(f"완료! 모든 사용자의 비밀번호가 '{default_password}'로 재설정되었습니다.")

cursor.close()
conn.close()
