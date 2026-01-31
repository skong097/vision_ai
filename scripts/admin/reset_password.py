#!/usr/bin/env python3
"""
사용자 비밀번호 재설정 스크립트
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

cursor = conn.cursor()

# 사용자명 입력
username = input("재설정할 사용자명: ")

# 새 비밀번호 입력
new_password = input("새 비밀번호: ")

# bcrypt 해시 생성
password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

print(f"\n생성된 해시: {password_hash[:50]}...")

# 비밀번호 업데이트
query = "UPDATE users SET password_hash = %s WHERE username = %s"
cursor.execute(query, (password_hash, username))
conn.commit()

if cursor.rowcount > 0:
    print(f"✅ {username} 사용자의 비밀번호가 재설정되었습니다!")
else:
    print(f"❌ {username} 사용자를 찾을 수 없습니다.")

cursor.close()
conn.close()
