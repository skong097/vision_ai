#!/usr/bin/env python3
"""
admin 계정 재생성 스크립트
"""

import bcrypt
import mysql.connector
from getpass import getpass

print("=" * 60)
print("🔐 admin 계정 재생성")
print("=" * 60)
print()

# MySQL 연결 정보
print("MySQL 연결 정보를 입력하세요:")
host = input("Host (기본: localhost): ").strip() or 'localhost'
database = input("Database (기본: home_safe): ").strip() or 'home_safe'
user = input("User (기본: homesafe): ").strip() or 'homesafe'
password = input("Password (기본: homesafe2026): ").strip() or 'homesafe2026'

print()
print("연결 중...")

try:
    # MySQL 연결
    conn = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    
    print("✅ 연결 성공!")
    print()
    
    cursor = conn.cursor()
    
    # 기존 admin 확인
    cursor.execute("SELECT user_id, username, name FROM users WHERE username='admin'")
    existing = cursor.fetchone()
    
    if existing:
        print(f"⚠️  기존 admin 계정 발견:")
        print(f"   user_id: {existing[0]}")
        print(f"   username: {existing[1]}")
        print(f"   name: {existing[2]}")
        print()
        response = input("기존 계정을 삭제하고 다시 만들까요? (y/N): ")
        if response.lower() != 'y':
            print("종료합니다.")
            exit(0)
        
        cursor.execute("DELETE FROM users WHERE username='admin'")
        print("🗑️  기존 계정 삭제됨")
    
    # 새 비밀번호 설정
    print()
    new_password = input("admin 계정 비밀번호 (기본: admin123): ").strip() or 'admin123'
    
    # 비밀번호 해시 생성
    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    print()
    print("계정 생성 중...")
    
    # admin 계정 생성
    cursor.execute("""
    INSERT INTO users (username, password_hash, name, gender, user_type)
    VALUES (%s, %s, %s, %s, %s)
    """, ('admin', password_hash, '관리자', '남성', '관리자'))
    
    conn.commit()
    
    # 확인
    cursor.execute("SELECT user_id, username, name, user_type FROM users WHERE username='admin'")
    result = cursor.fetchone()
    
    print()
    print("=" * 60)
    print("✅ admin 계정 생성 완료!")
    print("=" * 60)
    print()
    print(f"user_id: {result[0]}")
    print(f"username: {result[1]}")
    print(f"name: {result[2]}")
    print(f"user_type: {result[3]}")
    print()
    print("로그인 정보:")
    print(f"  아이디: admin")
    print(f"  비밀번호: {new_password}")
    print()
    print("이제 python main.py 로 프로그램을 실행하세요!")
    print()
    
    conn.close()
    
except mysql.connector.Error as e:
    print(f"❌ MySQL 오류: {e}")
    print()
    print("해결 방법:")
    print("1. MySQL 서비스 확인: sudo systemctl status mysql")
    print("2. 연결 정보 확인: user/password")
    print("3. 권한 확인: GRANT ALL PRIVILEGES ON home_safe.* TO 'homesafe'@'localhost';")
    exit(1)

except Exception as e:
    print(f"❌ 오류: {e}")
    exit(1)
