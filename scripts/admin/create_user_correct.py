#!/usr/bin/env python3
"""
올바른 방법으로 계정 생성
"""

from database_models import User, DatabaseManager

# DB 연결
db = DatabaseManager(
    host='localhost',
    database='home_safe',
    user='homesafe',
    password='homesafe2026'
)

user_model = User(db)

# 신규 계정 정보
username = input("사용자명: ")
password = input("비밀번호: ")
name = input("이름: ")
gender = input("성별 (남성/여성): ")

# 기존 계정 있으면 먼저 삭제
existing = user_model.get_by_username(username)
if existing:
    print(f"⚠️  {username} 계정이 이미 존재합니다. 삭제 후 재생성합니다.")
    # Soft delete
    user_model.delete(existing['user_id'])

# 올바른 방법으로 생성
user_id = user_model.create(
    username=username,
    password=password,  # 평문으로 입력 (자동으로 bcrypt 해시됨)
    name=name,
    gender=gender,
    user_type='일반유저'
)

if user_id:
    print(f"\n✅ 계정 생성 완료!")
    print(f"   사용자 ID: {user_id}")
    print(f"   사용자명: {username}")
    print(f"   비밀번호: {password}")
    print(f"\n이제 로그인할 수 있습니다.")
else:
    print("\n❌ 계정 생성 실패!")
