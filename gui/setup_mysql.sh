#!/bin/bash
# ============================================================
# MySQL 설치 및 Home Safe 데이터베이스 설정
# ============================================================

echo "============================================================"
echo "🗄️  MySQL 설치 및 설정"
echo "============================================================"
echo ""

# 1. MySQL 서버 설치
echo "📦 MySQL 서버 설치 중..."
sudo apt-get update
sudo apt-get install -y mysql-server

# 2. MySQL 시작
echo "🚀 MySQL 서비스 시작..."
sudo systemctl start mysql
sudo systemctl enable mysql

# 3. MySQL 상태 확인
echo "✅ MySQL 상태 확인..."
sudo systemctl status mysql --no-pager

echo ""
echo "============================================================"
echo "🔐 데이터베이스 설정"
echo "============================================================"
echo ""

# 4. 데이터베이스 및 사용자 생성
echo "데이터베이스를 생성합니다..."

sudo mysql -e "CREATE DATABASE IF NOT EXISTS home_safe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

echo "사용자를 생성합니다..."
sudo mysql -e "CREATE USER IF NOT EXISTS 'homesafe'@'localhost' IDENTIFIED BY 'homesafe2026';"

echo "권한을 부여합니다..."
sudo mysql -e "GRANT ALL PRIVILEGES ON home_safe.* TO 'homesafe'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"

echo ""
echo "============================================================"
echo "📋 테이블 생성"
echo "============================================================"
echo ""

# 5. 스키마 적용
if [ -f "database_schema.sql" ]; then
    echo "스키마를 적용합니다..."
    sudo mysql home_safe < database_schema.sql
    echo "✅ 스키마 적용 완료!"
else
    echo "⚠️  database_schema.sql 파일을 찾을 수 없습니다."
    echo "   나중에 수동으로 실행하세요:"
    echo "   sudo mysql home_safe < database_schema.sql"
fi

echo ""
echo "============================================================"
echo "✅ MySQL 설정 완료!"
echo "============================================================"
echo ""
echo "연결 정보:"
echo "  Host: localhost"
echo "  Database: home_safe"
echo "  User: homesafe"
echo "  Password: homesafe2026"
echo ""
echo "다음 단계:"
echo "  1. database_models.py 파일 수정"
echo "  2. python main.py 실행"
echo ""