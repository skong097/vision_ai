#!/bin/bash
# ============================================================
# Home Safe Solution - 자동 설치 스크립트
# Author: Home Safe Solution Team
# Date: 2026-01-28
# ============================================================

echo "============================================================"
echo "🏠 Home Safe Solution - 설치 시작"
echo "============================================================"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 에러 핸들링
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo -e "${RED}❌ 오류 발생: ${last_command}${NC}"' ERR

# 1. Python 패키지 설치
echo -e "${YELLOW}📦 1/4 Python 패키지 설치...${NC}"
pip install PyQt6 PyQt6-Charts mysql-connector-python bcrypt --break-system-packages
echo -e "${GREEN}✅ Python 패키지 설치 완료!${NC}"
echo ""

# 2. MySQL 확인
echo -e "${YELLOW}🗄️  2/4 MySQL 확인...${NC}"
if ! command -v mysql &> /dev/null; then
    echo -e "${RED}❌ MySQL이 설치되어 있지 않습니다.${NC}"
    echo "MySQL 설치 방법:"
    echo "  sudo apt-get install mysql-server"
    exit 1
fi
echo -e "${GREEN}✅ MySQL 확인 완료!${NC}"
echo ""

# 3. 디렉토리 생성
echo -e "${YELLOW}📁 3/4 디렉토리 구조 생성...${NC}"
BASE_DIR="/home/gjkong/dev_ws/yolo/myproj"
GUI_DIR="${BASE_DIR}/gui"

mkdir -p "${GUI_DIR}"
mkdir -p "${BASE_DIR}/recordings"
mkdir -p "${BASE_DIR}/events"
mkdir -p "${BASE_DIR}/logs"

echo -e "${GREEN}✅ 디렉토리 생성 완료!${NC}"
echo ""

# 4. 파일 복사
echo -e "${YELLOW}📋 4/4 파일 복사...${NC}"

# 현재 디렉토리의 모든 Python 파일 복사
CURRENT_DIR=$(pwd)
cp "${CURRENT_DIR}"/*.py "${GUI_DIR}/" 2>/dev/null || true
cp "${CURRENT_DIR}"/*.sql "${GUI_DIR}/" 2>/dev/null || true
cp "${CURRENT_DIR}"/*.md "${GUI_DIR}/" 2>/dev/null || true

echo -e "${GREEN}✅ 파일 복사 완료!${NC}"
echo ""

# 데이터베이스 설정 안내
echo "============================================================"
echo -e "${YELLOW}⚙️  데이터베이스 설정이 필요합니다${NC}"
echo "============================================================"
echo ""
echo "다음 명령어를 실행하여 데이터베이스를 생성하세요:"
echo ""
echo -e "${GREEN}  cd ${GUI_DIR}${NC}"
echo -e "${GREEN}  mysql -u root -p < database_schema.sql${NC}"
echo ""
echo "또는 MySQL 콘솔에서:"
echo ""
echo -e "${GREEN}  mysql -u root -p${NC}"
echo -e "${GREEN}  > source ${GUI_DIR}/database_schema.sql${NC}"
echo ""
echo "============================================================"
echo ""

# 비밀번호 설정 안내
echo "============================================================"
echo -e "${YELLOW}🔐 데이터베이스 연결 설정${NC}"
echo "============================================================"
echo ""
echo "database_models.py 파일에서 MySQL 비밀번호를 설정하세요:"
echo ""
echo -e "${GREEN}  nano ${GUI_DIR}/database_models.py${NC}"
echo ""
echo "36번째 줄 근처에서 수정:"
echo -e "${YELLOW}  'password': 'your_password'  ← 여기에 실제 비밀번호 입력${NC}"
echo ""
echo "============================================================"
echo ""

# 완료
echo "============================================================"
echo -e "${GREEN}✅ 설치 완료!${NC}"
echo "============================================================"
echo ""
echo "다음 단계:"
echo "  1. 데이터베이스 생성 (위 명령어 참조)"
echo "  2. database_models.py에서 MySQL 비밀번호 설정"
echo "  3. 프로그램 실행: cd ${GUI_DIR} && python main.py"
echo ""
echo "기본 로그인 정보:"
echo "  아이디: admin"
echo "  비밀번호: admin123"
echo ""
echo "자세한 사용법은 GUI_README.md 파일을 참조하세요."
echo ""
echo "🏠 Home Safe Solution - 설치가 완료되었습니다!"
echo "============================================================"
