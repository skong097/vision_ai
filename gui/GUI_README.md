# 🏠 Home Safe Solution - PyQt6 GUI 시스템

## 📋 개요

**Home Safe Solution**은 YOLO Pose와 Random Forest를 활용한 **실시간 낙상 감지 시스템**입니다.

### 주요 기능

✅ **사용자 관리**
- 사용자 생성/수정/삭제
- 관리자/일반유저 권한 관리
- 로그인/로그아웃

✅ **실시간 모니터링**
- 웹캠/RTSP 실시간 영상
- YOLO Pose Skeleton 오버레이
- 낙상 감지 (Normal/Falling/Fallen)
- 실시간 이벤트 알람

✅ **이벤트 로그**
- 이벤트 검색 및 조회
- 동영상 재생
- 통계 분석

✅ **자동 신고**
- 119/112 자동 신고
- 비상연락처 문자 발송
- 이벤트 동영상 전송

---

## 🚀 설치 방법

### 1. 필수 패키지 설치

```bash
# PyQt6 설치
pip install PyQt6 PyQt6-Charts --break-system-packages

# MySQL 클라이언트
pip install mysql-connector-python --break-system-packages

# 암호화 (bcrypt)
pip install bcrypt --break-system-packages

# 기타 (이미 설치되어 있을 것)
pip install opencv-python numpy pandas ultralytics --break-system-packages
```

### 2. MySQL 데이터베이스 설정

```bash
# MySQL 접속
mysql -u root -p

# 데이터베이스 및 테이블 생성
source /home/gjkong/dev_ws/yolo/myproj/gui/database_schema.sql
```

**또는 Python에서:**

```bash
mysql -u root -p < database_schema.sql
```

### 3. 데이터베이스 연결 설정

`database_models.py` 파일에서 MySQL 연결 정보 수정:

```python
# database_models.py 36번째 줄 근처
self.config = {
    'host': 'localhost',      # MySQL 호스트
    'database': 'home_safe',  # 데이터베이스 이름
    'user': 'root',           # MySQL 사용자명
    'password': 'your_password',  # ← 여기에 실제 비밀번호 입력!
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}
```

---

## 🎯 실행 방법

### 방법 1: 메인 스크립트 실행

```bash
cd /home/gjkong/dev_ws/yolo/myproj/gui
python main.py
```

### 방법 2: 직접 실행

```bash
python3 main.py
```

---

## 🔐 기본 로그인 정보

설치 시 자동으로 생성되는 관리자 계정:

```
아이디: admin
비밀번호: admin123
```

**⚠️ 보안 주의:** 첫 로그인 후 반드시 비밀번호를 변경하세요!

---

## 📁 파일 구조

```
/home/gjkong/dev_ws/yolo/myproj/gui/
├── main.py                          # 메인 실행 파일
├── login_window.py                  # 로그인 화면
├── main_window.py                   # 메인 윈도우
├── dashboard_page.py                # 대시보드
├── monitoring_page.py               # 실시간 모니터링
├── events_page.py                   # 이벤트 로그
├── users_page.py                    # 사용자 관리
├── settings_page.py                 # 설정
├── additional_pages.py              # 추가 페이지
├── database_models.py               # 데이터베이스 모델
├── database_schema.sql              # DB 스키마
└── README.md                        # 이 파일
```

---

## 🖼️ 화면 구성

### 1. 로그인 화면
```
┌────────────────────────────┐
│      🏠 Home Safe         │
│     낙상 감지 시스템        │
│                            │
│   아이디: [__________]    │
│   비밀번호: [__________]  │
│   □ 로그인 상태 유지       │
│                            │
│   [      로그인      ]    │
│   [      회원가입    ]    │
└────────────────────────────┘
```

### 2. 메인 대시보드
```
┌─────────────────────────────────────────────────────────┐
│  사이드바       │  대시보드                              │
│  ─────────     │  ───────────────────────────────────  │
│  🏠 Home Safe  │  [총 이벤트] [오늘] [낙상] [정상]    │
│  관리자님      │                                        │
│                 │  📊 차트 영역                         │
│  📊 대시보드    │  ┌─────────────┐  ┌──────────────┐  │
│  🎥 실시간모니터│  │ 이벤트 분포  │  │  시간대별    │  │
│  📋 이벤트로그  │  │  (파이차트)  │  │ (막대차트)   │  │
│  👥 사용자관리  │  └─────────────┘  └──────────────┘  │
│  ⚙️ 설정        │                                        │
│                 │  최근 이벤트 테이블                    │
│  [  로그아웃  ] │  ┌────────────────────────────────┐  │
└─────────────────┴──┴────────────────────────────────┴──┘
```

### 3. 실시간 모니터링
```
┌───────────────────────────────────────────────────────┐
│  영상 영역                    │  정보 패널              │
│  ┌─────────────────────────┐  │  ┌──────────────────┐│
│  │                          │  │  │ 현재 상태        ││
│  │    🎥 실시간 웹캠        │  │  │ 🟢 정상         ││
│  │    (Skeleton 오버레이)   │  │  │ 확률: 96.3%    ││
│  │                          │  │  └──────────────────┘│
│  │                          │  │  ┌──────────────────┐│
│  │                          │  │  │ 신체 정보        ││
│  └─────────────────────────┘  │  │ Hip: 320.5      ││
│  [▶ 시작]  [⏹ 중지]        │  │ Spine: 12.8°   ││
│                               │  └──────────────────┘│
│                               │  ┌──────────────────┐│
│                               │  │ 이벤트 로그      ││
│                               │  │ [15:30] 🟢 정상 ││
│                               │  └──────────────────┘│
└───────────────────────────────┴──────────────────────┘
```

---

## 🔧 설정

### MySQL 비밀번호 변경

`database_models.py` 36번째 줄:
```python
'password': 'YOUR_MYSQL_PASSWORD_HERE',
```

### 모델 경로 변경

`monitoring_page.py` 275번째 줄:
```python
model_path = '/home/gjkong/dev_ws/yolo/myproj/models/3class/random_forest_model.pkl'
```

### RTSP URL 설정

데이터베이스에서 사용자별로 설정하거나, 사용자 관리 화면에서 수정

---

## 🐛 문제 해결

### 1. PyQt6 설치 오류

```bash
# Qt platform plugin 오류
sudo apt-get install libxcb-cursor0

# 또는
pip uninstall PyQt6
pip install PyQt6 --break-system-packages --no-cache-dir
```

### 2. MySQL 연결 오류

```bash
# MySQL 서비스 시작
sudo systemctl start mysql

# 연결 확인
mysql -u root -p -e "SELECT 1"
```

### 3. 모델 파일 없음

```bash
# 모델 파일 확인
ls -lh /home/gjkong/dev_ws/yolo/myproj/models/3class/

# 없으면 모델 재학습
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python train_random_forest.py
```

### 4. 웹캠 인식 안 됨

```bash
# 웹캠 확인
ls /dev/video*

# 권한 확인
sudo usermod -a -G video $USER
```

---

## 📊 데이터베이스 관리

### 백업

```bash
mysqldump -u root -p home_safe > home_safe_backup.sql
```

### 복원

```bash
mysql -u root -p home_safe < home_safe_backup.sql
```

### 초기화

```bash
mysql -u root -p -e "DROP DATABASE home_safe"
mysql -u root -p < database_schema.sql
```

---

## 🎨 커스터마이징

### 색상 변경

각 페이지의 `setStyleSheet()` 메소드에서 색상 코드 수정:

```python
# 예: 메인 색상 변경
background-color: #3498db  # 파란색
```

### 폰트 변경

`main.py`에서:
```python
font = QFont('맑은 고딕', 10)  # ← 여기 수정
```

---

## 🚧 향후 개발 계획

- [ ] 이벤트 로그 상세 화면
- [ ] 사용자 관리 CRUD 완성
- [ ] 설정 화면 구현
- [ ] 자동 신고 시스템 연동
- [ ] 동영상 녹화 및 재생
- [ ] 통계 및 리포트 생성
- [ ] 모바일 앱 연동

---

## 📞 문의

문제가 발생하면:
1. 로그 파일 확인: `logs/`
2. 데이터베이스 확인: MySQL 로그
3. 시스템 로그 확인: `/var/log/`

---

**작성일**: 2026-01-28
**버전**: 1.0.0
**작성자**: Home Safe Solution Team

🏠 **Home Safe Solution** - 가정의 안전을 지키는 스마트 낙상 감지 시스템