# 🔧 MySQL 연결 오류 해결 가이드

## ❌ **발생한 오류**

```
mysql.connector.errors.ProgrammingError: 1698 (28000): 
Access denied for user 'root'@'localhost'
```

---

## ✅ **해결 방법 (2가지 중 선택)**

### **방법 1: SQLite로 변경** ⚡ (5분, 권장!)

MySQL 설치 없이 바로 사용 가능!

#### 1단계: 파일 교체

```bash
cd /home/gjkong/dev_ws/yolo/myproj/gui

# 기존 파일 백업
mv database_models.py database_models_mysql_backup.py

# SQLite 버전으로 교체
cp database_models_sqlite.py database_models.py
```

#### 2단계: 실행!

```bash
python main.py
```

**끝!** 🎉

---

### **방법 2: MySQL 설치 및 설정** 🗄️ (15분)

완전한 MySQL 데이터베이스 사용

#### 1단계: MySQL 설치

```bash
cd /home/gjkong/dev_ws/yolo/myproj/gui
chmod +x setup_mysql.sh
sudo ./setup_mysql.sh
```

#### 2단계: database_models.py 수정

```bash
nano database_models.py
```

**36번째 줄 근처 수정:**

```python
# 기존 (오류 발생)
self.config = {
    'host': 'localhost',
    'database': 'home_safe',
    'user': 'root',
    'password': 'your_password',  # ← 이게 문제!
    ...
}

# 수정 (올바름)
self.config = {
    'host': 'localhost',
    'database': 'home_safe',
    'user': 'homesafe',           # ← 변경
    'password': 'homesafe2026',    # ← 변경
    ...
}
```

저장: `Ctrl+X` → `Y` → `Enter`

#### 3단계: 실행!

```bash
python main.py
```

---

## 🔍 **어떤 방법을 선택할까?**

### SQLite (방법 1) - 권장! ⭐

**장점:**
- ✅ 설치 불필요
- ✅ 5분 만에 바로 사용
- ✅ 가볍고 빠름
- ✅ 파일 기반 (home_safe.db)

**단점:**
- ⚠️ 동시 접속 제한적
- ⚠️ 대용량 데이터 처리 느림

**추천 대상:** 
- 개인 사용자
- 프로토타입 테스트
- 빠른 시작 원하는 경우

### MySQL (방법 2)

**장점:**
- ✅ 대용량 데이터 처리
- ✅ 다중 사용자 지원
- ✅ 고급 기능 (트리거, 프로시저 등)

**단점:**
- ⚠️ 설치 및 설정 필요
- ⚠️ 메모리 사용량 높음

**추천 대상:**
- 실제 운영 환경
- 여러 사용자 동시 사용
- 대용량 데이터 처리

---

## 🚀 **추천 순서**

### 지금 바로 테스트하고 싶다면:

```bash
# 1. SQLite로 시작 (5분)
cd /home/gjkong/dev_ws/yolo/myproj/gui
mv database_models.py database_models_mysql.py
cp database_models_sqlite.py database_models.py
python main.py
```

### 나중에 MySQL로 마이그레이션:

```bash
# 2. 나중에 MySQL로 변경 (원할 때)
sudo ./setup_mysql.sh
mv database_models.py database_models_sqlite_backup.py
mv database_models_mysql.py database_models.py
nano database_models.py  # 비밀번호 수정
python main.py
```

---

## 🎯 **빠른 해결 (30초)**

지금 당장 실행하려면:

```bash
cd /home/gjkong/dev_ws/yolo/myproj/gui
mv database_models.py database_models.bak
cp database_models_sqlite.py database_models.py
python main.py
```

**로그인:**
- 아이디: `admin`
- 비밀번호: `admin123`

---

## 🐛 **추가 문제 해결**

### SQLite 오류 발생?

```bash
pip install pysqlite3 --break-system-packages
```

### 여전히 오류?

```bash
# 완전 초기화
rm -f home_safe.db
python main.py
```

### 데이터베이스 파일 위치

SQLite 사용 시:
```bash
ls -lh /home/gjkong/dev_ws/yolo/myproj/gui/home_safe.db
```

---

## 📊 **두 방법 비교**

| 항목 | SQLite | MySQL |
|------|--------|-------|
| 설치 시간 | 0분 | 15분 |
| 설정 난이도 | ⭐ | ⭐⭐⭐ |
| 성능 (소규모) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 성능 (대규모) | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 메모리 사용 | 낮음 | 높음 |
| 백업 | 파일 복사 | mysqldump |

---

## ✅ **최종 추천**

**지금 바로 테스트**: → **SQLite** ⚡
**실제 운영 환경**: → **MySQL** 🗄️

---

**결론**: SQLite로 시작하고, 필요하면 나중에 MySQL로 전환!

🏠 **Home Safe Solution** - 이제 바로 시작하세요! 🚀
