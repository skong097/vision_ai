# 🚀 Quick Start Guide

## ⚡ 5분 만에 시작하기

### 1단계: 설치 스크립트 실행 (1분)

```bash
cd /mnt/user-data/outputs
chmod +x install.sh
./install.sh
```

### 2단계: 데이터베이스 생성 (2분)

```bash
cd /home/gjkong/dev_ws/yolo/myproj/gui
mysql -u root -p < database_schema.sql
```

비밀번호 입력 후 완료!

### 3단계: MySQL 비밀번호 설정 (1분)

```bash
nano database_models.py
```

36번째 줄에서 수정:
```python
'password': 'YOUR_MYSQL_PASSWORD',  # ← 실제 비밀번호 입력
```

저장: `Ctrl+X` → `Y` → `Enter`

### 4단계: 실행! (1분)

```bash
python main.py
```

로그인:
- **아이디**: `admin`
- **비밀번호**: `admin123`

---

## 🎯 핵심 기능 테스트

### ✅ 1. 대시보드 확인
- 로그인 후 자동으로 대시보드 표시
- 통계 카드, 차트, 최근 이벤트 확인

### ✅ 2. 실시간 모니터링
1. **🎥 실시간 모니터링** 메뉴 클릭
2. **▶ 시작** 버튼 클릭
3. 웹캠 앞에서 **서있기** (전신이 보이도록!)
4. Skeleton이 표시되면 성공!
5. 상태 확인: 🟢 Normal

### ✅ 3. 낙상 테스트
1. 모니터링 중에 **천천히 쪼그려 앉기**
2. 🟡 Falling 감지
3. **바닥에 눕기**
4. 🔴 Fallen 감지
5. 이벤트 로그 확인

### ✅ 4. 이벤트 로그 확인
1. **📋 이벤트 로그** 메뉴 클릭
2. 기록된 이벤트 조회

---

## 🐛 문제 해결 (1분 해결)

### ❌ 문제 1: PyQt6 설치 오류

```bash
pip uninstall PyQt6
pip install PyQt6 --break-system-packages --no-cache-dir
```

### ❌ 문제 2: MySQL 연결 오류

```bash
# MySQL 시작
sudo systemctl start mysql

# 비밀번호 확인
mysql -u root -p -e "SELECT 1"
```

### ❌ 문제 3: 웹캠 인식 안 됨

```bash
# 웹캠 확인
ls /dev/video*

# 권한 부여
sudo chmod 666 /dev/video0
```

### ❌ 문제 4: Feature 값이 0

**원인**: 앉아서 테스트 → 하반신이 화면 밖

**해결**: 
1. **일어서기**
2. 카메라에서 1.5~2m 거리
3. **전신이 화면에 들어오도록**

---

## 📊 예상 결과

### 정상 작동 시

```
상태: 🟢 Normal (확률: 95.2%)
Hip Height: 320.5
Spine Angle: 12.8°
Confidence: 0.85
```

### 낙상 감지 시

```
상태: 🔴 Fallen (확률: 92.1%)
Hip Height: 150.3
Spine Angle: 85.5°
Confidence: 0.82
```

---

## 🎉 성공!

모든 것이 정상 작동하면:

1. ✅ 웹캠 영상 표시
2. ✅ Skeleton 오버레이
3. ✅ 실시간 상태 업데이트
4. ✅ 낙상 감지 및 이벤트 로그

**축하합니다! 🎊**

이제 다음 단계로:
- 사용자 추가
- RTSP 카메라 연결
- 자동 신고 설정
- 통계 분석

자세한 내용은 **GUI_README.md** 참조!

---

## 🆘 도움말

문제가 해결되지 않으면:

1. **로그 확인**: `logs/` 디렉토리
2. **README 읽기**: `GUI_README.md`
3. **데이터베이스 초기화**:
   ```bash
   mysql -u root -p -e "DROP DATABASE home_safe"
   mysql -u root -p < database_schema.sql
   ```

---

**제작**: Home Safe Solution Team  
**날짜**: 2026-01-28  
**버전**: 1.0.0

🏠 **Home Safe** - 가정의 안전을 지킵니다
