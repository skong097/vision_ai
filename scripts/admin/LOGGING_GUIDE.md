# 🔍 로깅 시스템 사용 가이드

## 📋 개요

실시간 낙상 감지 시스템에 **상세 로깅 기능**을 추가했습니다!

### 주요 기능
- ✅ 모든 이벤트 자동 기록
- ✅ 에러/경고 추적
- ✅ 예측 통계 저장
- ✅ 낙상 이벤트 상세 로그
- ✅ 자동 이슈 진단

---

## 🚀 실행 방법

### 1️⃣ 로깅이 포함된 실시간 감지

```bash
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python realtime_fall_detection_with_logging.py
```

**출력:**
```
📝 로그 파일: logs/fall_detection_20260128_143025.log
🤖 모델 로딩 중...
✅ Feature 순서 로드: 180개
✅ 모델 로드 완료!
...
```

### 2️⃣ 로그 디렉토리 지정

```bash
python realtime_fall_detection_with_logging.py --log-dir my_logs
```

### 3️⃣ 비디오 저장 + 로깅

```bash
python realtime_fall_detection_with_logging.py --save --output test.mp4
```

---

## 📁 생성되는 파일

### 로그 디렉토리 구조
```
logs/
├── fall_detection_20260128_143025.log    # 상세 로그
├── statistics_20260128_143520.json       # 통계 JSON
├── fall_detection_20260128_150130.log    # 다음 실행 로그
└── statistics_20260128_150625.json       # 다음 실행 통계
```

---

## 📄 로그 파일 내용

### 로그 파일 (.log)
```
2026-01-28 14:30:25 [INFO] ============================================================
2026-01-28 14:30:25 [INFO] 실시간 낙상 감지 시스템 초기화 시작
2026-01-28 14:30:25 [INFO] ============================================================
2026-01-28 14:30:26 [INFO] ✅ Random Forest 모델 로드 성공
2026-01-28 14:30:26 [INFO] ✅ YOLO Pose 모델 로드 성공
2026-01-28 14:30:26 [INFO] ✅ Feature 순서 로드: 180개
2026-01-28 14:30:26 [INFO] ✅ 초기화 완료!
2026-01-28 14:30:30 [INFO] Frame 100: Normal (확률: 96.2%)
2026-01-28 14:30:35 [INFO] Frame 200: Normal (확률: 95.8%)
2026-01-28 14:30:42 [WARNING] 🚨 낙상 감지! {
  "timestamp": "2026-01-28T14:30:42.123456",
  "frame": 253,
  "prediction": "Falling",
  "probability": 0.87,
  "hip_height": 320.5,
  "spine_angle": 65.2,
  "hip_velocity": -15.3
}
2026-01-28 14:30:45 [INFO] 사용자 종료 요청
```

### 통계 파일 (.json)
```json
{
  "total_frames": 500,
  "prediction_stats": {
    "Normal": 450,
    "Falling": 30,
    "Fallen": 20
  },
  "fall_events": [
    {
      "timestamp": "2026-01-28T14:30:42.123456",
      "frame": 253,
      "prediction": "Falling",
      "probability": 0.87,
      "hip_height": 320.5,
      "spine_angle": 65.2,
      "hip_velocity": -15.3
    }
  ],
  "avg_fps": 24.5
}
```

---

## 🔍 로그 분석

### 자동 분석 실행

```bash
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python analyze_logs.py
```

**출력:**
```
============================================================
📁 로그 분석 시작
============================================================
로그 디렉토리: logs
로그 파일: 3개
통계 파일: 3개

============================================================
📄 로그 분석: fall_detection_20260128_143025.log
============================================================

📊 기본 통계
  총 로그 라인: 245
  INFO:    230 (93.9%)
  WARNING: 10 (4.1%)
  ERROR:   5 (2.0%)

❌ 에러 발견: 5개
  1. 2026-01-28 14:30:30 [ERROR] 예측 오류: Feature names mismatch
  2. ...

⚠️  경고 발견: 10개
  1. 2026-01-28 14:30:42 [WARNING] 🚨 낙상 감지!
  2. ...

🎯 예측 로그: 15개
  예측 분포:
    Normal: 12회
    Falling: 2회
    Fallen: 1회

🚨 낙상 이벤트: 3개

============================================================
🔍 이슈 진단
============================================================

🔴 발견된 이슈:
  ❌ 에러 발견: 5개
  - Feature 순서 불일치 문제

💡 권장 조치:
  1. feature_columns.txt 파일이 존재하는지 확인
  2. 모델을 재학습하여 feature 순서 일치시키기
```

### 특정 파일 분석

```bash
python analyze_logs.py --file logs/fall_detection_20260128_143025.log
```

---

## 📊 이슈 진단 기능

### 자동으로 감지하는 이슈

#### 1️⃣ Feature 관련 문제
```
❌ Feature 순서 불일치 문제

💡 권장 조치:
  1. feature_columns.txt 파일 존재 확인
  2. 모델 재학습
```

#### 2️⃣ Keypoint 감지 문제
```
⚠️  Keypoint 감지 불안정

💡 권장 조치:
  1. 조명 확인 (밝은 곳에서 테스트)
  2. 카메라와의 거리 조정 (1~3m)
  3. 전신이 화면에 들어오도록
```

#### 3️⃣ 웹캠 연결 문제
```
❌ 웹캠 연결 문제

💡 권장 조치:
  1. 웹캠 연결 확인
  2. 다른 카메라 ID 시도
  3. 권한 확인
```

---

## 🧪 테스트 시나리오

### 정상 작동 확인

```bash
# 1. 실행
python realtime_fall_detection_with_logging.py

# 2. 웹캠 앞에서 정상 행동 (서있기, 걷기)

# 3. 종료 (q 키)

# 4. 로그 확인
python analyze_logs.py
```

**기대 결과:**
```
✅ 에러 없음!
✅ 경고 없음!
🎯 예측 로그: 50개
  예측 분포:
    Normal: 50회
✅ 이슈 없음! 시스템이 정상 작동 중입니다.
```

### 낙상 감지 확인

```bash
# 1. 실행
python realtime_fall_detection_with_logging.py

# 2. 천천히 바닥에 눕기

# 3. 종료

# 4. 로그 확인
python analyze_logs.py
```

**기대 결과:**
```
🚨 낙상 이벤트: 2개
  1. Frame 180: Falling (확률: 85.3%)
     Hip Height: 280.5, Spine Angle: 75.2°
  2. Frame 200: Fallen (확률: 92.1%)
     Hip Height: 150.3, Spine Angle: 85.5°
```

---

## 📈 로그 활용

### 1. 성능 모니터링
```bash
# 통계 파일에서 FPS 확인
cat logs/statistics_*.json | grep avg_fps
```

### 2. 예측 분포 분석
```bash
# 어떤 상태가 가장 많이 예측되었는지
python analyze_logs.py
```

### 3. 낙상 이벤트 추적
```bash
# 낙상이 언제 얼마나 발생했는지
grep "낙상 감지" logs/*.log
```

### 4. 에러 패턴 분석
```bash
# 어떤 에러가 반복되는지
grep ERROR logs/*.log | sort | uniq -c
```

---

## 💡 디버깅 팁

### 문제 1: 로그 파일이 생성되지 않음
```bash
# logs 디렉토리 권한 확인
ls -ld logs

# 권한이 없으면 생성
mkdir -p logs
chmod 755 logs
```

### 문제 2: 로그가 너무 많음
```python
# 로그 레벨 조정 (realtime_fall_detection_with_logging.py)
ch.setLevel(logging.WARNING)  # INFO → WARNING
```

### 문제 3: 특정 에러 추적
```bash
# 특정 키워드로 검색
grep -i "feature" logs/*.log
grep -i "keypoint" logs/*.log
```

---

## 🔧 고급 설정

### 로그 포맷 변경
```python
# setup_logger() 함수에서
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s'  # 함수명 추가
)
```

### 로그 파일 크기 제한
```python
# RotatingFileHandler 사용
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

## 📞 문제 해결

### 로그를 공유할 때

```bash
# 최근 로그 파일 압축
cd /home/gjkong/dev_ws/yolo/myproj/scripts
tar -czf logs_backup.tar.gz logs/*.log logs/*.json

# 압축 파일 공유
```

### 원격 지원 시

```bash
# 로그를 실시간으로 확인
tail -f logs/fall_detection_*.log
```

---

## ✅ 체크리스트

실행 후 다음을 확인:

- [ ] 로그 파일이 생성되었는가?
- [ ] 에러 메시지가 없는가?
- [ ] Feature 순서 로드 성공?
- [ ] FPS가 15 이상인가?
- [ ] 예측이 정상적으로 작동하는가?
- [ ] 낙상 이벤트가 기록되는가?

---

**작성일**: 2026-01-28  
**버전**: 2.0.0  
**작성자**: Home Safe Solution Team
