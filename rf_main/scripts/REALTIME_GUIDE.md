# 🎥 실시간 낙상 감지 시스템 사용 가이드

## 📋 기능

- ✅ 웹캠 실시간 영상 처리
- ✅ YOLO Pose skeleton 오버레이
- ✅ Random Forest 모델로 낙상 감지
- ✅ 실시간 상태 표시 (Normal/Falling/Fallen)
- ✅ 확률 및 주요 Feature 표시
- ✅ FPS 모니터링
- ✅ 비디오 녹화 기능
- ✅ 일시정지/재생 기능

---

## 🚀 실행 방법

### 1️⃣ 기본 실행 (3-Class 모델)

```bash
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python realtime_fall_detection.py
```

**결과:**
- 웹캠 영상 실시간 표시
- Skeleton 오버레이
- 낙상 상태 실시간 표시

---

### 2️⃣ Binary 모델 사용

```bash
python realtime_fall_detection.py --model-type binary
```

---

### 3️⃣ 비디오 녹화

```bash
python realtime_fall_detection.py --save --output my_detection.mp4
```

**결과:**
- 실시간 감지 + 비디오 파일 저장

---

### 4️⃣ 다른 카메라 사용

```bash
# 외장 웹캠 (카메라 ID = 1)
python realtime_fall_detection.py --camera 1
```

---

### 5️⃣ 모든 옵션 조합

```bash
python realtime_fall_detection.py \
    --model-type 3class \
    --camera 0 \
    --save \
    --output fall_detection_result.mp4
```

---

## ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `q` | 종료 |
| `p` | 일시정지/재생 |

---

## 📊 화면 정보

### 왼쪽 상단 (정보 패널)
```
Status: Normal / Falling / Fallen  (색상: 초록/주황/빨강)

Normal:  95.2%   ← 각 클래스별 확률
Falling:  3.1%
Fallen:   1.7%

Hip Height: 320.5    ← 주요 Feature
Spine Angle: 12.3
```

### 오른쪽 상단
```
FPS: 28.5       ← 처리 속도
Frame: 1523     ← 현재 프레임
```

### 화면 하단 (낙상 감지 시)
```
!!! FALL DETECTED !!!  (빨간색 경고)
```

---

## 🎨 색상 코드

### Binary 모델
- 🟢 **초록색**: Normal (정상)
- 🔴 **빨간색**: Fall (낙상)

### 3-Class 모델
- 🟢 **초록색**: Normal (정상)
- 🟠 **주황색**: Falling (낙상 중) ← 골든타임!
- 🔴 **빨간색**: Fallen (쓰러짐)

---

## 🔧 최적화 팁

### 1. FPS 향상

**GPU 사용 확인:**
```bash
nvidia-smi
```

**CPU 코어 사용:**
- Random Forest는 멀티코어 사용
- 16코어 시스템에서 약 25~30 FPS

### 2. 해상도 조정

스크립트에서 수정:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 더 낮추면 FPS 증가
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 3. 버퍼 크기 조정

```python
detector = RealtimeFallDetector(model_path, buffer_size=15)  # 30 → 15
```

---

## 🧪 테스트 시나리오

### 시나리오 1: 정상 행동
1. 웹캠 앞에서 서있기
2. 천천히 걷기
3. 앉았다 일어서기

**예상 결과:** `Status: Normal` (초록색)

---

### 시나리오 2: 낙상 시뮬레이션
1. 서있는 상태에서 시작
2. 천천히 바닥에 누우기
3. 누운 채로 유지

**예상 결과:**
- `Status: Falling` (주황색) → 넘어지는 동안
- `Status: Fallen` (빨간색) → 누운 후

---

### 시나리오 3: 빠른 낙상
1. 서있는 상태
2. 빠르게 쪼그려 앉기 또는 무릎 꿇기

**예상 결과:**
- 급격한 높이 변화 감지
- `Status: Falling` → `Status: Fallen`

---

## 📈 성능 지표

| 항목 | 값 |
|------|-----|
| **정확도** | 93~95% |
| **FPS (GPU)** | 25~30 |
| **FPS (CPU)** | 15~20 |
| **지연시간** | <50ms |
| **감지 거리** | 1~5m |

---

## ⚠️ 문제 해결

### 1. 웹캠이 안 열림
```bash
# 사용 가능한 카메라 확인
ls /dev/video*

# 다른 카메라 시도
python realtime_fall_detection.py --camera 1
```

### 2. FPS가 너무 낮음 (<10)
- 해상도 낮추기: 640x480 → 320x240
- 버퍼 크기 줄이기: 30 → 15
- YOLO 모델 변경: yolov8n → yolov8n (이미 가장 작은 모델)

### 3. Skeleton이 안 보임
- 조명 확인 (밝은 곳에서 테스트)
- 카메라와의 거리: 1~3m 권장
- 전신이 화면에 들어오도록

### 4. 예측이 이상함
- 모델 경로 확인
- Feature 계산 확인 (터미널 로그)
- 학습 데이터와 유사한 환경에서 테스트

---

## 🔬 디버그 모드

추가 정보 출력을 위해 스크립트 수정:

```python
# realtime_fall_detection.py 에서

def predict(self, features):
    # 예측 전에 feature 출력
    print(f"Features: {features}")
    
    # ... (기존 코드)
```

---

## 📝 로그 저장

```bash
# 터미널 출력을 파일로 저장
python realtime_fall_detection.py 2>&1 | tee detection_log.txt
```

---

## 🎯 다음 단계

### 1. 알람 시스템 추가
```python
if prediction == 1:  # Falling
    send_notification("낙상 감지!")
elif prediction == 2 and duration > 180:  # Fallen > 3분
    call_emergency()
```

### 2. 데이터베이스 연동
```python
# MySQL에 이벤트 저장
save_event_to_db(timestamp, prediction, features)
```

### 3. 다중 카메라 지원
```python
# 여러 카메라 동시 모니터링
cameras = [0, 1, 2]
detectors = [FallDetector(cam) for cam in cameras]
```

### 4. GUI 개발 (PyQt6)
- 실시간 영상 + 통계 대시보드
- 이벤트 로그 표시
- 설정 변경 UI

---

## 💡 사용 팁

1. **조명**: 밝은 곳에서 테스트 (YOLO Pose 성능 향상)
2. **거리**: 카메라와 1~3m 거리 유지
3. **각도**: 정면 또는 측면에서 전신이 보이도록
4. **배경**: 단순한 배경 (복잡한 배경은 감지 방해)
5. **의복**: 몸의 윤곽이 명확한 옷 착용

---

## 📞 지원

문제가 발생하면 다음 정보를 포함하여 문의:
- OS 및 Python 버전
- GPU 사용 여부 (nvidia-smi 출력)
- 오류 메시지 전문
- 실행 명령어

---

**작성일**: 2026-01-28  
**버전**: 1.0.0  
**작성자**: Home Safe Solution Team
