# Home Safe Solution - Vision AI 프로젝트 문서

## 프로젝트 개요

**Home Safe Solution**은 YOLO Pose 기반 키포인트 검출과 Random Forest 분류기를 활용한 **실시간 낙상 감지 시스템**입니다.

- **핵심 기술**: YOLO v8/v11 Pose, Random Forest, PyQt6

---

## 디렉토리 구조

```
myproj/
├── accel/              # 가속도계 센서 데이터 (CSV)
├── data/               # 원본 비디오 파일 (MP4)
├── dataset/            # 학습/검증/테스트 데이터셋
│   ├── 3class/         # 3클래스 분류용 (Normal/Falling/Fallen)
│   └── binary/         # 이진 분류용 (Normal/Fall)
├── features/           # 추출된 특징 데이터 (CSV)
├── gui/                # PyQt6 GUI 애플리케이션
├── labeled/            # 레이블링된 데이터
│   └── visualizations/ # 레이블링 시각화
├── models/             # 학습된 모델 파일
│   ├── 3class/         # 3클래스 Random Forest 모델
│   └── binary/         # 이진 분류 Random Forest 모델
├── scripts/            # 데이터 처리 및 학습 스크립트
│   ├── admin/          # 관리자용 스크립트
└── skeleton/           # YOLO Pose 키포인트 데이터 (CSV)
```

---

## 핵심 모듈 설명

### 1. scripts/ - 데이터 파이프라인

| 스크립트 | 기능 | 입력 | 출력 |
|---------|------|------|------|
| `extract_skeleton.py` | YOLO Pose 키포인트 추출 | MP4 + 가속도계 CSV | 스켈레톤 CSV |
| `feature_engineering.py` | 특징 추출 (180+ 특징) | 스켈레톤 CSV | 특징 CSV |
| `auto_labeling.py` | 자동 레이블링 | 특징 CSV | 레이블된 CSV |
| `create_dataset.py` | 데이터셋 분할 | 레이블된 데이터 | Train/Val/Test |
| `train_random_forest.py` | Random Forest 학습 | 데이터셋 CSV | 모델 (pkl) |
| `realtime_fall_detection.py` | 실시간 낙상 감지 | 웹캠 피드 | 예측 결과 |
| `run_pipeline.py` | 전체 파이프라인 오케스트레이터 | 원본 데이터 | 완성된 모델 |

### 2. gui/ - PyQt6 GUI 애플리케이션

| 파일 | 기능 |
|-----|------|
| `main.py` | 애플리케이션 진입점 |
| `login_window.py` | 로그인 (bcrypt 해시) |
| `main_window.py` | 메인 윈도우 + 사이드바 |
| `monitoring_page.py` | 실시간 웹캠 모니터링 |
| `dashboard_page.py` | 통계 대시보드 |
| `event_management_page.py` | 이벤트 관리 (CRUD) |
| `events_page.py` | 이벤트 로그 뷰어 |
| `users_page.py` | 사용자 관리 (관리자) |
| `database_models.py` | MySQL ORM |
| `fall_detector.py` | 낙상 감지 유틸리티 |
| `one_euro_filter.py` | 키포인트 스무딩 필터 |

### 3. models/ - 학습된 모델

```
models/
├── yolov8n-pose.pt          # YOLO v8 Nano Pose (6.8 MB)
├── yolo11s-pose.pt          # YOLO v11 Small Pose (20 MB)
├── binary/
│   ├── random_forest_model.pkl
│   └── feature_columns.txt   # 181개 특징 목록
└── 3class/
    ├── random_forest_model.pkl
    └── feature_columns.txt
```

---

## 데이터 파이프라인

```
원본 비디오 (fall-*.mp4) + 가속도계 (fall-*-acc.csv)
                    ↓
         [YOLO Pose Detection]
                    ↓
        스켈레톤 데이터 (17 키포인트 × 3)
                    ↓
         [Feature Engineering]
                    ↓
        특징 데이터 (180+ 특징)
                    ↓
           [Auto Labeling]
        ├─ Binary: Normal(0) / Fall(1)
        └─ 3-Class: Normal(0) / Falling(1) / Fallen(2)
                    ↓
         [Dataset Creation]
        Train(80%) / Val(10%) / Test(10%)
                    ↓
          [Model Training]
        Random Forest (100 trees, depth=20)
                    ↓
         [Real-time Inference]
        웹캠 → YOLO → RF → 이벤트 로깅
```

---

## 특징 구성 (181개)

### 정적 특징 (Static Features)
| 카테고리 | 특징 수 | 설명 |
|---------|--------|------|
| Raw Keypoints | 51 | 17 키포인트 × (x, y, confidence) |
| Joint Angles | 4 | 팔꿈치, 무릎 각도 |
| Body Metrics | 7 | 높이, 바운딩박스, 종횡비 |
| Orientation | 2 | 어깨 기울기, 평균 신뢰도 |
| Accelerometer | 4 | acc_x, acc_y, acc_z, acc_mag |

### 동적 특징 (Dynamic Features)
| 카테고리 | 특징 수 | 설명 |
|---------|--------|------|
| Keypoint Velocity | 34 | 각 키포인트 속도 (vx, vy) |
| Keypoint Acceleration | 34 | 각 키포인트 가속도 (ax, ay) |
| Rolling Statistics | 20+ | 이동 평균, 표준편차 |

---

## 주요 상수 및 설정

### YOLO 키포인트 (COCO 17-point)
```python
keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

### 분류 클래스
```python
# Binary
{0: 'Normal', 1: 'Fall'}

# 3-Class
{0: 'Normal', 1: 'Falling', 2: 'Fallen'}
```

### 감지 임계값
```python
confidence_threshold = 0.3      # 키포인트 신뢰도
fall_acc_threshold = 2.5        # 가속도 (g)
fallen_height_threshold = 0.6   # 정상 높이의 60%
model_confidence = 0.7          # 모델 예측 신뢰도
```

### Random Forest 설정
```python
RandomForestClassifier(
    n_estimators=100,        # 트리 개수
    max_depth=20,            # 최대 깊이
    class_weight='balanced', # 클래스 불균형 처리
    n_jobs=-1                # 병렬 처리
)
```

### 데이터베이스 (MySQL)
```python
config = {
    'host': 'localhost',
    'database': 'home_safe',
    'user': 'homesafe',
    'password': '*******'
}
```

---

## 데이터베이스 스키마

| 테이블 | 설명 |
|-------|------|
| `users` | 사용자 계정 (bcrypt 해시) |
| `event_types` | 이벤트 유형 (Fall, Fire, Flooding 등) |
| `event_logs` | 이벤트 기록 (신뢰도, 높이, 각도 등) |
| `auto_report_logs` | 119/112 자동 신고 로그 |
| `system_settings` | 시스템 설정 (key-value) |
| `login_history` | 로그인 이력 |

---

## 개발 가이드라인

### 파일 명명 규칙
```
# 비디오 파일
fall-{번호:02d}-cam0.mp4       예: fall-01-cam0.mp4

# 가속도계 데이터
fall-{번호:02d}-acc.csv        예: fall-01-acc.csv

# 스켈레톤 데이터
fall-{번호:02d}-skeleton.csv   예: fall-01-skeleton.csv

# 특징 데이터
fall-{번호:02d}-features.csv   예: fall-01-features.csv

# 레이블 데이터
fall-{번호:02d}-labeled.csv    예: fall-01-labeled.csv
```

### 코드 스타일
- Python 3.8+ 호환
- PEP 8 스타일 가이드 준수
- Type hints 권장
- Docstring 작성 (Google 스타일)

### 의존성 관리
```
ultralytics          # YOLO
opencv-python        # 영상 처리
numpy, pandas        # 데이터 처리
scikit-learn         # Random Forest
PyQt6                # GUI
mysql-connector-python  # DB
bcrypt               # 패스워드 해시
```

---

## 버전 관리 규칙

### 브랜치 전략
```
main              # 안정 버전
├── develop       # 개발 브랜치
├── feature/*     # 기능 개발
├── bugfix/*      # 버그 수정
└── release/*     # 릴리스 준비
```

### 커밋 메시지 규칙
```
[타입] 제목

본문 (선택)

타입:
- feat: 새 기능
- fix: 버그 수정
- docs: 문서 수정
- style: 코드 포맷팅
- refactor: 코드 리팩토링
- test: 테스트
- chore: 빌드/설정 변경

예시:
[feat] 3클래스 분류 모델 추가
[fix] 실시간 감지 시 메모리 누수 수정
[docs] README 업데이트
```

### 버전 태그
```
v{major}.{minor}.{patch}

예시:
v1.0.0  # 첫 안정 릴리스
v1.1.0  # 기능 추가
v1.1.1  # 버그 수정
```

---

## 체크리스트

### 새 기능 추가 시
- [ ] 기존 파이프라인과의 호환성 확인
- [ ] 특징 컬럼 순서 일관성 유지
- [ ] 단위 테스트 작성
- [ ] 문서 업데이트

### 모델 재학습 시
- [ ] 데이터셋 버전 기록
- [ ] 하이퍼파라미터 기록
- [ ] 성능 지표 저장 (Accuracy, F1, ROC-AUC)
- [ ] feature_columns.txt 업데이트
- [ ] 이전 모델 백업

### 배포 시
- [ ] 경로 하드코딩 제거
- [ ] 환경 변수 또는 config 파일 사용
- [ ] DB 마이그레이션 스크립트 확인
- [ ] 의존성 버전 고정 (requirements.txt)

---

## 주의사항

### 하드코딩된 경로 (수정 필요)
다음 경로들은 환경에 맞게 수정해야 합니다:
- `/home/gjkong/dev_ws/yolo/myproj/` (약 15개 파일에서 사용)
- `monitoring_page.py`의 모델 경로
- `realtime_fall_detection.py`의 모델 경로
- `database_models.py`의 DB 자격 증명

### 특징 컬럼 순서
**중요**: Random Forest 모델은 특징 컬럼 순서에 민감합니다.
- 항상 `feature_columns.txt` 순서를 따를 것
- 새 특징 추가 시 모델 재학습 필수

---

## 연락처 및 참고자료

- **YOLO 문서**: https://docs.ultralytics.com/
- **scikit-learn**: https://scikit-learn.org/
- **PyQt6**: https://www.riverbankcomputing.com/software/pyqt/

---

*최종 업데이트: 2026-01-31*
