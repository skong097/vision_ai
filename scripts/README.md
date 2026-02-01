# Home Safe Solution - 전체 파이프라인

## 📁 프로젝트 구조

```
/home/gjkong/dev_ws/yolo/myproj/
├── data/                    # 원본 동영상 (30개)
│   ├── fall-01-cam0.mp4
│   ├── fall-02-cam0.mp4
│   └── ...
├── accel/                   # 가속도 데이터 (30개)
│   ├── fall-01-acc.csv
│   ├── fall-02-acc.csv
│   └── ...
├── skeleton/                # Step 1: Skeleton 데이터
│   ├── fall-01-skeleton.csv
│   └── ...
├── features/                # Step 2: Feature Engineering
│   ├── fall-01-features.csv
│   └── ...
├── labeled/                 # Step 3: Auto Labeling
│   ├── fall-01-labeled.csv
│   ├── visualizations/      # 라벨링 시각화
│   └── ...
├── dataset/                 # Step 4: Train/Val/Test
│   ├── binary/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── feature_columns.txt
│   └── 3class/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── feature_columns.txt
├── models/                  # Step 5: 학습된 모델
│   ├── binary/
│   │   ├── random_forest_model.pkl
│   │   ├── feature_columns.txt
│   │   └── visualizations/
│   │       ├── confusion_matrix.png
│   │       ├── feature_importance.png
│   │       └── roc_curve.png
│   └── 3class/
│       ├── random_forest_model.pkl
│       ├── feature_columns.txt
│       └── visualizations/
│           ├── confusion_matrix.png
│           └── feature_importance.png
└── scripts/                 # 처리 스크립트
    ├── extract_skeleton.py
    ├── visualize_skeleton.py
    ├── feature_engineering.py
    ├── auto_labeling.py
    ├── create_dataset.py
    ├── train_random_forest.py
    ├── run_pipeline.py      # 🚀 전체 실행
    └── README.md
```

## 🚀 실행 방법

### 방법 1: 전체 파이프라인 자동 실행 (추천) ⭐

```bash
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python run_pipeline.py
```

**실행 내용**:
- Step 1: Skeleton 추출 (YOLO Pose)
- Step 2: Feature Engineering
- Step 3: Auto Labeling (Binary & 3-Class)
- Step 4: Dataset 생성 (Train/Val/Test)
- Step 5: Random Forest 학습 (Binary & 3-Class)

**예상 소요 시간**: 15~30분 (GPU 사용 시)

**Skeleton이 이미 추출된 경우**:
```bash
python run_pipeline.py --skip-skeleton
```

---

### 방법 2: 단계별 수동 실행

---

### 방법 2: 단계별 수동 실행

#### Step 1: Skeleton 추출

```bash
cd /home/gjkong/dev_ws/yolo/myproj/scripts
python extract_skeleton.py
```

**실행 시간**: 약 10-15분 (GPU 사용 시)
**출력**: `/skeleton/fall-XX-skeleton.csv` (55개 컬럼)

#### Step 2: Feature Engineering

```bash
python feature_engineering.py
```

**실행 시간**: 약 2-3분
**출력**: `/features/fall-XX-features.csv` (추가 feature 포함)

#### Step 3: Auto Labeling

```bash
python auto_labeling.py
```

**실행 시간**: 약 1-2분
**출력**: 
- `/labeled/fall-XX-labeled.csv` (Binary & 3-Class 라벨)
- `/labeled/visualizations/` (처음 5개 시각화)

#### Step 4: Dataset 생성

```bash
python create_dataset.py
```

**실행 시간**: 약 1분
**출력**:
- `/dataset/binary/train.csv, val.csv, test.csv`
- `/dataset/3class/train.csv, val.csv, test.csv`

#### Step 5: Random Forest 학습

```bash
python train_random_forest.py
```

**실행 시간**: 약 5-10분
**출력**:
- `/models/binary/random_forest_model.pkl`
- `/models/3class/random_forest_model.pkl`
- 각 모델의 시각화 결과

---

### 방법 3: 시각화 및 검증

#### Skeleton 시각화

---

### 방법 3: 시각화 및 검증

#### Skeleton 시각화

```bash
python visualize_skeleton.py
```

**기능**:
- 동영상에 skeleton 오버레이
- 가속도 데이터 실시간 표시
- 'q'를 눌러 종료

---

## 📊 데이터 구조

### Skeleton CSV (55개 컬럼)

```
frame_id            : 프레임 번호
timestamp_ms        : 타임스탬프 (밀리초)

# 17개 Keypoints × 3 = 51개
nose_x, nose_y, nose_conf
left_eye_x, left_eye_y, left_eye_conf
... (17개 keypoints)

# 가속도 데이터 4개
acc_x, acc_y, acc_z, acc_mag
```

### Features CSV (100+ 컬럼)

```
# 기본 skeleton 데이터 (55개)
+ 정적 특징 (~20개)
  - 관절 각도 (팔꿈치, 무릎, 척추 등)
  - 신체 높이 (hip, shoulder, head)
  - Bounding box (width, height, aspect ratio)
  - 신체 기울기

+ 동적 특징 (~50개)
  - 각 keypoint의 속도 (velocity)
  - 각 keypoint의 가속도 (acceleration)
  - Rolling window 통계 (mean, std)
  - 가속도 센서 변화율
```

### Labeled CSV

Features + 2개 라벨 컬럼:
- `label_binary`: 0=Normal, 1=Fall
- `label_3class`: 0=Normal, 1=Falling, 2=Fallen

---

## 🎯 라벨링 전략

### Binary Classification
```python
0: Normal   - 낙상 이전의 모든 행동
1: Fall     - 낙상 발생 시점부터 끝까지
```

**장점**: 단순, 빠름, 높은 정확도
**단점**: 상황 파악 어려움, 오탐 가능성

### 3-Class Classification
```python
0: Normal   - 서있기, 걷기, 앉기 등
1: Falling  - 넘어지는 순간 (1~2초)
2: Fallen   - 바닥에 쓰러진 상태
```

**장점**: 세밀한 상황 파악, 지능형 알람, 오탐 감소
**단점**: 복잡도 증가, 더 많은 데이터 필요

**자동 라벨링 로직**:
1. 가속도 급증(>2.5g) 시점 → Falling 시작
2. 높이가 정상의 60% 이하 → Fallen 시작
3. Falling 이전 → Normal

---

## 📈 모델 성능 지표

학습 완료 후 다음 지표들이 출력됩니다:

- **Accuracy**: 전체 정확도
- **Precision**: 양성 예측의 정확도
- **Recall**: 실제 양성을 찾아낸 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **Confusion Matrix**: 혼동 행렬
- **Feature Importance**: 중요 특징 순위
- **ROC Curve** (Binary만): 성능 곡선 및 AUC

---

## 🔍 COCO Keypoints (17개)

```
0:  nose            (코)
1:  left_eye        (왼쪽 눈)
2:  right_eye       (오른쪽 눈)
3:  left_ear        (왼쪽 귀)
4:  right_ear       (오른쪽 귀)
5:  left_shoulder   (왼쪽 어깨)
6:  right_shoulder  (오른쪽 어깨)
7:  left_elbow      (왼쪽 팔꿈치)
8:  right_elbow     (오른쪽 팔꿈치)
9:  left_wrist      (왼쪽 손목)
10: right_wrist     (오른쪽 손목)
11: left_hip        (왼쪽 골반)
12: right_hip       (오른쪽 골반)
13: left_knee       (왼쪽 무릎)
14: right_knee      (오른쪽 무릎)
15: left_ankle      (왼쪽 발목)
16: right_ankle     (오른쪽 발목)
```

## ⚙️ 필요 패키지

```bash
pip install ultralytics opencv-python pandas numpy tqdm matplotlib seaborn scikit-learn joblib
```

## 📋 전체 워크플로우

```
[원본 데이터]
   ↓
[Step 1] YOLO Pose → Skeleton 추출 (17 keypoints)
   ↓
[Step 2] Feature Engineering → 100+ features
   ↓
[Step 3] Auto Labeling → Binary & 3-Class 라벨
   ↓
[Step 4] Dataset 생성 → Train/Val/Test (80/10/10)
   ↓
[Step 5] Random Forest 학습
   ↓
[결과] 2개 모델 (Binary & 3-Class)
```

## 🎉 완료 후 확인사항

### 1. 모델 파일 확인
```bash
ls -lh /home/gjkong/dev_ws/yolo/myproj/models/binary/
ls -lh /home/gjkong/dev_ws/yolo/myproj/models/3class/
```

### 2. 시각화 결과 확인
```bash
# 라벨링 시각화
ls /home/gjkong/dev_ws/yolo/myproj/labeled/visualizations/

# 모델 성능 시각화
ls /home/gjkong/dev_ws/yolo/myproj/models/binary/visualizations/
ls /home/gjkong/dev_ws/yolo/myproj/models/3class/visualizations/
```

### 3. 성능 비교
학습 완료 후 터미널에 출력된 성능 지표를 비교하여 Binary vs 3-Class 중 선택

---

## 🚨 문제 해결

### CUDA out of memory
```bash
# CPU 모드로 실행
export CUDA_VISIBLE_DEVICES=""
python run_pipeline.py
```

### 패키지 누락
```bash
pip install -r requirements.txt
```

### 모델 로드 실패
```python
import joblib
model = joblib.load('models/binary/random_forest_model.pkl')
```

---

## 📞 다음 단계

### LSTM 모델 (예정)
- Sequence 데이터 생성
- LSTM 아키텍처 설계
- 성능 비교 (RF vs LSTM)
- 앙상블 고려


**작성일**: 2026-01-28
**버전**: 1.0.0
**저자**: Home Safe Solution Team
