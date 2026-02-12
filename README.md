# 🏠 Home Safe Solution — Vision AI 낙상 감지 시스템

실시간 영상 기반 낙상(Fall) 감지 시스템.  
YOLO Pose + Random Forest / ST-GCN 모델을 활용한 이중 감지 파이프라인.

## 프로젝트 구조

```
vision_ai/
├── path_config.py              # 경로 중앙 관리
├── rf_main/                    # Random Forest 기반 메인 애플리케이션
│   ├── gui/                    # PyQt6 GUI (모니터링, 대시보드, 이벤트 관리)
│   ├── models/                 # YOLO 모델
│   ├── models_integrated/      # RF 학습 모델 (binary_v3)
│   ├── pipeline/               # 학습 파이프라인
│   ├── scripts/                # 유틸리티 스크립트
│   └── utils/                  # 전처리/학습 도구
└── st_gcn/                     # ST-GCN 시계열 모델
    ├── checkpoints_v2/         # Fine-tuned 모델
    ├── models/                 # ST-GCN 네트워크 정의
    ├── pretrained/             # NTU60 사전학습 가중치
    └── scripts/                # 데이터 준비/학습 스크립트
```

## 모델 성능

| 모델 | Accuracy | F1 | AUC | 추론 속도 |
|------|----------|-----|-----|----------|
| 🌲 Random Forest (v3b) | 97.99% | 94.48% | 99.71% | 0.01ms |
| 🚀 ST-GCN Fine-tuned (v2) | 99.63% | 99.40% | 99.98% | 0.34ms |

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/skong097/vision_ai.git
cd vision_ai
```

### 2. 의존성 설치
```bash
pip install torch torchvision ultralytics mediapipe
pip install PyQt6 mysql-connector-python bcrypt
pip install numpy opencv-python scikit-learn matplotlib
```

### 3. MySQL 데이터베이스 구축
```bash
mysql -u root -p < rf_main/gui/database_schema.sql
```

### 4. 모델 파일 준비
모델 파일(.pth, .pkl, .pt)은 용량 문제로 Git에 포함되지 않습니다.  
별도로 다운로드하여 해당 경로에 배치하세요.

### 5. 경로 검증
```bash
python path_config.py
```

### 6. 실행
```bash
cd rf_main/gui
python main.py
```

## 기술 스택

- **Pose Estimation:** YOLO11s-Pose (17 keypoints)
- **Frame-level 감지:** Random Forest (181 features, bbox 정규화)
- **Temporal 감지:** ST-GCN (60-frame sequence, PYSKL pretrained)
- **GUI:** PyQt6
- **DB:** MySQL 8.0
- **GPU:** CUDA (ST-GCN 추론)

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다.
