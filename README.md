## 🏠 Home Safe Solution - Vision AI Project
- Home Safe Solution은 YOLO Pose 기반의 키포인트 검출과 Random Forest 분류기를 결합하여 실시간으로 낙상을 감지하는 지능형 안전 관리 시스템입니다.


## 🌟 주요 기능 (Key Features)
- 실시간 낙상 감지: 웹캠 피드를 통해 YOLO v8/v11 Pose 모델로 17개의 신체 키포인트를 추출하고 낙상 여부를 판별합니다.

- 지능형 데이터 파이프라인: 가속도계 데이터와 비디오 데이터를 통합하여 181개의 특징(정적 및 동적)을 생성합니다.

- 관리자 GUI 대시보드: PyQt6를 활용하여 실시간 모니터링, 이벤트 로그 관리, 통계 대시보드 및 사용자 관리 기능을 제공합니다.

- 다중 클래스 분류: 단순 이진 분류(정상/낙상)뿐만 아니라 3클래스(정상/낙상 중/낙상 완료) 분류 모델을 지원합니다.


## 📂 디렉토리 구조 (Directory Structure)

myproj/
├── gui/                # PyQt6 GUI 애플리케이션 소스 코드
├── scripts/            # 데이터 전처리, 특징 추출 및 모델 학습 스크립트
├── models/             # YOLO 및 학습된 Random Forest 모델 (.pt, .pkl)
├── dataset/            # 학습 및 검증용 데이터셋 (Binary / 3-Class)
├── data/               # 원본 비디오 데이터 (MP4)
├── features/           # 추출된 181개 특징 데이터 (CSV)
└── skeleton/           # YOLO Pose 키포인트 추출 데이터 (CSV)


## 🚀 시작하기 (Getting Started)
* 환경 설정 (Prerequisites)
 * Python 3.8 이상 권장
 * MySQL 데이터베이스 서버
* 설치 (Installation)
<code><pre>
git clone https://github.com/skong097/vision_ai.git
cd vision_ai
pip install -r requirements.txt
</code></pre>
* 실행 (Usage)
<code><pre>
python gui/main.py
</code></pre>

## 📊 분석 알고리즘 (Algorithm Details)
* Pose Estimation: YOLO v8n-pose / v11s-pose
* Classification: Random Forest (100 trees, depth=20) 
* feature Engineering:
  * 정적 특징: 관절 각도, 신체 비율, 가속도계 데이터 등
  * 동적 특징: 키포인트별 속도 및 가속도, 이동 평균 통계량 등 (총 181개)


## 🛠 기술 스택 (Tech Stack)
* Vision: Ultralytics YOLO, OpenCV
* Machine Learning: Scikit-learn, Pandas, NumPy
* GUI: PyQt6
* Database: MySQL (mysql-connector-python)
* Security: Bcrypt (Password Hashing)
