"""
학습된 Random Forest 모델 테스트
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO


class FallDetector:
    """학습된 모델을 사용한 낙상 감지기"""
    
    def __init__(self, model_path, model_type='3class'):
        """
        Args:
            model_path: 모델 파일 경로
            model_type: 'binary' or '3class'
        """
        self.model = joblib.load(model_path)
        self.model_type = model_type
        self.yolo_model = YOLO('yolov8n-pose.pt')
        
        # 클래스 이름
        if model_type == 'binary':
            self.class_names = {0: 'Normal', 1: 'Fall'}
        else:
            self.class_names = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
        
        # Feature 컬럼 로드
        feature_file = Path(model_path).parent / 'feature_columns.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()[2:]  # 헤더 스킵
                self.feature_cols = [line.strip().split('. ')[1] for line in lines if '. ' in line]
        else:
            self.feature_cols = None
        
        print(f"✅ 모델 로드 완료: {model_type}")
        print(f"   Features: {len(self.feature_cols) if self.feature_cols else 'Unknown'}")
    
    def predict_from_video(self, video_path, output_path=None):
        """동영상에서 낙상 감지"""
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"\n📹 처리 중: {video_path.name}")
        print("=" * 60)
        
        frame_idx = 0
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO Pose 추론
            results = self.yolo_model(frame, verbose=False)
            
            # TODO: Feature 추출 및 예측
            # (실제로는 feature_engineering.py의 로직을 사용해야 함)
            # 지금은 시연용으로 간단히 표시
            
            # 화면에 프레임 번호 표시
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if output_path:
                out.write(frame)
            else:
                cv2.imshow('Fall Detection', frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        cap.release()
        if output_path:
            out.release()
            print(f"✅ 저장 완료: {output_path}")
        else:
            cv2.destroyAllWindows()
    
    def predict_from_features(self, features_csv):
        """Features CSV 파일에서 예측"""
        
        df = pd.read_csv(features_csv)
        
        # Feature 선택
        if self.feature_cols:
            # 존재하는 feature만 선택
            available_features = [col for col in self.feature_cols if col in df.columns]
            X = df[available_features]
        else:
            # 라벨 컬럼 제외
            label_cols = ['label_binary', 'label_3class', 'frame_id', 'timestamp_ms']
            X = df[[col for col in df.columns if col not in label_cols]]
        
        # NaN 처리
        X = X.fillna(0)
        
        # 예측
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)
        
        # 결과 분석
        print(f"\n📊 예측 결과 - {features_csv.name}")
        print("=" * 60)
        print(f"총 프레임: {len(predictions)}")
        
        for label, name in self.class_names.items():
            count = np.sum(predictions == label)
            percentage = count / len(predictions) * 100
            print(f"{name}: {count} frames ({percentage:.1f}%)")
        
        # 시계열 예측 시각화
        self.plot_prediction_timeline(predictions, features_csv.stem)
        
        return predictions, proba
    
    def plot_prediction_timeline(self, predictions, title):
        """예측 결과 타임라인 시각화"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 4))
        
        colors = {0: 'green', 1: 'orange', 2: 'red'}
        color_map = [colors.get(pred, 'gray') for pred in predictions]
        
        plt.scatter(range(len(predictions)), predictions, c=color_map, s=20, alpha=0.6)
        plt.plot(predictions, 'k-', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Predicted Class', fontsize=12)
        plt.title(f'Fall Detection Timeline - {title}', fontsize=14, fontweight='bold')
        
        # Y축 라벨
        plt.yticks(list(self.class_names.keys()), list(self.class_names.values()))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'prediction_{title}.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 시각화 저장: prediction_{title}.png")
        plt.close()


def test_binary_model():
    """Binary 모델 테스트"""
    
    model_path = '/home/gjkong/dev_ws/yolo/myproj/models/binary/random_forest_model.pkl'
    detector = FallDetector(model_path, model_type='binary')
    
    # 테스트할 features 파일
    test_file = Path('/home/gjkong/dev_ws/yolo/myproj/labeled/fall-01-labeled.csv')
    
    if test_file.exists():
        predictions, proba = detector.predict_from_features(test_file)
    else:
        print(f"⚠️  파일을 찾을 수 없습니다: {test_file}")


def test_3class_model():
    """3-Class 모델 테스트"""
    
    model_path = '/home/gjkong/dev_ws/yolo/myproj/models/3class/random_forest_model.pkl'
    detector = FallDetector(model_path, model_type='3class')
    
    # 테스트할 features 파일
    test_file = Path('/home/gjkong/dev_ws/yolo/myproj/labeled/fall-01-labeled.csv')
    
    if test_file.exists():
        predictions, proba = detector.predict_from_features(test_file)
    else:
        print(f"⚠️  파일을 찾을 수 없습니다: {test_file}")


def main():
    """메인 함수"""
    
    print("=" * 60)
    print("🧪 모델 테스트")
    print("=" * 60)
    
    print("\n1️⃣  Binary Model Test")
    print("-" * 60)
    test_binary_model()
    
    print("\n2️⃣  3-Class Model Test")
    print("-" * 60)
    test_3class_model()


if __name__ == "__main__":
    main()