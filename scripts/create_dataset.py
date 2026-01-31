"""
데이터셋 생성 - Train/Val/Test Split (Binary & 3-Class)
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DatasetCreator:
    """Train/Val/Test 데이터셋 생성"""
    
    def __init__(self, labeled_dir, output_dir):
        self.labeled_dir = Path(labeled_dir)
        self.output_dir = Path(output_dir)
        
        # Binary & 3-Class 별도 디렉토리
        self.binary_dir = self.output_dir / 'binary'
        self.class3_dir = self.output_dir / '3class'
        
        self.binary_dir.mkdir(parents=True, exist_ok=True)
        self.class3_dir.mkdir(parents=True, exist_ok=True)
    
    def get_feature_columns(self, df):
        """Feature 컬럼 선택 (라벨 제외)"""
        
        # 제외할 컬럼
        exclude_cols = [
            'frame_id', 'timestamp_ms', 'label_binary', 'label_3class',
            # confidence 컬럼은 포함 (중요 feature)
        ]
        
        # Feature 컬럼 선택
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def create_binary_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """Binary Classification 데이터셋 생성"""
        
        print("\n" + "=" * 60)
        print("🔹 Binary Classification Dataset")
        print("=" * 60)
        
        # 모든 labeled 파일 로드
        labeled_files = sorted(self.labeled_dir.glob("fall-*-labeled.csv"))
        
        all_data = []
        for file in labeled_files:
            df = pd.read_csv(file)
            all_data.append(df)
        
        # 전체 데이터 병합
        df_all = pd.concat(all_data, ignore_index=True)
        
        print(f"📊 전체 데이터: {len(df_all)} frames")
        
        # Feature와 Label 분리
        feature_cols = self.get_feature_columns(df_all)
        X = df_all[feature_cols]
        y = df_all['label_binary']
        
        print(f"📊 Features: {len(feature_cols)}개")
        print(f"📊 Label 분포:")
        print(y.value_counts())
        
        # Train/Temp split (80/20)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Temp를 Val/Test로 split (50/50 of temp = 10/10 of total)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        
        # 저장
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(self.binary_dir / 'train.csv', index=False)
        val_df.to_csv(self.binary_dir / 'val.csv', index=False)
        test_df.to_csv(self.binary_dir / 'test.csv', index=False)
        
        print(f"\n✅ Train: {len(train_df)} frames → {self.binary_dir / 'train.csv'}")
        print(f"✅ Val:   {len(val_df)} frames → {self.binary_dir / 'val.csv'}")
        print(f"✅ Test:  {len(test_df)} frames → {self.binary_dir / 'test.csv'}")
        
        # 통계
        print(f"\n📊 Train Label 분포:")
        print(y_train.value_counts())
        print(f"\n📊 Val Label 분포:")
        print(y_val.value_counts())
        print(f"\n📊 Test Label 분포:")
        print(y_test.value_counts())
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'feature_cols': feature_cols
        }
    
    def create_3class_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """3-Class Classification 데이터셋 생성"""
        
        print("\n" + "=" * 60)
        print("🔹 3-Class Classification Dataset")
        print("=" * 60)
        
        # 모든 labeled 파일 로드
        labeled_files = sorted(self.labeled_dir.glob("fall-*-labeled.csv"))
        
        all_data = []
        for file in labeled_files:
            df = pd.read_csv(file)
            all_data.append(df)
        
        # 전체 데이터 병합
        df_all = pd.concat(all_data, ignore_index=True)
        
        print(f"📊 전체 데이터: {len(df_all)} frames")
        
        # Feature와 Label 분리
        feature_cols = self.get_feature_columns(df_all)
        X = df_all[feature_cols]
        y = df_all['label_3class']
        
        print(f"📊 Features: {len(feature_cols)}개")
        print(f"📊 Label 분포:")
        print(y.value_counts())
        
        # 모든 클래스가 존재하는지 확인
        unique_labels = y.unique()
        if len(unique_labels) < 3:
            print(f"\n⚠️  경고: 일부 클래스가 없습니다! 존재하는 클래스: {sorted(unique_labels)}")
            print("   라벨링 로직을 확인하거나 Binary Classification을 사용하세요.")
            
            # 클래스가 2개 이상이면 계속 진행 (stratify 없이)
            if len(unique_labels) < 2:
                print("   ❌ 클래스가 1개뿐이므로 학습할 수 없습니다!")
                return None
        
        # Train/Temp split (80/20)
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"\n⚠️  Stratify 실패: {e}")
            print("   Stratify 없이 진행합니다...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )
        
        # Temp를 Val/Test로 split (50/50 of temp = 10/10 of total)
        try:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
            )
        except ValueError as e:
            print(f"\n⚠️  Stratify 실패: {e}")
            print("   Stratify 없이 진행합니다...")
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=None
            )
        
        # 저장
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(self.class3_dir / 'train.csv', index=False)
        val_df.to_csv(self.class3_dir / 'val.csv', index=False)
        test_df.to_csv(self.class3_dir / 'test.csv', index=False)
        
        print(f"\n✅ Train: {len(train_df)} frames → {self.class3_dir / 'train.csv'}")
        print(f"✅ Val:   {len(val_df)} frames → {self.class3_dir / 'val.csv'}")
        print(f"✅ Test:  {len(test_df)} frames → {self.class3_dir / 'test.csv'}")
        
        # 통계
        print(f"\n📊 Train Label 분포:")
        print(y_train.value_counts())
        print(f"\n📊 Val Label 분포:")
        print(y_val.value_counts())
        print(f"\n📊 Test Label 분포:")
        print(y_test.value_counts())
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'feature_cols': feature_cols
        }
    
    def save_feature_list(self, feature_cols, dataset_type):
        """Feature 목록 저장"""
        
        if dataset_type == 'binary':
            output_dir = self.binary_dir
        else:
            output_dir = self.class3_dir
        
        with open(output_dir / 'feature_columns.txt', 'w') as f:
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write("=" * 60 + "\n\n")
            for i, col in enumerate(feature_cols, 1):
                f.write(f"{i}. {col}\n")
        
        print(f"📝 Feature 목록 저장: {output_dir / 'feature_columns.txt'}")
    
    def create_all(self):
        """Binary & 3-Class 데이터셋 모두 생성"""
        
        print("=" * 60)
        print("📦 Dataset Creation Pipeline")
        print("=" * 60)
        
        # Binary 데이터셋 생성
        binary_result = self.create_binary_dataset()
        self.save_feature_list(binary_result['feature_cols'], 'binary')
        
        # 3-Class 데이터셋 생성
        class3_result = self.create_3class_dataset()
        self.save_feature_list(class3_result['feature_cols'], '3class')
        
        print("\n" + "=" * 60)
        print("✅ 모든 데이터셋 생성 완료!")
        print("=" * 60)
        print(f"\n📁 Binary Dataset: {self.binary_dir}")
        print(f"   - train.csv")
        print(f"   - val.csv")
        print(f"   - test.csv")
        print(f"   - feature_columns.txt")
        
        print(f"\n📁 3-Class Dataset: {self.class3_dir}")
        print(f"   - train.csv")
        print(f"   - val.csv")
        print(f"   - test.csv")
        print(f"   - feature_columns.txt")
        
        print("\n🚀 다음 단계: Random Forest 모델 학습!")


def main():
    # 경로 설정
    labeled_dir = '/home/gjkong/dev_ws/yolo/myproj/labeled'
    output_dir = '/home/gjkong/dev_ws/yolo/myproj/dataset'
    
    # 데이터셋 생성
    creator = DatasetCreator(labeled_dir, output_dir)
    creator.create_all()


if __name__ == "__main__":
    main()
