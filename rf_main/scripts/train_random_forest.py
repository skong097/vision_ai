"""
Random Forest 모델 학습 및 평가 (Binary & 3-Class)
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class RandomForestTrainer:
    """Random Forest 학습 및 평가"""
    
    def __init__(self, dataset_dir, model_dir):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_cols = None
    
    def load_data(self, dataset_type='binary'):
        """데이터 로드"""
        
        if dataset_type == 'binary':
            data_dir = self.dataset_dir / 'binary'
            label_col = 'label_binary'
        else:
            data_dir = self.dataset_dir / '3class'
            label_col = 'label_3class'
        
        train_df = pd.read_csv(data_dir / 'train.csv')
        val_df = pd.read_csv(data_dir / 'val.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        
        # Feature 컬럼
        self.feature_cols = [col for col in train_df.columns if col != label_col]
        
        # X, y 분리
        X_train = train_df[self.feature_cols]
        y_train = train_df[label_col]
        
        X_val = val_df[self.feature_cols]
        y_val = val_df[label_col]
        
        X_test = test_df[self.feature_cols]
        y_test = test_df[label_col]
        
        # NaN 처리
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_model(self, X_train, y_train, n_estimators=100, max_depth=20, random_state=42):
        """Random Forest 학습"""
        
        print(f"🌲 Random Forest 학습 중...")
        print(f"   - Trees: {n_estimators}")
        print(f"   - Max Depth: {max_depth}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Samples: {X_train.shape[0]}")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight={0: 1.0, 1: 2.5, 2: 1.0},   # 수정
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        print("✅ 학습 완료!")
        
        return self.model
    
    def evaluate_model(self, X, y, dataset_name='Test'):
        """모델 평가"""
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # 기본 메트릭
        accuracy = accuracy_score(y, y_pred)
        
        # Binary vs Multi-class에 따라 다른 평균 방식 사용
        n_classes = len(np.unique(y))
        avg_method = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)
        
        print(f"\n📊 {dataset_name} Set 성능:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(y, y_pred, zero_division=0))
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
        """Confusion Matrix 시각화"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion Matrix 저장: {save_path}")
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Feature Importance 시각화"""
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Top N features
        top_indices = indices[:top_n]
        top_features = [self.feature_cols[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Feature Importance 저장: {save_path}")
        
        plt.close()
        
        # Feature Importance 리스트 출력
        print(f"\n🔝 Top {top_n} Important Features:")
        for i, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
            print(f"   {i}. {feat}: {imp:.4f}")
    
    def plot_roc_curve_binary(self, y_true, y_pred_proba, save_path):
        """ROC Curve (Binary Classification용)"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Binary Classification', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC Curve 저장: {save_path}")
        print(f"   AUC: {auc:.4f}")
    
    def save_model(self, model_path):
        """모델 저장"""
        joblib.dump(self.model, model_path)
        print(f"💾 모델 저장: {model_path}")
    
    def train_and_evaluate(self, dataset_type='binary'):
        """전체 파이프라인 실행"""
        
        print("=" * 60)
        print(f"🚀 Random Forest Training - {dataset_type.upper()}")
        print("=" * 60)
        
        # 데이터 로드
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(dataset_type)
        
        # 모델 학습
        self.train_model(X_train, y_train)
        
        # 평가
        print("\n" + "=" * 60)
        print("📊 모델 평가")
        print("=" * 60)
        
        train_results = self.evaluate_model(X_train, y_train, 'Train')
        val_results = self.evaluate_model(X_val, y_val, 'Validation')
        test_results = self.evaluate_model(X_test, y_test, 'Test')
        
        # 시각화
        viz_dir = self.model_dir / dataset_type / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        if dataset_type == 'binary':
            class_names = ['Normal', 'Fall']
        else:
            class_names = ['Normal', 'Falling', 'Fallen']
        
        self.plot_confusion_matrix(
            y_test, test_results['y_pred'], class_names,
            viz_dir / 'confusion_matrix.png'
        )
        
        # Feature Importance
        self.plot_feature_importance(
            top_n=20,
            save_path=viz_dir / 'feature_importance.png'
        )
        
        # ROC Curve (Binary만)
        if dataset_type == 'binary':
            self.plot_roc_curve_binary(
                y_test, test_results['y_pred_proba'],
                viz_dir / 'roc_curve.png'
            )
        
        # 모델 저장
        model_path = self.model_dir / dataset_type / 'random_forest_model.pkl'
        self.save_model(model_path)
        
        # Feature 목록 저장
        feature_path = self.model_dir / dataset_type / 'feature_columns.txt'
        with open(feature_path, 'w') as f:
            f.write(f"Total Features: {len(self.feature_cols)}\n")
            f.write("=" * 60 + "\n\n")
            for i, col in enumerate(self.feature_cols, 1):
                f.write(f"{i}. {col}\n")
        
        print("\n" + "=" * 60)
        print(f"✅ {dataset_type.upper()} 모델 학습 완료!")
        print(f"📁 저장 경로: {self.model_dir / dataset_type}")
        print("=" * 60)
        
        return {
            'train': train_results,
            'val': val_results,
            'test': test_results
        }


def main():
    # 경로 설정
    dataset_dir = '/home/gjkong/dev_ws/yolo/myproj/dataset'
    model_dir = '/home/gjkong/dev_ws/yolo/myproj/models'
    
    trainer = RandomForestTrainer(dataset_dir, model_dir)
    
    # Binary 모델 학습
    print("\n" + "🔹" * 30)
    print("BINARY CLASSIFICATION")
    print("🔹" * 30 + "\n")
    binary_results = trainer.train_and_evaluate('binary')
    
    # 3-Class 모델 학습
    print("\n" + "🔹" * 30)
    print("3-CLASS CLASSIFICATION")
    print("🔹" * 30 + "\n")
    class3_results = trainer.train_and_evaluate('3class')
    
    # 최종 비교
    print("\n" + "=" * 60)
    print("📊 최종 성능 비교")
    print("=" * 60)
    
    print(f"\n🔹 Binary Classification (Test Set):")
    print(f"   Accuracy:  {binary_results['test']['accuracy']:.4f}")
    print(f"   Precision: {binary_results['test']['precision']:.4f}")
    print(f"   Recall:    {binary_results['test']['recall']:.4f}")
    print(f"   F1-Score:  {binary_results['test']['f1']:.4f}")
    
    print(f"\n🔹 3-Class Classification (Test Set):")
    print(f"   Accuracy:  {class3_results['test']['accuracy']:.4f}")
    print(f"   Precision: {class3_results['test']['precision']:.4f}")
    print(f"   Recall:    {class3_results['test']['recall']:.4f}")
    print(f"   F1-Score:  {class3_results['test']['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 모든 모델 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()