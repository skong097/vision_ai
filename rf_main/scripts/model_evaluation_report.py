#!/usr/bin/env python3
"""
=============================================================================
Home Safe Solution - 모델 성능 비교 리포트 생성기
=============================================================================
Random Forest vs ST-GCN (Fine-tuned) 성능 비교

측정 지표:
  1. Accuracy (정확도)
  2. Recall (재현율)
  3. F1-Score
  4. AUC-ROC
  5. Inference Time (추론 속도)
  6. Parameters / FLOPs
  7. Robustness (강건성)
  8. Feature Importance
  9. Confusion Matrix

출력 파일 (3개):
  1. rf_performance_report.png         - Random Forest 성능 리포트
  2. stgcn_performance_report.png      - ST-GCN 성능 리포트
  3. model_comparison_matrix.png       - 두 모델 비교 매트릭스

생성 위치: /home/gjkong/dev_ws/yolo/myproj/scripts/
실행 방법: python model_evaluation_report.py
=============================================================================
작성일: 2026-02-05
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# matplotlib 설정 (한글 폰트 + 스타일)
# ============================================================
import matplotlib
matplotlib.use('Agg')  # GUI 없이 렌더링
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 경로 설정
# ============================================================
# 모델 경로
RF_MODEL_PATH = '/home/gjkong/dev_ws/yolo/myproj/models_integrated/3class/random_forest_model.pkl'
RF_FEATURE_PATH = '/home/gjkong/dev_ws/yolo/myproj/models_integrated/3class/feature_columns.txt'
STGCN_MODEL_PATH = '/home/gjkong/dev_ws/st_gcn/checkpoints_finetuned/best_model_finetuned.pth'

# ST-GCN 데이터 경로
STGCN_DATA_DIR = '/home/gjkong/dev_ws/st_gcn/data/binary'

# 출력 경로 (일시 폴더 자동 생성)
from datetime import datetime
_base_dir = '/home/gjkong/dev_ws/yolo/myproj/scripts/admin/Model_Compare_Report'
_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(_base_dir, _timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 색상 팔레트
# ============================================================
COLORS = {
    'rf_primary': '#2ECC71',       # Green
    'rf_secondary': '#27AE60',
    'rf_bg': '#E8F8F0',
    'stgcn_primary': '#3498DB',    # Blue
    'stgcn_secondary': '#2980B9',
    'stgcn_bg': '#EBF5FB',
    'accent': '#E74C3C',           # Red
    'warning': '#F39C12',          # Orange
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'text': '#34495E',
    'grid': '#BDC3C7',
    'normal': '#2ECC71',
    'fall': '#E74C3C',
}


# ============================================================
# 1. 모델 로드 함수
# ============================================================
def load_random_forest():
    """Random Forest 모델 로드"""
    import joblib
    
    if not os.path.exists(RF_MODEL_PATH):
        print(f"[ERROR] RF 모델 파일 없음: {RF_MODEL_PATH}")
        return None, None
    
    model = joblib.load(RF_MODEL_PATH)
    
    # Feature columns 로드
    feature_columns = []
    if os.path.exists(RF_FEATURE_PATH):
        with open(RF_FEATURE_PATH, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:
                if '. ' in line:
                    feature_name = line.strip().split('. ', 1)[1]
                    feature_columns.append(feature_name)
    
    print(f"[OK] RF 모델 로드 완료 (Features: {len(feature_columns)})")
    return model, feature_columns


def load_stgcn_model():
    """ST-GCN Fine-tuned 모델 로드"""
    import torch
    import torch.nn as nn
    
    if not os.path.exists(STGCN_MODEL_PATH):
        print(f"[ERROR] ST-GCN 모델 파일 없음: {STGCN_MODEL_PATH}")
        return None
    
    # Fine-tuned 모델 구조 정의 (3-partition adjacency matrix)
    class Graph:
        """COCO 17 keypoints 스켈레톤 그래프"""
        def __init__(self):
            self.num_node = 17
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12),
                (11, 12), (5, 11), (6, 12), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (1, 3), (2, 4), (0, 1), (0, 2),
                (1, 2), (0, 5), (0, 6),
            ]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.A = self.get_adjacency_matrix()
        
        def get_adjacency_matrix(self):
            A = np.zeros((3, self.num_node, self.num_node))
            for i, j in self.self_link:
                A[0, i, j] = 1
            for i, j in self.inward:
                A[1, i, j] = 1
            for i, j in self.outward:
                A[2, i, j] = 1
            return A
    
    class STGCNBlock(nn.Module):
        def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
            super().__init__()
            self.gcn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            self.tcn = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
                nn.BatchNorm2d(out_channels),
            )
            self.relu = nn.ReLU(inplace=True)
            
            if not residual:
                self.residual = lambda x: 0
            elif in_channels == out_channels and stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
            # 3-partition adjacency matrix 등록
            self.register_buffer('A', torch.FloatTensor(A))
        
        def forward(self, x):
            res = self.residual(x)
            N, C, T, V = x.size()
            x = x.view(N, C * T, V)
            x = torch.matmul(x, self.A.sum(0))
            x = x.view(N, C, T, V)
            x = self.gcn(x)
            x = self.tcn(x) + res
            return self.relu(x)
    
    class STGCNFineTuned(nn.Module):
        def __init__(self, num_class=2, num_point=17, num_person=1, in_channels=3):
            super().__init__()
            self.graph = Graph()
            A = self.graph.A  # (3, 17, 17)
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
            self.layers = nn.ModuleList([
                STGCNBlock(in_channels, 64, A, residual=False),
                STGCNBlock(64, 64, A),
                STGCNBlock(64, 64, A),
                STGCNBlock(64, 128, A, stride=2),
                STGCNBlock(128, 128, A),
                STGCNBlock(128, 128, A),
                STGCNBlock(128, 256, A, stride=2),
                STGCNBlock(256, 256, A),
                STGCNBlock(256, 256, A),
            ])
            self.fc = nn.Linear(256, num_class)
            self.num_point = num_point
            self.num_person = num_person
            self.in_channels = in_channels
        
        def forward(self, x):
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()
            x = x.view(N, C, T, V * M)
            for layer in self.layers:
                x = layer(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCNFineTuned(num_class=2, num_point=17, num_person=1, in_channels=3)
    
    checkpoint = torch.load(STGCN_MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[OK] ST-GCN Fine-tuned 모델 로드 완료 (Device: {device})")
    return model


def load_stgcn_data():
    """ST-GCN 검증 데이터 로드"""
    val_data = np.load(os.path.join(STGCN_DATA_DIR, 'val_data.npy'))
    val_labels = np.load(os.path.join(STGCN_DATA_DIR, 'val_labels.npy'))
    train_data = np.load(os.path.join(STGCN_DATA_DIR, 'train_data.npy'))
    train_labels = np.load(os.path.join(STGCN_DATA_DIR, 'train_labels.npy'))
    
    print(f"[OK] ST-GCN 데이터 로드: Train={len(train_data)}, Val={len(val_data)}")
    return train_data, train_labels, val_data, val_labels


# ============================================================
# 2. 평가 함수
# ============================================================
def evaluate_random_forest(model, feature_columns, stgcn_val_data, stgcn_val_labels):
    """
    Random Forest 모델 평가
    ST-GCN 검증 데이터에서 keypoints를 추출하여 RF 특성으로 변환 후 평가
    """
    from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                                 confusion_matrix, roc_auc_score, classification_report)
    
    results = {}
    
    # ST-GCN val 데이터에서 프레임별 키포인트 추출
    # stgcn_val_data shape: (N, 3, 60, 17, 1)
    # RF는 프레임 단위이므로, 각 시퀀스의 중간 프레임을 사용
    y_true_list = []
    y_pred_list = []
    y_proba_list = []
    inference_times = []
    
    for i in range(len(stgcn_val_data)):
        seq = stgcn_val_data[i]  # (3, 60, 17, 1)
        label = int(stgcn_val_labels[i])
        
        # 여러 프레임에서 특성 추출하여 투표
        frame_preds = []
        frame_probas = []
        
        # 시퀀스에서 10개 프레임 샘플링
        frame_indices = np.linspace(10, 59, 10, dtype=int)
        
        for fi in frame_indices:
            # 키포인트 추출: (3, 17, 1) → x, y, confidence for 17 joints
            kpts = seq[:, fi, :, 0]  # (3, 17)
            x_coords = kpts[0]  # 17개 x좌표
            y_coords = kpts[1]  # 17개 y좌표
            confs = kpts[2]     # 17개 confidence
            
            # RF에 필요한 특성 생성 (간단한 특성)
            features = extract_rf_features_from_keypoints(x_coords, y_coords, confs)
            
            if features is not None and len(features) == len(feature_columns):
                start_t = time.perf_counter()
                pred = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                elapsed = time.perf_counter() - start_t
                
                inference_times.append(elapsed)
                frame_preds.append(pred)
                frame_probas.append(proba)
        
        if len(frame_preds) > 0:
            # 다수결 투표
            from collections import Counter
            vote = Counter(frame_preds).most_common(1)[0][0]
            avg_proba = np.mean(frame_probas, axis=0)
            
            # RF는 3-class (0=Normal, 1=Falling, 2=Fallen)
            # Binary 변환: Falling+Fallen → Fall(1)
            binary_pred = 0 if vote == 0 else 1
            # Fall 확률 = Falling 확률 + Fallen 확률
            if len(avg_proba) >= 3:
                fall_prob = avg_proba[1] + avg_proba[2]
            else:
                fall_prob = avg_proba[1] if len(avg_proba) > 1 else 0
            
            y_true_list.append(label)
            y_pred_list.append(binary_pred)
            y_proba_list.append(fall_prob)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_proba = np.array(y_proba_list)
    
    # 지표 계산
    results['accuracy'] = accuracy_score(y_true, y_pred) * 100
    results['recall'] = recall_score(y_true, y_pred, zero_division=0) * 100  # Fall class recall
    results['recall_normal'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    results['f1'] = f1_score(y_true, y_pred, zero_division=0) * 100
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    try:
        results['auc_roc'] = roc_auc_score(y_true, y_proba) * 100
    except:
        results['auc_roc'] = 0.0
    
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['y_proba'] = y_proba
    
    # Inference Time
    if len(inference_times) > 0:
        results['inference_time_mean'] = np.mean(inference_times) * 1000  # ms
        results['inference_time_std'] = np.std(inference_times) * 1000
        results['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    else:
        results['inference_time_mean'] = 0
        results['inference_time_std'] = 0
        results['fps'] = 0
    
    # Parameters
    results['parameters'] = count_rf_parameters(model)
    results['flops'] = 'N/A (Tree-based)'
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        results['feature_importance'] = {
            'names': [feature_columns[i] if i < len(feature_columns) else f'F{i}' for i in top_idx],
            'values': importances[top_idx]
        }
    else:
        results['feature_importance'] = {'names': [], 'values': []}
    
    # Robustness (노이즈 추가 후 정확도 변화)
    results['robustness'] = evaluate_rf_robustness(model, feature_columns, 
                                                    stgcn_val_data, stgcn_val_labels)
    
    print(f"[OK] RF 평가 완료: Acc={results['accuracy']:.2f}%, Recall={results['recall']:.2f}%")
    return results


def evaluate_stgcn(model, val_data, val_labels):
    """ST-GCN Fine-tuned 모델 평가"""
    import torch
    from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                                 confusion_matrix, roc_auc_score)
    
    device = next(model.parameters()).device
    results = {}
    
    y_true_list = []
    y_pred_list = []
    y_proba_list = []
    inference_times = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(val_data)):
            x = torch.tensor(val_data[i:i+1], dtype=torch.float32).to(device)
            label = int(val_labels[i])
            
            start_t = time.perf_counter()
            output = model(x)
            elapsed = time.perf_counter() - start_t
            
            proba = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(proba))
            
            y_true_list.append(label)
            y_pred_list.append(pred)
            y_proba_list.append(proba[1])  # Fall 확률
            inference_times.append(elapsed)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_proba = np.array(y_proba_list)
    
    # 지표 계산
    results['accuracy'] = accuracy_score(y_true, y_pred) * 100
    results['recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    results['recall_normal'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    results['f1'] = f1_score(y_true, y_pred, zero_division=0) * 100
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    try:
        results['auc_roc'] = roc_auc_score(y_true, y_proba) * 100
    except:
        results['auc_roc'] = 0.0
    
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['y_proba'] = y_proba
    
    # Inference Time
    results['inference_time_mean'] = np.mean(inference_times) * 1000
    results['inference_time_std'] = np.std(inference_times) * 1000
    results['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    
    # Parameters & FLOPs
    results['parameters'] = count_torch_parameters(model)
    results['flops'] = estimate_stgcn_flops(model)
    
    # Feature Importance (ST-GCN: Joint Importance via gradient)
    results['feature_importance'] = compute_stgcn_joint_importance(model, val_data, val_labels, device)
    
    # Robustness
    results['robustness'] = evaluate_stgcn_robustness(model, val_data, val_labels, device)
    
    print(f"[OK] ST-GCN 평가 완료: Acc={results['accuracy']:.2f}%, Recall={results['recall']:.2f}%")
    return results


# ============================================================
# 3. 헬퍼 함수
# ============================================================
def extract_rf_features_from_keypoints(x_coords, y_coords, confs):
    """키포인트에서 RF 특성 추출 (모니터링 페이지와 동일한 방식)"""
    try:
        features = []
        
        # 기본 좌표 특성 (x, y for 17 joints = 34)
        for i in range(17):
            features.append(x_coords[i])
            features.append(y_coords[i])
        
        # 관절 각도 및 거리 특성
        # Hip center (5,6 평균)
        hip_x = (x_coords[5] + x_coords[6]) / 2
        hip_y = (y_coords[5] + y_coords[6]) / 2
        
        # Shoulder center (5,6)
        shoulder_x = (x_coords[5] + x_coords[6]) / 2
        shoulder_y = (y_coords[5] + y_coords[6]) / 2
        
        # Spine angle
        nose_x, nose_y = x_coords[0], y_coords[0]
        spine_angle = np.arctan2(nose_y - hip_y, nose_x - hip_x) if (nose_x - hip_x) != 0 else 0
        features.append(spine_angle)
        
        # Hip height (normalized)
        features.append(hip_y)
        
        # Body bounding box
        valid_x = x_coords[confs > 0.3]
        valid_y = y_coords[confs > 0.3]
        if len(valid_x) > 0:
            bbox_w = np.max(valid_x) - np.min(valid_x)
            bbox_h = np.max(valid_y) - np.min(valid_y)
            aspect_ratio = bbox_w / max(bbox_h, 1e-6)
        else:
            bbox_w, bbox_h, aspect_ratio = 0, 0, 0
        
        features.extend([bbox_w, bbox_h, aspect_ratio])
        
        # Knee angles
        for side in [(11, 13, 15), (12, 14, 16)]:  # L/R
            hip_idx, knee_idx, ankle_idx = side
            v1 = np.array([x_coords[hip_idx] - x_coords[knee_idx],
                          y_coords[hip_idx] - y_coords[knee_idx]])
            v2 = np.array([x_coords[ankle_idx] - x_coords[knee_idx],
                          y_coords[ankle_idx] - y_coords[knee_idx]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(angle)
        
        # Velocity placeholders (single frame → 0)
        features.extend([0.0] * 4)  # hip_vx, hip_vy, head_vx, head_vy
        
        return features
        
    except Exception as e:
        return None


def count_rf_parameters(model):
    """RF 모델 파라미터 수 추정"""
    total = 0
    if hasattr(model, 'estimators_'):
        for tree in model.estimators_:
            total += tree.tree_.node_count * 3  # threshold + feature + value
    return total


def count_torch_parameters(model):
    """PyTorch 모델 파라미터 수"""
    return sum(p.numel() for p in model.parameters())


def estimate_stgcn_flops(model):
    """ST-GCN FLOPs 추정"""
    total_params = count_torch_parameters(model)
    # 대략적 추정: params * 2 (곱셈 + 덧셈)
    flops = total_params * 2
    if flops > 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops > 1e6:
        return f"{flops/1e6:.2f}M"
    else:
        return f"{flops/1e3:.2f}K"


def evaluate_rf_robustness(model, feature_columns, val_data, val_labels):
    """RF 강건성 평가 (노이즈 레벨별 정확도)"""
    from sklearn.metrics import accuracy_score
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    accuracies = []
    
    for noise_std in noise_levels:
        y_true_list = []
        y_pred_list = []
        
        for i in range(len(val_data)):
            seq = val_data[i].copy()
            label = int(val_labels[i])
            
            # 노이즈 추가
            if noise_std > 0:
                seq[:2] += np.random.normal(0, noise_std, seq[:2].shape)
            
            # 중간 프레임에서 특성 추출
            fi = 30  # 중간 프레임
            kpts = seq[:, fi, :, 0]
            features = extract_rf_features_from_keypoints(kpts[0], kpts[1], kpts[2])
            
            if features is not None and len(features) == len(feature_columns):
                pred = model.predict([features])[0]
                binary_pred = 0 if pred == 0 else 1
                y_true_list.append(label)
                y_pred_list.append(binary_pred)
        
        if len(y_true_list) > 0:
            acc = accuracy_score(y_true_list, y_pred_list) * 100
        else:
            acc = 0
        accuracies.append(acc)
    
    return {'noise_levels': noise_levels, 'accuracies': accuracies}


def evaluate_stgcn_robustness(model, val_data, val_labels, device):
    """ST-GCN 강건성 평가"""
    import torch
    from sklearn.metrics import accuracy_score
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    accuracies = []
    
    model.eval()
    for noise_std in noise_levels:
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for i in range(len(val_data)):
                data = val_data[i:i+1].copy()
                label = int(val_labels[i])
                
                if noise_std > 0:
                    data[:, :2] += np.random.normal(0, noise_std, data[:, :2].shape)
                
                x = torch.tensor(data, dtype=torch.float32).to(device)
                output = model(x)
                pred = int(torch.argmax(output, dim=1).cpu().item())
                
                y_true_list.append(label)
                y_pred_list.append(pred)
        
        acc = accuracy_score(y_true_list, y_pred_list) * 100
        accuracies.append(acc)
    
    return {'noise_levels': noise_levels, 'accuracies': accuracies}


def compute_stgcn_joint_importance(model, val_data, val_labels, device):
    """ST-GCN Joint Importance (Gradient 기반)"""
    import torch
    
    model.eval()
    joint_names = [
        'Nose', 'L-Eye', 'R-Eye', 'L-Ear', 'R-Ear',
        'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow',
        'L-Wrist', 'R-Wrist', 'L-Hip', 'R-Hip',
        'L-Knee', 'R-Knee', 'L-Ankle', 'R-Ankle'
    ]
    
    joint_importance = np.zeros(17)
    count = 0
    
    for i in range(min(len(val_data), 30)):  # 최대 30개 샘플
        x = torch.tensor(val_data[i:i+1], dtype=torch.float32).to(device)
        x.requires_grad = True
        
        output = model(x)
        pred_class = torch.argmax(output, dim=1)
        loss = output[0, pred_class[0]]
        loss.backward()
        
        if x.grad is not None:
            # grad shape: (1, 3, 60, 17, 1)
            grad = x.grad.abs().cpu().numpy()[0]  # (3, 60, 17, 1)
            # 채널, 시간 축 평균 → Joint별 중요도
            joint_grad = grad.mean(axis=(0, 1, 3))  # (17,)
            joint_importance += joint_grad
            count += 1
        
        model.zero_grad()
    
    if count > 0:
        joint_importance /= count
    
    # Top 10
    top_idx = np.argsort(joint_importance)[::-1][:10]
    
    return {
        'names': [joint_names[i] for i in top_idx],
        'values': joint_importance[top_idx]
    }


# ============================================================
# 4. 시각화 - 개별 모델 리포트
# ============================================================
def create_model_report(results, model_name, model_icon, primary_color, bg_color, output_path):
    """개별 모델 성능 리포트 PNG 생성"""
    
    fig = plt.figure(figsize=(20, 24), facecolor='white')
    fig.suptitle('', fontsize=1)
    
    # 전체 그리드
    gs = gridspec.GridSpec(5, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.94, top=0.92, bottom=0.03)
    
    # ====== 헤더 ======
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')
    
    # 배경
    header_bg = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor=primary_color, alpha=0.15,
                                edgecolor=primary_color, linewidth=2)
    ax_header.add_patch(header_bg)
    
    ax_header.text(0.5, 0.7, f"{model_icon}  {model_name} Performance Report",
                   fontsize=22, fontweight='bold', ha='center', va='center',
                   color=COLORS['dark'])
    ax_header.text(0.5, 0.35,
                   f"Accuracy: {results['accuracy']:.2f}%  |  "
                   f"Recall (Fall): {results['recall']:.2f}%  |  "
                   f"F1-Score: {results['f1']:.2f}%  |  "
                   f"AUC-ROC: {results['auc_roc']:.2f}%",
                   fontsize=13, ha='center', va='center', color=COLORS['text'])
    ax_header.text(0.5, 0.1, "Home Safe Solution - Fall Detection System  |  2026-02-05",
                   fontsize=10, ha='center', va='center', color=COLORS['grid'])
    
    # ====== Row 1: Metrics Bars + Confusion Matrix ======
    # 1-1: 주요 지표 바 차트
    ax_metrics = fig.add_subplot(gs[1, :2])
    metrics_names = ['Accuracy', 'Recall\n(Fall)', 'Recall\n(Normal)', 'F1-Score\n(Fall)', 'F1-Score\n(Macro)', 'AUC-ROC']
    metrics_values = [results['accuracy'], results['recall'], results['recall_normal'],
                      results['f1'], results['f1_macro'], results['auc_roc']]
    
    bars = ax_metrics.barh(range(len(metrics_names)), metrics_values,
                           color=[primary_color]*len(metrics_names), alpha=0.8,
                           edgecolor='white', linewidth=1.5, height=0.6)
    
    for bar, val in zip(bars, metrics_values):
        ax_metrics.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}%', va='center', fontsize=11, fontweight='bold',
                       color=COLORS['dark'])
    
    ax_metrics.set_yticks(range(len(metrics_names)))
    ax_metrics.set_yticklabels(metrics_names, fontsize=11)
    ax_metrics.set_xlim(0, 110)
    ax_metrics.set_xlabel('Score (%)', fontsize=11)
    ax_metrics.set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=10)
    ax_metrics.axvline(x=90, color=COLORS['warning'], linestyle='--', alpha=0.5, label='90% threshold')
    ax_metrics.legend(loc='lower right', fontsize=9)
    ax_metrics.invert_yaxis()
    ax_metrics.grid(axis='x', alpha=0.3)
    
    # 1-2: Confusion Matrix
    ax_cm = fig.add_subplot(gs[1, 2])
    cm = results['confusion_matrix']
    
    im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
    
    classes = ['Normal', 'Fall']
    ax_cm.set_xticks(range(len(classes)))
    ax_cm.set_yticks(range(len(classes)))
    ax_cm.set_xticklabels(classes, fontsize=11)
    ax_cm.set_yticklabels(classes, fontsize=11)
    ax_cm.set_xlabel('Predicted', fontsize=11)
    ax_cm.set_ylabel('Actual', fontsize=11)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else COLORS['dark']
            ax_cm.text(j, i, str(cm[i, j]),
                      ha='center', va='center', fontsize=16, fontweight='bold',
                      color=color)
    
    plt.colorbar(im, ax=ax_cm, fraction=0.046)
    
    # ====== Row 2: Inference Time + ROC Curve ======
    # 2-1: Inference Time
    ax_time = fig.add_subplot(gs[2, 0])
    
    time_data = [results['inference_time_mean']]
    time_labels = ['Mean']
    
    bar = ax_time.bar(time_labels, time_data, color=primary_color, alpha=0.8,
                     width=0.4, edgecolor='white')
    ax_time.errorbar(time_labels, time_data,
                    yerr=[results['inference_time_std']], fmt='none',
                    ecolor=COLORS['dark'], capsize=10, linewidth=2)
    
    ax_time.text(0, time_data[0] + results['inference_time_std'] + 0.5,
                f"{time_data[0]:.3f} ms\n({results['fps']:.0f} FPS)",
                ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
    
    ax_time.set_title('Inference Time', fontsize=14, fontweight='bold', pad=10)
    ax_time.set_ylabel('Time (ms)', fontsize=11)
    ax_time.grid(axis='y', alpha=0.3)
    
    # 2-2: ROC Curve
    ax_roc = fig.add_subplot(gs[2, 1])
    
    from sklearn.metrics import roc_curve
    y_true = results['y_true']
    y_proba = results['y_proba']
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax_roc.plot(fpr, tpr, color=primary_color, linewidth=2.5,
                   label=f'AUC = {results["auc_roc"]:.2f}%')
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color=primary_color)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax_roc.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=10)
    ax_roc.set_xlabel('False Positive Rate', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate', fontsize=11)
    ax_roc.legend(loc='lower right', fontsize=11)
    ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    
    # 2-3: Parameters Info
    ax_params = fig.add_subplot(gs[2, 2])
    ax_params.axis('off')
    
    params_bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                boxstyle="round,pad=0.03",
                                facecolor=bg_color, edgecolor=primary_color,
                                linewidth=1.5)
    ax_params.add_patch(params_bg)
    
    param_count = results['parameters']
    if param_count > 1e6:
        param_str = f"{param_count/1e6:.2f}M"
    elif param_count > 1e3:
        param_str = f"{param_count/1e3:.1f}K"
    else:
        param_str = str(param_count)
    
    ax_params.text(0.5, 0.8, 'Model Complexity', fontsize=14, fontweight='bold',
                  ha='center', va='center', color=COLORS['dark'])
    ax_params.text(0.5, 0.6, f'Parameters: {param_str}',
                  fontsize=12, ha='center', va='center', color=COLORS['text'])
    ax_params.text(0.5, 0.45, f'FLOPs: {results["flops"]}',
                  fontsize=12, ha='center', va='center', color=COLORS['text'])
    ax_params.text(0.5, 0.3, f'Inference: {results["inference_time_mean"]:.3f} ms',
                  fontsize=12, ha='center', va='center', color=COLORS['text'])
    ax_params.text(0.5, 0.15, f'FPS: {results["fps"]:.0f}',
                  fontsize=14, fontweight='bold', ha='center', va='center',
                  color=primary_color)
    
    # ====== Row 3: Feature Importance ======
    ax_feat = fig.add_subplot(gs[3, :2])
    
    fi = results['feature_importance']
    if len(fi['names']) > 0:
        y_pos = range(len(fi['names']))
        bars = ax_feat.barh(y_pos, fi['values'],
                           color=primary_color, alpha=0.8,
                           edgecolor='white', linewidth=1)
        
        for bar, val in zip(bars, fi['values']):
            ax_feat.text(bar.get_width() + max(fi['values']) * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=10, color=COLORS['dark'])
        
        ax_feat.set_yticks(y_pos)
        ax_feat.set_yticklabels(fi['names'], fontsize=10)
        ax_feat.invert_yaxis()
    
    ax_feat.set_title('Feature / Joint Importance (Top 10)', fontsize=14, fontweight='bold', pad=10)
    ax_feat.set_xlabel('Importance', fontsize=11)
    ax_feat.grid(axis='x', alpha=0.3)
    
    # ====== Row 3-right: Robustness ======
    ax_robust = fig.add_subplot(gs[3, 2])
    
    rob = results['robustness']
    noise_pct = [f"{n*100:.0f}%" for n in rob['noise_levels']]
    
    ax_robust.plot(noise_pct, rob['accuracies'], 'o-',
                  color=primary_color, linewidth=2.5, markersize=8)
    ax_robust.fill_between(range(len(noise_pct)), rob['accuracies'],
                          alpha=0.15, color=primary_color)
    
    for i, (x, y) in enumerate(zip(range(len(noise_pct)), rob['accuracies'])):
        ax_robust.annotate(f'{y:.1f}%', (x, y),
                          textcoords="offset points", xytext=(0, 10),
                          ha='center', fontsize=9, fontweight='bold')
    
    ax_robust.set_title('Robustness (Noise Test)', fontsize=14, fontweight='bold', pad=10)
    ax_robust.set_xlabel('Noise Level (std)', fontsize=11)
    ax_robust.set_ylabel('Accuracy (%)', fontsize=11)
    ax_robust.grid(alpha=0.3)
    ax_robust.set_ylim(max(0, min(rob['accuracies']) - 15), 105)
    
    # ====== Row 4: Summary ======
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    summary_bg = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS['light'],
                                 edgecolor=COLORS['grid'], linewidth=1)
    ax_summary.add_patch(summary_bg)
    
    # 강건성 점수 계산
    if len(rob['accuracies']) >= 2:
        robustness_score = rob['accuracies'][-1] / max(rob['accuracies'][0], 1) * 100
    else:
        robustness_score = 100
    
    summary_text = (
        f"Summary  |  "
        f"Accuracy: {results['accuracy']:.2f}%  |  "
        f"Fall Recall: {results['recall']:.2f}%  |  "
        f"F1-Score: {results['f1']:.2f}%  |  "
        f"AUC-ROC: {results['auc_roc']:.2f}%  |  "
        f"Speed: {results['fps']:.0f} FPS  |  "
        f"Robustness: {robustness_score:.1f}%"
    )
    ax_summary.text(0.5, 0.5, summary_text,
                   fontsize=12, ha='center', va='center',
                   fontweight='bold', color=COLORS['dark'])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SAVED] {output_path}")


# ============================================================
# 5. 시각화 - 모델 비교 매트릭스
# ============================================================
def create_comparison_matrix(rf_results, stgcn_results, output_path):
    """두 모델 비교 매트릭스 PNG 생성"""
    
    fig = plt.figure(figsize=(22, 26), facecolor='white')
    gs = gridspec.GridSpec(6, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.94, top=0.93, bottom=0.03)
    
    # ====== 헤더 ======
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')
    
    header_bg = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['dark'], alpha=0.9,
                                edgecolor=COLORS['dark'], linewidth=2)
    ax_header.add_patch(header_bg)
    
    ax_header.text(0.5, 0.65,
                   "Model Comparison Matrix",
                   fontsize=24, fontweight='bold', ha='center', va='center',
                   color='white')
    ax_header.text(0.5, 0.3,
                   "Random Forest  vs  ST-GCN (Fine-tuned)  |  Home Safe Solution  |  2026-02-05",
                   fontsize=13, ha='center', va='center', color=COLORS['light'])
    
    # ====== Row 1: 주요 지표 비교 (레이더 차트 + 바 차트) ======
    # 1-1: 레이더 차트
    ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
    
    categories = ['Accuracy', 'Recall\n(Fall)', 'Recall\n(Normal)', 'F1-Score', 'AUC-ROC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    rf_vals = [rf_results['accuracy'], rf_results['recall'], rf_results['recall_normal'],
               rf_results['f1'], rf_results['auc_roc']]
    stgcn_vals = [stgcn_results['accuracy'], stgcn_results['recall'], stgcn_results['recall_normal'],
                  stgcn_results['f1'], stgcn_results['auc_roc']]
    
    rf_vals += rf_vals[:1]
    stgcn_vals += stgcn_vals[:1]
    
    ax_radar.plot(angles, rf_vals, 'o-', color=COLORS['rf_primary'],
                 linewidth=2.5, label='Random Forest', markersize=6)
    ax_radar.fill(angles, rf_vals, alpha=0.15, color=COLORS['rf_primary'])
    
    ax_radar.plot(angles, stgcn_vals, 's-', color=COLORS['stgcn_primary'],
                 linewidth=2.5, label='ST-GCN', markersize=6)
    ax_radar.fill(angles, stgcn_vals, alpha=0.15, color=COLORS['stgcn_primary'])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=9)
    ax_radar.set_ylim(0, 105)
    ax_radar.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # 1-2: 비교 바 차트
    ax_bar = fig.add_subplot(gs[1, 1:])
    
    metric_names = ['Accuracy', 'Recall (Fall)', 'Recall (Normal)', 'F1-Score (Fall)', 'F1-Score (Macro)', 'AUC-ROC']
    rf_metrics = [rf_results['accuracy'], rf_results['recall'], rf_results['recall_normal'],
                  rf_results['f1'], rf_results['f1_macro'], rf_results['auc_roc']]
    stgcn_metrics = [stgcn_results['accuracy'], stgcn_results['recall'], stgcn_results['recall_normal'],
                     stgcn_results['f1'], stgcn_results['f1_macro'], stgcn_results['auc_roc']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, rf_metrics, width, label='Random Forest',
                       color=COLORS['rf_primary'], alpha=0.85, edgecolor='white')
    bars2 = ax_bar.bar(x + width/2, stgcn_metrics, width, label='ST-GCN (Fine-tuned)',
                       color=COLORS['stgcn_primary'], alpha=0.85, edgecolor='white')
    
    for bar, val in zip(bars1, rf_metrics):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', fontsize=9, fontweight='bold',
                   color=COLORS['rf_secondary'])
    for bar, val in zip(bars2, stgcn_metrics):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', fontsize=9, fontweight='bold',
                   color=COLORS['stgcn_secondary'])
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=10, rotation=15)
    ax_bar.set_ylabel('Score (%)', fontsize=11)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold', pad=10)
    ax_bar.legend(fontsize=11)
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.axhline(y=90, color=COLORS['warning'], linestyle='--', alpha=0.4)
    
    # ====== Row 2: Confusion Matrix 비교 ======
    # 2-1: RF Confusion Matrix
    ax_cm_rf = fig.add_subplot(gs[2, 0])
    cm_rf = rf_results['confusion_matrix']
    im1 = ax_cm_rf.imshow(cm_rf, interpolation='nearest', cmap='Greens', aspect='auto')
    ax_cm_rf.set_title('RF Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    classes = ['Normal', 'Fall']
    ax_cm_rf.set_xticks(range(2))
    ax_cm_rf.set_yticks(range(2))
    ax_cm_rf.set_xticklabels(classes, fontsize=11)
    ax_cm_rf.set_yticklabels(classes, fontsize=11)
    ax_cm_rf.set_xlabel('Predicted', fontsize=10)
    ax_cm_rf.set_ylabel('Actual', fontsize=10)
    for i in range(cm_rf.shape[0]):
        for j in range(cm_rf.shape[1]):
            color = 'white' if cm_rf[i,j] > cm_rf.max()/2 else COLORS['dark']
            ax_cm_rf.text(j, i, str(cm_rf[i,j]), ha='center', va='center',
                         fontsize=18, fontweight='bold', color=color)
    
    # 2-2: ST-GCN Confusion Matrix
    ax_cm_stgcn = fig.add_subplot(gs[2, 1])
    cm_stgcn = stgcn_results['confusion_matrix']
    im2 = ax_cm_stgcn.imshow(cm_stgcn, interpolation='nearest', cmap='Blues', aspect='auto')
    ax_cm_stgcn.set_title('ST-GCN Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    ax_cm_stgcn.set_xticks(range(2))
    ax_cm_stgcn.set_yticks(range(2))
    ax_cm_stgcn.set_xticklabels(classes, fontsize=11)
    ax_cm_stgcn.set_yticklabels(classes, fontsize=11)
    ax_cm_stgcn.set_xlabel('Predicted', fontsize=10)
    ax_cm_stgcn.set_ylabel('Actual', fontsize=10)
    for i in range(cm_stgcn.shape[0]):
        for j in range(cm_stgcn.shape[1]):
            color = 'white' if cm_stgcn[i,j] > cm_stgcn.max()/2 else COLORS['dark']
            ax_cm_stgcn.text(j, i, str(cm_stgcn[i,j]), ha='center', va='center',
                            fontsize=18, fontweight='bold', color=color)
    
    # 2-3: ROC Curve 비교
    ax_roc = fig.add_subplot(gs[2, 2])
    from sklearn.metrics import roc_curve
    
    for res, name, color in [(rf_results, 'RF', COLORS['rf_primary']),
                              (stgcn_results, 'ST-GCN', COLORS['stgcn_primary'])]:
        if len(np.unique(res['y_true'])) > 1:
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
            ax_roc.plot(fpr, tpr, color=color, linewidth=2.5,
                       label=f'{name} (AUC={res["auc_roc"]:.1f}%)')
            ax_roc.fill_between(fpr, tpr, alpha=0.1, color=color)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold', pad=10)
    ax_roc.set_xlabel('FPR', fontsize=11)
    ax_roc.set_ylabel('TPR', fontsize=11)
    ax_roc.legend(fontsize=10)
    ax_roc.grid(alpha=0.3)
    
    # ====== Row 3: Inference Time + Robustness ======
    # 3-1: Inference Time 비교
    ax_time = fig.add_subplot(gs[3, 0])
    
    models = ['RF', 'ST-GCN']
    times = [rf_results['inference_time_mean'], stgcn_results['inference_time_mean']]
    fps_vals = [rf_results['fps'], stgcn_results['fps']]
    colors = [COLORS['rf_primary'], COLORS['stgcn_primary']]
    
    bars = ax_time.bar(models, times, color=colors, alpha=0.85, width=0.5, edgecolor='white')
    ax_time.errorbar(models,
                    times,
                    yerr=[rf_results['inference_time_std'], stgcn_results['inference_time_std']],
                    fmt='none', ecolor=COLORS['dark'], capsize=10, linewidth=2)
    
    for bar, t, f in zip(bars, times, fps_vals):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{t:.3f} ms\n({f:.0f} FPS)',
                    ha='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
    
    ax_time.set_title('Inference Time', fontsize=13, fontweight='bold', pad=10)
    ax_time.set_ylabel('Time (ms)', fontsize=11)
    ax_time.grid(axis='y', alpha=0.3)
    
    # 3-2: Robustness 비교
    ax_robust = fig.add_subplot(gs[3, 1:])
    
    rf_rob = rf_results['robustness']
    stgcn_rob = stgcn_results['robustness']
    
    noise_labels = [f"{n*100:.0f}%" for n in rf_rob['noise_levels']]
    
    ax_robust.plot(noise_labels, rf_rob['accuracies'], 'o-',
                  color=COLORS['rf_primary'], linewidth=2.5, markersize=8,
                  label='Random Forest')
    ax_robust.fill_between(range(len(noise_labels)), rf_rob['accuracies'],
                          alpha=0.1, color=COLORS['rf_primary'])
    
    ax_robust.plot(noise_labels, stgcn_rob['accuracies'], 's-',
                  color=COLORS['stgcn_primary'], linewidth=2.5, markersize=8,
                  label='ST-GCN (Fine-tuned)')
    ax_robust.fill_between(range(len(noise_labels)), stgcn_rob['accuracies'],
                          alpha=0.1, color=COLORS['stgcn_primary'])
    
    for i, (rf_a, st_a) in enumerate(zip(rf_rob['accuracies'], stgcn_rob['accuracies'])):
        ax_robust.annotate(f'{rf_a:.1f}', (i, rf_a),
                          textcoords="offset points", xytext=(0, 10),
                          ha='center', fontsize=9, color=COLORS['rf_secondary'])
        ax_robust.annotate(f'{st_a:.1f}', (i, st_a),
                          textcoords="offset points", xytext=(0, -15),
                          ha='center', fontsize=9, color=COLORS['stgcn_secondary'])
    
    ax_robust.set_title('Robustness: Accuracy vs Noise Level', fontsize=13, fontweight='bold', pad=10)
    ax_robust.set_xlabel('Gaussian Noise Std (%)', fontsize=11)
    ax_robust.set_ylabel('Accuracy (%)', fontsize=11)
    ax_robust.legend(fontsize=11)
    ax_robust.grid(alpha=0.3)
    all_accs = rf_rob['accuracies'] + stgcn_rob['accuracies']
    ax_robust.set_ylim(max(0, min(all_accs) - 15), 105)
    
    # ====== Row 4: Feature Importance 비교 ======
    # 4-1: RF Feature Importance
    ax_fi_rf = fig.add_subplot(gs[4, 0:1])
    fi_rf = rf_results['feature_importance']
    if len(fi_rf['names']) > 0:
        y_pos = range(min(8, len(fi_rf['names'])))
        ax_fi_rf.barh(y_pos, fi_rf['values'][:8],
                     color=COLORS['rf_primary'], alpha=0.8, edgecolor='white')
        ax_fi_rf.set_yticks(y_pos)
        ax_fi_rf.set_yticklabels(fi_rf['names'][:8], fontsize=9)
        ax_fi_rf.invert_yaxis()
    ax_fi_rf.set_title('RF Feature Importance', fontsize=13, fontweight='bold', pad=10)
    ax_fi_rf.set_xlabel('Importance', fontsize=10)
    ax_fi_rf.grid(axis='x', alpha=0.3)
    
    # 4-2: ST-GCN Joint Importance
    ax_fi_stgcn = fig.add_subplot(gs[4, 1:2])
    fi_stgcn = stgcn_results['feature_importance']
    if len(fi_stgcn['names']) > 0:
        y_pos = range(min(8, len(fi_stgcn['names'])))
        ax_fi_stgcn.barh(y_pos, fi_stgcn['values'][:8],
                        color=COLORS['stgcn_primary'], alpha=0.8, edgecolor='white')
        ax_fi_stgcn.set_yticks(y_pos)
        ax_fi_stgcn.set_yticklabels(fi_stgcn['names'][:8], fontsize=9)
        ax_fi_stgcn.invert_yaxis()
    ax_fi_stgcn.set_title('ST-GCN Joint Importance', fontsize=13, fontweight='bold', pad=10)
    ax_fi_stgcn.set_xlabel('Gradient Importance', fontsize=10)
    ax_fi_stgcn.grid(axis='x', alpha=0.3)
    
    # 4-3: Model Specs 비교 테이블
    ax_specs = fig.add_subplot(gs[4, 2])
    ax_specs.axis('off')
    
    specs_bg = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                               boxstyle="round,pad=0.02",
                               facecolor=COLORS['light'],
                               edgecolor=COLORS['grid'], linewidth=1)
    ax_specs.add_patch(specs_bg)
    
    rf_params = rf_results['parameters']
    stgcn_params = stgcn_results['parameters']
    
    def fmt_params(p):
        if p > 1e6: return f"{p/1e6:.2f}M"
        elif p > 1e3: return f"{p/1e3:.1f}K"
        return str(p)
    
    specs_text = (
        f"Model Specifications\n"
        f"{'─' * 30}\n"
        f"{'':8s}  {'RF':>10s}  {'ST-GCN':>10s}\n"
        f"{'─' * 30}\n"
        f"{'Params':8s}  {fmt_params(rf_params):>10s}  {fmt_params(stgcn_params):>10s}\n"
        f"{'FLOPs':8s}  {'N/A':>10s}  {stgcn_results['flops']:>10s}\n"
        f"{'Speed':8s}  {rf_results['fps']:>8.0f}fps  {stgcn_results['fps']:>8.0f}fps\n"
        f"{'Input':8s}  {'Frame':>10s}  {'60-Seq':>10s}\n"
        f"{'Classes':8s}  {'3':>10s}  {'2':>10s}\n"
        f"{'─' * 30}\n"
    )
    
    ax_specs.text(0.5, 0.5, specs_text, fontsize=10, ha='center', va='center',
                 fontfamily='monospace', color=COLORS['dark'])
    
    # ====== Row 5: 종합 비교 요약 ======
    ax_summary = fig.add_subplot(gs[5, :])
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    
    # RF 요약
    rf_box = FancyBboxPatch((0.02, 0.1), 0.46, 0.85,
                             boxstyle="round,pad=0.02",
                             facecolor=COLORS['rf_bg'],
                             edgecolor=COLORS['rf_primary'], linewidth=2)
    ax_summary.add_patch(rf_box)
    
    ax_summary.text(0.25, 0.85, "Random Forest", fontsize=15, fontweight='bold',
                   ha='center', va='center', color=COLORS['rf_secondary'])
    ax_summary.text(0.25, 0.65, f"Accuracy: {rf_results['accuracy']:.2f}%",
                   fontsize=12, ha='center', va='center', color=COLORS['dark'])
    ax_summary.text(0.25, 0.50, f"Recall (Fall): {rf_results['recall']:.2f}%  |  F1: {rf_results['f1']:.2f}%",
                   fontsize=11, ha='center', va='center', color=COLORS['text'])
    ax_summary.text(0.25, 0.35, f"Speed: {rf_results['fps']:.0f} FPS  |  AUC: {rf_results['auc_roc']:.1f}%",
                   fontsize=11, ha='center', va='center', color=COLORS['text'])
    
    # 장점
    rf_strength = "Fast inference, Frame-level detection"
    ax_summary.text(0.25, 0.18, f"Strength: {rf_strength}",
                   fontsize=9, ha='center', va='center', color=COLORS['rf_secondary'],
                   style='italic')
    
    # ST-GCN 요약
    stgcn_box = FancyBboxPatch((0.52, 0.1), 0.46, 0.85,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['stgcn_bg'],
                                edgecolor=COLORS['stgcn_primary'], linewidth=2)
    ax_summary.add_patch(stgcn_box)
    
    ax_summary.text(0.75, 0.85, "ST-GCN (Fine-tuned)", fontsize=15, fontweight='bold',
                   ha='center', va='center', color=COLORS['stgcn_secondary'])
    ax_summary.text(0.75, 0.65, f"Accuracy: {stgcn_results['accuracy']:.2f}%",
                   fontsize=12, ha='center', va='center', color=COLORS['dark'])
    ax_summary.text(0.75, 0.50, f"Recall (Fall): {stgcn_results['recall']:.2f}%  |  F1: {stgcn_results['f1']:.2f}%",
                   fontsize=11, ha='center', va='center', color=COLORS['text'])
    ax_summary.text(0.75, 0.35, f"Speed: {stgcn_results['fps']:.0f} FPS  |  AUC: {stgcn_results['auc_roc']:.1f}%",
                   fontsize=11, ha='center', va='center', color=COLORS['text'])
    
    stgcn_strength = "Temporal pattern analysis, Graph-based"
    ax_summary.text(0.75, 0.18, f"Strength: {stgcn_strength}",
                   fontsize=9, ha='center', va='center', color=COLORS['stgcn_secondary'],
                   style='italic')
    
    # VS
    ax_summary.text(0.5, 0.55, "VS", fontsize=20, fontweight='bold',
                   ha='center', va='center', color=COLORS['accent'])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SAVED] {output_path}")


# ============================================================
# 6. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("  Home Safe Solution - Model Evaluation Report Generator")
    print("=" * 60)
    print()
    
    # 1. 모델 및 데이터 로드
    print("[Step 1] Loading models and data...")
    print("-" * 40)
    
    rf_model, rf_features = load_random_forest()
    stgcn_model = load_stgcn_model()
    train_data, train_labels, val_data, val_labels = load_stgcn_data()
    
    if rf_model is None or stgcn_model is None:
        print("[ERROR] 모델 로드 실패. 경로를 확인하세요.")
        sys.exit(1)
    
    print()
    
    # 2. 모델 평가
    print("[Step 2] Evaluating models...")
    print("-" * 40)
    
    rf_results = evaluate_random_forest(rf_model, rf_features, val_data, val_labels)
    stgcn_results = evaluate_stgcn(stgcn_model, val_data, val_labels)
    
    print()
    
    # 3. 리포트 생성
    print("[Step 3] Generating reports...")
    print("-" * 40)
    
    # 3-1: RF 리포트
    rf_report_path = os.path.join(OUTPUT_DIR, 'rf_performance_report.png')
    create_model_report(
        rf_results,
        model_name="Random Forest",
        model_icon="🌲",
        primary_color=COLORS['rf_primary'],
        bg_color=COLORS['rf_bg'],
        output_path=rf_report_path
    )
    
    # 3-2: ST-GCN 리포트
    stgcn_report_path = os.path.join(OUTPUT_DIR, 'stgcn_performance_report.png')
    create_model_report(
        stgcn_results,
        model_name="ST-GCN (Fine-tuned)",
        model_icon="🚀",
        primary_color=COLORS['stgcn_primary'],
        bg_color=COLORS['stgcn_bg'],
        output_path=stgcn_report_path
    )
    
    # 3-3: 비교 매트릭스
    comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison_matrix.png')
    create_comparison_matrix(rf_results, stgcn_results, comparison_path)
    
    print()
    print("=" * 60)
    print("  Report Generation Complete!")
    print("=" * 60)
    print()
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  1. rf_performance_report.png")
    print(f"  2. stgcn_performance_report.png")
    print(f"  3. model_comparison_matrix.png")
    print()
    print("  Quick Summary:")
    print(f"  {'Metric':<20s} {'RF':>10s} {'ST-GCN':>10s} {'Winner':>10s}")
    print(f"  {'-'*50}")
    
    comparisons = [
        ('Accuracy', rf_results['accuracy'], stgcn_results['accuracy']),
        ('Recall (Fall)', rf_results['recall'], stgcn_results['recall']),
        ('F1-Score', rf_results['f1'], stgcn_results['f1']),
        ('AUC-ROC', rf_results['auc_roc'], stgcn_results['auc_roc']),
        ('FPS', rf_results['fps'], stgcn_results['fps']),
    ]
    
    for name, rf_val, stgcn_val in comparisons:
        winner = 'RF' if rf_val >= stgcn_val else 'ST-GCN'
        print(f"  {name:<20s} {rf_val:>9.2f}% {stgcn_val:>9.2f}% {winner:>10s}")
    
    print()


if __name__ == '__main__':
    main()
