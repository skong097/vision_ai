#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - 낙상 감지 모델 성능 비교 분석
============================================================
각 모델의 고유 테스트 파이프라인을 사용하여 공정하게 비교합니다.

  1) Random Forest  : feature_columns 기반 프레임 단위 추론
  2) ST-GCN Original: 60프레임 시퀀스 (st_gcn_networks 구조)
  3) ST-GCN Fine-tuned: 60프레임 시퀀스 (layers + data_bn 구조)

출력:
  ~/dev_ws/yolo/myproj/scripts/admin/Model_Compare_Report/YYYYMMDD_HHMMSS/
    ├── MODEL_COMPARISON_REPORT.md
    ├── dashboard_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── inference_time.png
    └── model_size.png

사용법:
  python compare_models.py
============================================================
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

# ── 시각화 ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── 평가 지표 ──
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
)


# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path("/home/gjkong/dev_ws")
ST_GCN_DIR = BASE_DIR / "st_gcn"
GUI_DIR = BASE_DIR / "yolo/myproj/gui"
MODELS_DIR = BASE_DIR / "yolo/myproj/models"

# 결과 저장 (일시 폴더)
REPORT_BASE_DIR = BASE_DIR / "yolo/myproj/scripts/admin/Model_Compare_Report"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = REPORT_BASE_DIR / TIMESTAMP

CLASS_NAMES = ["Normal", "Fall"]
INFERENCE_REPEAT = 50  # 추론 속도 측정 반복 횟수

# ── 시각화 설정 ──
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
COLORS = ["#22c55e", "#3b82f6", "#ef4444"]

# matplotlib에서 이모지 렌더링 불가 → 차트용 텍스트 아이콘
CHART_ICONS = {"🌲": "[RF]", "📊": "[Orig]", "🚀": "[FT]"}


def chart_label(r):
    """차트용 라벨 (이모지 → 텍스트)"""
    icon = CHART_ICONS.get(r["icon"], r["icon"])
    return f"{icon} {r['short']}"


# ============================================================
# 유틸리티
# ============================================================

def get_model_size_mb(path):
    p = Path(path)
    return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0


def count_parameters(model):
    if isinstance(model, nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    return 0, 0


def format_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ============================================================
# 모델 파이프라인 추상 클래스
# ============================================================

class ModelPipeline(ABC):
    """각 모델의 고유 테스트 파이프라인 인터페이스"""

    def __init__(self, name, short, icon, model_path):
        self.name = name
        self.short = short
        self.icon = icon
        self.model_path = Path(model_path)
        self.model = None

        # 평가 결과 (evaluate() 호출 후 채워짐)
        self.y_true = None
        self.y_pred = None
        self.y_prob = None
        self.avg_time_ms = 0.0
        self.n_samples = 0
        self.data_description = ""

        # 모델 정보
        self.model_size_mb = get_model_size_mb(model_path)
        self.total_params = "N/A"
        self.trainable_params = "N/A"

    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass

    @abstractmethod
    def load_test_data(self):
        """모델 고유 테스트 데이터 로드 → (X, y_true)"""
        pass

    @abstractmethod
    def predict(self, X):
        """추론 → (y_pred, y_prob)"""
        pass

    def measure_speed(self, X):
        """추론 속도 측정 (INFERENCE_REPEAT회 반복)"""
        times = []
        for _ in range(INFERENCE_REPEAT):
            start = time.perf_counter()
            self.predict(X)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        # 전체 배치 시간 / 샘플 수 → ms
        self.avg_time_ms = (np.mean(times) / self.n_samples) * 1000

    def evaluate(self):
        """전체 평가 파이프라인 실행"""
        print(f"\n{'─' * 60}")
        print(f"  {self.icon} {self.name}")
        print(f"{'─' * 60}")

        # 1. 모델 로드
        if not self.model_path.exists():
            print(f"  ⚠ 모델 파일 없음: {self.model_path}")
            return False
        self.load_model()
        print(f"  모델 로드: {self.model_path.name} ({self.model_size_mb:.2f} MB)")

        # 2. 테스트 데이터 로드
        X, self.y_true = self.load_test_data()
        self.n_samples = len(self.y_true)
        n_normal = np.sum(self.y_true == 0)
        n_fall = np.sum(self.y_true == 1)
        print(f"  테스트 데이터: {self.n_samples} samples (Normal: {n_normal}, Fall: {n_fall})")
        print(f"  데이터 소스: {self.data_description}")

        # 3. 추론
        self.y_pred, self.y_prob = self.predict(X)

        # 4. 속도 측정
        self.measure_speed(X)

        # 5. 결과 요약
        acc = accuracy_score(self.y_true, self.y_pred) * 100
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        print(f"  Accuracy: {acc:.2f}%  |  F1: {f1:.4f}  |  Speed: {self.avg_time_ms:.2f} ms/sample")
        print(f"  Params: {self.total_params}")

        return True

    @property
    def accuracy(self):
        if self.y_true is not None and self.y_pred is not None:
            return accuracy_score(self.y_true, self.y_pred) * 100
        return 0.0

    def to_dict(self):
        """시각화/보고서용 딕셔너리"""
        return {
            "name": self.name,
            "short": self.short,
            "icon": self.icon,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "y_prob": self.y_prob,
            "accuracy": self.accuracy,
            "avg_time_ms": self.avg_time_ms,
            "model_size_mb": self.model_size_mb,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "model_path": str(self.model_path),
            "n_samples": self.n_samples,
            "data_description": self.data_description,
        }


# ============================================================
# 1. Random Forest 파이프라인
# ============================================================

class RandomForestPipeline(ModelPipeline):
    """
    RF 고유 파이프라인:
    - feature_columns.txt 기반 특징 추출
    - 프레임 단위 예측 → 시퀀스에 대해서는 다수결 투표
    - 자체 테스트 데이터 사용 (RF 학습 시 분리한 test set)
    """

    def __init__(self):
        super().__init__(
            name="Random Forest",
            short="RF",
            icon="🌲",
            model_path=MODELS_DIR / "binary" / "random_forest_model.pkl",
        )
        self.feature_cols_path = MODELS_DIR / "binary" / "feature_columns.txt"
        self.feature_names = []

        # RF 전용 binary dataset
        self.dataset_dir = BASE_DIR / "yolo/myproj/dataset/binary"
        self.test_csv_path = self.dataset_dir / "test.csv"
        self.feature_cols_dataset_path = self.dataset_dir / "feature_columns.txt"

    def load_model(self):
        self.model = joblib.load(str(self.model_path))

        # feature columns
        if self.feature_cols_path.exists():
            with open(str(self.feature_cols_path), "r") as f:
                self.feature_names = [l.strip() for l in f if l.strip()]
            print(f"  Feature columns: {len(self.feature_names)}개")

        # RF 모델 정보
        n_estimators = getattr(self.model, "n_estimators", "?")
        n_features = getattr(self.model, "n_features_in_", len(self.feature_names))
        self.total_params = f"{n_estimators} trees, {n_features} features"
        self.trainable_params = "N/A (ML)"

    def load_test_data(self):
        """
        RF 고유 테스트 데이터: ~/dev_ws/yolo/myproj/dataset/binary/test.csv
        CSV 형태 → feature columns + label column
        """
        import pandas as pd

        if not self.test_csv_path.exists():
            raise FileNotFoundError(f"RF 테스트 데이터 없음: {self.test_csv_path}")

        df = pd.read_csv(str(self.test_csv_path))
        print(f"  test.csv 로드: {df.shape[0]} rows × {df.shape[1]} cols")

        # 라벨 컬럼 탐색 (일반적인 이름들)
        label_col = None
        for candidate in ["label", "Label", "class", "Class", "target", "Target", "y"]:
            if candidate in df.columns:
                label_col = candidate
                break

        # 마지막 컬럼이 라벨일 가능성
        if label_col is None:
            label_col = df.columns[-1]
            print(f"  ⚠ 라벨 컬럼 자동 추정: '{label_col}' (마지막 컬럼)")

        y = df[label_col].values.astype(int)
        X = df.drop(columns=[label_col]).values

        # feature_columns.txt와 일치 확인
        if self.feature_names:
            feature_cols_in_csv = [c for c in df.columns if c != label_col]
            if len(feature_cols_in_csv) != len(self.feature_names):
                print(f"  ⚠ Feature 수 불일치: CSV {len(feature_cols_in_csv)} vs model {len(self.feature_names)}")
            else:
                # feature_columns 순서대로 정렬
                try:
                    X = df[self.feature_names].values
                    print(f"  Feature columns 순서 정렬 완료")
                except KeyError:
                    print(f"  ⚠ Feature column 이름 불일치 → CSV 순서 그대로 사용")

        self.data_description = f"binary/test.csv ({len(y)} samples)"
        print(f"  라벨 분포: Normal={np.sum(y == 0)}, Fall={np.sum(y == 1)}")

        return X, y

    def predict(self, X):
        """RF 추론: feature 벡터 (2D array) → 예측/확률"""
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds.astype(int), probs

    def measure_speed(self, X):
        """RF 추론 속도: 프레임 단위"""
        times = []
        for _ in range(INFERENCE_REPEAT):
            start = time.perf_counter()
            self.model.predict(X)
            times.append(time.perf_counter() - start)
        self.avg_time_ms = (np.mean(times) / len(X)) * 1000


# ============================================================
# ST-GCN 공통 블록 & Graph
# ============================================================

class Graph:
    """COCO 17-keypoint graph with optional 3-partition strategy"""
    
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    NUM_NODE = 17
    CENTER = 0  # nose as center

    @staticmethod
    def uniform():
        """1-partition: (1, 17, 17)"""
        A = np.zeros((1, Graph.NUM_NODE, Graph.NUM_NODE), dtype=np.float32)
        for i, j in Graph.EDGES:
            A[0, i, j] = 1
            A[0, j, i] = 1
        for k in range(Graph.NUM_NODE):
            A[0, k, k] = 1
        return A

    @staticmethod
    def spatial():
        """3-partition: (3, 17, 17) — self-loop, inward, outward"""
        num = Graph.NUM_NODE
        A = np.zeros((3, num, num), dtype=np.float32)

        # partition 0: self-loop (identity)
        for k in range(num):
            A[0, k, k] = 1

        # BFS로 각 노드의 center까지 거리 계산
        hop = np.full(num, -1, dtype=int)
        hop[Graph.CENTER] = 0
        queue = [Graph.CENTER]
        adj = {n: [] for n in range(num)}
        for i, j in Graph.EDGES:
            adj[i].append(j)
            adj[j].append(i)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if hop[nb] < 0:
                    hop[nb] = hop[node] + 1
                    queue.append(nb)

        # partition 1: inward (closer to center), partition 2: outward (farther)
        for i, j in Graph.EDGES:
            if hop[j] < hop[i]:
                A[1, j, i] = 1  # inward: j → i (j closer)
                A[2, i, j] = 1  # outward: i → j
            else:
                A[1, i, j] = 1
                A[2, j, i] = 1

        return A



class OriginalBlock(nn.Module):
    """Original ST-GCN block: TCN = BN->ReLU->Conv->BN->Dropout"""
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True, dropout=0):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        self.gcn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )
        # checkpoint key mapping: tcn.0=BN, tcn.1=ReLU, tcn.2=Conv, tcn.3=BN, tcn.4=Dropout
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.residual(x)
        A = self.A.sum(0) if self.A.dim() == 3 and self.A.size(0) > 1 else self.A.squeeze(0)
        x_g = torch.einsum("nctv,vw->nctw", x, A)
        x_g = self.gcn(x_g)
        x_t = self.tcn(x_g)
        return self.relu(x_t + res)


class FineTunedBlock(nn.Module):
    """Fine-tuned ST-GCN block: A=(3,17,17) summed, GCN=out_ch, TCN=Conv->BN"""
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        self.gcn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.residual(x)
        A = self.A.sum(0)  # (3,17,17) -> (17,17)
        x_g = torch.einsum("nctv,vw->nctw", x, A)
        x_g = self.gcn(x_g)
        x_t = self.tcn(x_g)
        return self.relu(x_t + res)


# ============================================================
# 2. ST-GCN Original 파이프라인
# ============================================================

class _STGCNOriginalModel(nn.Module):
    """st_gcn_networks + BN-ReLU-Conv-BN-Dropout TCN (1-partition)"""
    def __init__(self, num_classes=2):
        super().__init__()
        A = Graph.uniform()  # (1, 17, 17)
        self.st_gcn_networks = nn.ModuleList([
            OriginalBlock(3, 64, A, residual=False),
            OriginalBlock(64, 64, A),
            OriginalBlock(64, 64, A),
            OriginalBlock(64, 128, A, stride=2),
            OriginalBlock(128, 128, A),
            OriginalBlock(128, 128, A),
            OriginalBlock(128, 256, A, stride=2),
            OriginalBlock(256, 256, A),
            OriginalBlock(256, 256, A),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(-1)
        for layer in self.st_gcn_networks:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.fc(x)


class _STGCNFineTunedModel(nn.Module):
    """layers + data_bn + Conv-BN TCN (3-partition)"""
    def __init__(self, num_classes=2):
        super().__init__()
        A = Graph.spatial()  # (3, 17, 17)
        self.data_bn = nn.BatchNorm1d(3 * 17)
        self.layers = nn.ModuleList([
            FineTunedBlock(3, 64, A, residual=False),
            FineTunedBlock(64, 64, A),
            FineTunedBlock(64, 64, A),
            FineTunedBlock(64, 128, A, stride=2),
            FineTunedBlock(128, 128, A),
            FineTunedBlock(128, 128, A),
            FineTunedBlock(128, 256, A, stride=2),
            FineTunedBlock(256, 256, A),
            FineTunedBlock(256, 256, A),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(-1)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.fc(x)


class STGCNOriginalPipeline(ModelPipeline):
    """
    ST-GCN Original 파이프라인:
    - test_data.npy (N, 3, 60, 17, 1) 시퀀스 입력
    - st_gcn_networks 구조 모델
    """

    def __init__(self):
        super().__init__(
            name="ST-GCN (Original)",
            short="Original",
            icon="📊",
            model_path=ST_GCN_DIR / "checkpoints" / "best_model_binary.pth",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = _STGCNOriginalModel(num_classes=2)

        ckpt = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        total, trainable = count_parameters(self.model)
        self.total_params = format_params(total)
        self.trainable_params = format_params(trainable)
        print(f"  Device: {self.device}  |  Params: {self.total_params}")

    def load_test_data(self):
        data_path = ST_GCN_DIR / "data/binary/test_data.npy"
        label_path = ST_GCN_DIR / "data/binary/test_labels.npy"
        self.data_description = f"binary/test_data.npy (60-frame sequences)"

        X = np.load(str(data_path))    # (N, 3, 60, 17, 1)
        y = np.load(str(label_path))
        return X, y.astype(int)

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds, probs

    def measure_speed(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.model(X_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(INFERENCE_REPEAT):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                self.model(X_t)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        self.avg_time_ms = (np.mean(times) / len(X)) * 1000


class STGCNFineTunedPipeline(ModelPipeline):
    """
    ST-GCN Fine-tuned 파이프라인:
    - test_data.npy (N, 3, 60, 17, 1) 시퀀스 입력
    - layers + data_bn 구조 모델
    """

    def __init__(self):
        super().__init__(
            name="ST-GCN (Fine-tuned)",
            short="Fine-tuned",
            icon="🚀",
            model_path=ST_GCN_DIR / "checkpoints_finetuned" / "best_model_finetuned.pth",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = _STGCNFineTunedModel(num_classes=2)

        ckpt = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        total, trainable = count_parameters(self.model)
        self.total_params = format_params(total)
        self.trainable_params = format_params(trainable)
        print(f"  Device: {self.device}  |  Params: {self.total_params}")

    def load_test_data(self):
        data_path = ST_GCN_DIR / "data/binary/test_data.npy"
        label_path = ST_GCN_DIR / "data/binary/test_labels.npy"
        self.data_description = f"binary/test_data.npy (60-frame sequences)"

        X = np.load(str(data_path))
        y = np.load(str(label_path))
        return X, y.astype(int)

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds, probs

    def measure_speed(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.model(X_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(INFERENCE_REPEAT):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                self.model(X_t)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        self.avg_time_ms = (np.mean(times) / len(X)) * 1000


# ============================================================
# 시각화 함수
# ============================================================

def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    cmap = LinearSegmentedColormap.from_list("c", ["#f0f4ff", "#2563eb"])

    for idx, (r, ax) in enumerate(zip(results, axes)):
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=20, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{r['icon']} {r['short']}\nAcc: {r['accuracy']:.2f}% (n={r['n_samples']})",
                      fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrix Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {save_path.name}")


def plot_roc_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        r["auc"] = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[idx % 3], linewidth=2.5,
                label=f"{r['icon']} {r['short']} (AUC={r['auc']:.4f}, n={r['n_samples']})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {save_path.name}")


def plot_inference_time(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [chart_label(r) for r in results]
    times = [r["avg_time_ms"] for r in results]
    bars = ax.barh(names, times, color=[COLORS[i % 3] for i in range(len(results))],
                   height=0.5, edgecolor="white", linewidth=1.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.03, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f} ms", ha="left", va="center", fontsize=12, fontweight="bold")

    # 추론 방식 주석
    for idx, r in enumerate(results):
        note = "frame-level" if r["short"] == "RF" else "60-frame batch"
        ax.text(max(times) * 0.5, idx, f"({note})", ha="center", va="center",
                fontsize=9, color="gray", style="italic")

    ax.set_xlabel("Inference Time (ms/sample)"); ax.invert_yaxis()
    ax.set_title("Inference Speed Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {save_path.name}")


def plot_model_size(results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [chart_label(r) for r in results]

    # 파일 크기
    sizes = [r["model_size_mb"] for r in results]
    bars = ax1.barh(names, sizes, color=[COLORS[i % 3] for i in range(len(results))],
                    height=0.5, edgecolor="white")
    for bar, s in zip(bars, sizes):
        ax1.text(bar.get_width() + max(sizes) * 0.03, bar.get_y() + bar.get_height() / 2,
                 f"{s:.2f} MB", ha="left", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("File Size (MB)"); ax1.invert_yaxis()
    ax1.set_title("Model File Size", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)

    # 파라미터 수 (DL 모델만)
    dl_results = [r for r in results if r["short"] != "RF"]
    if dl_results:
        dl_names = [chart_label(r) for r in dl_results]
        dl_params = []
        for r in dl_results:
            ps = r["total_params"]
            if isinstance(ps, str) and ps.endswith("M"):
                dl_params.append(float(ps[:-1]) * 1e6)
            elif isinstance(ps, str) and ps.endswith("K"):
                dl_params.append(float(ps[:-1]) * 1e3)
            else:
                dl_params.append(0)
        bars2 = ax2.barh(dl_names, dl_params,
                         color=[COLORS[(i + 1) % 3] for i in range(len(dl_results))],
                         height=0.5, edgecolor="white")
        for bar, v, r in zip(bars2, dl_params, dl_results):
            ax2.text(bar.get_width() + max(dl_params) * 0.03,
                     bar.get_y() + bar.get_height() / 2,
                     r["total_params"], ha="left", va="center", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Parameters"); ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
    ax2.set_title("Model Parameters", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {save_path.name}")


def plot_dashboard(results, save_path):
    """종합 대시보드"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # (1) Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    names = [chart_label(r) for r in results]
    accs = [r["accuracy"] for r in results]
    bars = ax1.bar(names, accs, color=COLORS[:len(results)], width=0.5, edgecolor="white")
    for bar, a, r in zip(bars, accs, results):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{a:.1f}%\n(n={r['n_samples']})", ha="center", va="bottom",
                 fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 110); ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Accuracy", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # (2) F1-Score
    ax2 = fig.add_subplot(gs[0, 1])
    f1s = [f1_score(r["y_true"], r["y_pred"], zero_division=0) for r in results]
    bars = ax2.bar(names, f1s, color=COLORS[:len(results)], width=0.5, edgecolor="white")
    for bar, f in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{f:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("F1-Score"); ax2.set_title("F1-Score", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    # (3) ROC
    ax3 = fig.add_subplot(gs[0, 2])
    for idx, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax3.plot(fpr, tpr, color=COLORS[idx], linewidth=2,
                 label=f"{r['short']} ({r.get('auc', 0):.3f})")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR"); ax3.set_title("ROC", fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # (4-6) Confusion Matrices
    cmap = LinearSegmentedColormap.from_list("c", ["#f0f4ff", "#2563eb"])
    for idx, r in enumerate(results):
        ax = fig.add_subplot(gs[1, idx])
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{r['icon']} {r['short']} (n={r['n_samples']})", fontweight="bold")

    fig.suptitle("Home Safe Solution - Model Comparison Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {save_path.name}")


# ============================================================
# 보고서 생성
# ============================================================

def generate_report(results, save_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []  # lines

    L.append("# Home Safe Solution - 모델 성능 비교 분석 보고서")
    L.append(f"## 생성일: {now}")
    L.append("")
    L.append("---")
    L.append("")

    # ── 1. 개요 ──
    L.append("## 1. 분석 개요")
    L.append("")
    L.append(f"- **비교 모델:** {len(results)}개")
    L.append(f"- **추론 반복 횟수:** {INFERENCE_REPEAT}회 (속도 측정)")
    L.append(f"- **평가 방식:** 각 모델의 고유 테스트 파이프라인 사용")
    L.append("")
    L.append("### 모델별 테스트 데이터")
    L.append("")
    L.append("| 모델 | 테스트 데이터 | 샘플 수 | 추론 방식 |")
    L.append("|------|-------------|---------|----------|")
    for r in results:
        infer_type = "프레임 단위 → 다수결" if r["short"] == "RF" else "60프레임 시퀀스"
        L.append(f"| {r['icon']} {r['name']} | {r['data_description'].split(chr(10))[0]} | "
                 f"{r['n_samples']} | {infer_type} |")
    L.append("")
    L.append("> ⚠ **참고:** 각 모델은 자체 테스트 파이프라인으로 평가되었습니다. "
             "RF는 feature 기반, ST-GCN은 시퀀스 기반으로 동일 데이터라도 입력 형태가 다릅니다.")
    L.append("")

    # ── 2. 종합 비교 ──
    L.append("## 2. 종합 성능 비교")
    L.append("")
    L.append("| 항목 | " + " | ".join([f"{r['icon']} {r['name']}" for r in results]) + " |")
    L.append("|------|" + "|".join(["------" for _ in results]) + "|")

    L.append("| **Accuracy** | " +
             " | ".join([f"**{r['accuracy']:.2f}%**" for r in results]) + " |")

    precs = [precision_score(r["y_true"], r["y_pred"], zero_division=0) for r in results]
    L.append("| **Precision** | " + " | ".join([f"{p:.4f}" for p in precs]) + " |")

    recs = [recall_score(r["y_true"], r["y_pred"], zero_division=0) for r in results]
    L.append("| **Recall** | " + " | ".join([f"{rc:.4f}" for rc in recs]) + " |")

    f1s = [f1_score(r["y_true"], r["y_pred"], zero_division=0) for r in results]
    L.append("| **F1-Score** | " + " | ".join([f"{f:.4f}" for f in f1s]) + " |")

    aucs = []
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        aucs.append(auc(fpr, tpr))
    L.append("| **AUC** | " + " | ".join([f"{a:.4f}" for a in aucs]) + " |")

    L.append("| **Inference Time** | " +
             " | ".join([f"{r['avg_time_ms']:.2f} ms" for r in results]) + " |")
    L.append("| **Model Size** | " +
             " | ".join([f"{r['model_size_mb']:.2f} MB" for r in results]) + " |")
    L.append("| **Parameters** | " +
             " | ".join([str(r['total_params']) for r in results]) + " |")
    L.append("| **Test Samples** | " +
             " | ".join([str(r['n_samples']) for r in results]) + " |")
    L.append("")

    # ── 3. Classification Report ──
    L.append("## 3. 상세 Classification Report")
    L.append("")
    for r in results:
        L.append(f"### {r['icon']} {r['name']}")
        L.append("")
        L.append("```")
        L.append(classification_report(r["y_true"], r["y_pred"],
                                       target_names=CLASS_NAMES, zero_division=0))
        L.append("```")
        L.append("")

    # ── 4. Confusion Matrix ──
    L.append("## 4. Confusion Matrix")
    L.append("")
    for r in results:
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        L.append(f"### {r['icon']} {r['name']} (n={r['n_samples']})")
        L.append("")
        L.append("|  | Pred: Normal | Pred: Fall |")
        L.append("|--|-------------|-----------|")
        L.append(f"| **Actual: Normal** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |")
        L.append(f"| **Actual: Fall** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |")
        L.append("")

    # ── 5. 시각화 ──
    L.append("## 5. 시각화")
    L.append("")
    L.append("![종합 대시보드](dashboard_comparison.png)")
    L.append("")
    L.append("![Confusion Matrix](confusion_matrices.png)")
    L.append("")
    L.append("![ROC Curve](roc_curves.png)")
    L.append("")
    L.append("![Inference Time](inference_time.png)")
    L.append("")
    L.append("![Model Size](model_size.png)")
    L.append("")

    # ── 6. 결론 ──
    L.append("## 6. 분석 결론")
    L.append("")

    best_acc = max(results, key=lambda x: x["accuracy"])
    best_f1_r = max(results, key=lambda x: f1_score(x["y_true"], x["y_pred"], zero_division=0))
    fastest = min(results, key=lambda x: x["avg_time_ms"])
    best_recall_r = max(results, key=lambda x: recall_score(x["y_true"], x["y_pred"], zero_division=0))

    L.append(f"- **최고 정확도:** {best_acc['icon']} {best_acc['name']} ({best_acc['accuracy']:.2f}%)")
    best_f1_val = f1_score(best_f1_r["y_true"], best_f1_r["y_pred"], zero_division=0)
    L.append(f"- **최고 F1-Score:** {best_f1_r['icon']} {best_f1_r['name']} ({best_f1_val:.4f})")
    best_recall_val = recall_score(best_recall_r["y_true"], best_recall_r["y_pred"], zero_division=0)
    L.append(f"- **최고 Recall (낙상 감지율):** {best_recall_r['icon']} {best_recall_r['name']} ({best_recall_val:.4f})")
    L.append(f"- **최고 속도:** {fastest['icon']} {fastest['name']} ({fastest['avg_time_ms']:.2f} ms)")
    L.append("")

    L.append("### 권장 사용 시나리오")
    L.append("")
    L.append("| 시나리오 | 권장 모델 | 이유 |")
    L.append("|---------|----------|------|")
    L.append(f"| 실시간 빠른 응답 | {fastest['icon']} {fastest['name']} | 최소 지연시간 ({fastest['avg_time_ms']:.2f} ms) |")
    L.append(f"| 최고 정확도 우선 | {best_acc['icon']} {best_acc['name']} | 최고 정확도 ({best_acc['accuracy']:.2f}%) |")
    L.append(f"| 낙상 놓침 최소화 | {best_recall_r['icon']} {best_recall_r['name']} | 최고 Recall ({best_recall_val:.4f}) |")
    L.append(f"| 균형 잡힌 선택 | {best_f1_r['icon']} {best_f1_r['name']} | 최고 F1-Score ({best_f1_val:.4f}) |")
    L.append("")

    L.append("### 참고 사항")
    L.append("")
    L.append("- RF와 ST-GCN은 입력 형태가 다르므로 (feature vs sequence) 직접 비교 시 해석에 주의가 필요합니다.")
    L.append("- 실제 운영 환경에서는 YOLO 포즈 추정 → 특징 추출/시퀀스 구성 시간도 고려해야 합니다.")
    L.append("- 낙상 감지 시스템에서는 Recall (낙상을 놓치지 않는 것)이 Precision보다 중요할 수 있습니다.")
    L.append("")

    # ── 7. 모델 경로 ──
    L.append("## 7. 모델 경로")
    L.append("")
    for r in results:
        L.append(f"- {r['icon']} {r['name']}: `{r['model_path']}`")
    L.append("")

    L.append("---")
    L.append("")
    L.append(f"**생성 도구:** `compare_models.py`")
    L.append(f"**생성일:** {now}")
    L.append(f"**저장 경로:** `{RESULTS_DIR}`")

    with open(str(save_path), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  ✅ {save_path.name}")


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 60)
    print("  Home Safe Solution - 모델 성능 비교 분석")
    print("  각 모델의 고유 테스트 파이프라인 사용")
    print("=" * 60)

    # 결과 폴더 생성
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  결과 저장: {RESULTS_DIR}")

    # ── 파이프라인 등록 ──
    pipelines = [
        RandomForestPipeline(),
        STGCNOriginalPipeline(),
        STGCNFineTunedPipeline(),
    ]

    # ── 평가 실행 ──
    results = []
    for pipe in pipelines:
        try:
            success = pipe.evaluate()
            if success:
                results.append(pipe.to_dict())
        except Exception as e:
            print(f"  ⚠ {pipe.name} 평가 실패: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n⚠ 평가 가능한 모델이 없습니다.")
        return

    # ── 시각화 ──
    print(f"\n{'=' * 60}")
    print("  시각화 생성")
    print(f"{'=' * 60}")

    plot_confusion_matrices(results, RESULTS_DIR / "confusion_matrices.png")
    plot_roc_curves(results, RESULTS_DIR / "roc_curves.png")
    plot_inference_time(results, RESULTS_DIR / "inference_time.png")
    plot_model_size(results, RESULTS_DIR / "model_size.png")
    plot_dashboard(results, RESULTS_DIR / "dashboard_comparison.png")

    # ── 보고서 ──
    print(f"\n{'=' * 60}")
    print("  보고서 생성")
    print(f"{'=' * 60}")

    generate_report(results, RESULTS_DIR / "MODEL_COMPARISON_REPORT.md")

    # ── 콘솔 요약 ──
    print(f"\n{'=' * 60}")
    print("  종합 결과 요약")
    print(f"{'=' * 60}")
    hdr = f"{'Model':<28} {'Acc':>7} {'F1':>7} {'AUC':>7} {'ms':>8} {'MB':>8} {'n':>5}"
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        f1 = f1_score(r["y_true"], r["y_pred"], zero_division=0)
        a = r.get("auc", 0)
        print(f"{r['icon']} {r['name']:<25} {r['accuracy']:>6.2f}% {f1:>7.4f} "
              f"{a:>7.4f} {r['avg_time_ms']:>7.2f} {r['model_size_mb']:>7.2f} {r['n_samples']:>5}")

    print(f"\n{'=' * 60}")
    print(f"  저장 완료: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
