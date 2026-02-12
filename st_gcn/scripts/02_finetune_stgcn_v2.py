#!/usr/bin/env python3
"""
============================================================
ST-GCN Fine-tuning with PYSKL Pre-trained Backbone
============================================================
Pre-trained: PYSKL ST-GCN NTU60 HRNet (17 keypoints)
Fine-tune:   binary_v2 데이터 (normal 1629 + fallen 301 영상)
구조:        Fine-tuned (data_bn + layers + 3-partition)

출력:
  /home/gjkong/dev_ws/st_gcn/checkpoints_v2/
    ├── best_model.pth
    └── training_log.json
============================================================
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS

# ============================================================
# 설정
# ============================================================

# DATA_DIR = Path("/home/gjkong/dev_ws/st_gcn/data/binary_v2")
# PRETRAINED_PATH = Path("/home/gjkong/dev_ws/st_gcn/pretrained/stgcn_ntu60_hrnet.pth")
# CHECKPOINT_DIR = Path("/home/gjkong/dev_ws/st_gcn/checkpoints_v2")

DATA_DIR = PATHS.STGCN_DATA_V2
PRETRAINED_PATH = PATHS.STGCN_PRETRAINED
CHECKPOINT_DIR = PATHS.ST_GCN / "checkpoints_v2"



NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 80
LR_FC = 1e-3          # FC layer 학습률
LR_BACKBONE = 1e-4    # Backbone 학습률
WEIGHT_DECAY = 1e-4
PATIENCE = 15          # Early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Graph (3-partition, Fine-tuned 구조와 동일)
# ============================================================

class Graph:
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    NUM_NODE = 17
    CENTER = 0

    @staticmethod
    def spatial():
        """3-partition: (3, 17, 17) — self-loop, inward, outward"""
        num = Graph.NUM_NODE
        A = np.zeros((3, num, num), dtype=np.float32)

        A[0] = np.eye(num)  # self-loop

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

        for i, j in Graph.EDGES:
            if hop[j] < hop[i]:
                A[1, j, i] = 1
                A[2, i, j] = 1
            else:
                A[1, i, j] = 1
                A[2, j, i] = 1

        return A


# ============================================================
# Fine-tuned Block (기존과 동일 구조)
# ============================================================

class FineTunedBlock(nn.Module):
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
        A = self.A.sum(0)
        x_g = torch.einsum("nctv,vw->nctw", x, A)
        x_g = self.gcn(x_g)
        x_t = self.tcn(x_g)
        return self.relu(x_t + res)


# ============================================================
# ST-GCN Fine-tuned Model
# ============================================================

class STGCNFineTuned(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        A = Graph.spatial()
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


# ============================================================
# Pre-trained 가중치 로드
# ============================================================

def load_pretrained_weights(model, pretrained_path):
    """
    PYSKL Pre-trained 가중치를 Fine-tuned 구조에 로드
    
    PYSKL 키: backbone.data_bn.*, backbone.gcn.0.*, ...
    우리 키:   data_bn.*, layers.0.gcn.*, ...
    """
    print(f"\n📦 Pre-trained 가중치 로드: {pretrained_path.name}")
    
    ckpt = torch.load(str(pretrained_path), map_location='cpu', weights_only=False)
    pretrained_state = ckpt.get('state_dict', ckpt)
    
    # PYSKL 키 매핑 분석
    print(f"  Pre-trained keys: {len(pretrained_state)}개")
    
    # backbone. prefix 제거
    cleaned = {}
    for k, v in pretrained_state.items():
        new_k = k.replace('backbone.', '')
        cleaned[new_k] = v
    
    model_state = model.state_dict()
    
    # 매칭 시도
    loaded = 0
    skipped_fc = 0
    skipped_shape = 0
    skipped_missing = 0
    
    for key in model_state.keys():
        if 'fc' in key:
            skipped_fc += 1
            continue
        
        if key in cleaned:
            if cleaned[key].shape == model_state[key].shape:
                model_state[key] = cleaned[key]
                loaded += 1
            else:
                skipped_shape += 1
                print(f"  ⚠ Shape 불일치: {key} — model {model_state[key].shape} vs pretrained {cleaned[key].shape}")
        else:
            skipped_missing += 1
    
    model.load_state_dict(model_state, strict=False)
    
    print(f"  ✅ 로드: {loaded}개")
    print(f"  ⏭ FC 스킵: {skipped_fc}개")
    print(f"  ⚠ Shape 불일치: {skipped_shape}개")
    print(f"  ❌ 매칭 안됨: {skipped_missing}개")
    
    return loaded


# ============================================================
# 학습
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    
    return total_loss / total, correct / total * 100


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return total_loss / total, correct / total * 100, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 60)
    print("  ST-GCN Fine-tuning with PYSKL Pre-trained Backbone")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Data:   {DATA_DIR}")
    print(f"  Pre-trained: {PRETRAINED_PATH.name}")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── 데이터 로드 ──
    print(f"\n📦 데이터 로드...")
    train_data = np.load(DATA_DIR / "train_data.npy")
    train_labels = np.load(DATA_DIR / "train_labels.npy")
    test_data = np.load(DATA_DIR / "test_data.npy")
    test_labels = np.load(DATA_DIR / "test_labels.npy")
    
    print(f"  Train: {train_data.shape} (Normal={sum(train_labels==0)}, Fallen={sum(train_labels==1)})")
    print(f"  Test:  {test_data.shape} (Normal={sum(test_labels==0)}, Fallen={sum(test_labels==1)})")
    
    # DataLoader
    train_ds = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    test_ds = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # ── 모델 생성 ──
    model = STGCNFineTuned(num_classes=NUM_CLASSES).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗 모델 파라미터: {total_params:,}")
    
    # ── Pre-trained 가중치 로드 ──
    if PRETRAINED_PATH.exists():
        loaded = load_pretrained_weights(model, PRETRAINED_PATH)
        if loaded == 0:
            print("  ⚠ Pre-trained 가중치 로드 실패 → scratch 학습")
    else:
        print(f"  ⚠ Pre-trained 파일 없음: {PRETRAINED_PATH}")
        print("  → Scratch 학습으로 진행")
    
    # ── Optimizer (차등 learning rate) ──
    fc_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    
    optimizer = optim.Adam([
        {'params': fc_params, 'lr': LR_FC},
        {'params': backbone_params, 'lr': LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)
    
    # Class weight (불균형 보정)
    n_normal = sum(train_labels == 0)
    n_fallen = sum(train_labels == 1)
    weight = torch.tensor([1.0, n_normal / n_fallen], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    print(f"\n⚙ 학습 설정:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  LR (FC): {LR_FC}, LR (Backbone): {LR_BACKBONE}")
    print(f"  Class weight: Normal=1.0, Fallen={n_normal/n_fallen:.2f}")
    print(f"  Early stopping: {PATIENCE} epochs")
    
    # ── Scheduler ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    # ── 학습 루프 ──
    print(f"\n{'=' * 60}")
    print(f"  학습 시작")
    print(f"{'=' * 60}")
    
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    training_log = []
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate
        test_loss, test_acc, test_preds, test_true, test_probs = evaluate(model, test_loader, criterion)
        
        # Scheduler
        scheduler.step(test_acc)
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc, 2),
        }
        training_log.append(log_entry)
        
        # Print
        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            no_improve = 0
            marker = " ⭐ Best!"
            
            # 모델 저장
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }, CHECKPOINT_DIR / "best_model.pth")
        else:
            no_improve += 1
        
        if epoch % 5 == 0 or marker:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_acc:5.1f}% (loss={train_loss:.4f}) | "
                  f"Test: {test_acc:5.1f}% (loss={test_loss:.4f}) | "
                  f"{elapsed:.0f}s{marker}")
        
        # Early stopping
        if no_improve >= PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break
    
    # ── 최종 평가 ──
    print(f"\n{'=' * 60}")
    print(f"  최종 결과")
    print(f"{'=' * 60}")
    
    # Best 모델 로드
    best_ckpt = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['state_dict'])
    
    _, final_acc, final_preds, final_true, final_probs = evaluate(model, test_loader, criterion)
    
    # 상세 메트릭
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
    
    f1 = f1_score(final_true, final_preds, zero_division=0) * 100
    prec = precision_score(final_true, final_preds, zero_division=0) * 100
    rec = recall_score(final_true, final_preds, zero_division=0) * 100
    auc_val = roc_auc_score(final_true, final_probs) * 100
    cm = confusion_matrix(final_true, final_preds)
    
    print(f"\n  Best Epoch: {best_epoch}")
    print(f"  Accuracy:  {final_acc:.1f}%")
    print(f"  F1-Score:  {f1:.1f}%")
    print(f"  Precision: {prec:.1f}%")
    print(f"  Recall:    {rec:.1f}%")
    print(f"  AUC:       {auc_val:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    
    # 학습 로그 저장
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'pretrained': str(PRETRAINED_PATH),
        'data_dir': str(DATA_DIR),
        'train_samples': len(train_labels),
        'test_samples': len(test_labels),
        'best_epoch': best_epoch,
        'best_accuracy': round(best_acc, 2),
        'final_metrics': {
            'accuracy': round(final_acc, 2),
            'f1': round(f1, 2),
            'precision': round(prec, 2),
            'recall': round(rec, 2),
            'auc': round(auc_val, 2),
        },
        'confusion_matrix': cm.tolist(),
        'training_log': training_log,
    }
    
    with open(CHECKPOINT_DIR / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✅ 학습 완료! ({elapsed:.0f}초)")
    print(f"  모델: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"  로그: {CHECKPOINT_DIR / 'training_log.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
