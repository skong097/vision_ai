#!/usr/bin/env python3
"""
ST-GCN Pre-trained 모델 Fine-tuning 스크립트
PYSKL NTU RGB+D 60 HRNet → Fall Detection (2 classes)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import time
from datetime import datetime

# ST-GCN 모델 경로 추가
sys.path.insert(0, '/home/gjkong/dev_ws/yolo/myproj/gui')


# ============================================================================
# 설정
# ============================================================================

CONFIG = {
    # 데이터
    'data_dir': '/home/gjkong/dev_ws/st_gcn/data/binary',
    'train_data': 'train_data.npy',
    'train_label': 'train_labels.npy',
    'val_data': 'val_data.npy',
    'val_label': 'val_labels.npy',
    
    # Pre-trained 모델
    'pretrained_path': '/home/gjkong/dev_ws/st_gcn/pretrained/stgcn_ntu60_hrnet.pth',
    
    # 출력
    'output_dir': '/home/gjkong/dev_ws/st_gcn/checkpoints_finetuned',
    
    # 학습 설정
    'num_classes': 2,           # Normal, Fall
    'num_point': 17,            # COCO keypoints
    'num_person': 1,
    'in_channels': 3,           # x, y, confidence
    
    # 하이퍼파라미터
    'epochs': 50,
    'batch_size': 16,
    'lr_backbone': 1e-5,        # backbone 낮은 lr
    'lr_head': 1e-3,            # FC head 높은 lr
    'weight_decay': 1e-4,
    
    # Fine-tuning 전략
    'freeze_backbone': False,   # True: backbone 완전 고정
    'freeze_epochs': 5,         # 처음 N epoch는 backbone 고정
    
    # 기타
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'seed': 42,
}


# ============================================================================
# 데이터셋
# ============================================================================

class FallDataset(Dataset):
    """낙상 감지 데이터셋"""
    
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)   # (N, C, T, V, M)
        self.labels = np.load(label_path)
        
        # 레이블이 1D인 경우 처리
        if len(self.labels.shape) == 2:
            self.labels = np.argmax(self.labels, axis=1)
        
        print(f"[Dataset] Loaded: {len(self.data)} samples")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Labels: {np.bincount(self.labels.astype(int))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y


# ============================================================================
# 모델 (간소화된 ST-GCN)
# ============================================================================

class Graph:
    """COCO 17 keypoints 그래프"""
    
    def __init__(self, layout='coco'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        # COCO 17 keypoints 연결
        self.inward = [
            (15, 13), (13, 11), (16, 14), (14, 12),  # legs
            (11, 12),  # hip connection
            (5, 11), (6, 12),  # torso-leg connection
            (5, 6),  # shoulder connection
            (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (1, 3), (2, 4),  # ears-eyes
            (0, 1), (0, 2),  # nose-eyes
            (1, 2),  # eye connection
            (0, 5), (0, 6),  # nose-shoulders
        ]
        
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        
        self.A = self.get_adjacency_matrix()
    
    def get_adjacency_matrix(self):
        """인접 행렬 생성"""
        A = np.zeros((3, self.num_node, self.num_node))
        
        # Self-loop
        for i, j in self.self_link:
            A[0, i, j] = 1
        
        # Inward
        for i, j in self.inward:
            A[1, i, j] = 1
        
        # Outward
        for i, j in self.outward:
            A[2, i, j] = 1
        
        return A


class STGCNBlock(nn.Module):
    """ST-GCN 기본 블록"""
    
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
        
        # 인접 행렬 등록
        self.register_buffer('A', torch.FloatTensor(A))
    
    def forward(self, x):
        # x: (N, C, T, V)
        res = self.residual(x)
        
        # GCN
        N, C, T, V = x.size()
        x = x.view(N, C * T, V)
        x = torch.matmul(x, self.A.sum(0))
        x = x.view(N, C, T, V)
        x = self.gcn(x)
        
        # TCN
        x = self.tcn(x) + res
        return self.relu(x)


class STGCN(nn.Module):
    """ST-GCN 모델 (Fine-tuning 용)"""
    
    def __init__(self, num_class=2, num_point=17, num_person=1, 
                 in_channels=3, graph_layout='coco'):
        super().__init__()
        
        self.graph = Graph(graph_layout)
        A = self.graph.A
        
        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # ST-GCN layers (PYSKL 구조와 유사)
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
        
        # Classification head
        self.fc = nn.Linear(256, num_class)
        
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
    
    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        
        # (N, M, V, C, T) -> (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        
        # (N, C, T, V*M)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(N, C, T, V * M)
        
        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=(2, 3))  # (N, C)
        
        # Classification
        x = self.fc(x)
        
        return x


# ============================================================================
# Fine-tuning 함수
# ============================================================================

def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    Pre-trained 가중치 로드 (FC layer 제외)
    
    Args:
        model: 대상 모델
        pretrained_path: Pre-trained 체크포인트 경로
        strict: 엄격한 매칭 여부
    
    Returns:
        로드 성공 여부
    """
    print(f"\n[Loading Pre-trained Weights]")
    print(f"  Path: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        print(f"  [ERROR] File not found!")
        return False
    
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    # state_dict 추출
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # PYSKL 형식 처리 (backbone. prefix 제거)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value
    
    # 현재 모델과 매칭
    model_dict = model.state_dict()
    
    # FC layer 제외하고 매칭되는 것만 로드
    matched = {}
    unmatched = []
    
    for key, value in new_state_dict.items():
        if 'fc' in key.lower() or 'cls' in key.lower():
            continue
        
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                matched[key] = value
            else:
                unmatched.append(f"{key}: shape mismatch {value.shape} vs {model_dict[key].shape}")
        else:
            unmatched.append(f"{key}: not in model")
    
    print(f"  Matched layers: {len(matched)}/{len(model_dict)}")
    
    if unmatched and len(unmatched) <= 10:
        print(f"  Unmatched:")
        for msg in unmatched[:5]:
            print(f"    - {msg}")
    
    # 로드
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"  [OK] Pre-trained weights loaded!")
    return True


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, freeze_backbone=False):
    """한 에폭 학습"""
    model.train()
    
    if freeze_backbone:
        # Backbone freeze
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    else:
        # 모든 파라미터 학습
        for param in model.parameters():
            param.requires_grad = True
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """평가"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def fine_tune(config):
    """Fine-tuning 메인 함수"""
    
    print("\n" + "="*60)
    print("ST-GCN Fine-tuning for Fall Detection")
    print("="*60)
    
    # 시드 설정
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device(config['device'])
    print(f"\n[Device] {device}")
    
    # 데이터 로드
    print("\n[1/5] Loading data...")
    train_data_path = os.path.join(config['data_dir'], config['train_data'])
    train_label_path = os.path.join(config['data_dir'], config['train_label'])
    val_data_path = os.path.join(config['data_dir'], config['val_data'])
    val_label_path = os.path.join(config['data_dir'], config['val_label'])
    
    train_dataset = FallDataset(train_data_path, train_label_path)
    val_dataset = FallDataset(val_data_path, val_label_path)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'])
    
    # 모델 생성
    print("\n[2/5] Creating model...")
    model = STGCN(
        num_class=config['num_classes'],
        num_point=config['num_point'],
        num_person=config['num_person'],
        in_channels=config['in_channels']
    )
    
    # Pre-trained 가중치 로드
    print("\n[3/5] Loading pre-trained weights...")
    if os.path.exists(config['pretrained_path']):
        load_pretrained_weights(model, config['pretrained_path'])
    else:
        print(f"  [WARNING] Pre-trained model not found: {config['pretrained_path']}")
        print(f"  Training from scratch...")
    
    model = model.to(device)
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # 옵티마이저 (차등 learning rate)
    print("\n[4/5] Setting up training...")
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    head_params = [p for n, p in model.named_parameters() if 'fc' in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['lr_backbone']},
        {'params': head_params, 'lr': config['lr_head']}
    ], weight_decay=config['weight_decay'])
    
    # 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 출력 디렉토리
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 학습
    print("\n[5/5] Training...")
    print("-"*60)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'LR':>10}")
    print("-"*60)
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(1, config['epochs'] + 1):
        # Freeze backbone for first N epochs
        freeze = config['freeze_backbone'] or (epoch <= config['freeze_epochs'])
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, freeze
        )
        
        # 평가
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 출력
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.2f}% | {val_loss:>10.4f} | {val_acc:>9.2f}% | {current_lr:>10.2e}")
        
        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            save_path = os.path.join(config['output_dir'], 'best_model_finetuned.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, save_path)
            print(f"  [BEST] Saved: {save_path}")
    
    # 최종 모델 저장
    final_path = os.path.join(config['output_dir'], 'final_model_finetuned.pth')
    torch.save({
        'epoch': config['epochs'],
        'state_dict': model.state_dict(),
        'val_acc': val_acc,
        'config': config,
    }, final_path)
    
    print("-"*60)
    print(f"\n[Result]")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Models saved to: {config['output_dir']}")
    
    return best_val_acc


# ============================================================================
# 메인
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ST-GCN Fine-tuning')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr-backbone', type=float, default=CONFIG['lr_backbone'])
    parser.add_argument('--lr-head', type=float, default=CONFIG['lr_head'])
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--freeze-epochs', type=int, default=CONFIG['freeze_epochs'])
    parser.add_argument('--pretrained', type=str, default=CONFIG['pretrained_path'])
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'])
    
    args = parser.parse_args()
    
    # Config 업데이트
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['lr_backbone'] = args.lr_backbone
    CONFIG['lr_head'] = args.lr_head
    CONFIG['freeze_backbone'] = args.freeze_backbone
    CONFIG['freeze_epochs'] = args.freeze_epochs
    CONFIG['pretrained_path'] = args.pretrained
    CONFIG['output_dir'] = args.output_dir
    
    # Fine-tuning 실행
    fine_tune(CONFIG)
