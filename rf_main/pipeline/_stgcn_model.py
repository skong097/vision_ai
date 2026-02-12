#!/usr/bin/env python3
"""
============================================================
ST-GCN Model Definition for Training Pipeline
============================================================
compare_models.py의 _STGCNFineTunedModel과 동일한 구조
"""

import numpy as np
import torch
import torch.nn as nn


class Graph:
    """COCO 17-keypoint graph"""
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
        
        # self-loop
        for k in range(num):
            A[0, k, k] = 1
        
        # BFS for hop distance
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
        
        # inward/outward
        for i, j in Graph.EDGES:
            if hop[j] < hop[i]:
                A[1, j, i] = 1
                A[2, i, j] = 1
            else:
                A[1, i, j] = 1
                A[2, j, i] = 1
        
        return A


class STGCNBlock(nn.Module):
    """ST-GCN block: GCN + TCN with residual"""
    
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


class STGCNFineTunedModel(nn.Module):
    """ST-GCN Fine-tuned model: layers + data_bn + 3-partition"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        A = Graph.spatial()
        self.data_bn = nn.BatchNorm1d(3 * 17)
        self.layers = nn.ModuleList([
            STGCNBlock(3, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (N, C, T, V, M) or (N, C, T, V)
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


# Original 모델 (compare_models.py 호환)
class STGCNOriginalModel(nn.Module):
    """ST-GCN Original model: st_gcn_networks + 1-partition"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        A = self._uniform_graph()
        self.st_gcn_networks = nn.ModuleList([
            self._OriginalBlock(3, 64, A, residual=False),
            self._OriginalBlock(64, 64, A),
            self._OriginalBlock(64, 64, A),
            self._OriginalBlock(64, 128, A, stride=2),
            self._OriginalBlock(128, 128, A),
            self._OriginalBlock(128, 128, A),
            self._OriginalBlock(128, 256, A, stride=2),
            self._OriginalBlock(256, 256, A),
            self._OriginalBlock(256, 256, A),
        ])
        self.fc = nn.Linear(256, num_classes)

    def _uniform_graph(self):
        A = np.zeros((1, 17, 17), dtype=np.float32)
        for i, j in Graph.EDGES:
            A[0, i, j] = 1
            A[0, j, i] = 1
        for k in range(17):
            A[0, k, k] = 1
        return A

    class _OriginalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, A, stride=1, residual=True, dropout=0):
            super().__init__()
            self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
            self.gcn = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, (9, 1), (stride, 1), (4, 0)),
                nn.BatchNorm2d(out_ch), nn.Dropout(dropout, inplace=True),
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
            A = self.A.sum(0) if self.A.dim() == 3 else self.A.squeeze(0)
            x_g = torch.einsum("nctv,vw->nctw", x, A)
            x_g = self.gcn(x_g)
            x_t = self.tcn(x_g)
            return self.relu(x_t + res)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(-1)
        for layer in self.st_gcn_networks:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.fc(x)
