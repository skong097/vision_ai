"""
ST-GCN 모델 구현
Spatial-Temporal Graph Convolutional Network for Fall Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import Graph


class Model(nn.Module):
    """
    ST-GCN for Fall Detection
    
    Input: (N, C, T, V, M)
        N: Batch size
        C: Channels (3: x, y, confidence)
        T: Time frames (60)
        V: Vertices (17 keypoints)
        M: Number of people (1)
    
    Output: (N, num_class)
        num_class: 2 (Normal, Fall) for binary classification
    """
    
    def __init__(self, num_class=2, num_point=17, num_person=1, 
                 in_channels=3, graph=None, graph_args=dict()):
        super(Model, self).__init__()
        
        if graph is None:
            self.graph = Graph(**graph_args)
        else:
            self.graph = graph
        
        # Adjacency Matrix
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # ST-GCN Networks (9 layers)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, A, residual=False),
            st_gcn(64, 64, A),
            st_gcn(64, 64, A),
            st_gcn(64, 128, A, stride=2),
            st_gcn(128, 128, A),
            st_gcn(128, 128, A),
            st_gcn(128, 256, A, stride=2),
            st_gcn(256, 256, A),
            st_gcn(256, 256, A),
        ))
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc = nn.Linear(256, num_class)
        
        # Dropout
        self.drop_out = nn.Dropout(0.5)
        
        # Initialize
        bn_init(self.fc, 1)
    
    def forward(self, x):
        # x: (N, C, T, V, M)
        
        N, C, T, V, M = x.size()
        
        # (N, C, T, V, M) -> (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        
        # ST-GCN Layers
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A)
        
        # Global Pooling
        # (N*M, C, T, V) -> (N*M, C, 1, 1)
        x = self.pool(x)
        
        # (N*M, C, 1, 1) -> (N*M, C)
        x = x.view(N, M, -1).mean(dim=1)
        
        # Dropout
        x = self.drop_out(x)
        
        # Classification
        x = self.fc(x)
        
        return x


class st_gcn(nn.Module):
    """
    ST-GCN Block: Spatial GCN + Temporal Conv
    """
    
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(st_gcn, self).__init__()
        
        # Spatial GCN
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, A.size(0))
        
        # Temporal Conv
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1),
                (stride, 1),
                (4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True),
        )
        
        # Residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        # Spatial GCN
        res = self.residual(x)
        x, A = self.gcn(x, A)
        
        # Temporal Conv
        x = self.tcn(x) + res
        
        return self.relu(x), A


class ConvTemporalGraphical(nn.Module):
    """
    Graph Convolution Layer
    """
    
    def __init__(self, in_channels, out_channels, num_node):
        super(ConvTemporalGraphical, self).__init__()
        
        self.num_node = num_node
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (V, V)
        
        N, C, T, V = x.size()
        
        # Graph Convolution
        # (N, C, T, V) @ (V, V) -> (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', (x, A))
        
        # 1x1 Conv
        x = self.conv(x)
        
        return x.contiguous(), A


def bn_init(module, scale):
    """BatchNorm 초기화"""
    if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, scale)
        nn.init.constant_(module.bias, 0)


def conv_init(module):
    """Conv 초기화"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
