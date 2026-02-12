"""
ST-GCN 모델 패키지
- Fall Detection을 위한 Spatial-Temporal Graph Convolutional Network
"""

from .st_gcn import Model
from .graph import Graph

__all__ = ['Model', 'Graph']
