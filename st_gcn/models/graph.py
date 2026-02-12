"""
Skeleton Graph 정의 (COCO 17 keypoints)
"""

import numpy as np

class Graph:
    """
    인체 Skeleton 그래프
    
    COCO 17 Keypoints:
    0: nose
    1: left_eye, 2: right_eye
    3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """
    
    def __init__(self):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor_link = [
            # 머리
            (0, 1), (0, 2),  # nose - eyes
            (1, 3), (2, 4),  # eyes - ears
            
            # 몸통
            (5, 6),          # shoulders
            (11, 12),        # hips
            (5, 11), (6, 12), # shoulders - hips
            
            # 팔
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10), # right arm
            
            # 다리
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]
        self.edge = self.self_link + self.neighbor_link
        self.center = 0  # nose
        
        # Adjacency Matrix
        self.A = self.get_adjacency_matrix()
    
    def get_adjacency_matrix(self):
        """
        인접 행렬 생성
        
        Returns:
            A: (17, 17) adjacency matrix
        """
        A = np.zeros((self.num_node, self.num_node))
        
        for i, j in self.edge:
            A[i, j] = 1
            A[j, i] = 1
        
        return A
