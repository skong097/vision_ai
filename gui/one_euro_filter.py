"""
OneEuroFilter - Keypoint 안정화 필터
Reference: http://cristal.univ-lille.fr/~casiez/1euro/
"""

import numpy as np


class OneEuroFilter:
    """
    OneEuroFilter - 저지연 스무딩 필터
    
    Parameters:
    - min_cutoff: 최소 컷오프 주파수 (낮을수록 더 부드럽게, 기본: 1.0)
    - beta: 속도 계수 (높을수록 빠른 움직임에 민감, 기본: 0.007)
    - d_cutoff: 미분 컷오프 주파수 (기본: 1.0)
    """
    
    def __init__(self, freq=20, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Args:
            freq: 샘플링 주파수 (Hz) - 20 FPS = 20
            min_cutoff: 최소 컷오프 주파수 (낮을수록 부드럽게)
            beta: 속도 감응도 (높을수록 빠른 움직임 추적)
            d_cutoff: 미분 컷오프 주파수
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # 상태 변수
        self.x_prev = None
        self.dx_prev = 0.0
        self.timestamp_prev = None
    
    def __call__(self, x, timestamp=None):
        """
        필터 적용
        
        Args:
            x: 현재 값 (float or np.array)
            timestamp: 타임스탬프 (None이면 자동 계산)
        
        Returns:
            필터링된 값
        """
        # 첫 번째 호출
        if self.x_prev is None:
            self.x_prev = x
            if timestamp is not None:
                self.timestamp_prev = timestamp
            return x
        
        # 시간 간격 계산
        if timestamp is not None and self.timestamp_prev is not None:
            te = timestamp - self.timestamp_prev
            self.timestamp_prev = timestamp
        else:
            te = 1.0 / self.freq
        
        # 속도 계산 (미분)
        dx = (x - self.x_prev) / te
        
        # 속도 필터링
        alpha_d = self.smoothing_factor(te, self.d_cutoff)
        dx_filtered = self.exponential_smoothing(alpha_d, dx, self.dx_prev)
        
        # 컷오프 주파수 계산 (속도에 따라 조정)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_filtered)
        
        # 위치 필터링
        alpha = self.smoothing_factor(te, cutoff)
        x_filtered = self.exponential_smoothing(alpha, x, self.x_prev)
        
        # 상태 업데이트
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered
        
        return x_filtered
    
    def smoothing_factor(self, te, cutoff):
        """
        스무딩 계수 계산
        
        Args:
            te: 시간 간격
            cutoff: 컷오프 주파수
        
        Returns:
            alpha (0~1)
        """
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)
    
    def exponential_smoothing(self, alpha, x, x_prev):
        """
        지수 평활
        
        Args:
            alpha: 스무딩 계수
            x: 현재 값
            x_prev: 이전 값
        
        Returns:
            평활된 값
        """
        return alpha * x + (1 - alpha) * x_prev
    
    def reset(self):
        """필터 리셋"""
        self.x_prev = None
        self.dx_prev = 0.0
        self.timestamp_prev = None


class KeypointFilter:
    """
    17개 Keypoint를 위한 필터 관리 클래스
    """
    
    def __init__(self, num_keypoints=17, filter_strength='medium'):
        """
        Args:
            num_keypoints: Keypoint 개수 (17)
            filter_strength: 필터 강도 ('light', 'medium', 'strong')
        """
        self.num_keypoints = num_keypoints
        self.filter_strength = filter_strength
        
        # 필터 강도별 파라미터
        self.presets = {
            'none': {
                'enabled': False
            },
            'light': {
                'enabled': True,
                'min_cutoff': 1.5,
                'beta': 0.01,
                'd_cutoff': 1.0
            },
            'medium': {
                'enabled': True,
                'min_cutoff': 1.0,
                'beta': 0.007,
                'd_cutoff': 1.0
            },
            'strong': {
                'enabled': True,
                'min_cutoff': 0.5,
                'beta': 0.005,
                'd_cutoff': 1.0
            }
        }
        
        # 필터 생성
        self.filters_x = []
        self.filters_y = []
        self.filters_conf = []
        
        params = self.presets.get(filter_strength, self.presets['medium'])
        
        if params['enabled']:
            for _ in range(num_keypoints):
                self.filters_x.append(OneEuroFilter(
                    freq=20,
                    min_cutoff=params['min_cutoff'],
                    beta=params['beta'],
                    d_cutoff=params['d_cutoff']
                ))
                self.filters_y.append(OneEuroFilter(
                    freq=20,
                    min_cutoff=params['min_cutoff'],
                    beta=params['beta'],
                    d_cutoff=params['d_cutoff']
                ))
                # Confidence는 약하게 필터링
                self.filters_conf.append(OneEuroFilter(
                    freq=20,
                    min_cutoff=2.0,
                    beta=0.01,
                    d_cutoff=1.0
                ))
        
        self.enabled = params['enabled']
    
    def apply(self, keypoints):
        """
        Keypoints 필터링
        
        Args:
            keypoints: (17, 3) - [x, y, confidence]
        
        Returns:
            filtered_keypoints: (17, 3)
        """
        if not self.enabled:
            return keypoints
        
        filtered_kps = np.zeros_like(keypoints)
        
        for i in range(self.num_keypoints):
            # Confidence가 낮으면 필터링 스킵
            if keypoints[i, 2] < 0.3:
                filtered_kps[i] = keypoints[i]
                continue
            
            # x, y, confidence 각각 필터링
            filtered_kps[i, 0] = self.filters_x[i](keypoints[i, 0])
            filtered_kps[i, 1] = self.filters_y[i](keypoints[i, 1])
            filtered_kps[i, 2] = self.filters_conf[i](keypoints[i, 2])
        
        return filtered_kps
    
    def reset(self):
        """모든 필터 리셋"""
        for f_x, f_y, f_conf in zip(self.filters_x, self.filters_y, self.filters_conf):
            f_x.reset()
            f_y.reset()
            f_conf.reset()
    
    def set_strength(self, strength):
        """
        필터 강도 변경
        
        Args:
            strength: 'none', 'light', 'medium', 'strong'
        """
        self.filter_strength = strength
        self.__init__(self.num_keypoints, strength)


# 테스트
if __name__ == "__main__":
    # 단일 값 테스트
    filter_1d = OneEuroFilter(freq=20, min_cutoff=1.0, beta=0.007)
    
    # 노이즈가 있는 신호
    import random
    true_signal = [100 + i for i in range(50)]
    noisy_signal = [s + random.uniform(-5, 5) for s in true_signal]
    
    filtered_signal = []
    for val in noisy_signal:
        filtered_signal.append(filter_1d(val))
    
    print("Original:", noisy_signal[:10])
    print("Filtered:", [f"{x:.2f}" for x in filtered_signal[:10]])
    
    # Keypoint 테스트
    kp_filter = KeypointFilter(filter_strength='medium')
    
    # 가상의 keypoints
    keypoints = np.random.rand(17, 3) * 100
    keypoints[:, 2] = 0.9  # confidence
    
    filtered_kps = kp_filter.apply(keypoints)
    
    print("\nKeypoint filtering test:")
    print("Original:", keypoints[0])
    print("Filtered:", filtered_kps[0])
