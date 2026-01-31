"""
자동 라벨링 - 가속도 데이터 기반 Binary & 3-Class 라벨 생성
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AutoLabeler:
    """가속도 데이터 기반 자동 라벨링"""
    
    def __init__(self, features_dir, output_dir):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 라벨링 임계값 설정
        self.fall_acc_threshold = 2.5  # 낙상 감지 가속도 임계값 (g)
        self.fall_duration = 30  # 낙상 판단 최소 지속 프레임 (약 1초 @ 30fps)
        self.fallen_height_threshold = 0.6  # 쓰러짐 판단 높이 비율 (정상 높이의 60%)
    
    def detect_fall_point(self, df):
        """가속도 급증 시점 감지 (낙상 시작점)"""
        
        # 가속도 크기 분석
        acc_mag = df['acc_mag'].values
        
        # 급증 시점 찾기 (threshold 초과)
        fall_candidates = np.where(acc_mag > self.fall_acc_threshold)[0]
        
        if len(fall_candidates) == 0:
            # 임계값 초과가 없으면 최대값 위치 사용
            fall_start = np.argmax(acc_mag)
        else:
            # 첫 번째 급증 시점
            fall_start = fall_candidates[0]
        
        return fall_start
    
    def label_binary(self, df):
        """이진 분류 라벨링 (0: Normal, 1: Fall)"""
        
        # 낙상 시작점 감지
        fall_start = self.detect_fall_point(df)
        
        # 라벨 초기화
        labels = np.zeros(len(df), dtype=int)
        
        # 낙상 시작점 이전: Normal (0)
        # 낙상 시작점 이후: Fall (1)
        labels[fall_start:] = 1
        
        return labels, fall_start
    
    def label_3class(self, df):
        """3-Class 라벨링 (0: Normal, 1: Falling, 2: Fallen)"""
        
        # 낙상 시작점 감지
        fall_start = self.detect_fall_point(df)
        
        # 라벨 초기화
        labels = np.zeros(len(df), dtype=int)
        
        # Phase 1: 낙상 이전 = Normal (0)
        labels[:fall_start] = 0
        
        # Phase 2: Falling 구간 설정
        # Falling 구간은 최소 15프레임(0.5초), 최대 60프레임(2초)
        falling_duration = 30  # 기본 1초
        
        if 'hip_height' in df.columns:
            hip_heights = df['hip_height'].values
            
            # 정상 구간의 평균 높이
            if fall_start > 10:
                normal_height = hip_heights[:fall_start].mean()
            else:
                normal_height = hip_heights[:max(10, fall_start)].mean()
            
            # Falling 구간 찾기 (높이가 급격히 감소하는 구간)
            falling_end = fall_start + falling_duration  # 기본값
            
            for i in range(fall_start, min(fall_start + 90, len(df))):  # 최대 3초(90프레임) 탐색
                current_height = hip_heights[i]
                
                # 높이가 정상의 60% 이하로 떨어지면 Fallen 시작
                if current_height < normal_height * self.fallen_height_threshold:
                    falling_end = i
                    break
            
            # Falling 구간이 너무 짧으면 최소 15프레임 보장
            if falling_end - fall_start < 15:
                falling_end = min(fall_start + 15, len(df))
            
            # Falling 구간이 너무 길면 60프레임으로 제한
            if falling_end - fall_start > 60:
                falling_end = fall_start + 60
        else:
            # hip_height가 없으면 30프레임(1초)을 Falling으로 설정
            falling_end = min(fall_start + falling_duration, len(df))
        
        # Falling 구간 라벨링 (최소 1개 이상 보장)
        if falling_end > fall_start:
            labels[fall_start:falling_end] = 1
        else:
            # Falling 구간이 없으면 최소 1프레임이라도 설정
            labels[fall_start] = 1
            falling_end = fall_start + 1
        
        # Fallen 구간 라벨링
        if falling_end < len(df):
            labels[falling_end:] = 2
        
        return labels, fall_start, falling_end
    
    def visualize_labels(self, df, labels_binary, labels_3class, fall_start, filename):
        """라벨링 결과 시각화"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        time_ms = df['timestamp_ms'].values
        acc_mag = df['acc_mag'].values
        hip_height = df['hip_height'].values if 'hip_height' in df.columns else np.zeros(len(df))
        
        # 1. 가속도 + Binary 라벨
        ax1 = axes[0]
        ax1.plot(time_ms, acc_mag, 'b-', label='Acc Magnitude', linewidth=2)
        ax1.axhline(y=self.fall_acc_threshold, color='r', linestyle='--', label='Threshold')
        ax1.axvline(x=time_ms[fall_start], color='orange', linestyle='--', label='Fall Start')
        
        # Binary 라벨 배경색
        for i in range(len(df)):
            if labels_binary[i] == 1:
                ax1.axvspan(time_ms[i], time_ms[min(i+1, len(df)-1)], alpha=0.3, color='red')
        
        ax1.set_ylabel('Acceleration (g)', fontsize=12)
        ax1.set_title('Binary Labeling: Normal vs Fall', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 높이 + 3-Class 라벨
        ax2 = axes[1]
        ax2.plot(time_ms, hip_height, 'g-', label='Hip Height', linewidth=2)
        
        # 3-Class 라벨 배경색
        colors = {0: 'white', 1: 'yellow', 2: 'red'}
        labels_text = {0: 'Normal', 1: 'Falling', 2: 'Fallen'}
        
        for i in range(len(df)):
            label = labels_3class[i]
            ax2.axvspan(time_ms[i], time_ms[min(i+1, len(df)-1)], 
                       alpha=0.3, color=colors[label])
        
        ax2.set_ylabel('Hip Height (pixels)', fontsize=12)
        ax2.set_title('3-Class Labeling: Normal vs Falling vs Fallen', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 라벨 타임라인
        ax3 = axes[2]
        ax3.plot(time_ms, labels_binary, 'b-', label='Binary (0=Normal, 1=Fall)', linewidth=2, alpha=0.7)
        ax3.plot(time_ms, labels_3class, 'r-', label='3-Class (0=Normal, 1=Falling, 2=Fallen)', linewidth=2, alpha=0.7)
        ax3.set_ylabel('Label', fontsize=12)
        ax3.set_xlabel('Time (ms)', fontsize=12)
        ax3.set_title('Label Timeline Comparison', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, 2.5)
        
        plt.tight_layout()
        
        # 저장
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / f'{filename}_labeling.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_file(self, features_file, visualize=False):
        """단일 features 파일 처리"""
        
        # Features 데이터 로드
        df = pd.read_csv(features_file)
        
        # Binary 라벨링
        labels_binary, fall_start = self.label_binary(df)
        df['label_binary'] = labels_binary
        
        # 3-Class 라벨링
        labels_3class, fall_start, falling_end = self.label_3class(df)
        df['label_3class'] = labels_3class
        
        # 저장
        output_file = self.output_dir / features_file.name.replace('-features', '-labeled')
        df.to_csv(output_file, index=False)
        
        # 시각화 (선택)
        if visualize:
            self.visualize_labels(df, labels_binary, labels_3class, fall_start, features_file.stem)
        
        # 통계 출력
        binary_counts = pd.Series(labels_binary).value_counts().to_dict()
        class3_counts = pd.Series(labels_3class).value_counts().to_dict()
        
        return {
            'file': features_file.name,
            'total_frames': len(df),
            'fall_start': fall_start,
            'binary': binary_counts,
            '3class': class3_counts
        }
    
    def process_all(self, visualize_first_n=5):
        """모든 features 파일 처리"""
        
        print("=" * 60)
        print("🏷️  Auto-Labeling Pipeline (Binary & 3-Class)")
        print("=" * 60)
        
        features_files = sorted(self.features_dir.glob("fall-*-features.csv"))
        print(f"\n📂 찾은 파일: {len(features_files)}개")
        
        stats = []
        
        for idx, features_file in enumerate(tqdm(features_files, desc="Labeling")):
            # 처음 N개만 시각화
            visualize = (idx < visualize_first_n)
            stat = self.process_file(features_file, visualize=visualize)
            stats.append(stat)
        
        # 전체 통계 출력
        print("\n" + "=" * 60)
        print("📊 라벨링 통계")
        print("=" * 60)
        
        total_frames = sum([s['total_frames'] for s in stats])
        
        # Binary 통계
        binary_normal = sum([s['binary'].get(0, 0) for s in stats])
        binary_fall = sum([s['binary'].get(1, 0) for s in stats])
        
        print(f"\n🔹 Binary Classification:")
        print(f"   Normal (0): {binary_normal} frames ({binary_normal/total_frames*100:.1f}%)")
        print(f"   Fall   (1): {binary_fall} frames ({binary_fall/total_frames*100:.1f}%)")
        
        # 3-Class 통계
        class3_normal = sum([s['3class'].get(0, 0) for s in stats])
        class3_falling = sum([s['3class'].get(1, 0) for s in stats])
        class3_fallen = sum([s['3class'].get(2, 0) for s in stats])
        
        print(f"\n🔹 3-Class Classification:")
        print(f"   Normal  (0): {class3_normal} frames ({class3_normal/total_frames*100:.1f}%)")
        print(f"   Falling (1): {class3_falling} frames ({class3_falling/total_frames*100:.1f}%)")
        print(f"   Fallen  (2): {class3_fallen} frames ({class3_fallen/total_frames*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print(f"✅ 완료! 출력 경로: {self.output_dir}")
        print(f"📊 시각화: {self.output_dir / 'visualizations'} (처음 {visualize_first_n}개)")
        print("=" * 60)


def main():
    # 경로 설정
    features_dir = '/home/gjkong/dev_ws/yolo/myproj/features'
    output_dir = '/home/gjkong/dev_ws/yolo/myproj/labeled'
    
    # Auto-Labeling 실행
    labeler = AutoLabeler(features_dir, output_dir)
    labeler.process_all(visualize_first_n=5)


if __name__ == "__main__":
    main()
