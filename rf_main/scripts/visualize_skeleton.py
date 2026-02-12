"""
Skeleton 데이터 시각화 및 검증 유틸리티
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


class SkeletonVisualizer:
    """Skeleton 데이터 시각화 및 검증"""
    
    def __init__(self):
        # COCO Skeleton 연결선 (17 keypoints)
        self.skeleton_connections = [
            (0, 1), (0, 2),  # 코 -> 눈
            (1, 3), (2, 4),  # 눈 -> 귀
            (0, 5), (0, 6),  # 코 -> 어깨
            (5, 7), (7, 9),  # 왼팔
            (6, 8), (8, 10), # 오른팔
            (5, 11), (6, 12),# 어깨 -> 골반
            (11, 12),        # 골반 연결
            (11, 13), (13, 15),  # 왼다리
            (12, 14), (14, 16)   # 오른다리
        ]
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def visualize_skeleton_on_video(self, video_file, skeleton_file, output_file=None):
        """동영상에 skeleton을 오버레이하여 시각화"""
        
        print(f"📹 시각화 중: {video_file.name}")
        
        # Skeleton 데이터 로드
        df = pd.read_csv(skeleton_file)
        
        # 동영상 열기
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 동영상 설정
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 해당 프레임의 skeleton 데이터 가져오기
            if frame_idx < len(df):
                row = df.iloc[frame_idx]
                
                # Keypoint 좌표 추출
                keypoints = []
                for kp_name in self.keypoint_names:
                    x = row[f'{kp_name}_x']
                    y = row[f'{kp_name}_y']
                    conf = row[f'{kp_name}_conf']
                    keypoints.append((x, y, conf))
                
                # Skeleton 그리기
                frame = self.draw_skeleton(frame, keypoints)
                
                # 가속도 정보 표시
                acc_x = row['acc_x']
                acc_y = row['acc_y']
                acc_z = row['acc_z']
                acc_mag = row['acc_mag']
                
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Acc: ({acc_x:.2f}, {acc_y:.2f}, {acc_z:.2f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Mag: {acc_mag:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 프레임 저장 또는 표시
            if output_file:
                out.write(frame)
            else:
                cv2.imshow('Skeleton Visualization', frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        cap.release()
        if output_file:
            out.release()
            print(f"✅ 저장 완료: {output_file}")
        else:
            cv2.destroyAllWindows()
    
    def draw_skeleton(self, frame, keypoints):
        """프레임에 skeleton 그리기"""
        
        # Keypoint 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # confidence threshold
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # 연결선 그리기
        for start_idx, end_idx in self.skeleton_connections:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if start_point[2] > 0.5 and end_point[2] > 0.5:
                cv2.line(frame, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        (255, 0, 0), 2)
        
        return frame
    
    def plot_acceleration_timeline(self, skeleton_file):
        """가속도 데이터 시계열 플롯"""
        
        df = pd.read_csv(skeleton_file)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # X축 가속도
        axes[0].plot(df['timestamp_ms'], df['acc_x'], label='Acc X', color='red')
        axes[0].set_ylabel('Acc X (g)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Y축 가속도
        axes[1].plot(df['timestamp_ms'], df['acc_y'], label='Acc Y', color='green')
        axes[1].set_ylabel('Acc Y (g)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Z축 가속도
        axes[2].plot(df['timestamp_ms'], df['acc_z'], label='Acc Z', color='blue')
        axes[2].set_ylabel('Acc Z (g)')
        axes[2].legend()
        axes[2].grid(True)
        
        # 가속도 크기
        axes[3].plot(df['timestamp_ms'], df['acc_mag'], label='Acc Magnitude', color='purple', linewidth=2)
        axes[3].set_ylabel('Acc Mag (g)')
        axes[3].set_xlabel('Time (ms)')
        axes[3].legend()
        axes[3].grid(True)
        
        # 낙상 추정 시점 표시 (가속도 급증 구간)
        threshold = df['acc_mag'].mean() + 2 * df['acc_mag'].std()
        fall_indices = df[df['acc_mag'] > threshold].index
        if len(fall_indices) > 0:
            for ax in axes:
                ax.axvline(x=df.loc[fall_indices[0], 'timestamp_ms'], 
                          color='red', linestyle='--', label='Estimated Fall')
        
        plt.suptitle(f'Acceleration Timeline - {skeleton_file.stem}')
        plt.tight_layout()
        plt.show()
    
    def analyze_skeleton_quality(self, skeleton_dir):
        """추출된 skeleton 데이터 품질 분석"""
        
        skeleton_files = sorted(Path(skeleton_dir).glob("fall-*-skeleton.csv"))
        
        print("=" * 60)
        print("📊 Skeleton 데이터 품질 분석")
        print("=" * 60)
        
        for skeleton_file in skeleton_files:
            df = pd.read_csv(skeleton_file)
            
            # 통계 계산
            total_frames = len(df)
            
            # 각 keypoint의 평균 confidence
            conf_cols = [col for col in df.columns if col.endswith('_conf')]
            avg_conf = df[conf_cols].mean().mean()
            
            # 가속도 통계
            acc_mag_max = df['acc_mag'].max()
            acc_mag_mean = df['acc_mag'].mean()
            
            # 낙상 추정 프레임 (가속도 급증)
            threshold = df['acc_mag'].mean() + 2 * df['acc_mag'].std()
            fall_frames = df[df['acc_mag'] > threshold]
            
            print(f"\n📁 {skeleton_file.name}")
            print(f"   총 프레임: {total_frames}")
            print(f"   평균 Confidence: {avg_conf:.3f}")
            print(f"   최대 가속도: {acc_mag_max:.3f}g")
            print(f"   평균 가속도: {acc_mag_mean:.3f}g")
            if len(fall_frames) > 0:
                fall_time = fall_frames.iloc[0]['timestamp_ms']
                print(f"   추정 낙상 시점: {fall_time:.0f}ms (프레임 {fall_frames.index[0]})")


def main():
    """메인 함수"""
    
    # 경로 설정
    video_dir = Path('/home/gjkong/dev_ws/yolo/myproj/data')
    skeleton_dir = Path('/home/gjkong/dev_ws/yolo/myproj/skeleton')
    
    visualizer = SkeletonVisualizer()
    
    # 예시: fall-01 시각화
    video_file = video_dir / 'fall-01-cam0.mp4'
    skeleton_file = skeleton_dir / 'fall-01-skeleton.csv'
    
    if video_file.exists() and skeleton_file.exists():
        # 1. 동영상에 skeleton 오버레이 (화면 출력)
        print("💡 팁: 'q'를 눌러 종료")
        visualizer.visualize_skeleton_on_video(video_file, skeleton_file)
        
        # 2. 가속도 타임라인 플롯
        # visualizer.plot_acceleration_timeline(skeleton_file)
    
    # 3. 전체 skeleton 데이터 품질 분석
    # visualizer.analyze_skeleton_quality(skeleton_dir)


if __name__ == "__main__":
    main()