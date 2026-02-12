"""
YOLO Pose Skeleton 추출 및 가속도 데이터 동기화
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SkeletonExtractor:
    """YOLO Pose를 사용하여 skeleton을 추출하고 가속도 데이터와 동기화"""
    
    def __init__(self, video_dir, accel_dir, output_dir):
        self.video_dir = Path(video_dir)
        self.accel_dir = Path(accel_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO Pose 모델 로드 (yolo11s-pose.pt 사용)
        print("🤖 YOLO Pose 모델 로딩 중...")
        self.model = YOLO('yolo11s-pose.pt')
        print("✅ 모델 로드 완료!")
        
        # COCO Keypoint 이름 (17개)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def load_acceleration_data(self, accel_file):
        """가속도 CSV 파일 로드"""
        df = pd.read_csv(accel_file, header=None)
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'acc_mag']
        return df
    
    def process_video(self, video_file, accel_file):
        """단일 동영상에서 skeleton 추출 및 가속도 데이터 동기화"""
        
        video_name = video_file.stem
        print(f"\n📹 처리 중: {video_name}")
        
        # 동영상 열기
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 가속도 데이터 로드
        acc_df = self.load_acceleration_data(accel_file)
        
        # 결과 저장용 리스트
        results_data = []
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Extracting skeleton")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 프레임의 timestamp (ms)
            timestamp_ms = (frame_idx / fps) * 1000
            
            # YOLO Pose 추론
            results = self.model(frame, verbose=False)
            
            # Skeleton 데이터 추출
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data.cpu().numpy()
                
                if len(keypoints) > 0:
                    # 첫 번째 사람만 사용 (다중 인물 감지 시)
                    kp = keypoints[0]  # shape: (17, 3) -> x, y, confidence
                    
                    # 가속도 데이터 동기화 (가장 가까운 timestamp 찾기)
                    acc_row = acc_df.iloc[(acc_df['timestamp'] - timestamp_ms).abs().argsort()[:1]]
                    
                    # 데이터 행 생성
                    row = {
                        'frame_id': frame_idx,
                        'timestamp_ms': timestamp_ms,
                    }
                    
                    # 17개 keypoint 추가
                    for i, kp_name in enumerate(self.keypoint_names):
                        row[f'{kp_name}_x'] = kp[i, 0]
                        row[f'{kp_name}_y'] = kp[i, 1]
                        row[f'{kp_name}_conf'] = kp[i, 2]
                    
                    # 가속도 데이터 추가
                    if len(acc_row) > 0:
                        row['acc_x'] = acc_row['acc_x'].values[0]
                        row['acc_y'] = acc_row['acc_y'].values[0]
                        row['acc_z'] = acc_row['acc_z'].values[0]
                        row['acc_mag'] = acc_row['acc_mag'].values[0]
                    else:
                        row['acc_x'] = 0
                        row['acc_y'] = 0
                        row['acc_z'] = 0
                        row['acc_mag'] = 0
                    
                    results_data.append(row)
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # DataFrame으로 변환 및 저장
        if len(results_data) > 0:
            df_result = pd.DataFrame(results_data)
            output_file = self.output_dir / f"{video_name.replace('-cam1', '')}-skeleton.csv"
            df_result.to_csv(output_file, index=False)
            print(f"✅ 저장 완료: {output_file}")
            print(f"   총 프레임: {len(df_result)}, Keypoints: 17개, Features: {len(df_result.columns)}")
            return df_result
        else:
            print(f"⚠️  경고: {video_name}에서 skeleton을 추출하지 못했습니다.")
            return None
    
    def process_all(self):
        """모든 동영상 처리"""
        print("=" * 60)
        print("🏠 Home Safe Solution - Skeleton Extraction Pipeline")
        print("=" * 60)
        
        # 동영상 파일 목록
        video_files = sorted(self.video_dir.glob("fall-*-cam0.mp4"))
        print(f"\n📂 찾은 동영상: {len(video_files)}개")
        
        success_count = 0
        
        for video_file in video_files:
            # 대응하는 가속도 파일 찾기
            video_num = video_file.stem.split('-')[1]  # fall-01-cam0 -> 01
            accel_file = self.accel_dir / f"fall-{video_num}-acc.csv"
            
            if not accel_file.exists():
                print(f"⚠️  경고: {accel_file.name}을 찾을 수 없습니다. 스킵합니다.")
                continue
            
            # 처리
            result = self.process_video(video_file, accel_file)
            if result is not None:
                success_count += 1
        
        print("\n" + "=" * 60)
        print(f"✅ 완료! 성공: {success_count}/{len(video_files)}")
        print(f"📁 출력 경로: {self.output_dir}")
        print("=" * 60)


def main():
    # 경로 설정
    video_dir = '/home/gjkong/dev_ws/yolo/myproj/data'
    accel_dir = '/home/gjkong/dev_ws/yolo/myproj/accel'
    output_dir = '/home/gjkong/dev_ws/yolo/myproj/skeleton'
    
    # Skeleton 추출 실행
    extractor = SkeletonExtractor(video_dir, accel_dir, output_dir)
    extractor.process_all()


if __name__ == "__main__":
    main()