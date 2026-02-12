"""
동영상 → 181개 Feature CSV 전처리 스크립트
- YOLO Pose로 keypoint 추출
- 181개 feature 계산 (기존 binary 모델과 동일)
- 가속도 센서 feature = 0 (센서 없음)

Usage:
    # 소량 테스트 (normal 3개 + fallen 2개)
    python preprocess_videos.py --test
    
    # 전체 실행
    python preprocess_videos.py
    
    # 특정 폴더만
    python preprocess_videos.py --folder normal --limit 10
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


# ===== 설정 =====
BASE_DIR = str(PATHS.RF_MAIN)
# BASE_DIR = '/home/gjkong/dev_ws/yolo/myproj'
NEW_DATA_DIR = os.path.join(BASE_DIR, 'new_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'new_data', 'features')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11s-pose.pt')

# COCO 17 keypoint 이름
KP_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def calc_angle(a, b, c):
    """세 점의 각도 계산 (degree)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_features(keypoints, prev_kp=None, prev2_kp=None, feature_history=None):
    """
    단일 프레임의 keypoints에서 181개 feature 추출
    
    Args:
        keypoints: (17, 3) array - x, y, conf
        prev_kp: 이전 프레임 keypoints
        prev2_kp: 2프레임 전 keypoints
        feature_history: 시계열 통계용 히스토리 리스트
    
    Returns:
        features: dict (181개)
    """
    features = {}
    
    # ===== 1~51: keypoint x, y, conf =====
    for i, name in enumerate(KP_NAMES):
        features[f'{name}_x'] = float(keypoints[i][0])
        features[f'{name}_y'] = float(keypoints[i][1])
        features[f'{name}_conf'] = float(keypoints[i][2])
    
    # ===== 52~55: 가속도 (센서 없음 → 0) =====
    features['acc_x'] = 0.0
    features['acc_y'] = 0.0
    features['acc_z'] = 0.0
    features['acc_mag'] = 0.0
    
    # ===== 56~60: 관절 각도 =====
    features['left_elbow_angle'] = calc_angle(keypoints[5], keypoints[7], keypoints[9])
    features['right_elbow_angle'] = calc_angle(keypoints[6], keypoints[8], keypoints[10])
    features['left_knee_angle'] = calc_angle(keypoints[11], keypoints[13], keypoints[15])
    features['right_knee_angle'] = calc_angle(keypoints[12], keypoints[14], keypoints[16])
    
    shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
    hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
    vertical = np.array([hip_mid[0], hip_mid[1] - 100])
    features['spine_angle'] = calc_angle(shoulder_mid, hip_mid, vertical)
    
    # ===== 61~68: 높이/bbox/기타 =====
    features['hip_height'] = float(hip_mid[1])
    features['shoulder_height'] = float(shoulder_mid[1])
    features['head_height'] = float(keypoints[0][1])
    
    valid = keypoints[:, 2] > 0.3
    if np.any(valid):
        xs = keypoints[valid, 0]
        ys = keypoints[valid, 1]
        features['bbox_width'] = float(np.max(xs) - np.min(xs))
        features['bbox_height'] = float(np.max(ys) - np.min(ys))
        features['bbox_aspect_ratio'] = features['bbox_width'] / (features['bbox_height'] + 1e-6)
    else:
        features['bbox_width'] = 0.0
        features['bbox_height'] = 0.0
        features['bbox_aspect_ratio'] = 1.0
    
    features['shoulder_tilt'] = float(abs(keypoints[5][1] - keypoints[6][1]))
    features['avg_confidence'] = float(np.mean(keypoints[:, 2]))
    
    # ===== 69~170: 속도/가속도 =====
    for i, name in enumerate(KP_NAMES):
        if prev_kp is not None:
            vx = float(keypoints[i][0] - prev_kp[i][0])
            vy = float(keypoints[i][1] - prev_kp[i][1])
        else:
            vx, vy = 0.0, 0.0
        
        speed = float(np.sqrt(vx**2 + vy**2))
        features[f'{name}_vx'] = vx
        features[f'{name}_vy'] = vy
        features[f'{name}_speed'] = speed
        
        if prev2_kp is not None and prev_kp is not None:
            prev_vx = float(prev_kp[i][0] - prev2_kp[i][0])
            prev_vy = float(prev_kp[i][1] - prev2_kp[i][1])
            ax = vx - prev_vx
            ay = vy - prev_vy
        else:
            ax, ay = 0.0, 0.0
        
        features[f'{name}_ax'] = ax
        features[f'{name}_ay'] = ay
        features[f'{name}_accel'] = float(np.sqrt(ax**2 + ay**2))
    
    # ===== 171~172: hip 속도/가속도 =====
    features['hip_velocity'] = (features.get('left_hip_speed', 0) + features.get('right_hip_speed', 0)) / 2
    features['hip_acceleration'] = (features.get('left_hip_accel', 0) + features.get('right_hip_accel', 0)) / 2
    
    # ===== 173~181: 시계열 통계 (5프레임 윈도우) =====
    if feature_history is not None:
        feature_history.append({
            'hip_height': features['hip_height'],
            'shoulder_height': features['shoulder_height'],
            'head_height': features['head_height'],
            'acc_mag': features['acc_mag'],
        })
        if len(feature_history) > 5:
            del feature_history[:-5]
    
    hist = feature_history if feature_history else [{'hip_height': 0, 'shoulder_height': 0, 'head_height': 0, 'acc_mag': 0}]
    
    for key in ['hip_height', 'shoulder_height', 'head_height']:
        vals = [h[key] for h in hist]
        features[f'{key}_mean_5'] = float(np.mean(vals))
        features[f'{key}_std_5'] = float(np.std(vals))
    
    features['acc_mag_diff'] = 0.0
    vals = [h['acc_mag'] for h in hist]
    features['acc_mag_mean_5'] = float(np.mean(vals))
    features['acc_mag_std_5'] = float(np.std(vals))
    
    return features


def process_video(video_path, yolo_model, label):
    """
    단일 동영상 처리 → feature list 반환
    
    Args:
        video_path: 동영상 경로
        yolo_model: YOLO 모델
        label: 0 (Normal) or 1 (Fallen)
    
    Returns:
        list of dict (각 프레임의 181개 feature + label)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️ 열기 실패: {video_path}")
        return []
    
    all_features = []
    prev_kp = None
    prev2_kp = None
    feature_history = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO 추론
        results = yolo_model(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_all = results[0].keypoints.data.cpu().numpy()
            
            if len(keypoints_all) > 0:
                # 가장 큰 사람 선택 (bbox 면적 기준)
                if len(keypoints_all) > 1:
                    areas = []
                    for kp in keypoints_all:
                        valid = kp[:, 2] > 0.3
                        if np.any(valid):
                            xs = kp[valid, 0]
                            ys = kp[valid, 1]
                            area = (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))
                        else:
                            area = 0
                        areas.append(area)
                    target_idx = np.argmax(areas)
                else:
                    target_idx = 0
                
                keypoints = keypoints_all[target_idx]
                
                # Feature 추출
                features = extract_features(keypoints, prev_kp, prev2_kp, feature_history)
                features['label'] = label
                features['source_file'] = os.path.basename(video_path)
                features['frame_num'] = frame_count
                
                all_features.append(features)
                
                # 이전 프레임 저장
                prev2_kp = prev_kp.copy() if prev_kp is not None else None
                prev_kp = keypoints.copy()
    
    cap.release()
    return all_features


def main():
    parser = argparse.ArgumentParser(description='동영상 → Feature CSV 전처리')
    parser.add_argument('--test', action='store_true', help='소량 테스트 (normal 3 + fallen 2)')
    parser.add_argument('--folder', type=str, help='특정 폴더만 처리 (normal/fallen)')
    parser.add_argument('--limit', type=int, default=0, help='처리할 최대 파일 수')
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # YOLO 모델 로드
    print("🔄 YOLO 모델 로딩...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 모델 로드 완료")
    
    # 처리할 폴더 설정
    folders = {
        'normal': {'path': os.path.join(NEW_DATA_DIR, 'normal'), 'label': 0},
        'fallen': {'path': os.path.join(NEW_DATA_DIR, 'fallen'), 'label': 1},
    }
    
    if args.folder:
        folders = {args.folder: folders[args.folder]}
    
    # 테스트 모드
    if args.test:
        limits = {'normal': 3, 'fallen': 2}
    elif args.limit > 0:
        limits = {k: args.limit for k in folders}
    else:
        limits = {k: 0 for k in folders}  # 0 = 전체
    
    total_start = time.time()
    
    for folder_name, info in folders.items():
        folder_path = info['path']
        label = info['label']
        
        # 동영상 파일 목록
        videos = sorted([
            f for f in os.listdir(folder_path) 
            if f.endswith(('.avi', '.mp4', '.mkv'))
        ])
        
        limit = limits.get(folder_name, 0)
        if limit > 0:
            videos = videos[:limit]
        
        print(f"\n{'='*60}")
        print(f"📁 {folder_name} ({len(videos)}개 동영상, label={label})")
        print(f"{'='*60}")
        
        all_data = []
        
        for idx, video_file in enumerate(videos):
            video_path = os.path.join(folder_path, video_file)
            start = time.time()
            
            features_list = process_video(video_path, yolo_model, label)
            elapsed = time.time() - start
            
            all_data.extend(features_list)
            
            print(f"  [{idx+1}/{len(videos)}] {video_file}: "
                  f"{len(features_list)}프레임, {elapsed:.1f}초")
        
        if all_data:
            # CSV 저장
            df = pd.DataFrame(all_data)
            output_path = os.path.join(OUTPUT_DIR, f'{folder_name}_features.csv')
            df.to_csv(output_path, index=False)
            
            print(f"\n✅ {folder_name} 저장 완료!")
            print(f"   파일: {output_path}")
            print(f"   행: {len(df)}, 열: {len(df.columns)}")
            print(f"   라벨 분포: {df['label'].value_counts().to_dict()}")
        else:
            print(f"\n⚠️ {folder_name}: 추출된 데이터 없음")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 전체 완료! ({total_elapsed:.1f}초)")
    print(f"   출력 디렉토리: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
