#!/usr/bin/env python3
"""
============================================================
ST-GCN 데이터 준비 — 동영상 → YOLO Keypoint → npy 변환
============================================================
RF와 동일한 데이터(normal 1629 + fallen 301)를 ST-GCN 입력 형식으로 변환

입력:
  /home/gjkong/dev_ws/yolo/myproj/new_data/normal/*.avi  (읽기만)
  /home/gjkong/dev_ws/yolo/myproj/new_data/fallen/*.mp4  (읽기만)

출력:
  /home/gjkong/dev_ws/st_gcn/data/binary_v2/
    ├── train_data.npy      (N, 3, 60, 17, 1)
    ├── train_labels.npy    (N,)
    ├── test_data.npy
    ├── test_labels.npy
    └── video_info.pkl

정규화: 기존 ST-GCN과 동일 (hip center 기준, max distance 스케일링)
"""

import os
import sys
import numpy as np
import pickle
import time
from pathlib import Path
from collections import Counter
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS

# YOLO
from ultralytics import YOLO

# ============================================================
# 설정
# ============================================================

# RF 데이터 경로 (읽기만!)
RF_DATA_DIR = PATHS.NEW_DATA_DIR
# RF_DATA_DIR = Path("/home/gjkong/dev_ws/yolo/myproj/new_data")
NORMAL_DIR = RF_DATA_DIR / "normal"
FALLEN_DIR = RF_DATA_DIR / "fallen"

# ST-GCN 출력 경로
OUTPUT_DIR = PATHS.STGCN_DATA_V2
# OUTPUT_DIR = Path("/home/gjkong/dev_ws/st_gcn/data/binary_v2")

# YOLO 모델
YOLO_MODEL_PATH = str(PATHS.YOLO_MODEL_N)
# YOLO_MODEL_PATH = "/home/gjkong/dev_ws/yolo/myproj/models/yolo11n-pose.pt"

# 시퀀스 설정
SEQ_LEN = 60       # 60프레임 (약 2-3초)
STRIDE = 30         # 50% overlap
TEST_RATIO = 0.2    # 동영상 단위 20% 테스트
RANDOM_SEED = 42

# ============================================================
# YOLO Keypoint 추출
# ============================================================

def extract_keypoints_from_video(yolo_model, video_path):
    """
    동영상에서 YOLO Pose로 keypoints 추출
    
    Returns:
        keypoints: (T, 17, 3) — x, y, confidence
        None if failed
    """
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    all_keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Pose 추론
        results = yolo_model(frame, verbose=False)
        
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            
            if kps.data is not None and len(kps.data) > 0:
                # 가장 큰 사람 선택 (bbox 면적 기준)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_idx = areas.argmax()
                else:
                    best_idx = 0
                
                kp = kps.data[best_idx].cpu().numpy()  # (17, 3)
                
                if kp.shape == (17, 3):
                    all_keypoints.append(kp)
                else:
                    # keypoint 감지 실패 → 0으로 채움
                    all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
            else:
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
        else:
            all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
    
    cap.release()
    
    if len(all_keypoints) == 0:
        return None
    
    return np.array(all_keypoints, dtype=np.float32)  # (T, 17, 3)


# ============================================================
# 정규화 (기존 ST-GCN과 동일)
# ============================================================

def normalize_skeleton(seq):
    """
    기존 01_prepare_data.py와 동일한 정규화
    
    1. Hip center (keypoint 11, 12)를 원점으로
    2. 최대 거리로 스케일링 → -1 ~ 1
    
    Args:
        seq: (3, T, 17, 1) — C, T, V, M
    
    Returns:
        normalized seq: (3, T, 17, 1)
    """
    C, T, V, M = seq.shape
    
    # Hip center (left_hip=11, right_hip=12)
    left_hip = seq[:2, :, 11, :]    # (2, T, M)
    right_hip = seq[:2, :, 12, :]
    hip_center = (left_hip + right_hip) / 2  # (2, T, M)
    
    # Center: x, y를 hip center 기준으로 이동
    seq[:2, :, :, :] -= hip_center[:, :, np.newaxis, :]
    
    # Scale: 최대 거리로 나누기
    max_dist = np.abs(seq[:2, :, :, :]).max()
    if max_dist > 0:
        seq[:2, :, :, :] /= max_dist
    
    return seq


# ============================================================
# 시퀀스 생성
# ============================================================

def create_sequences(keypoints, label, seq_len=SEQ_LEN, stride=STRIDE):
    """
    동영상의 keypoints를 고정 길이 시퀀스로 분할
    
    Args:
        keypoints: (T, 17, 3)
        label: 0 (Normal) or 1 (Fallen)
        seq_len: 시퀀스 길이
        stride: 슬라이딩 윈도우 이동량
    
    Returns:
        sequences: list of (3, seq_len, 17, 1)
        labels: list of int
    """
    T = keypoints.shape[0]
    
    if T < seq_len:
        # 프레임이 부족하면 패딩 (마지막 프레임 반복)
        pad_len = seq_len - T
        padding = np.tile(keypoints[-1:], (pad_len, 1, 1))
        keypoints = np.concatenate([keypoints, padding], axis=0)
        T = seq_len
    
    sequences = []
    labels = []
    
    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        seq = keypoints[start:end]  # (seq_len, 17, 3)
        
        # (T, V, C) → (C, T, V, M)
        seq = seq.transpose(2, 0, 1)       # (3, seq_len, 17)
        seq = seq[..., np.newaxis]          # (3, seq_len, 17, 1)
        
        # 정규화
        seq = normalize_skeleton(seq.copy())
        
        sequences.append(seq)
        labels.append(label)
    
    return sequences, labels


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 60)
    print("  ST-GCN 데이터 준비 (RF 동일 데이터)")
    print("  동영상 → YOLO Keypoint → 60프레임 시퀀스 → npy")
    print("=" * 60)
    
    # 출력 폴더 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # YOLO 모델 로드
    print("\n🔄 YOLO 모델 로딩...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 로드 완료")
    
    # ── 동영상 목록 ──
    normal_videos = sorted(list(NORMAL_DIR.glob("*.avi")) + list(NORMAL_DIR.glob("*.mp4")))
    fallen_videos = sorted(list(FALLEN_DIR.glob("*.avi")) + list(FALLEN_DIR.glob("*.mp4")))
    
    print(f"\n📁 Normal: {len(normal_videos)}개")
    print(f"📁 Fallen: {len(fallen_videos)}개")
    
    # ── 동영상 단위 Split (RF와 동일한 seed) ──
    np.random.seed(RANDOM_SEED)
    
    n_test_n = max(1, int(len(normal_videos) * TEST_RATIO))
    n_test_f = max(1, int(len(fallen_videos) * TEST_RATIO))
    
    test_normal_idx = set(np.random.choice(len(normal_videos), n_test_n, replace=False))
    test_fallen_idx = set(np.random.choice(len(fallen_videos), n_test_f, replace=False))
    
    print(f"\n📋 Split (seed={RANDOM_SEED}):")
    print(f"   Normal — Train: {len(normal_videos)-n_test_n}, Test: {n_test_n}")
    print(f"   Fallen — Train: {len(fallen_videos)-n_test_f}, Test: {n_test_f}")
    
    # ── 처리 ──
    train_seqs, train_labels = [], []
    test_seqs, test_labels = [], []
    video_info = []
    
    start_time = time.time()
    
    # --- Normal ---
    print(f"\n{'=' * 60}")
    print(f"📁 Normal ({len(normal_videos)}개, label=0)")
    print(f"{'=' * 60}")
    
    for i, vpath in enumerate(normal_videos):
        if (i + 1) % 50 == 0 or (i + 1) == len(normal_videos):
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(normal_videos)}] {vpath.name} ({elapsed:.0f}s)")
        
        kps = extract_keypoints_from_video(yolo_model, vpath)
        if kps is None or len(kps) < 10:
            continue
        
        seqs, lbls = create_sequences(kps, label=0)
        
        if i in test_normal_idx:
            test_seqs.extend(seqs)
            test_labels.extend(lbls)
        else:
            train_seqs.extend(seqs)
            train_labels.extend(lbls)
        
        video_info.append({
            'video': vpath.name, 'label': 0, 'frames': len(kps),
            'sequences': len(seqs), 'split': 'test' if i in test_normal_idx else 'train'
        })
    
    print(f"✅ Normal 완료: train={sum(1 for l in train_labels if l==0)}, test={sum(1 for l in test_labels if l==0)} 시퀀스")
    
    # --- Fallen ---
    print(f"\n{'=' * 60}")
    print(f"📁 Fallen ({len(fallen_videos)}개, label=1)")
    print(f"{'=' * 60}")
    
    for i, vpath in enumerate(fallen_videos):
        if (i + 1) % 10 == 0 or (i + 1) == len(fallen_videos):
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(fallen_videos)}] {vpath.name} ({elapsed:.0f}s)")
        
        kps = extract_keypoints_from_video(yolo_model, vpath)
        if kps is None or len(kps) < 10:
            continue
        
        seqs, lbls = create_sequences(kps, label=1)
        
        if i in test_fallen_idx:
            test_seqs.extend(seqs)
            test_labels.extend(lbls)
        else:
            train_seqs.extend(seqs)
            train_labels.extend(lbls)
        
        video_info.append({
            'video': vpath.name, 'label': 1, 'frames': len(kps),
            'sequences': len(seqs), 'split': 'test' if i in test_fallen_idx else 'train'
        })
    
    print(f"✅ Fallen 완료: train={sum(1 for l in train_labels if l==1)}, test={sum(1 for l in test_labels if l==1)} 시퀀스")
    
    # ── NumPy 변환 ──
    train_data = np.array(train_seqs, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_data = np.array(test_seqs, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    # ── 검증 ──
    print(f"\n{'=' * 60}")
    print(f"📊 데이터 검증")
    print(f"{'=' * 60}")
    print(f"  Train: {train_data.shape} (Normal={sum(train_labels==0)}, Fallen={sum(train_labels==1)})")
    print(f"  Test:  {test_data.shape} (Normal={sum(test_labels==0)}, Fallen={sum(test_labels==1)})")
    
    # 정규화 범위 확인
    for name, data in [("Train", train_data), ("Test", test_data)]:
        print(f"\n  {name} 정규화 범위:")
        for c, ch_name in enumerate(["x", "y", "conf"]):
            print(f"    {ch_name}: min={data[:,c].min():.4f}, max={data[:,c].max():.4f}, mean={data[:,c].mean():.4f}")
    
    # ── 저장 ──
    np.save(OUTPUT_DIR / "train_data.npy", train_data)
    np.save(OUTPUT_DIR / "train_labels.npy", train_labels)
    np.save(OUTPUT_DIR / "test_data.npy", test_data)
    np.save(OUTPUT_DIR / "test_labels.npy", test_labels)
    
    with open(OUTPUT_DIR / "video_info.pkl", "wb") as f:
        pickle.dump(video_info, f)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ 저장 완료! ({elapsed:.0f}초)")
    print(f"   {OUTPUT_DIR}/")
    print(f"   train_data.npy:   {train_data.shape}")
    print(f"   train_labels.npy: {train_labels.shape}")
    print(f"   test_data.npy:    {test_data.shape}")
    print(f"   test_labels.npy:  {test_labels.shape}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
