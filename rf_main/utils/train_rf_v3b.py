"""
RF 모델 재학습 v3b — 좌표 정규화 (버그 수정)
- ⭐ conf < 0.3인 keypoint는 정규화 시 0으로 처리
- ⭐ 최적 모델 선택: Recall 우선 (낙상 놓치면 위험)
- 동영상 단위 train/test split

Usage:
    python train_rf_v3b.py --test     # 소량 테스트
    python train_rf_v3b.py            # 전체 실행
    python train_rf_v3b.py --train-only  # 전처리 건너뛰기
"""

import cv2
import numpy as np
import pandas as pd
import joblib
import os
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from ultralytics import YOLO
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


# ===== 설정 =====
BASE_DIR = str(PATHS.RF_MAIN)
# BASE_DIR = '/home/gjkong/dev_ws/yolo/myproj'
NEW_DATA_DIR = os.path.join(BASE_DIR, 'new_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models_integrated', 'binary_v3')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11s-pose.pt')
FEATURE_DIR = os.path.join(NEW_DATA_DIR, 'features_normalized')

KP_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

CONF_THRESHOLD = 0.3
META_COLS = ['label', 'source_file', 'frame_num']


def calc_angle(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_normalized_features(keypoints, prev_kp=None, prev2_kp=None, feature_history=None):
    """
    ⭐ 정규화된 181개 Feature 추출 (v3b: 버그 수정)
    - conf < 0.3인 keypoint는 x=0, y=0으로 처리
    - 모든 좌표를 bbox 기준 0~1로 클램핑
    """
    features = {}
    
    # ===== bbox 계산 =====
    valid = keypoints[:, 2] > CONF_THRESHOLD
    if np.any(valid):
        xs = keypoints[valid, 0]
        ys = keypoints[valid, 1]
        bbox_x_min = float(np.min(xs))
        bbox_y_min = float(np.min(ys))
        bbox_w = float(np.max(xs) - bbox_x_min)
        bbox_h = float(np.max(ys) - bbox_y_min)
        if bbox_w < 1: bbox_w = 1.0
        if bbox_h < 1: bbox_h = 1.0
    else:
        bbox_x_min, bbox_y_min = 0.0, 0.0
        bbox_w, bbox_h = 1.0, 1.0
    
    # ===== 정규화 (0~1 클램핑) =====
    kp_norm = np.zeros((17, 3))
    for i in range(17):
        if keypoints[i][2] > CONF_THRESHOLD:
            kp_norm[i][0] = np.clip((keypoints[i][0] - bbox_x_min) / bbox_w, 0, 1)
            kp_norm[i][1] = np.clip((keypoints[i][1] - bbox_y_min) / bbox_h, 0, 1)
        else:
            kp_norm[i][0] = 0.0
            kp_norm[i][1] = 0.0
        kp_norm[i][2] = float(keypoints[i][2])
    
    # prev 정규화 (현재 bbox 기준)
    def norm_prev(kp):
        if kp is None:
            return None
        normed = np.zeros((17, 3))
        for i in range(17):
            if kp[i][2] > CONF_THRESHOLD:
                normed[i][0] = np.clip((kp[i][0] - bbox_x_min) / bbox_w, 0, 1)
                normed[i][1] = np.clip((kp[i][1] - bbox_y_min) / bbox_h, 0, 1)
            else:
                normed[i][0] = 0.0
                normed[i][1] = 0.0
            normed[i][2] = float(kp[i][2])
        return normed
    
    prev_norm = norm_prev(prev_kp)
    prev2_norm = norm_prev(prev2_kp)
    
    # ===== 1~51: 정규화된 keypoint =====
    for i, name in enumerate(KP_NAMES):
        features[f'{name}_x'] = float(kp_norm[i][0])
        features[f'{name}_y'] = float(kp_norm[i][1])
        features[f'{name}_conf'] = float(kp_norm[i][2])
    
    # ===== 52~55: 가속도 =====
    features['acc_x'] = 0.0
    features['acc_y'] = 0.0
    features['acc_z'] = 0.0
    features['acc_mag'] = 0.0
    
    # ===== 56~60: 각도 (원본 좌표 사용 — 각도는 스케일 불변) =====
    features['left_elbow_angle'] = calc_angle(keypoints[5], keypoints[7], keypoints[9])
    features['right_elbow_angle'] = calc_angle(keypoints[6], keypoints[8], keypoints[10])
    features['left_knee_angle'] = calc_angle(keypoints[11], keypoints[13], keypoints[15])
    features['right_knee_angle'] = calc_angle(keypoints[12], keypoints[14], keypoints[16])
    
    shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
    hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
    vertical = np.array([hip_mid[0], hip_mid[1] - 100])
    features['spine_angle'] = calc_angle(shoulder_mid, hip_mid, vertical)
    
    # ===== 61~68: 정규화된 높이/비율 =====
    hip_mid_n = (kp_norm[11][:2] + kp_norm[12][:2]) / 2
    shoulder_mid_n = (kp_norm[5][:2] + kp_norm[6][:2]) / 2
    
    features['hip_height'] = float(hip_mid_n[1])
    features['shoulder_height'] = float(shoulder_mid_n[1])
    features['head_height'] = float(kp_norm[0][1])
    
    features['bbox_width'] = float(bbox_w / (bbox_w + bbox_h))
    features['bbox_height'] = float(bbox_h / (bbox_w + bbox_h))
    features['bbox_aspect_ratio'] = float(bbox_w / bbox_h)
    
    features['shoulder_tilt'] = float(abs(kp_norm[5][1] - kp_norm[6][1]))
    features['avg_confidence'] = float(np.mean(keypoints[:, 2]))
    
    # ===== 69~170: 정규화된 속도/가속도 =====
    for i, name in enumerate(KP_NAMES):
        if prev_norm is not None and kp_norm[i][2] > CONF_THRESHOLD and prev_norm[i][2] > CONF_THRESHOLD:
            vx = float(kp_norm[i][0] - prev_norm[i][0])
            vy = float(kp_norm[i][1] - prev_norm[i][1])
        else:
            vx, vy = 0.0, 0.0
        
        speed = float(np.sqrt(vx**2 + vy**2))
        features[f'{name}_vx'] = vx
        features[f'{name}_vy'] = vy
        features[f'{name}_speed'] = speed
        
        if (prev2_norm is not None and prev_norm is not None and
            kp_norm[i][2] > CONF_THRESHOLD and prev_norm[i][2] > CONF_THRESHOLD and prev2_norm[i][2] > CONF_THRESHOLD):
            prev_vx = float(prev_norm[i][0] - prev2_norm[i][0])
            prev_vy = float(prev_norm[i][1] - prev2_norm[i][1])
            ax = vx - prev_vx
            ay = vy - prev_vy
        else:
            ax, ay = 0.0, 0.0
        
        features[f'{name}_ax'] = ax
        features[f'{name}_ay'] = ay
        features[f'{name}_accel'] = float(np.sqrt(ax**2 + ay**2))
    
    # ===== 171~172 =====
    features['hip_velocity'] = (features.get('left_hip_speed', 0) + features.get('right_hip_speed', 0)) / 2
    features['hip_acceleration'] = (features.get('left_hip_accel', 0) + features.get('right_hip_accel', 0)) / 2
    
    # ===== 173~181: 시계열 =====
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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
        results = yolo_model(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_all = results[0].keypoints.data.cpu().numpy()
            
            if len(keypoints_all) > 0:
                if len(keypoints_all) > 1:
                    areas = []
                    for kp in keypoints_all:
                        v = kp[:, 2] > 0.3
                        if np.any(v):
                            area = (np.max(kp[v, 0]) - np.min(kp[v, 0])) * (np.max(kp[v, 1]) - np.min(kp[v, 1]))
                        else:
                            area = 0
                        areas.append(area)
                    target_idx = np.argmax(areas)
                else:
                    target_idx = 0
                
                keypoints = keypoints_all[target_idx]
                features = extract_normalized_features(keypoints, prev_kp, prev2_kp, feature_history)
                features['label'] = label
                features['source_file'] = os.path.basename(video_path)
                features['frame_num'] = frame_count
                
                all_features.append(features)
                prev2_kp = prev_kp.copy() if prev_kp is not None else None
                prev_kp = keypoints.copy()
    
    cap.release()
    return all_features


def preprocess_all(test_mode=False):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    
    print("🔄 YOLO 모델 로딩...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 로드 완료")
    
    folders = {
        'normal': {'path': os.path.join(NEW_DATA_DIR, 'normal'), 'label': 0},
        'fallen': {'path': os.path.join(NEW_DATA_DIR, 'fallen'), 'label': 1},
    }
    
    limits = {'normal': 5, 'fallen': 3} if test_mode else {'normal': 0, 'fallen': 0}
    
    for folder_name, info in folders.items():
        videos = sorted([f for f in os.listdir(info['path']) if f.endswith(('.avi', '.mp4'))])
        limit = limits.get(folder_name, 0)
        if limit > 0:
            videos = videos[:limit]
        
        print(f"\n{'='*60}")
        print(f"📁 {folder_name} ({len(videos)}개, label={info['label']})")
        print(f"{'='*60}")
        
        all_data = []
        for idx, vf in enumerate(videos):
            start = time.time()
            fl = process_video(os.path.join(info['path'], vf), yolo_model, info['label'])
            all_data.extend(fl)
            if (idx + 1) % 50 == 0 or idx == len(videos) - 1:
                print(f"  [{idx+1}/{len(videos)}] {vf}: {len(fl)}프레임, {time.time()-start:.1f}초")
        
        if all_data:
            df = pd.DataFrame(all_data)
            path = os.path.join(FEATURE_DIR, f'{folder_name}_features.csv')
            df.to_csv(path, index=False)
            print(f"✅ {folder_name}: {len(df)}행, {len(df.columns)}열")
            
            # ⭐ 정규화 범위 검증
            for k in ['hip_height', 'head_height', 'shoulder_height', 'nose_x', 'nose_y']:
                mn, mx = df[k].min(), df[k].max()
                print(f"   {k}: min={mn:.3f}, max={mx:.3f} {'✅' if 0 <= mn and mx <= 1 else '⚠️'}")


def train_model():
    print("\n" + "="*60)
    print("🔧 모델 학습 시작")
    print("="*60)
    
    normal_df = pd.read_csv(os.path.join(FEATURE_DIR, 'normal_features.csv'))
    fallen_df = pd.read_csv(os.path.join(FEATURE_DIR, 'fallen_features.csv'))
    df = pd.concat([normal_df, fallen_df], ignore_index=True)
    
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    print(f"   Normal: {len(normal_df)}행 ({normal_df['source_file'].nunique()}영상)")
    print(f"   Fallen: {len(fallen_df)}행 ({fallen_df['source_file'].nunique()}영상)")
    print(f"   Feature: {len(feature_cols)}개")
    
    # 정규화 검증
    print(f"\n📊 정규화 검증:")
    for k in ['hip_height', 'bbox_aspect_ratio', 'spine_angle', 'head_height']:
        nm, fm = normal_df[k].mean(), fallen_df[k].mean()
        print(f"   {k}: Normal={nm:.3f}, Fallen={fm:.3f}")
    
    # 동영상 단위 split
    video_labels = df.groupby('source_file')['label'].first()
    normal_videos = video_labels[video_labels == 0].index.tolist()
    fallen_videos = video_labels[video_labels == 1].index.tolist()
    
    np.random.seed(42)
    n_test_n = max(1, int(len(normal_videos) * 0.2))
    n_test_f = max(1, int(len(fallen_videos) * 0.2))
    
    test_videos = set(
        list(np.random.choice(normal_videos, n_test_n, replace=False)) +
        list(np.random.choice(fallen_videos, n_test_f, replace=False))
    )
    
    train_df = df[~df['source_file'].isin(test_videos)]
    test_df = df[df['source_file'].isin(test_videos)]
    
    print(f"\n📋 동영상 단위 Split:")
    print(f"   Train: {len(train_df)}행 (Normal={sum(train_df['label']==0)}, Fallen={sum(train_df['label']==1)})")
    print(f"   Test:  {len(test_df)}행 (Normal={sum(test_df['label']==0)}, Fallen={sum(test_df['label']==1)})")
    
    # 전략 비교
    strategies = [
        ("Balanced", 'none', 0, dict(n_estimators=100, class_weight='balanced', max_depth=20)),
        ("Undersample 5:1", 'undersample', 5, dict(n_estimators=100, max_depth=20)),
        ("Undersample 3:1 + Bal", 'undersample', 3, dict(n_estimators=200, class_weight='balanced', max_depth=25)),
        ("Hybrid 5:1 + Bal", 'hybrid', 5, dict(n_estimators=150, class_weight='balanced', max_depth=20)),
    ]
    
    results = []
    
    for name, bal_method, bal_ratio, rf_params in strategies:
        print(f"\n--- {name} ---")
        
        if bal_method == 'undersample':
            n_f = sum(train_df['label'] == 1)
            ns = train_df[train_df['label']==0].sample(n=min(n_f*bal_ratio, sum(train_df['label']==0)), random_state=42)
            bal_df = pd.concat([ns, train_df[train_df['label']==1]])
        elif bal_method == 'hybrid':
            n_f = sum(train_df['label'] == 1)
            ns = train_df[train_df['label']==0].sample(n=min(n_f*bal_ratio, sum(train_df['label']==0)), random_state=42)
            fo = train_df[train_df['label']==1].sample(n=n_f*2, replace=True, random_state=42)
            bal_df = pd.concat([ns, fo])
        else:
            bal_df = train_df
        
        X_train = bal_df[feature_cols].fillna(0)
        y_train = bal_df['label'].values
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['label'].values
        
        rf = RandomForestClassifier(n_jobs=1, random_state=42, verbose=0, **rf_params)
        start = time.time()
        rf.fit(X_train, y_train)
        t = time.time() - start
        
        y_pred = rf.predict(X_test)
        fi = list(rf.classes_).index(1)
        y_proba = rf.predict_proba(X_test)[:, fi]
        
        acc = np.mean(y_pred == y_test) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        auc = roc_auc_score(y_test, y_proba) * 100 if len(np.unique(y_test)) > 1 else 0
        cm = confusion_matrix(y_test, y_pred)
        fp = cm[0][1]/(cm[0][0]+cm[0][1])*100 if (cm[0][0]+cm[0][1])>0 else 0
        fn = cm[1][0]/(cm[1][0]+cm[1][1])*100 if (cm[1][0]+cm[1][1])>0 else 0
        
        print(f"  Acc={acc:.1f}% F1={f1:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% AUC={auc:.1f}% FP={fp:.1f}% FN={fn:.1f}% ({t:.1f}s)")
        print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        
        results.append({'name': name, 'model': rf, 'acc': acc, 'f1': f1, 'prec': prec,
                        'rec': rec, 'auc': auc, 'fp': fp, 'fn': fn, 'cm': cm, 'cols': feature_cols})
    
    # 요약
    print(f"\n{'='*80}")
    print(f"📊 전략 비교 (정규화 v3b)")
    print(f"{'='*80}")
    print(f"{'전략':<28} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'FP%':>5} {'FN%':>5}")
    print("-"*75)
    for r in results:
        print(f"{r['name']:<28} {r['acc']:>5.1f}% {r['f1']:>5.1f}% {r['prec']:>5.1f}% "
              f"{r['rec']:>5.1f}% {r['auc']:>5.1f}% {r['fp']:>4.1f}% {r['fn']:>4.1f}%")
    
    # ⭐ Recall 우선 선택 (낙상 시스템 → 놓치면 위험)
    # Recall > 80% 중에서 F1이 높은 것
    high_recall = [r for r in results if r['rec'] >= 80]
    if high_recall:
        best = max(high_recall, key=lambda r: r['f1'])
    else:
        best = max(results, key=lambda r: (r['rec'], r['f1']))
    
    print(f"\n🏆 최적 (Recall 우선): {best['name']}")
    print(f"   F1={best['f1']:.1f}%, Recall={best['rec']:.1f}%, Precision={best['prec']:.1f}%")
    
    # 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUT_DIR, 'random_forest_model.pkl')
    joblib.dump(best['model'], model_path)
    
    feature_path = os.path.join(OUTPUT_DIR, 'feature_columns.txt')
    with open(feature_path, 'w') as f:
        f.write(f"Total Features: {len(best['cols'])}\n")
        f.write("="*60 + "\n")
        for i, col in enumerate(best['cols']):
            f.write(f"{i+1}. {col}\n")
    
    report_path = os.path.join(OUTPUT_DIR, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("RF v3b 리포트 (정규화 + Recall 우선)\n")
        f.write(f"날짜: 2026-02-07\n\n")
        f.write(f"{'전략':<28} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'FP%':>5} {'FN%':>5}\n")
        f.write("-"*75 + "\n")
        for r in results:
            f.write(f"{r['name']:<28} {r['acc']:>5.1f}% {r['f1']:>5.1f}% {r['prec']:>5.1f}% "
                    f"{r['rec']:>5.1f}% {r['auc']:>5.1f}% {r['fp']:>4.1f}% {r['fn']:>4.1f}%\n")
        f.write(f"\n최적: {best['name']} (Recall={best['rec']:.1f}%, F1={best['f1']:.1f}%)\n")
    
    print(f"\n💾 저장: {model_path}")
    print(f"   feature_names_in_: {hasattr(best['model'], 'feature_names_in_')}")
    print(f"   classes: {best['model'].classes_}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    
    total_start = time.time()
    
    if not args.train_only:
        print("="*60)
        print("📦 STEP 1: 전처리 (정규화 v3b)")
        print("="*60)
        preprocess_all(test_mode=args.test)
    
    print("\n" + "="*60)
    print("📦 STEP 2: 학습")
    print("="*60)
    train_model()
    
    print(f"\n🏁 전체 완료! ({time.time()-total_start:.1f}초)")


if __name__ == '__main__':
    main()
