#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Preprocessing Engine (Stage 2)
============================================================
비디오 → YOLO Pose Estimation → RF Feature / ST-GCN Sequence

파이프라인:
  1. 비디오 로드 및 FPS 샘플링
  2. YOLO Pose 추정 → 17 키포인트 추출
  3a. RF: 키포인트 → 특징 벡터 (관절 각도, 거리, 속도 등)
  3b. ST-GCN: 키포인트 → (3, 60, 17, 1) 시퀀스 텐서
  4. Train / Val / Test 분할

사용법:
    from pipeline.preprocessor import PreprocessEngine
    from pipeline.config import PreprocessConfig
    
    config = PreprocessConfig()
    engine = PreprocessEngine(config)
    result = engine.run("/path/to/raw_videos")
============================================================
"""

import os
import math
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Callable

try:
    from pipeline.config import PreprocessConfig, SUPPORTED_VIDEO_FORMATS, ST_GCN_DIR, DATASET_DIR
except ImportError:
    from config import PreprocessConfig, SUPPORTED_VIDEO_FORMATS, ST_GCN_DIR, DATASET_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# COCO 17 Keypoints 정의
# ============================================================
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 키포인트 인덱스 매핑
KP = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}


class PreprocessEngine:
    """전처리 엔진"""

    def __init__(
        self,
        config: PreprocessConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Args:
            config: 전처리 설정
            progress_callback: 진행 콜백 (current, total, message)
        """
        self.config = config
        self.progress_callback = progress_callback
        self.yolo_model = None

    def run(self, raw_video_dir: str) -> dict:
        """
        전체 전처리 파이프라인 실행

        Args:
            raw_video_dir: 원본 비디오 디렉토리 (fall/, normal/ 하위 구조)

        Returns:
            결과 요약 딕셔너리
        """
        raw_dir = Path(raw_video_dir)
        output_dir = Path(self.config.output_dir)

        logger.info("=" * 60)
        logger.info("  전처리 시작")
        logger.info("=" * 60)

        # 1. 비디오 목록 수집
        videos = self._collect_videos(raw_dir)
        if not videos:
            raise ValueError(f"비디오가 없습니다: {raw_dir}")
        logger.info(f"비디오 수집: {len(videos)}개")

        # 2. YOLO 모델 로드
        self._load_yolo()

        # 3. 각 비디오에서 키포인트 추출
        all_keypoints = []  # [(keypoints_array, label, video_name), ...]
        total = len(videos)

        for idx, (video_path, label) in enumerate(videos):
            self._emit_progress(idx, total, f"포즈 추출: {video_path.name}")
            try:
                keypoints = self._extract_keypoints(video_path)
                if keypoints is not None and len(keypoints) > 0:
                    all_keypoints.append((keypoints, label, video_path.name))
                    logger.info(f"  ✅ {video_path.name}: {len(keypoints)} frames")
                else:
                    logger.warning(f"  ⚠ {video_path.name}: 키포인트 추출 실패")
            except Exception as e:
                logger.error(f"  ❌ {video_path.name}: {e}")

        if not all_keypoints:
            raise RuntimeError("키포인트 추출된 비디오가 없습니다")

        logger.info(f"키포인트 추출 완료: {len(all_keypoints)}/{len(videos)} 비디오")

        # 4. RF 특징 추출 + 데이터셋 생성
        self._emit_progress(total, total + 2, "RF 특징 추출 중...")
        rf_result = self._build_rf_dataset(all_keypoints, output_dir / "binary")

        # 5. ST-GCN 시퀀스 생성 + 데이터셋 생성
        self._emit_progress(total + 1, total + 2, "ST-GCN 시퀀스 생성 중...")
        stgcn_result = self._build_stgcn_dataset(
            all_keypoints,
            Path(ST_GCN_DIR) / "data/binary",
        )

        self._emit_progress(total + 2, total + 2, "전처리 완료")

        logger.info("=" * 60)
        logger.info("  전처리 완료")
        logger.info(f"  RF: {rf_result.get('train_samples', 0)} train, "
                   f"{rf_result.get('test_samples', 0)} test")
        logger.info(f"  ST-GCN: {stgcn_result.get('total_sequences', 0)} sequences")
        logger.info("=" * 60)

        return {
            "total_videos": len(videos),
            "processed_videos": len(all_keypoints),
            "rf": rf_result,
            "stgcn": stgcn_result,
        }

    # ================================================================
    # 비디오 수집
    # ================================================================

    def _collect_videos(self, raw_dir: Path) -> List[Tuple[Path, int]]:
        """비디오 목록 수집 → [(path, label), ...]"""
        videos = []
        for label_name in ["fall", "normal"]:
            label_dir = raw_dir / label_name
            if not label_dir.exists():
                continue
            label_int = 1 if label_name == "fall" else 0
            for f in sorted(label_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                    videos.append((f, label_int))
        return videos

    # ================================================================
    # YOLO 포즈 추정
    # ================================================================

    def _load_yolo(self):
        """YOLO Pose 모델 로드"""
        if self.yolo_model is not None:
            return
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.config.yolo_model)
            logger.info(f"YOLO 모델 로드: {self.config.yolo_model}")
        except ImportError:
            raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics")

    def _extract_keypoints(self, video_path: Path) -> Optional[np.ndarray]:
        """
        비디오에서 프레임별 키포인트 추출

        Returns:
            np.ndarray shape (T, 17, 3) - T프레임, 17키포인트, (x, y, confidence)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        # FPS 기반 프레임 샘플링
        target_fps = self.config.target_fps
        sample_interval = max(1, round(fps / target_fps))

        all_kps = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                results = self.yolo_model(
                    frame,
                    verbose=False,
                    conf=self.config.confidence_threshold,
                )

                if (results and results[0].keypoints is not None
                        and len(results[0].keypoints.data) > 0):

                    kps_data = results[0].keypoints.data.cpu().numpy()

                    # 대상자 선택
                    if len(kps_data) > 1:
                        target_idx = self._select_target(results[0])
                        kps = kps_data[target_idx]
                    else:
                        kps = kps_data[0]

                    all_kps.append(kps)
                else:
                    # 감지 실패 시 이전 프레임 복사 또는 제로
                    if all_kps:
                        all_kps.append(all_kps[-1].copy())
                    else:
                        all_kps.append(np.zeros((17, 3), dtype=np.float32))

            frame_idx += 1

        cap.release()
        return np.array(all_kps, dtype=np.float32) if all_kps else None

    def _select_target(self, result) -> int:
        """다중 감지 시 대상자 선택"""
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) <= 1:
            return 0

        method = self.config.select_target_method

        if method == "largest":
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return int(np.argmax(areas))

        elif method == "center":
            h, w = result.orig_shape[:2]
            cx, cy = w / 2, h / 2
            box_cx = (boxes[:, 0] + boxes[:, 2]) / 2
            box_cy = (boxes[:, 1] + boxes[:, 3]) / 2
            dists = np.sqrt((box_cx - cx)**2 + (box_cy - cy)**2)
            return int(np.argmin(dists))

        elif method == "combined":
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            h, w = result.orig_shape[:2]
            cx, cy = w / 2, h / 2
            box_cx = (boxes[:, 0] + boxes[:, 2]) / 2
            box_cy = (boxes[:, 1] + boxes[:, 3]) / 2
            dists = np.sqrt((box_cx - cx)**2 + (box_cy - cy)**2)
            area_score = areas / (areas.max() + 1e-6)
            dist_score = 1 - dists / (dists.max() + 1e-6)
            combined = 0.6 * area_score + 0.4 * dist_score
            return int(np.argmax(combined))

        return 0

    # ================================================================
    # RF 특징 추출
    # ================================================================

    def _build_rf_dataset(self, all_keypoints: list, output_dir: Path) -> dict:
        """키포인트 → RF 특징 벡터 → CSV 데이터셋"""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        output_dir.mkdir(parents=True, exist_ok=True)

        all_features = []
        all_labels = []

        for keypoints, label, name in all_keypoints:
            for frame_idx in range(len(keypoints)):
                kps = keypoints[frame_idx]
                features = self._extract_rf_features(kps, keypoints, frame_idx)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label)

        if not all_features:
            return {"status": "error", "message": "특징 추출 실패"}

        feature_names = self._get_rf_feature_names()
        df = pd.DataFrame(all_features, columns=feature_names)
        df["label"] = all_labels

        # 데이터 분할 - 샘플 수가 적으면 분할하지 않음
        n_samples = len(df)
        min_samples_for_split = 5
        
        if n_samples < min_samples_for_split:
            logger.warning(f"RF 샘플 수 부족 ({n_samples}개) - 분할 없이 전체를 train으로")
            train_df = df
            val_df = df.iloc[:0]  # 빈 DataFrame
            test_df = df.iloc[:0]
        else:
            unique, counts = df["label"].value_counts().to_dict(), None
            counts = df["label"].value_counts()
            can_stratify = self.config.stratify and all(c >= 2 for c in counts)
            
            try:
                train_df, temp_df = train_test_split(
                    df,
                    test_size=(1 - self.config.train_ratio),
                    random_state=self.config.random_seed,
                    stratify=df["label"] if can_stratify else None,
                )

                if len(temp_df) >= 2:
                    val_ratio_adj = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
                    val_df, test_df = train_test_split(
                        temp_df,
                        test_size=(1 - val_ratio_adj),
                        random_state=self.config.random_seed,
                        stratify=temp_df["label"] if can_stratify else None,
                    )
                else:
                    val_df = temp_df
                    test_df = df.iloc[:0]
            except ValueError as e:
                logger.warning(f"RF 분할 오류, 전체를 train으로: {e}")
                train_df = df
                val_df = df.iloc[:0]
                test_df = df.iloc[:0]

        # 저장
        train_df.to_csv(str(output_dir / "train.csv"), index=False)
        val_df.to_csv(str(output_dir / "val.csv"), index=False)
        test_df.to_csv(str(output_dir / "test.csv"), index=False)

        with open(str(output_dir / "feature_columns.txt"), "w") as f:
            for name in feature_names:
                f.write(name + "\n")

        logger.info(f"RF 데이터셋 저장: {output_dir}")

        return {
            "status": "ok",
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "n_features": len(feature_names),
            "output_dir": str(output_dir),
        }

    def _extract_rf_features(self, kps: np.ndarray, all_kps: np.ndarray, frame_idx: int) -> Optional[list]:
        """단일 프레임 → RF 특징 벡터"""
        features = []
        xy = kps[:, :2]
        conf = kps[:, 2]

        if np.mean(conf) < 0.3:
            return None

        # 관절 각도
        angle_joints = [
            (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]),
            (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]),
            (KP["left_hip"], KP["left_knee"], KP["left_ankle"]),
            (KP["right_hip"], KP["right_knee"], KP["right_ankle"]),
            (KP["left_shoulder"], KP["left_hip"], KP["left_knee"]),
            (KP["right_shoulder"], KP["right_hip"], KP["right_knee"]),
        ]
        for a, b, c in angle_joints:
            features.append(self._calc_angle(xy[a], xy[b], xy[c]))

        # 정규화 좌표
        hip_center = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) / 2
        shoulder_center = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) / 2
        body_height = np.linalg.norm(shoulder_center - hip_center) + 1e-6

        for i in range(17):
            normalized = (xy[i] - hip_center) / body_height
            features.extend(normalized.tolist())

        # 거리/비율 특징
        head_y = xy[KP["nose"]][1]
        foot_y = max(xy[KP["left_ankle"]][1], xy[KP["right_ankle"]][1])
        features.append((foot_y - head_y) / (body_height + 1e-6))

        shoulder_width = np.linalg.norm(xy[KP["left_shoulder"]] - xy[KP["right_shoulder"]])
        features.append(shoulder_width / (body_height + 1e-6))

        body_points = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
        center_y = np.mean([xy[p][1] for p in body_points])
        features.append(center_y / (body_height + 1e-6))

        # 속도 특징
        if frame_idx > 0:
            prev_xy = all_kps[frame_idx - 1, :, :2]
            velocity = xy - prev_xy
            for joint in [KP["nose"], KP["left_hip"], KP["right_hip"], KP["left_ankle"], KP["right_ankle"]]:
                features.append(np.linalg.norm(velocity[joint]))
            prev_center = np.mean([prev_xy[p] for p in body_points], axis=0)
            curr_center = np.mean([xy[p] for p in body_points], axis=0)
            features.append(np.linalg.norm(curr_center - prev_center))
        else:
            features.extend([0.0] * 6)

        features.append(np.mean(conf))
        return features

    def _calc_angle(self, a, b, c) -> float:
        """세 점의 각도 (degree)"""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _get_rf_feature_names(self) -> list:
        """RF 특징 이름"""
        names = [
            "left_elbow_angle", "right_elbow_angle",
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle",
        ]
        for kp_name in COCO_KEYPOINTS:
            names.extend([f"{kp_name}_norm_x", f"{kp_name}_norm_y"])
        names.extend(["head_foot_ratio", "shoulder_height_ratio", "center_y_ratio"])
        names.extend([
            "nose_velocity", "left_hip_velocity", "right_hip_velocity",
            "left_ankle_velocity", "right_ankle_velocity", "center_velocity",
        ])
        names.append("avg_confidence")
        return names

    # ================================================================
    # ST-GCN 시퀀스 생성
    # ================================================================

    def _build_stgcn_dataset(self, all_keypoints: list, output_dir: Path) -> dict:
        """키포인트 → ST-GCN 시퀀스 텐서 → npy"""
        from sklearn.model_selection import train_test_split

        output_dir.mkdir(parents=True, exist_ok=True)

        seq_len = self.config.sequence_length
        stride = self.config.sequence_stride

        all_sequences = []
        all_labels = []

        for keypoints, label, name in all_keypoints:
            T = len(keypoints)
            if T < seq_len:
                padded = np.tile(keypoints, (math.ceil(seq_len / T), 1, 1))[:seq_len]
                seq = self._keypoints_to_stgcn_tensor(padded)
                all_sequences.append(seq)
                all_labels.append(label)
            else:
                for start in range(0, T - seq_len + 1, stride):
                    window = keypoints[start:start + seq_len]
                    seq = self._keypoints_to_stgcn_tensor(window)
                    all_sequences.append(seq)
                    all_labels.append(label)

        if not all_sequences:
            return {"status": "error", "message": "시퀀스 생성 실패"}

        data = np.array(all_sequences, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)

        # 분할 - 샘플 수가 적으면 분할하지 않고 전체를 train으로
        n_samples = len(data)
        min_samples_for_split = 5  # 최소 5개 이상이어야 분할
        
        if n_samples < min_samples_for_split:
            logger.warning(f"샘플 수 부족 ({n_samples}개) - 분할 없이 전체를 train으로 사용")
            train_idx = np.arange(n_samples)
            val_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)
        else:
            indices = np.arange(n_samples)
            
            # stratify는 클래스별 최소 2개 이상 필요
            unique, counts = np.unique(labels, return_counts=True)
            can_stratify = self.config.stratify and all(c >= 2 for c in counts)
            
            try:
                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=(1 - self.config.train_ratio),
                    random_state=self.config.random_seed,
                    stratify=labels if can_stratify else None,
                )

                if len(temp_idx) >= 2:
                    val_ratio_adj = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
                    val_idx, test_idx = train_test_split(
                        temp_idx,
                        test_size=(1 - val_ratio_adj),
                        random_state=self.config.random_seed,
                        stratify=labels[temp_idx] if can_stratify else None,
                    )
                else:
                    val_idx = temp_idx
                    test_idx = np.array([], dtype=int)
            except ValueError as e:
                logger.warning(f"분할 오류, 전체를 train으로: {e}")
                train_idx = np.arange(n_samples)
                val_idx = np.array([], dtype=int)
                test_idx = np.array([], dtype=int)

        # 저장
        np.save(str(output_dir / "train_data.npy"), data[train_idx])
        np.save(str(output_dir / "train_labels.npy"), labels[train_idx])
        np.save(str(output_dir / "val_data.npy"), data[val_idx])
        np.save(str(output_dir / "val_labels.npy"), labels[val_idx])
        np.save(str(output_dir / "test_data.npy"), data[test_idx])
        np.save(str(output_dir / "test_labels.npy"), labels[test_idx])

        logger.info(f"ST-GCN 데이터셋 저장: {output_dir}")

        return {
            "status": "ok",
            "total_sequences": len(data),
            "train_sequences": len(train_idx),
            "val_sequences": len(val_idx),
            "test_sequences": len(test_idx),
            "shape": data.shape,
            "output_dir": str(output_dir),
        }

    def _keypoints_to_stgcn_tensor(self, keypoints: np.ndarray) -> np.ndarray:
        """(T, 17, 3) → (3, T, 17, 1)"""
        T, V, C = keypoints.shape

        if self.config.normalize_method == "center":
            hip_center = (keypoints[:, KP["left_hip"], :2] + keypoints[:, KP["right_hip"], :2]) / 2
            keypoints[:, :, 0] -= hip_center[:, 0:1]
            keypoints[:, :, 1] -= hip_center[:, 1:2]
        elif self.config.normalize_method == "minmax":
            xy = keypoints[:, :, :2]
            xy_min = xy.min(axis=(0, 1), keepdims=True)
            xy_max = xy.max(axis=(0, 1), keepdims=True)
            keypoints[:, :, :2] = (xy - xy_min) / (xy_max - xy_min + 1e-6)

        tensor = keypoints.transpose(2, 0, 1)[:, :, :, np.newaxis]
        return tensor.astype(np.float32)

    def _emit_progress(self, current: int, total: int, message: str):
        if self.progress_callback:
            self.progress_callback(current, total, message)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전처리 엔진")
    parser.add_argument("--data-dir", required=True, help="원본 비디오 디렉토리 (fall/, normal/ 구조)")
    parser.add_argument("--output-dir", default=None, help="출력 디렉토리")
    parser.add_argument("--seq-length", type=int, default=60, help="ST-GCN 시퀀스 길이")
    parser.add_argument("--seq-stride", type=int, default=30, help="슬라이딩 윈도우 stride")
    parser.add_argument("--fps", type=int, default=30, help="타겟 FPS")
    args = parser.parse_args()

    config = PreprocessConfig(
        target_fps=args.fps,
        sequence_length=args.seq_length,
        sequence_stride=args.seq_stride,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    engine = PreprocessEngine(
        config,
        progress_callback=lambda c, t, m: print(f"  [{c}/{t}] {m}"),
    )

    result = engine.run(args.data_dir)
    print(f"\n결과: {result}")
