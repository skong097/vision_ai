#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Training Pipeline Configuration
============================================================
모든 파이프라인 스테이지의 설정을 통합 관리합니다.

사용법:
    from pipeline.config import PipelineConfig
    
    # 기본 설정 생성
    config = PipelineConfig()
    
    # 설정 수정
    config.stgcn_train.epochs = 100
    config.preprocess.sequence_length = 60
    
    # JSON 저장/로드
    config.save("my_config.json")
    config = PipelineConfig.load("my_config.json")
    
    # 유효성 검사
    errors = config.validate()
    if errors:
        print("설정 오류:", errors)
============================================================
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


# ============================================================
# 경로 설정 (프로젝트 환경에 맞게 수정)
# ============================================================
# BASE_DIR = Path("/home/gjkong/dev_ws")
BASE_DIR = PATHS.DEV_ROOT
PROJECT_DIR = BASE_DIR / "yolo/myproj"
ST_GCN_DIR = BASE_DIR / "st_gcn"
GUI_DIR = PROJECT_DIR / "gui"
DATASET_DIR = PROJECT_DIR / "dataset"
PIPELINE_DIR = PROJECT_DIR / "pipeline"
REPORT_DIR = PROJECT_DIR / "scripts/admin/Model_Compare_Report"

# 모델 경로
MODEL_PATHS = {
    "rf": PROJECT_DIR / "models_integrated/3class/random_forest_model.pkl",
    "rf_features": PROJECT_DIR / "models_integrated/3class/feature_columns.txt",
    "stgcn_original": ST_GCN_DIR / "checkpoints/best_model_binary.pth",
    "stgcn_finetuned": ST_GCN_DIR / "checkpoints_finetuned/best_model_finetuned.pth",
    "stgcn_pretrained": ST_GCN_DIR / "pretrained/stgcn_ntu60_hrnet.pth",
    "yolo_pose": "yolov8m-pose.pt",
}

# 지원 동영상 포맷
SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
}


# ============================================================
# Stage 1: 데이터 수집 설정
# ============================================================
@dataclass
class DataIngestConfig:
    """
    데이터 수집 설정
    
    Attributes:
        raw_video_dir: 원본 비디오 저장 디렉토리
        label_strategy: 라벨링 방식 (folder/filename/csv/manual)
        youtube_format: yt-dlp 다운로드 포맷
        youtube_max_duration: 최대 비디오 길이 (초)
        download_timeout: 다운로드 타임아웃 (초)
        max_file_size_mb: 최대 파일 크기 (MB)
        sources: 데이터 소스 목록
    """
    # 출력 경로
    raw_video_dir: str = str(DATASET_DIR / "raw_videos")
    
    # 라벨링 전략
    # - folder: fall/, normal/ 폴더 구조
    # - filename: fall-01.mp4, normal-03.mp4 접두어
    # - csv: manifest.csv 파일 참조
    # - manual: 소스 등록 시 직접 지정
    label_strategy: str = "folder"
    
    # YouTube 다운로드 옵션
    youtube_format: str = "bestvideo[height<=720]+bestaudio/best[height<=720]"
    youtube_max_duration: int = 300  # 최대 5분
    
    # URL 다운로드 옵션
    download_timeout: int = 120
    max_file_size_mb: int = 500
    
    # 데이터 소스 목록 (런타임에 채워짐)
    # 각 소스: {"type": "youtube"|"url"|"local", "path": "...", "label": "fall"|"normal"}
    sources: List[Dict[str, str]] = field(default_factory=list)


# ============================================================
# Stage 2: 전처리 설정
# ============================================================
@dataclass
class PreprocessConfig:
    """
    전처리 설정
    
    비디오 → YOLO Pose → RF Feature / ST-GCN Sequence 변환
    
    Attributes:
        target_resolution: 비디오 정규화 해상도 (width, height)
        target_fps: 목표 FPS
        yolo_model: YOLO Pose 모델 경로
        confidence_threshold: 키포인트 검출 신뢰도 임계값
        select_target_method: 다중 감지 시 대상자 선택 방법
        sequence_length: ST-GCN 시퀀스 프레임 수
        sequence_stride: 슬라이딩 윈도우 stride
        normalize_method: 키포인트 정규화 방법
        train_ratio/val_ratio/test_ratio: 데이터 분할 비율
    """
    # 비디오 정규화
    target_resolution: tuple = (640, 480)
    target_fps: int = 30
    
    # YOLO Pose Estimation
    yolo_model: str = MODEL_PATHS["yolo_pose"]
    confidence_threshold: float = 0.5
    select_target_method: str = "largest"  # largest / center / combined
    
    # ST-GCN 시퀀스 생성
    sequence_length: int = 60       # 프레임 수 (60프레임 = 2초 @ 30fps)
    sequence_stride: int = 30       # 슬라이딩 윈도우 stride
    normalize_method: str = "center"  # center / minmax / none
    
    # RF 특징 추출
    rf_feature_set: str = "full"    # full / minimal / custom
    
    # 데이터 분할
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    stratify: bool = True           # 라벨 비율 유지
    
    # 출력 경로
    output_dir: str = str(DATASET_DIR)


# ============================================================
# Stage 3a: Random Forest 학습 설정
# ============================================================
@dataclass
class RFTrainConfig:
    """
    Random Forest 학습 설정
    
    Attributes:
        enabled: RF 학습 활성화 여부
        n_estimators: 트리 개수
        max_depth: 최대 깊이 (None=무제한)
        tuning_enabled: 하이퍼파라미터 튜닝 활성화
        tuning_method: 튜닝 방법 (grid/random/bayesian)
        cv_folds: 교차검증 폴드 수
        scoring: 평가 지표
    """
    enabled: bool = True
    
    # 하이퍼파라미터 (고정 학습 시)
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    
    # 하이퍼파라미터 튜닝
    tuning_enabled: bool = False
    tuning_method: str = "grid"     # grid / random / bayesian
    cv_folds: int = 5
    scoring: str = "f1"             # accuracy / f1 / recall / precision
    
    # 튜닝 탐색 범위
    tuning_n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200, 300])
    tuning_max_depth: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20, 30])
    tuning_min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    tuning_min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # 모델 저장 경로
    model_save_path: str = str(MODEL_PATHS["rf"])
    feature_columns_path: str = str(MODEL_PATHS["rf_features"])


# ============================================================
# Stage 3b: ST-GCN 학습 설정
# ============================================================
@dataclass
class STGCNTrainConfig:
    """
    ST-GCN Fine-tuning 학습 설정
    
    Attributes:
        enabled: ST-GCN 학습 활성화 여부
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        backbone_lr: Backbone 학습률 (낮게 설정)
        head_lr: FC Head 학습률 (높게 설정)
        scheduler: 학습률 스케줄러
        early_stopping: Early Stopping 활성화
        patience: Early Stopping patience
        use_pretrained: Pre-trained 가중치 사용
    """
    enabled: bool = True
    
    # 기본 하이퍼파라미터
    epochs: int = 50
    batch_size: int = 16
    
    # Learning Rate (차등 적용)
    backbone_lr: float = 1e-5       # backbone: 낮은 lr (pre-trained 유지)
    head_lr: float = 1e-3           # FC head: 높은 lr (새로 학습)
    weight_decay: float = 1e-4
    
    # 스케줄러
    scheduler: str = "cosine"       # cosine / step / plateau
    step_size: int = 10             # StepLR용
    gamma: float = 0.1              # StepLR용
    
    # Early Stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Fine-tuning 옵션
    use_pretrained: bool = True
    pretrained_path: str = str(MODEL_PATHS["stgcn_pretrained"])
    freeze_backbone_epochs: int = 5  # 초기 N 에포크 backbone 동결
    
    # Optuna 하이퍼파라미터 튜닝
    tuning_enabled: bool = False
    n_trials: int = 20
    tuning_params: Dict[str, Any] = field(default_factory=lambda: {
        "backbone_lr": {"low": 1e-6, "high": 1e-4, "log": True},
        "head_lr": {"low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"choices": [8, 16, 32]},
        "weight_decay": {"low": 1e-5, "high": 1e-3, "log": True},
    })
    
    # 데이터 경로
    data_dir: str = str(ST_GCN_DIR / "data/binary")
    
    # 모델 저장 경로
    checkpoint_dir: str = str(ST_GCN_DIR / "checkpoints_finetuned")
    model_save_path: str = str(MODEL_PATHS["stgcn_finetuned"])


# ============================================================
# Stage 4: 자동 비교 설정
# ============================================================
@dataclass
class AutoCompareConfig:
    """
    자동 모델 비교 설정
    
    학습 완료 후 compare_models.py를 자동 실행하여
    기존 모델과 새 모델의 성능을 비교합니다.
    
    Attributes:
        enabled: 자동 비교 활성화
        compare_script: compare_models.py 경로
        report_base_dir: 리포트 저장 기본 경로
        auto_replace: 성능 향상 시 자동 모델 교체
        replace_metric: 비교 기준 메트릭
        replace_threshold: 교체 임계값 (새 모델이 이만큼 이상 좋아야 함)
    """
    enabled: bool = True
    
    # compare_models.py 경로
    compare_script: str = str(ST_GCN_DIR / "compare_models.py")
    
    # 리포트 저장 경로
    report_base_dir: str = str(REPORT_DIR)
    
    # 자동 모델 교체 설정
    auto_replace: bool = False      # True면 자동 교체, False면 사용자 승인 필요
    replace_metric: str = "f1"      # 비교 기준 메트릭
    replace_threshold: float = 0.01  # 새 모델이 이만큼 이상 좋아야 교체
    
    # 추론 속도 측정 반복 횟수
    inference_repeat: int = 50


# ============================================================
# 통합 파이프라인 설정
# ============================================================
@dataclass
class PipelineConfig:
    """
    전체 파이프라인 통합 설정
    
    모든 스테이지의 설정을 하나로 묶어 관리합니다.
    JSON 파일로 저장/로드할 수 있어 설정 재사용이 편리합니다.
    
    Example:
        # 설정 생성 및 수정
        config = PipelineConfig(name="my_experiment")
        config.stgcn_train.epochs = 100
        config.preprocess.sequence_length = 90
        
        # 저장
        config.save("configs/exp1.json")
        
        # 로드
        config = PipelineConfig.load("configs/exp1.json")
        
        # 스테이지 범위 지정 실행
        config.start_stage = 2  # 전처리부터
        config.end_stage = 4    # 비교까지
    """
    # 메타 정보
    name: str = "default"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 각 스테이지 설정
    data_ingest: DataIngestConfig = field(default_factory=DataIngestConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    rf_train: RFTrainConfig = field(default_factory=RFTrainConfig)
    stgcn_train: STGCNTrainConfig = field(default_factory=STGCNTrainConfig)
    auto_compare: AutoCompareConfig = field(default_factory=AutoCompareConfig)
    
    # 실행 옵션
    start_stage: int = 1    # 1=수집, 2=전처리, 3=학습, 4=비교
    end_stage: int = 4
    gpu_device: int = 0     # CUDA 디바이스 번호
    
    def save(self, path: str) -> None:
        """
        설정을 JSON 파일로 저장
        
        Args:
            path: 저장 경로 (예: "configs/my_config.json")
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(p), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 설정 저장: {path}")
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """
        JSON 파일에서 설정 로드
        
        Args:
            path: JSON 파일 경로
            
        Returns:
            PipelineConfig 인스턴스
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        config = cls()
        config.name = data.get("name", "loaded")
        config.description = data.get("description", "")
        config.created_at = data.get("created_at", "")
        
        # 각 스테이지 설정 로드
        if "data_ingest" in data:
            for k, v in data["data_ingest"].items():
                if hasattr(config.data_ingest, k):
                    setattr(config.data_ingest, k, v)
        
        if "preprocess" in data:
            for k, v in data["preprocess"].items():
                if hasattr(config.preprocess, k):
                    # tuple 복원 (JSON에서는 list로 저장됨)
                    if k == "target_resolution" and isinstance(v, list):
                        v = tuple(v)
                    setattr(config.preprocess, k, v)
        
        if "rf_train" in data:
            for k, v in data["rf_train"].items():
                if hasattr(config.rf_train, k):
                    setattr(config.rf_train, k, v)
        
        if "stgcn_train" in data:
            for k, v in data["stgcn_train"].items():
                if hasattr(config.stgcn_train, k):
                    setattr(config.stgcn_train, k, v)
        
        if "auto_compare" in data:
            for k, v in data["auto_compare"].items():
                if hasattr(config.auto_compare, k):
                    setattr(config.auto_compare, k, v)
        
        config.start_stage = data.get("start_stage", 1)
        config.end_stage = data.get("end_stage", 4)
        config.gpu_device = data.get("gpu_device", 0)
        
        print(f"✅ 설정 로드: {path}")
        return config
    
    def validate(self) -> List[str]:
        """
        설정 유효성 검사
        
        Returns:
            에러 메시지 리스트 (비어있으면 유효함)
        """
        errors = []
        
        # 데이터 분할 비율 합계 확인
        ratio_sum = (
            self.preprocess.train_ratio
            + self.preprocess.val_ratio
            + self.preprocess.test_ratio
        )
        if abs(ratio_sum - 1.0) > 0.01:
            errors.append(f"데이터 분할 비율 합계가 1.0이 아닙니다: {ratio_sum:.2f}")
        
        # ST-GCN 시퀀스 길이 확인
        if self.preprocess.sequence_length < 10:
            errors.append(f"시퀀스 길이가 너무 짧습니다: {self.preprocess.sequence_length}")
        
        # 학습 에포크 확인
        if self.stgcn_train.enabled and self.stgcn_train.epochs < 1:
            errors.append(f"에포크 수가 유효하지 않습니다: {self.stgcn_train.epochs}")
        
        # Learning Rate 확인
        if self.stgcn_train.backbone_lr >= self.stgcn_train.head_lr:
            errors.append(
                f"Backbone LR({self.stgcn_train.backbone_lr})이 "
                f"Head LR({self.stgcn_train.head_lr})보다 크거나 같습니다. "
                f"일반적으로 backbone_lr < head_lr 권장"
            )
        
        # 스테이지 범위 확인
        if not (1 <= self.start_stage <= 4):
            errors.append(f"시작 스테이지 범위 오류: {self.start_stage} (1-4)")
        if not (1 <= self.end_stage <= 4):
            errors.append(f"종료 스테이지 범위 오류: {self.end_stage} (1-4)")
        if self.start_stage > self.end_stage:
            errors.append(
                f"시작 스테이지({self.start_stage})가 "
                f"종료 스테이지({self.end_stage})보다 큽니다"
            )
        
        return errors
    
    def summary(self) -> str:
        """설정 요약 문자열 반환"""
        lines = [
            f"{'='*60}",
            f"Pipeline Config: {self.name}",
            f"{'='*60}",
            f"스테이지: {self.start_stage} → {self.end_stage}",
            f"",
            f"[데이터 수집]",
            f"  라벨 전략: {self.data_ingest.label_strategy}",
            f"  소스 수: {len(self.data_ingest.sources)}",
            f"",
            f"[전처리]",
            f"  FPS: {self.preprocess.target_fps}",
            f"  시퀀스 길이: {self.preprocess.sequence_length}",
            f"  분할: train={self.preprocess.train_ratio}, "
            f"val={self.preprocess.val_ratio}, test={self.preprocess.test_ratio}",
            f"",
            f"[RF 학습] {'✅ 활성' if self.rf_train.enabled else '❌ 비활성'}",
            f"  n_estimators: {self.rf_train.n_estimators}",
            f"  튜닝: {'✅' if self.rf_train.tuning_enabled else '❌'}",
            f"",
            f"[ST-GCN 학습] {'✅ 활성' if self.stgcn_train.enabled else '❌ 비활성'}",
            f"  epochs: {self.stgcn_train.epochs}",
            f"  batch_size: {self.stgcn_train.batch_size}",
            f"  LR: backbone={self.stgcn_train.backbone_lr}, head={self.stgcn_train.head_lr}",
            f"  Early Stopping: {'✅' if self.stgcn_train.early_stopping else '❌'} "
            f"(patience={self.stgcn_train.patience})",
            f"  튜닝: {'✅' if self.stgcn_train.tuning_enabled else '❌'}",
            f"",
            f"[자동 비교] {'✅ 활성' if self.auto_compare.enabled else '❌ 비활성'}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ============================================================
# 편의 함수: 자주 사용하는 설정 프리셋
# ============================================================

def get_default_config() -> PipelineConfig:
    """기본 설정 반환"""
    return PipelineConfig(
        name="default",
        description="기본 학습 파이프라인 설정"
    )


def get_quick_rf_config() -> PipelineConfig:
    """RF만 빠르게 학습하는 설정"""
    config = PipelineConfig(
        name="quick_rf",
        description="RF 빠른 학습 (ST-GCN 비활성)"
    )
    config.stgcn_train.enabled = False
    config.auto_compare.enabled = False
    return config


def get_quick_stgcn_config() -> PipelineConfig:
    """ST-GCN만 빠르게 학습하는 설정"""
    config = PipelineConfig(
        name="quick_stgcn",
        description="ST-GCN 빠른 학습 (RF 비활성)"
    )
    config.rf_train.enabled = False
    config.stgcn_train.epochs = 30
    return config


def get_full_tuning_config() -> PipelineConfig:
    """전체 하이퍼파라미터 튜닝 설정"""
    config = PipelineConfig(
        name="full_tuning",
        description="RF + ST-GCN 하이퍼파라미터 튜닝"
    )
    config.rf_train.tuning_enabled = True
    config.rf_train.tuning_method = "grid"
    config.stgcn_train.tuning_enabled = True
    config.stgcn_train.n_trials = 30
    config.stgcn_train.epochs = 50
    return config


# ============================================================
# CLI 테스트
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Pipeline Config 테스트")
    print("=" * 60)
    
    # 기본 설정 생성
    config = get_default_config()
    
    # 요약 출력
    print(config.summary())
    
    # 유효성 검사
    errors = config.validate()
    if errors:
        print("\n⚠️ 설정 오류:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ 설정 유효")
    
    # JSON 저장 테스트
    test_path = "/tmp/test_pipeline_config.json"
    config.save(test_path)
    
    # 로드 테스트
    loaded = PipelineConfig.load(test_path)
    print(f"\n로드된 설정 이름: {loaded.name}")
