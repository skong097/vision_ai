"""
============================================================
Home Safe Solution - Training Pipeline Package
============================================================

사용법:
    from pipeline import PipelineConfig, TrainingPipelineOrchestrator
    
    config = PipelineConfig()
    orchestrator = TrainingPipelineOrchestrator(config)
    orchestrator.run()
"""

from pipeline.config import (
    PipelineConfig,
    DataIngestConfig,
    PreprocessConfig,
    RFTrainConfig,
    STGCNTrainConfig,
    AutoCompareConfig,
    get_default_config,
    get_quick_rf_config,
    get_quick_stgcn_config,
    get_full_tuning_config,
    SUPPORTED_VIDEO_FORMATS,
)

from pipeline.data_ingest import DataIngestEngine, DataSource
from pipeline.preprocessor import PreprocessEngine
from pipeline.trainer import RFTrainer, STGCNTrainer, TrainResult
from pipeline.orchestrator import TrainingPipelineOrchestrator, PipelineState

__all__ = [
    # Config
    "PipelineConfig",
    "DataIngestConfig",
    "PreprocessConfig", 
    "RFTrainConfig",
    "STGCNTrainConfig",
    "AutoCompareConfig",
    "get_default_config",
    "get_quick_rf_config",
    "get_quick_stgcn_config",
    "get_full_tuning_config",
    "SUPPORTED_VIDEO_FORMATS",
    # Engines
    "DataIngestEngine",
    "DataSource",
    "PreprocessEngine",
    "RFTrainer",
    "STGCNTrainer",
    "TrainResult",
    # Orchestrator
    "TrainingPipelineOrchestrator",
    "PipelineState",
]

__version__ = "1.0.0"
