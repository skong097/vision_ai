#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Training Pipeline Orchestrator
============================================================
4단계 파이프라인을 통합 관리하고 GUI와 연동합니다.

Stage 1: 데이터 수집 (data_ingest)
Stage 2: 전처리 (preprocessor)
Stage 3: 학습 (trainer)
Stage 4: 자동 비교 (compare_models.py)

사용법:
    from pipeline.orchestrator import TrainingPipelineOrchestrator
    from pipeline.config import PipelineConfig
    
    config = PipelineConfig()
    orchestrator = TrainingPipelineOrchestrator(config)
    orchestrator.on_log = print
    orchestrator.run()

CLI:
    python orchestrator.py --data-dir ./videos --train-rf --train-stgcn --auto-compare
============================================================
"""

import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any

try:
    from pipeline.config import PipelineConfig, DATASET_DIR, ST_GCN_DIR, REPORT_DIR
    from pipeline.data_ingest import DataIngestEngine
    from pipeline.preprocessor import PreprocessEngine
    from pipeline.trainer import RFTrainer, STGCNTrainer
except ImportError:
    from config import PipelineConfig, DATASET_DIR, ST_GCN_DIR, REPORT_DIR
    from data_ingest import DataIngestEngine
    from preprocessor import PreprocessEngine
    from trainer import RFTrainer, STGCNTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class PipelineState:
    """파이프라인 실행 상태"""
    STAGES = {1: "데이터 수집", 2: "전처리", 3: "학습", 4: "모델 비교"}

    def __init__(self):
        self.current_stage = 0
        self.is_running = False
        self.is_cancelled = False
        self.start_time = None
        self.results: Dict[int, Any] = {}
        self.errors = []

    @property
    def stage_name(self) -> str:
        return self.STAGES.get(self.current_stage, "대기")

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0


class TrainingPipelineOrchestrator:
    """
    전체 학습 파이프라인 오케스트레이터
    
    콜백:
        on_stage_changed(stage: int, name: str)
        on_progress(current: int, total: int, message: str)
        on_metric(data: dict)
        on_log(message: str)
        on_finished(results: dict)
        on_error(error_msg: str)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState()

        # 콜백
        self.on_stage_changed: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        self.on_metric: Optional[Callable] = None
        self.on_log: Optional[Callable] = None
        self.on_finished: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        self._stgcn_trainer: Optional[STGCNTrainer] = None

    def run(self):
        """전체 파이프라인 실행"""
        self.state = PipelineState()
        self.state.is_running = True
        self.state.start_time = time.time()

        # 설정 검증
        errors = self.config.validate()
        if errors:
            for e in errors:
                self._log(f"⚠ 설정 오류: {e}")
            self._emit_error("설정 오류로 시작할 수 없습니다")
            return

        self._log("=" * 60)
        self._log("  Home Safe Solution - Training Pipeline")
        self._log(f"  설정: {self.config.name}")
        self._log(f"  스테이지: {self.config.start_stage} → {self.config.end_stage}")
        self._log("=" * 60)

        try:
            # Stage 1
            if self.config.start_stage <= 1 <= self.config.end_stage:
                if self.state.is_cancelled:
                    return
                self._run_stage_1()

            # Stage 2
            if self.config.start_stage <= 2 <= self.config.end_stage:
                if self.state.is_cancelled:
                    return
                self._run_stage_2()

            # Stage 3
            if self.config.start_stage <= 3 <= self.config.end_stage:
                if self.state.is_cancelled:
                    return
                self._run_stage_3()

            # Stage 4
            if self.config.start_stage <= 4 <= self.config.end_stage:
                if self.state.is_cancelled:
                    return
                self._run_stage_4()

            self.state.is_running = False
            self._log("=" * 60)
            self._log(f"  파이프라인 완료! (총 {self.state.elapsed_sec:.1f}초)")
            self._log("=" * 60)

            if self.on_finished:
                self.on_finished(self.state.results)

        except Exception as e:
            self.state.is_running = False
            logger.error(f"파이프라인 오류: {e}")
            import traceback
            traceback.print_exc()
            self._emit_error(str(e))

    def _run_stage_1(self):
        """Stage 1: 데이터 수집"""
        self._set_stage(1)
        self._log("📥 Stage 1: 데이터 수집")

        config = self.config.data_ingest
        engine = DataIngestEngine(config, progress_callback=self._on_progress)

        # 소스 등록
        for src in config.sources:
            engine.add_source(src.get("type", "local"), src.get("path", ""), src.get("label", "normal"))

        if not engine.sources:
            raw_dir = Path(config.raw_video_dir)
            existing = self._count_videos(raw_dir)
            if existing > 0:
                self._log(f"  기존 데이터 사용: {existing}개 비디오")
                self.state.results[1] = {"status": "skip", "existing": existing}
                return
            else:
                self._log("  ⚠ 데이터 소스 없음 → 건너뜀")
                self.state.results[1] = {"status": "skip", "reason": "no_sources"}
                return

        result = engine.process_all()
        self.state.results[1] = result
        self._log(f"  ✅ 수집 완료: 성공={result['success']}, 오류={result['error']}")

    def _run_stage_2(self):
        """Stage 2: 전처리"""
        self._set_stage(2)
        self._log("⚙️ Stage 2: 전처리")

        config = self.config.preprocess
        engine = PreprocessEngine(config, progress_callback=self._on_progress)

        raw_dir = self.config.data_ingest.raw_video_dir
        result = engine.run(raw_dir)
        self.state.results[2] = result

        self._log(f"  ✅ 전처리 완료: {result['processed_videos']}/{result['total_videos']} 비디오")
        if result.get("rf", {}).get("status") == "ok":
            rf = result["rf"]
            self._log(f"    RF: train={rf['train_samples']}, val={rf['val_samples']}, test={rf['test_samples']}")
        if result.get("stgcn", {}).get("status") == "ok":
            stgcn = result["stgcn"]
            self._log(f"    ST-GCN: {stgcn['total_sequences']} sequences")

    def _run_stage_3(self):
        """Stage 3: 모델 학습"""
        self._set_stage(3)
        self._log("🎯 Stage 3: 모델 학습")

        train_results = {}

        # RF 학습
        if self.config.rf_train.enabled:
            self._log("\n  ── Random Forest ──")
            rf_trainer = RFTrainer(self.config.rf_train, metric_callback=self._on_metric)
            binary_dir = Path(self.config.preprocess.output_dir) / "binary"
            rf_result = rf_trainer.train(
                str(binary_dir / "train.csv"),
                str(binary_dir / "val.csv"),
            )
            train_results["rf"] = rf_result.to_dict()
            self._log(f"  RF: {rf_result.status}, F1={rf_result.best_metric:.4f}")

        # ST-GCN 학습
        if self.config.stgcn_train.enabled:
            self._log("\n  ── ST-GCN Fine-tuning ──")
            self._stgcn_trainer = STGCNTrainer(self.config.stgcn_train, metric_callback=self._on_metric)
            stgcn_result = self._stgcn_trainer.train()
            train_results["stgcn"] = stgcn_result.to_dict()
            self._log(f"  ST-GCN: {stgcn_result.status}, Val Acc={stgcn_result.best_metric*100:.2f}%")
            self._stgcn_trainer = None

        self.state.results[3] = train_results

    def _run_stage_4(self):
        """Stage 4: 자동 모델 비교"""
        self._set_stage(4)
        self._log("📊 Stage 4: 모델 비교")

        compare_config = self.config.auto_compare
        if not compare_config.enabled:
            self._log("  자동 비교 비활성화 → 건너뜀")
            self.state.results[4] = {"status": "skip"}
            return

        compare_script = Path(compare_config.compare_script)
        if not compare_script.exists():
            self._log(f"  ⚠ compare_models.py 없음: {compare_script}")
            self.state.results[4] = {"status": "error", "reason": "script_not_found"}
            return

        self._log(f"  실행: python {compare_script.name}")
        try:
            result = subprocess.run(
                [sys.executable, str(compare_script)],
                capture_output=True, text=True,
                cwd=str(compare_script.parent),
                timeout=300,
            )

            if result.returncode == 0:
                report_dir = self._find_latest_report()
                self.state.results[4] = {
                    "status": "ok",
                    "report_dir": str(report_dir) if report_dir else "",
                }
                self._log(f"  ✅ 비교 완료: {report_dir}")
            else:
                self.state.results[4] = {"status": "error", "stderr": result.stderr[:200]}
                self._log(f"  ❌ 비교 실패: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            self.state.results[4] = {"status": "error", "reason": "timeout"}
            self._log("  ❌ 타임아웃")
        except Exception as e:
            self.state.results[4] = {"status": "error", "reason": str(e)}
            self._log(f"  ❌ 오류: {e}")

    def cancel(self):
        """파이프라인 취소"""
        self.state.is_cancelled = True
        if self._stgcn_trainer:
            self._stgcn_trainer.cancel()
        self._log("⏹ 파이프라인 취소 요청")

    def _set_stage(self, stage: int):
        self.state.current_stage = stage
        if self.on_stage_changed:
            self.on_stage_changed(stage, self.state.stage_name)

    def _on_progress(self, current: int, total: int, message: str):
        if self.on_progress:
            self.on_progress(current, total, message)

    def _on_metric(self, data: dict):
        if self.on_metric:
            self.on_metric(data)

    def _log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def _emit_error(self, error_msg: str):
        self.state.errors.append(error_msg)
        if self.on_error:
            self.on_error(error_msg)

    def _count_videos(self, directory: Path) -> int:
        from pipeline.config import SUPPORTED_VIDEO_FORMATS
        count = 0
        if directory.exists():
            for f in directory.rglob("*"):
                if f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                    count += 1
        return count

    def _find_latest_report(self) -> Optional[Path]:
        report_base = Path(REPORT_DIR)
        if not report_base.exists():
            return None
        dirs = [d for d in report_base.iterdir() if d.is_dir()]
        return max(dirs, key=lambda d: d.name) if dirs else None


# ============================================================
# PyQt5 Worker (GUI용)
# ============================================================
try:
    from PyQt5.QtCore import QThread, pyqtSignal

    class PipelineWorker(QThread):
        """GUI용 비동기 Worker Thread"""
        stage_changed = pyqtSignal(int, str)
        progress = pyqtSignal(int, int, str)
        metric = pyqtSignal(dict)
        log_message = pyqtSignal(str)
        finished_signal = pyqtSignal(dict)
        error_signal = pyqtSignal(str)

        def __init__(self, config: PipelineConfig, parent=None):
            super().__init__(parent)
            self.config = config
            self.orchestrator: Optional[TrainingPipelineOrchestrator] = None

        def run(self):
            self.orchestrator = TrainingPipelineOrchestrator(self.config)
            self.orchestrator.on_stage_changed = lambda s, n: self.stage_changed.emit(s, n)
            self.orchestrator.on_progress = lambda c, t, m: self.progress.emit(c, t, m)
            self.orchestrator.on_metric = lambda d: self.metric.emit(d)
            self.orchestrator.on_log = lambda m: self.log_message.emit(m)
            self.orchestrator.on_finished = lambda r: self.finished_signal.emit(r)
            self.orchestrator.on_error = lambda e: self.error_signal.emit(e)
            self.orchestrator.run()

        def cancel(self):
            if self.orchestrator:
                self.orchestrator.cancel()

except ImportError:
    pass


# ============================================================
# CLI
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training Pipeline Orchestrator")
    parser.add_argument("--config", help="설정 파일 (JSON)")
    parser.add_argument("--data-dir", help="비디오 데이터 디렉토리")
    parser.add_argument("--stage", type=int, help="특정 스테이지만 실행 (1-4)")
    parser.add_argument("--train-rf", action="store_true", help="RF 학습")
    parser.add_argument("--train-stgcn", action="store_true", help="ST-GCN 학습")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--auto-compare", action="store_true", help="자동 비교 활성화")
    parser.add_argument("--no-compare", action="store_true", help="자동 비교 비활성화")
    args = parser.parse_args()

    # 설정
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig(name="cli_run")

    if args.data_dir:
        config.data_ingest.raw_video_dir = args.data_dir

    if args.stage:
        config.start_stage = args.stage
        config.end_stage = args.stage

    if args.train_rf or args.train_stgcn:
        config.rf_train.enabled = args.train_rf
        config.stgcn_train.enabled = args.train_stgcn

    config.stgcn_train.epochs = args.epochs

    if args.no_compare:
        config.auto_compare.enabled = False
    elif args.auto_compare:
        config.auto_compare.enabled = True

    # 실행
    orchestrator = TrainingPipelineOrchestrator(config)
    orchestrator.on_log = lambda m: print(m)
    orchestrator.run()


if __name__ == "__main__":
    main()
