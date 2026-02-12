#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Training Engine (Stage 3)
============================================================
RF 및 ST-GCN 모델 학습 + 하이퍼파라미터 튜닝

지원 모델:
  1) Random Forest (sklearn) - Grid/Random/Bayesian Search
  2) ST-GCN Fine-tuned (PyTorch) - 차등 LR + Optuna

사용법:
    from pipeline.trainer import RFTrainer, STGCNTrainer
    from pipeline.config import RFTrainConfig, STGCNTrainConfig
    
    # RF 학습
    rf_trainer = RFTrainer(RFTrainConfig())
    rf_result = rf_trainer.train("train.csv", "val.csv")
    
    # ST-GCN 학습
    stgcn_trainer = STGCNTrainer(STGCNTrainConfig())
    stgcn_result = stgcn_trainer.train()
============================================================
"""

import os
import time
import logging
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

try:
    from pipeline.config import RFTrainConfig, STGCNTrainConfig, DATASET_DIR
except ImportError:
    from config import RFTrainConfig, STGCNTrainConfig, DATASET_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 학습 결과 컨테이너
# ============================================================
@dataclass
class TrainResult:
    """학습 결과"""
    model_name: str
    status: str = "pending"
    best_metric: float = 0.0
    best_params: Dict[str, Any] = None
    model_path: str = ""
    training_time_sec: float = 0.0
    history: list = None
    error_msg: str = ""

    def __post_init__(self):
        if self.best_params is None:
            self.best_params = {}
        if self.history is None:
            self.history = []

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "best_metric": self.best_metric,
            "best_params": self.best_params,
            "model_path": self.model_path,
            "training_time_sec": self.training_time_sec,
            "history_length": len(self.history),
            "error_msg": self.error_msg,
        }


# ============================================================
# RF 학습 엔진
# ============================================================
class RFTrainer:
    """Random Forest 학습기"""

    def __init__(
        self,
        config: RFTrainConfig,
        metric_callback: Optional[Callable[[dict], None]] = None
    ):
        self.config = config
        self.metric_callback = metric_callback

    def train(self, train_csv: str, val_csv: Optional[str] = None) -> TrainResult:
        """RF 학습 실행"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score

        result = TrainResult("Random Forest")
        result.status = "running"
        start_time = time.time()

        try:
            # 데이터 로드
            train_df = pd.read_csv(train_csv)
            X_train = train_df.drop(columns=["label"]).values
            y_train = train_df["label"].values

            X_val, y_val = None, None
            if val_csv and Path(val_csv).exists():
                val_df = pd.read_csv(val_csv)
                X_val = val_df.drop(columns=["label"]).values
                y_val = val_df["label"].values

            logger.info(f"RF 데이터: train={len(X_train)}, val={len(X_val) if X_val is not None else 'CV'}")

            if self.config.tuning_enabled:
                best_model, best_params, best_score = self._tune(X_train, y_train)
                result.best_params = best_params
                result.best_metric = best_score
                logger.info(f"RF 튜닝 완료: {best_params} → F1={best_score:.4f}")
            else:
                best_model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    class_weight=self.config.class_weight,
                    random_state=42,
                    n_jobs=-1,
                )
                best_model.fit(X_train, y_train)
                result.best_params = {"n_estimators": self.config.n_estimators, "max_depth": self.config.max_depth}

            # 검증 평가
            if X_val is not None:
                y_pred = best_model.predict(X_val)
                result.best_metric = f1_score(y_val, y_pred)
                acc = accuracy_score(y_val, y_pred)
                logger.info(f"RF Val: Acc={acc:.4f}, F1={result.best_metric:.4f}")

            # 저장
            save_path = Path(self.config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, str(save_path))
            result.model_path = str(save_path)

            result.status = "done"
            result.training_time_sec = time.time() - start_time
            logger.info(f"RF 학습 완료: {result.training_time_sec:.1f}초")

            self._emit_metric({"model": "RF", "event": "done", "metric": result.best_metric, "params": result.best_params})

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            logger.error(f"RF 학습 오류: {e}")

        return result

    def _tune(self, X, y):
        """하이퍼파라미터 튜닝"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        param_grid = {
            "n_estimators": self.config.tuning_n_estimators,
            "max_depth": self.config.tuning_max_depth,
            "min_samples_split": self.config.tuning_min_samples_split,
            "min_samples_leaf": self.config.tuning_min_samples_leaf,
        }

        base_model = RandomForestClassifier(
            class_weight=self.config.class_weight,
            random_state=42,
            n_jobs=-1,
        )

        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                base_model, param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=-1, verbose=1,
            )
        else:  # random
            from scipy.stats import randint
            param_dist = {
                "n_estimators": randint(50, 500),
                "max_depth": [None] + list(range(5, 50, 5)),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
            }
            search = RandomizedSearchCV(
                base_model, param_dist,
                n_iter=50, cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=-1, random_state=42, verbose=1,
            )

        search.fit(X, y)
        return search.best_estimator_, search.best_params_, search.best_score_

    def _emit_metric(self, data: dict):
        if self.metric_callback:
            self.metric_callback(data)


# ============================================================
# ST-GCN 학습 엔진
# ============================================================
class STGCNTrainer:
    """ST-GCN Fine-tuning 학습기"""

    def __init__(
        self,
        config: STGCNTrainConfig,
        metric_callback: Optional[Callable[[dict], None]] = None
    ):
        self.config = config
        self.metric_callback = metric_callback
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def train(self) -> TrainResult:
        """ST-GCN Fine-tuning 학습"""
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
        from torch.utils.data import TensorDataset, DataLoader

        result = TrainResult("ST-GCN Fine-tuned")
        result.status = "running"
        start_time = time.time()
        self._cancelled = False

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"ST-GCN 디바이스: {device}")

            # 데이터 로드
            data_dir = Path(self.config.data_dir)
            train_data = np.load(str(data_dir / "train_data.npy"))
            train_labels = np.load(str(data_dir / "train_labels.npy"))
            val_data = np.load(str(data_dir / "val_data.npy"))
            val_labels = np.load(str(data_dir / "val_labels.npy"))

            logger.info(f"ST-GCN 데이터: train={len(train_data)}, val={len(val_data)}")

            train_ds = TensorDataset(
                torch.tensor(train_data, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.long),
            )
            val_ds = TensorDataset(
                torch.tensor(val_data, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

            # 모델 생성
            model = self._build_model(device)

            # 옵티마이저
            backbone_params, head_params = [], []
            for name, param in model.named_parameters():
                if "fc" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            optimizer = AdamW([
                {"params": backbone_params, "lr": self.config.backbone_lr},
                {"params": head_params, "lr": self.config.head_lr},
            ], weight_decay=self.config.weight_decay)

            # 스케줄러
            if self.config.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
            elif self.config.scheduler == "step":
                scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
            else:
                scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)

            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0

            ckpt_dir = Path(self.config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # 학습 루프
            for epoch in range(1, self.config.epochs + 1):
                if self._cancelled:
                    logger.info("학습 취소됨")
                    result.status = "cancelled"
                    break

                # Backbone 동결
                freeze = epoch <= self.config.freeze_backbone_epochs
                for p in backbone_params:
                    p.requires_grad = not freeze

                # Train
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(batch_y)
                    train_correct += (logits.argmax(1) == batch_y).sum().item()
                    train_total += len(batch_y)

                train_loss /= train_total
                train_acc = train_correct / train_total * 100

                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        val_loss += loss.item() * len(batch_y)
                        val_correct += (logits.argmax(1) == batch_y).sum().item()
                        val_total += len(batch_y)

                val_loss /= val_total
                val_acc = val_correct / val_total * 100

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                result.history.append({
                    "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                })

                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        {"state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                        str(ckpt_dir / "best_model_finetuned.pth"),
                    )
                else:
                    patience_counter += 1

                mark = " ⭐" if is_best else ""
                logger.info(f"  Epoch {epoch:3d}/{self.config.epochs} | "
                           f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                           f"Val: {val_loss:.4f}/{val_acc:.1f}%{mark}")

                self._emit_metric({
                    "model": "ST-GCN", "event": "epoch", "epoch": epoch,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                    "patience": f"{patience_counter}/{self.config.patience}",
                })

                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early Stopping at epoch {epoch}")
                    break

            result.best_metric = best_val_acc / 100
            result.best_params = {
                "best_epoch": best_epoch, "best_val_acc": best_val_acc,
                "backbone_lr": self.config.backbone_lr, "head_lr": self.config.head_lr,
            }
            result.model_path = str(ckpt_dir / "best_model_finetuned.pth")
            result.training_time_sec = time.time() - start_time

            if result.status != "cancelled":
                result.status = "done"

            logger.info(f"ST-GCN 완료: Best={best_val_acc:.2f}% (Epoch {best_epoch}), {result.training_time_sec:.1f}초")

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            logger.error(f"ST-GCN 오류: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _build_model(self, device):
        """ST-GCN 모델 생성"""
        import torch
        from pipeline._stgcn_model import STGCNFineTunedModel

        model = STGCNFineTunedModel(num_classes=2)

        if self.config.use_pretrained and Path(self.config.pretrained_path).exists():
            pretrained = torch.load(self.config.pretrained_path, map_location=device, weights_only=False)
            state = pretrained.get("state_dict", pretrained)
            matched = 0
            model_dict = model.state_dict()
            for k, v in state.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    matched += 1
            model.load_state_dict(model_dict)
            logger.info(f"Pre-trained: {matched}/{len(state)} layers matched")

        return model.to(device)

    def _emit_metric(self, data: dict):
        if self.metric_callback:
            self.metric_callback(data)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="모델 학습")
    parser.add_argument("--model", choices=["rf", "stgcn", "both"], default="both")
    parser.add_argument("--train-csv", default=str(DATASET_DIR / "binary/train.csv"))
    parser.add_argument("--val-csv", default=str(DATASET_DIR / "binary/val.csv"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    if args.model in ("rf", "both"):
        rf_config = RFTrainConfig(tuning_enabled=args.tune)
        rf_trainer = RFTrainer(rf_config)
        rf_result = rf_trainer.train(args.train_csv, args.val_csv)
        print(f"RF: {rf_result.to_dict()}")

    if args.model in ("stgcn", "both"):
        stgcn_config = STGCNTrainConfig(epochs=args.epochs)
        stgcn_trainer = STGCNTrainer(stgcn_config)
        stgcn_result = stgcn_trainer.train()
        print(f"ST-GCN: {stgcn_result.to_dict()}")
