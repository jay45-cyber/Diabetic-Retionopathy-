"""Reproducible Training Pipeline for Track B: Binary Diabetes Risk Screening."""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.utils.logging import get_logger

logger = get_logger("diabetes_trainer")


class DiabetesRiskTrainer:
    """Trainer implementing AdamW, Cosine LR scheduling, AMP, and Validation AUROC early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Union[str, Path] = "models/checkpoints",
        model_name: str = "resnet50_diabetes",
        early_stopping_patience: int = 5,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm

        # Default criterion: BCEWithLogitsLoss with positive class weighting
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

        # AMP configuration
        self.use_amp = use_amp and (self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.best_val_auroc = -1.0
        self.patience_counter = 0

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Unpack either (images, targets) or (images, clinical, targets)."""
        if len(batch) == 2:
            images, targets = batch
            return images.to(self.device), None, targets.to(self.device)
        elif len(batch) == 3:
            images, clinical, targets = batch
            return images.to(self.device), clinical.to(self.device), targets.to(self.device)
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            images, clinical, targets = self._unpack_batch(batch)

            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                if clinical is not None:
                    logits = self.model(images, clinical)
                else:
                    logits = self.model(images)

                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch in self.val_loader:
            images, clinical, targets = self._unpack_batch(batch)

            if clinical is not None:
                logits = self.model(images, clinical)
            else:
                logits = self.model(images)

            loss = self.criterion(logits, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        val_loss = total_loss / len(self.val_loader.dataset)

        # Compute metrics
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auroc = 0.5

        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except Exception:
            auprc = float(np.mean(y_true))

        brier = float(brier_score_loss(y_true, y_prob))

        return {
            "val_loss": round(val_loss, 4),
            "val_auroc": round(auroc, 4),
            "val_auprc": round(auprc, 4),
            "val_brier": round(brier, 4)
        }

    def fit(self, epochs: int = 30) -> Dict[str, Any]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": []
        }

        logger.info(f"Starting Track B training for {self.model_name} on {self.device} (Max Epochs: {epochs})")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_auroc"].append(val_metrics["val_auroc"])
            history["val_auprc"].append(val_metrics["val_auprc"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val AUROC: {val_metrics['val_auroc']:.4f} | Val AUPRC: {val_metrics['val_auprc']:.4f}"
            )

            # Early stopping on Validation AUROC
            current_auroc = val_metrics["val_auroc"]
            if current_auroc > self.best_val_auroc:
                self.best_val_auroc = current_auroc
                self.patience_counter = 0
                ckpt_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_val_auroc": self.best_val_auroc,
                    "val_metrics": val_metrics,
                    "model_name": self.model_name
                }, ckpt_path)
                logger.info(f" -> Best checkpoint saved: {ckpt_path} (AUROC: {current_auroc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience}).")
                    break

        return history
