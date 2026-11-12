"""Reproducible 5-Class Diabetic Retinopathy Training Pipeline."""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("dr_trainer")


class DRTrainer:
    """Trainer implementing AdamW, Cosine LR scheduling, mixed precision, and Macro-F1 early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str | Path = "models/checkpoints",
        model_name: str = "vit_base_patch16_224",
        early_stopping_patience: int = 5,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        
        # AMP only supported on CUDA
        self.use_amp = use_amp and (self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.best_macro_f1 = -1.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps * images.size(0)

        epoch_loss = total_loss / len(self.train_loader.dataset)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluate model on validation set and return (val_loss, val_acc, val_macro_f1)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        val_loss = total_loss / len(self.val_loader.dataset)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        val_acc = float(np.mean(all_preds == all_targets))
        _, _, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

        return val_loss, val_acc, float(macro_f1)

    def fit(self, epochs: int = 30) -> Dict[str, Any]:
        """Execute full training loop with early stopping on validation Macro-F1."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": []
        }

        logger.info(f"Starting training for {self.model_name} on {self.device} (Max Epochs: {epochs})")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.evaluate()

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 4))
            history["val_macro_f1"].append(round(val_f1, 4))

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}"
            )

            # Checkpoint on Macro-F1 improvement
            if val_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_f1
                self.patience_counter = 0
                ckpt_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_macro_f1": self.best_macro_f1,
                    "val_acc": val_acc,
                    "model_name": self.model_name
                }, ckpt_path)
                logger.info(f" -> Best checkpoint saved: {ckpt_path} (Macro-F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience}).")
                    break

        return history
