"""CLI Training Script for 5-Class Diabetic Retinopathy Severity Grading.

Usage:
  python scripts/train_dr_model.py --config config/dr_grading_config.yaml
  python scripts/train_dr_model.py --model_name resnet50 --epochs 10 --batch_size 16
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dr_dataset import DRGradingDataset, compute_inverse_class_weights, create_stratified_dr_splits
from src.models.dr_models import build_dr_model
from src.training.dr_trainer import DRTrainer
from src.utils.logging import get_logger

logger = get_logger("train_dr_model")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 5-Class DR Severity Model")
    parser.add_argument("--config", type=str, default="config/dr_grading_config.yaml", help="Path to config YAML")
    parser.add_argument("--model_name", type=str, default=None, help="Backbone model name override")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--image_size", type=int, default=None, help="Square image resolution (e.g. 224 or 384)")
    parser.add_argument("--use_clahe", action="store_true", help="Enable CLAHE preprocessing")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch with small subset to verify pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Overrides
    model_name = args.model_name or cfg["model"]["model_name"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    seed = cfg["project"].get("random_seed", 42)
    img_size_val = args.image_size or cfg["data"]["image_size"][0]
    img_size = (img_size_val, img_size_val)
    use_clahe = args.use_clahe or cfg["data"].get("use_clahe", False)
    crop_circle = cfg["data"].get("crop_circle", True)
    patience = cfg["training"].get("early_stopping_patience", 5)

    device_str = args.device or cfg["training"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    set_seed(seed)
    logger.info(f"Target Device: {device} | Model: {model_name} | Image Size: {img_size}")

    # Paths
    train_csv_path = Path(cfg["data"]["train_csv"])
    val_csv_path = Path(cfg["data"]["val_csv"])
    image_dir = Path(cfg["data"]["image_dir"])
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check splits existence
    if not (train_csv_path.is_file() and val_csv_path.is_file()):
        logger.info("Stratified split files not found at specified paths. Checking for raw labels CSV...")
        raw_csv_candidates = [
            Path(cfg["data"]["dataset_root"]) / "train.csv",
            image_dir.parent / "train.csv",
            Path("data/raw/aptos2019/train.csv")
        ]
        raw_csv = next((p for p in raw_csv_candidates if p.is_file()), None)
        if raw_csv:
            logger.info(f"Generating stratified splits from {raw_csv}...")
            splits = create_stratified_dr_splits(
                csv_path=raw_csv,
                output_dir="data/processed/splits",
                random_state=seed
            )
            train_df = splits["train"]
            val_df = splits["val"]
        else:
            logger.error(
                f"No split files found at {train_csv_path} and no raw CSV found at candidates. "
                "Please run scripts/build_splits.py or provide valid CSV path."
            )
            return
    else:
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

    if args.dry_run:
        logger.info("Executing dry run: sampling 32 train and 16 val records.")
        train_df = train_df.sample(min(32, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(min(16, len(val_df)), random_state=seed).reset_index(drop=True)
        epochs = 1

    logger.info(f"Loaded {len(train_df)} training and {len(val_df)} validation samples.")

    # Datasets and Loaders
    train_dataset = DRGradingDataset(
        df=train_df,
        image_dir=image_dir,
        image_size=img_size,
        is_training=True,
        use_clahe=use_clahe,
        crop_circle=crop_circle
    )
    val_dataset = DRGradingDataset(
        df=val_df,
        image_dir=image_dir,
        image_size=img_size,
        is_training=False,
        use_clahe=use_clahe,
        crop_circle=crop_circle
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(device.type == "cuda")
    )

    # Class Weights for Imbalanced Loss
    class_weights = compute_inverse_class_weights(train_df["diagnosis"].values, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Computed normalized inverse class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    # Model
    model = build_dr_model(
        model_name=model_name,
        num_classes=5,
        pretrained=cfg["model"].get("pretrained", True),
        dropout_rate=cfg["model"].get("dropout_rate", 0.2)
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Trainer
    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        early_stopping_patience=patience,
        grad_accum_steps=cfg["training"].get("grad_accum_steps", 1),
        use_amp=cfg["training"].get("use_amp", True)
    )

    history = trainer.fit(epochs=epochs)

    # Save training history
    history_file = output_dir / f"{model_name}_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
