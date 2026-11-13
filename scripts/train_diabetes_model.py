"""CLI Training Script for Track B: Binary Systemic Diabetes Risk Prediction.

Usage:
  python scripts/train_diabetes_model.py --model_name resnet50 --epochs 15
  python scripts/train_diabetes_model.py --model_name convnext --dry_run
  python scripts/train_diabetes_model.py --multimodal --epochs 10
"""
import argparse
import json
import sys
from pathlib import Path
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diabetes_models import build_diabetes_model
from src.data.diabetes_dataset import DiabetesScreeningDataset
from src.training.diabetes_trainer import DiabetesRiskTrainer
from src.utils.logging import get_logger

logger = get_logger("train_diabetes_model")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Track B Systemic Diabetes Model")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model: resnet50, convnext, foundation_vit, multimodal")
    parser.add_argument("--train_manifest", type=str, default="data/manifests/train_manifest.csv")
    parser.add_argument("--val_manifest", type=str, default="data/manifests/val_manifest.csv")
    parser.add_argument("--image_dir", type=str, default="data/raw/brset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_clahe", action="store_true")
    parser.add_argument("--multimodal", action="store_true", help="Enable tabular clinical feature fusion")
    parser.add_argument("--allow_self_reported", action="store_true", help="Allow self-reported diabetes labels")
    parser.add_argument("--dr0_only", action="store_true", help="Filter for participants without visible retinopathy (DR=0)")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch smoke test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info(f"Target Device: {device} | Model: {args.model_name} | Task: Track B Diabetes Screening")

    train_path = Path(args.train_manifest)
    val_path = Path(args.val_manifest)
    image_dir = Path(args.image_dir)

    # Check manifest availability or generate synthetic proxy for verification
    if not (train_path.is_file() and val_path.is_file()):
        logger.warning("Clinical manifests not found on disk. Generating synthetic verification cohort...")
        proxy_dir = Path("outputs/temp_diabetes_data")
        proxy_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        records = []
        for i in range(40):
            id_code = f"proxy_patient_{i:03d}"
            p_img = proxy_dir / f"{id_code}.png"
            if not p_img.is_file():
                dummy = np.random.randint(30, 220, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(dummy).save(p_img)

            records.append({
                "participant_id": id_code,
                "image_id": id_code,
                "image_path": str(p_img),
                "diabetes_label": 1 if i % 3 == 0 else 0,
                "diabetes_label_source": "EHR_Diagnosis",
                "dr_grade": 0 if i % 2 == 0 else 1,
                "age": 45 + (i % 30),
                "sex": i % 2,
                "bmi": 24.5 + (i % 10),
                "systolic_bp": 120 + (i % 25)
            })

        df_all = pd.DataFrame(records)
        train_df = df_all.iloc[:28].reset_index(drop=True)
        val_df = df_all.iloc[28:].reset_index(drop=True)
        image_dir = proxy_dir
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    clinical_cols = ["age", "sex", "bmi", "systolic_bp"] if args.multimodal else None

    # Datasets
    train_dataset = DiabetesScreeningDataset(
        df=train_df,
        image_dir=image_dir,
        image_size=(args.image_size, args.image_size),
        is_training=True,
        use_clahe=args.use_clahe,
        allow_self_reported=args.allow_self_reported,
        dr0_only=args.dr0_only,
        clinical_feature_cols=clinical_cols
    )
    val_dataset = DiabetesScreeningDataset(
        df=val_df,
        image_dir=image_dir,
        image_size=(args.image_size, args.image_size),
        is_training=False,
        use_clahe=args.use_clahe,
        allow_self_reported=args.allow_self_reported,
        dr0_only=args.dr0_only,
        clinical_feature_cols=clinical_cols,
        clinical_scaler_stats=train_dataset.scaler_stats
    )

    logger.info(f"Loaded {len(train_dataset)} valid train samples and {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate class imbalance pos_weight
    labels = train_dataset.df["_clean_label"].values
    n_pos = np.sum(labels == 1.0)
    n_neg = np.sum(labels == 0.0)
    pos_weight_val = float(n_neg / max(n_pos, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    logger.info(f"Class distribution: {n_pos} positive, {n_neg} negative | pos_weight: {pos_weight_val:.2f}")

    # Model
    model = build_diabetes_model(
        model_name=args.model_name,
        pretrained=True,
        num_clinical_features=len(clinical_cols) if clinical_cols else 4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = DiabetesRiskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        model_name=f"{args.model_name}_diabetes",
        early_stopping_patience=5
    )

    history = trainer.fit(epochs=epochs)

    out_dir = Path("outputs/diabetes_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model_name}_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Track B training completed successfully!")


if __name__ == "__main__":
    main()
