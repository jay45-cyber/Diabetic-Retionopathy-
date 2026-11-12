"""CLI Evaluation Script for 5-Class Diabetic Retinopathy Grading.

Computes:
- Accuracy, Macro-F1, Macro-Precision, Macro-Recall
- Per-class F1, Precision, Recall, Support
- Quadratic Weighted Kappa (QWK)
- Referable DR Sensitivity & Specificity (Grade >= 2)
- One-vs-Rest AUROC & Multi-Class ECE
- 95% Bootstrap Confidence Intervals
- Saves: metrics.json, classification_report.csv, confusion_matrix.png, evaluation_summary.txt
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dr_dataset import DRGradingDataset
from src.models.dr_models import build_dr_model
from src.evaluation.dr_metrics import (
    compute_dr_evaluation_metrics,
    compute_dr_bootstrap_ci,
    save_dr_evaluation_artifacts,
    DR_CLASS_NAMES
)
from src.utils.logging import get_logger

logger = get_logger("evaluate_dr_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5-Class DR Severity Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default=None, help="Model architecture (e.g. vit_base_patch16_224, resnet50, swin_base_patch4_window7_224)")
    parser.add_argument("--test_csv", type=str, default="data/processed/splits/dr_test.csv", help="Path to test CSV")
    parser.add_argument("--image_dir", type=str, default="data/raw/aptos2019/train_images", help="Path to images directory")
    parser.add_argument("--image_size", type=int, default=224, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--use_clahe", action="store_true", help="Enable CLAHE")
    parser.add_argument("--output_dir", type=str, default="outputs/dr_experiments/eval_results", help="Directory to save evaluation artifacts")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test set if images are not present")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Evaluation Device: {device}")

    # Checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and args.model_name is None:
        model_name = checkpoint["model_name"]
    else:
        model_name = args.model_name or "vit_base_patch16_224"

    logger.info(f"Loading model architecture: {model_name}")
    model = build_dr_model(model_name=model_name, num_classes=5, pretrained=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset loading
    test_csv_path = Path(args.test_csv)
    image_dir = Path(args.image_dir)

    if not test_csv_path.is_file() and not args.synthetic:
        logger.error(f"Test CSV not found at {test_csv_path}. Pass --synthetic to run evaluation smoke-test.")
        return

    if args.synthetic or not test_csv_path.is_file():
        logger.info("Generating synthetic test dataset for evaluation pipeline validation...")
        # Create 50 synthetic test samples
        from PIL import Image
        temp_img_dir = output_dir / "synthetic_test_imgs"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        synthetic_records = []
        for i in range(50):
            id_code = f"synth_test_{i:03d}"
            label = i % 5
            img_path = temp_img_dir / f"{id_code}.png"
            if not img_path.is_file():
                dummy_arr = np.random.randint(20, 200, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(dummy_arr).save(img_path)
            synthetic_records.append({"id_code": id_code, "diagnosis": label})
        test_df = pd.DataFrame(synthetic_records)
        image_dir = temp_img_dir
    else:
        test_df = pd.read_csv(test_csv_path)

    test_dataset = DRGradingDataset(
        df=test_df,
        image_dir=image_dir,
        image_size=(args.image_size, args.image_size),
        is_training=False,
        use_clahe=args.use_clahe
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    logger.info(f"Running evaluation loop on {len(test_dataset)} samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute metrics
    metrics = compute_dr_evaluation_metrics(y_true, y_pred, y_prob=y_prob, num_classes=5)

    # Compute 95% bootstrap confidence intervals
    logger.info("Computing 95% bootstrap confidence intervals (500 iterations)...")
    bootstrap_cis = compute_dr_bootstrap_ci(y_true, y_pred, n_bootstraps=500, random_state=42)
    metrics["bootstrap_95ci"] = bootstrap_cis

    # Save artifacts
    saved_files = save_dr_evaluation_artifacts(
        metrics,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        model_name=model_name,
        prefix=f"{model_name}_test"
    )

    logger.info("=" * 60)
    logger.info(f"Evaluation Completed for Model: {model_name}")
    logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}% (95% CI: [{bootstrap_cis['accuracy']['ci_lower']*100:.2f}%, {bootstrap_cis['accuracy']['ci_upper']*100:.2f}%])")
    logger.info(f"Macro-F1: {metrics['macro_f1']:.4f} (95% CI: [{bootstrap_cis['macro_f1']['ci_lower']:.4f}, {bootstrap_cis['macro_f1']['ci_upper']:.4f}])")
    logger.info(f"Quadratic Weighted Kappa (QWK): {metrics['quadratic_weighted_kappa']:.4f} (95% CI: [{bootstrap_cis['quadratic_weighted_kappa']['ci_lower']:.4f}, {bootstrap_cis['quadratic_weighted_kappa']['ci_upper']:.4f}])")
    logger.info(f"Referable DR Sensitivity: {metrics['referable_dr']['sensitivity'] * 100:.2f}%")
    logger.info(f"Referable DR Specificity: {metrics['referable_dr']['specificity'] * 100:.2f}%")
    logger.info(f"One-vs-Rest AUROC: {metrics['one_vs_rest_auroc']}")
    logger.info(f"Expected Calibration Error: {metrics['expected_calibration_error']}")
    logger.info(f"Saved artifacts to {output_dir}:")
    for k, p in saved_files.items():
        logger.info(f"  - {k}: {p}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
