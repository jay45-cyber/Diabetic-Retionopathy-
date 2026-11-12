"""Comprehensive 5-Class Diabetic Retinopathy Evaluation Module.

Computes:
- Overall Accuracy
- Macro Precision, Macro Recall, Macro-F1
- Per-Class Precision, Recall, F1
- Confusion Matrix
- Quadratic Weighted Kappa (QWK)
- Referable DR Sensitivity & Specificity (Grade >= 2 vs Grade < 2)
- One-vs-Rest AUROC
- Multi-Class Expected Calibration Error (ECE)
- Bootstrap 95% Confidence Intervals
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score
)

DR_CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]


def compute_dr_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 5
) -> Dict[str, Any]:
    """Calculate full suite of clinical classification and DR-specific metrics."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    # 1. Standard classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 2. Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class_metrics = {}
    for i in range(num_classes):
        name = DR_CLASS_NAMES[i] if i < len(DR_CLASS_NAMES) else f"Class_{i}"
        per_class_metrics[name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_support[i])
        }

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 4. Quadratic Weighted Kappa (QWK)
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        qwk = float("nan")

    # 5. Referable DR Metrics (Referable defined as Grade >= 2)
    # Binary target: 1 if grade >= 2, 0 if grade < 2
    ref_true = (y_true >= 2).astype(np.int32)
    ref_pred = (y_pred >= 2).astype(np.int32)
    ref_cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    tn_ref, fp_ref, fn_ref, tp_ref = ref_cm.ravel()

    ref_sensitivity = float(tp_ref / (tp_ref + fn_ref)) if (tp_ref + fn_ref) > 0 else 0.0
    ref_specificity = float(tn_ref / (tn_ref + fp_ref)) if (tn_ref + fp_ref) > 0 else 0.0

    # 6. One-vs-Rest AUROC (if probabilities provided)
    ovr_auc = float("nan")
    if y_prob is not None:
        try:
            ovr_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            ovr_auc = float("nan")

    # 7. Multi-Class Expected Calibration Error (ECE)
    ece = float("nan")
    if y_prob is not None:
        ece = compute_multiclass_ece(y_true, y_prob, n_bins=10)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "referable_dr_sensitivity": round(ref_sensitivity, 4),
        "referable_dr_specificity": round(ref_specificity, 4),
        "referable_dr": {
            "sensitivity": round(ref_sensitivity, 4),
            "specificity": round(ref_specificity, 4)
        },
        "ovr_auc": round(ovr_auc, 4) if not np.isnan(ovr_auc) else None,
        "one_vs_rest_auroc": round(ovr_auc, 4) if not np.isnan(ovr_auc) else None,
        "expected_calibration_error": round(ece, 4) if not np.isnan(ece) else None,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }


def compute_multiclass_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error for multi-class classification."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    class_names: List[str] = DR_CLASS_NAMES,
    title: str = "Diabetic Retinopathy Confusion Matrix"
):
    """Plot normalized confusion matrix heatmap and save to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Severity Grade",
        xlabel="Predicted Severity Grade"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_evaluation_artifacts(
    metrics: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    output_dir: str | Path = "outputs",
    model_name: str = "model",
    prefix: Optional[str] = None
) -> Dict[str, Path]:
    """Save all 4 mandatory evaluation artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    saved_paths = {}

    # 1. metrics.json
    metrics_path = out_dir / f"{pfx}metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    saved_paths["metrics_json"] = metrics_path

    # 2. classification_report.csv
    if y_true is not None and y_pred is not None:
        report_path = out_dir / f"{pfx}classification_report.csv"
        report_dict = classification_report(y_true, y_pred, target_names=DR_CLASS_NAMES, output_dict=True, zero_division=0)
        pd.DataFrame(report_dict).transpose().to_csv(report_path)
        saved_paths["classification_report_csv"] = report_path

    # 3. confusion_matrix.png
    if "confusion_matrix" in metrics:
        cm_path = out_dir / f"{pfx}confusion_matrix.png"
        cm = np.array(metrics["confusion_matrix"])
        plot_and_save_confusion_matrix(cm, cm_path, title=f"Confusion Matrix: {model_name}")
        saved_paths["confusion_matrix_png"] = cm_path

    # 4. evaluation_summary.txt
    summary_path = out_dir / f"{pfx}evaluation_summary.txt"
    summary_lines = [
        f"===========================================================",
        f" CLINICAL DR EVALUATION SUMMARY: {model_name}",
        f"===========================================================",
        f" Overall Accuracy:             {metrics.get('accuracy', 0.0) * 100:.2f}%",
        f" Macro F1-Score:               {metrics.get('macro_f1', 0.0):.4f}",
        f" Weighted F1-Score:            {metrics.get('weighted_f1', 0.0):.4f}",
        f" Quadratic Weighted Kappa:     {metrics.get('quadratic_weighted_kappa', 0.0):.4f}",
        f" Referable DR Sensitivity:     {metrics.get('referable_dr_sensitivity', 0.0) * 100:.2f}% (Grade >= 2)",
        f" Referable DR Specificity:     {metrics.get('referable_dr_specificity', 0.0) * 100:.2f}% (Grade < 2)",
        f" One-vs-Rest AUROC:            {metrics.get('ovr_auc', 'N/A')}",
        f" Expected Calibration Error:   {metrics.get('expected_calibration_error', 'N/A')}",
        f"-----------------------------------------------------------",
        f" Per-Class Metrics:"
    ]
    for c_name, vals in metrics.get("per_class_metrics", {}).items():
        summary_lines.append(
            f"   - {c_name:18s} | F1: {vals['f1_score']:.4f} | Prec: {vals['precision']:.4f} | Rec: {vals['recall']:.4f} | N={vals['support']}"
        )
    summary_lines.append(f"===========================================================")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    saved_paths["evaluation_summary_txt"] = summary_path

    return saved_paths


save_dr_evaluation_artifacts = save_evaluation_artifacts


def compute_dr_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Compute 95% bootstrap confidence intervals for Macro-F1 and QWK."""
    seed = random_state if random_state is not None else random_seed
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_list, qwk_list, acc_list = [], [], []

    alpha_low = (1.0 - confidence_level) / 2.0 * 100.0
    alpha_high = (1.0 - (1.0 - confidence_level) / 2.0) * 100.0

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]

        acc = float(accuracy_score(sample_true, sample_pred))
        _, _, mf1, _ = precision_recall_fscore_support(sample_true, sample_pred, average="macro", zero_division=0)
        qwk = cohen_kappa_score(sample_true, sample_pred, weights="quadratic")
        acc_list.append(acc)
        f1_list.append(mf1)
        qwk_list.append(qwk)

    return {
        "accuracy": {
            "mean": round(float(np.mean(acc_list)), 4),
            "ci_lower": round(float(np.percentile(acc_list, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(acc_list, alpha_high)), 4)
        },
        "macro_f1": {
            "mean": round(float(np.mean(f1_list)), 4),
            "ci_lower": round(float(np.percentile(f1_list, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(f1_list, alpha_high)), 4)
        },
        "quadratic_weighted_kappa": {
            "mean": round(float(np.mean(qwk_list)), 4),
            "ci_lower": round(float(np.percentile(qwk_list, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(qwk_list, alpha_high)), 4)
        },
        "macro_f1_ci": (
            round(float(np.percentile(f1_list, alpha_low)), 4),
            round(float(np.percentile(f1_list, alpha_high)), 4)
        ),
        "qwk_ci": (
            round(float(np.percentile(qwk_list, alpha_low)), 4),
            round(float(np.percentile(qwk_list, alpha_high)), 4)
        )
    }
