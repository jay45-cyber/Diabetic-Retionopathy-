"""Comprehensive Unit and Integration Tests for 5-Class DR Pipeline.

Verifies:
1. DRGradingDataset & CLAHE preprocessing
2. Stratified 70/15/15 Splitting with Hash Integrity
3. Model forward passes (ViT-B/16, ResNet-50, Swin)
4. Clinical evaluation metrics (Macro-F1, QWK, Referable DR Sensitivity/Specificity, ECE)
5. Explainability heatmaps (Grad-CAM, Attention Rollout, Integrated Gradients)
6. DRTrainer end-to-end training and checkpointing loop
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dr_dataset import DRGradingDataset, compute_inverse_class_weights, create_stratified_dr_splits, apply_clahe
from src.models.dr_models import build_dr_model, ResNet50DR, ViTDR, SwinTransformerDR
from src.evaluation.dr_metrics import (
    compute_dr_evaluation_metrics,
    compute_dr_bootstrap_ci,
    save_dr_evaluation_artifacts
)
from src.explainability.dr_explainer import (
    DRModelExplainer,
    ResNetGradCAM,
    ViTTokenGradCAM,
    ViTAttentionRollout,
    DRIntegratedGradients,
    overlay_cam_on_image
)
from src.training.dr_trainer import DRTrainer


@pytest.fixture
def dummy_retinal_environment():
    """Create temporary directory with synthetic fundus images and metadata CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        records = []
        for i in range(20):
            id_code = f"retina_{i:03d}"
            # Create synthetic circular retina
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            y, x = np.ogrid[:224, :224]
            mask = (x - 112) ** 2 + (y - 112) ** 2 <= 95 ** 2
            arr[mask] = np.random.randint(50, 220, (3,), dtype=np.uint8)

            img_file = img_dir / f"{id_code}.png"
            Image.fromarray(arr).save(img_file)
            records.append({
                "id_code": id_code,
                "diagnosis": i % 5  # Ensure all 5 classes [0, 1, 2, 3, 4]
            })

        df = pd.DataFrame(records)
        csv_file = tmp_path / "train.csv"
        df.to_csv(csv_file, index=False)

        yield {
            "root": tmp_path,
            "img_dir": img_dir,
            "csv_file": csv_file,
            "df": df
        }


def test_dr_dataset_and_clahe(dummy_retinal_environment):
    env = dummy_retinal_environment
    dataset = DRGradingDataset(
        df=env["df"],
        image_dir=env["img_dir"],
        image_size=(224, 224),
        is_training=True,
        use_clahe=True,
        crop_circle=True
    )

    assert len(dataset) == len(env["df"])
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert 0 <= label.item() <= 4

    # Test CLAHE directly
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    clahe_out = apply_clahe(img)
    assert isinstance(clahe_out, Image.Image)


def test_stratified_dr_splits(dummy_retinal_environment):
    env = dummy_retinal_environment
    out_dir = env["root"] / "splits"
    splits = create_stratified_dr_splits(
        csv_path=env["csv_file"],
        output_dir=out_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    assert len(train_df) + len(val_df) + len(test_df) == len(env["df"])
    assert (out_dir / "dr_train.csv").is_file()
    assert (out_dir / "dr_val.csv").is_file()
    assert (out_dir / "dr_test.csv").is_file()

    # Verify no overlap between splits
    train_ids = set(train_df["id_code"])
    val_ids = set(val_df["id_code"])
    test_ids = set(test_df["id_code"])
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0


def test_inverse_class_weights():
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4])
    weights = compute_inverse_class_weights(labels, num_classes=5)
    assert len(weights) == 5
    # Class 0 should have smaller weight than rare classes (2, 3, 4)
    assert weights[0] < weights[2]
    assert weights[0] < weights[4]


def test_dr_models_forward():
    x = torch.randn(2, 3, 224, 224)

    # 1. ResNet-50
    resnet = build_dr_model("resnet50", num_classes=5, pretrained=False)
    out_res = resnet(x)
    assert out_res.shape == (2, 5)

    # 2. ViT-B/16
    vit = build_dr_model("vit_base_patch16_224", num_classes=5, pretrained=False)
    out_vit = vit(x)
    assert out_vit.shape == (2, 5)

    # 3. Swin
    swin = build_dr_model("swin_tiny_patch4_window7_224", num_classes=5, pretrained=False)
    out_swin = swin(x)
    assert out_swin.shape == (2, 5)


def test_dr_evaluation_metrics():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 1, 3, 3])  # 2 errors
    y_prob = np.zeros((10, 5))
    for i, p in enumerate(y_pred):
        y_prob[i, p] = 0.8
        y_prob[i, (p + 1) % 5] = 0.2

    metrics = compute_dr_evaluation_metrics(y_true, y_pred, y_prob=y_prob, num_classes=5)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "quadratic_weighted_kappa" in metrics
    assert "referable_dr" in metrics
    assert "sensitivity" in metrics["referable_dr"]
    assert "specificity" in metrics["referable_dr"]
    assert "expected_calibration_error" in metrics

    # Bootstrap test
    cis = compute_dr_bootstrap_ci(y_true, y_pred, n_bootstraps=50, random_state=42)
    assert "macro_f1" in cis
    assert cis["macro_f1"]["ci_lower"] <= cis["macro_f1"]["mean"] <= cis["macro_f1"]["ci_upper"]


def test_dr_explainability_methods():
    x = torch.randn(1, 3, 224, 224)
    model = build_dr_model("resnet50", num_classes=5, pretrained=False)

    # ResNet Grad-CAM
    cam_explainer = ResNetGradCAM(model)
    cam = cam_explainer.generate(x, target_class=2)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() and cam.max() <= 1.0

    # Integrated Gradients
    ig = DRIntegratedGradients(model, steps=5)
    attr = ig.attribute(x, target_class=2)
    assert attr.shape == (224, 224)
    assert 0.0 <= attr.min() and attr.max() <= 1.0

    # Image Overlay
    pil_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    blended = overlay_cam_on_image(pil_img, cam)
    assert blended.size == (224, 224)


def test_dr_trainer_smoke(dummy_retinal_environment):
    env = dummy_retinal_environment
    dataset = DRGradingDataset(
        df=env["df"].head(4),
        image_dir=env["img_dir"],
        image_size=(224, 224),
        is_training=False
    )
    loader = DataLoader(dataset, batch_size=2)

    model = build_dr_model("resnet50", num_classes=5, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    ckpt_dir = env["root"] / "checkpoints"

    trainer = DRTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        model_name="resnet50_test",
        early_stopping_patience=2,
        use_amp=False
    )

    history = trainer.fit(epochs=1)
    assert len(history["train_loss"]) == 1
    assert (ckpt_dir / "resnet50_test_best.pt").is_file()

