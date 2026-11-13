"""Comprehensive Unit and Integration Tests for Track B: Systemic Diabetes Risk Screening.

Verifies:
1. Label Provenance & Evidentiary Tiers (rejection of DR-inferred labels).
2. Prevention of Target Leakage in tabular features.
3. DiabetesScreeningDataset loading & DR=0 cohort isolation.
4. Model forward passes (ResNet-50, ConvNeXt, Foundation ViT, Multimodal).
5. DiabetesRiskTrainer end-to-end training and checkpointing loop.
6. Screening evaluation metrics, calibration (ECE), and patient bootstrap CIs.
7. DiabetesRiskExplainer saliency map generation on binary logits.
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

from src.data.label_provenance import (
    validate_diabetes_label_provenance,
    assert_no_target_leakage,
    DiabetesProvenanceTier,
    TargetLeakageError
)
from src.data.diabetes_dataset import DiabetesScreeningDataset
from src.models.diabetes_models import build_diabetes_model, ResNet50Diabetes
from src.training.diabetes_trainer import DiabetesRiskTrainer
from src.evaluation.classification_metrics import compute_screening_metrics
from src.evaluation.calibration import PlattCalibrator, compute_expected_calibration_error
from src.evaluation.confidence_intervals import compute_patient_bootstrap_ci
from src.explainability.diabetes_explainer import DiabetesRiskExplainer


@pytest.fixture
def dummy_diabetes_env():
    """Create temporary directory with synthetic fundus images and clinical metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "brset_images"
        img_dir.mkdir()

        records = []
        for i in range(24):
            pid = f"patient_{i:03d}"
            # Synthetic retinal image
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            y, x = np.ogrid[:224, :224]
            mask = (x - 112) ** 2 + (y - 112) ** 2 <= 95 ** 2
            arr[mask] = np.random.randint(40, 210, (3,), dtype=np.uint8)

            img_file = img_dir / f"{pid}.png"
            Image.fromarray(arr).save(img_file)

            records.append({
                "participant_id": pid,
                "image_id": pid,
                "image_path": str(img_file),
                "diabetes_label": 1 if i % 2 == 0 else 0,
                "diabetes_label_source": "EHR_Diagnosis",
                "dr_grade": 0 if i < 12 else (i % 4 + 1),  # First 12 are DR = 0
                "age": 50 + (i % 20),
                "sex": i % 2,
                "bmi": 25.0 + (i % 10),
                "systolic_bp": 120 + (i % 30)
            })

        df = pd.DataFrame(records)
        yield {
            "root": tmp_path,
            "img_dir": img_dir,
            "df": df
        }


def test_diabetes_label_provenance():
    # 1. Valid Laboratory Confirmation
    valid, tier, _ = validate_diabetes_label_provenance(1, "LAB_HBA1C_VENOUS")
    assert valid is True
    assert tier == DiabetesProvenanceTier.LABORATORY_CONFIRMED

    # 2. Valid Clinician Documentation (EHR)
    valid, tier, _ = validate_diabetes_label_provenance(0, "EHR_HOSPITAL_DIAGNOSIS")
    assert valid is True
    assert tier == DiabetesProvenanceTier.CLINICIAN_DOCUMENTED

    # 3. Self-reported: excluded by default, accepted with flag
    valid_default, tier, _ = validate_diabetes_label_provenance(1, "PATIENT_SURVEY", allow_self_reported=False)
    assert valid_default is False
    assert tier == DiabetesProvenanceTier.SELF_REPORTED

    valid_allowed, tier, _ = validate_diabetes_label_provenance(1, "PATIENT_SURVEY", allow_self_reported=True)
    assert valid_allowed is True

    # 4. Unknown / Missing Label: Must NEVER be converted to 0
    valid_none, tier, _ = validate_diabetes_label_provenance(None, "EHR")
    assert valid_none is False
    assert tier == DiabetesProvenanceTier.UNKNOWN

    # 5. Invalid DR-inferred Label: Prohibited
    valid_dr, tier, _ = validate_diabetes_label_provenance(0, "APTOS_DR_GRADE_0")
    assert valid_dr is False
    assert tier == DiabetesProvenanceTier.INVALID_DR_INFERRED


def test_target_leakage_assertion():
    # Legitimate screening covariates: Should PASS
    valid_features = ["age", "biological_sex", "bmi", "systolic_blood_pressure", "family_history"]
    assert_no_target_leakage(valid_features)

    # Tautological diagnostic biomarkers: Must RAISE TargetLeakageError
    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["age", "bmi", "hba1c"])

    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["fasting_plasma_glucose", "systolic_bp"])

    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["age", "insulin_prescribed"])


def test_diabetes_dataset_and_dr0_filtering(dummy_diabetes_env):
    env = dummy_diabetes_env
    
    # Full dataset
    full_dataset = DiabetesScreeningDataset(
        df=env["df"],
        image_dir=env["img_dir"],
        image_size=(224, 224),
        is_training=False,
        dr0_only=False
    )
    assert len(full_dataset) == 24
    img, target = full_dataset[0]
    assert img.shape == (3, 224, 224)
    assert target.shape == ()

    # DR = 0 Subgroup dataset (only first 12 records had dr_grade == 0)
    dr0_dataset = DiabetesScreeningDataset(
        df=env["df"],
        image_dir=env["img_dir"],
        image_size=(224, 224),
        is_training=False,
        dr0_only=True
    )
    assert len(dr0_dataset) == 12

    # Multimodal dataset
    multimodal_dataset = DiabetesScreeningDataset(
        df=env["df"],
        image_dir=env["img_dir"],
        is_training=False,
        clinical_feature_cols=["age", "sex", "bmi", "systolic_bp"]
    )
    img, clin, target = multimodal_dataset[0]
    assert img.shape == (3, 224, 224)
    assert clin.shape == (4,)
    assert target.shape == ()


def test_diabetes_models_forward():
    x = torch.randn(2, 3, 224, 224)
    c = torch.randn(2, 4)

    # 1. ResNet-50
    resnet = build_diabetes_model("resnet50", pretrained=False)
    out_res = resnet(x)
    assert out_res.shape == (2,)

    # 2. ConvNeXt
    convnext = build_diabetes_model("convnext", pretrained=False)
    out_conv = convnext(x)
    assert out_conv.shape == (2,)

    # 3. Foundation Model Adapter
    fnd = build_diabetes_model("foundation_vit", pretrained=False)
    out_fnd = fnd(x)
    assert out_fnd.shape == (2,)

    # 4. Multimodal Fusion
    multimodal = build_diabetes_model("multimodal", pretrained=False, num_clinical_features=4)
    out_multi = multimodal(x, c)
    assert out_multi.shape == (2,)


def test_diabetes_trainer_smoke(dummy_diabetes_env):
    env = dummy_diabetes_env
    dataset = DiabetesScreeningDataset(
        df=env["df"].head(4),
        image_dir=env["img_dir"],
        image_size=(224, 224),
        is_training=True
    )
    loader = DataLoader(dataset, batch_size=2)

    model = build_diabetes_model("resnet50", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    ckpt_dir = env["root"] / "checkpoints"

    trainer = DiabetesRiskTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        model_name="resnet50_diabetes_test",
        early_stopping_patience=2,
        use_amp=False
    )

    history = trainer.fit(epochs=1)
    assert len(history["train_loss"]) == 1
    assert (ckpt_dir / "resnet50_diabetes_test_best.pt").is_file()


def test_diabetes_evaluation_and_calibration():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85])
    pids = np.array([f"P_{i}" for i in range(10)])

    # Screening metrics
    metrics = compute_screening_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0

    # Calibration (ECE)
    ece_res = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert "expected_calibration_error" in ece_res
    assert 0.0 <= ece_res["expected_calibration_error"] <= 1.0

    # Platt Calibrator
    logits = np.array([-2.0, 2.0, -1.5, 1.5, -1.0, 1.0, -0.5, 0.5, -2.2, 2.2])
    calibrator = PlattCalibrator()
    calibrator.fit(logits, y_true)
    cal_probs = calibrator.predict_proba(logits)
    assert len(cal_probs) == 10
    assert 0.0 <= cal_probs.min() and cal_probs.max() <= 1.0

    # Patient Bootstrap CI
    cis = compute_patient_bootstrap_ci(pids, y_true, y_prob, n_bootstraps=50, random_seed=42)
    assert "auroc" in cis
    assert cis["auroc"]["ci_lower"] <= cis["auroc"]["mean"] <= cis["auroc"]["ci_upper"]


def test_diabetes_explainer():
    x = torch.randn(1, 3, 224, 224)
    model = build_diabetes_model("resnet50", pretrained=False)

    explainer = DiabetesRiskExplainer(model, model_architecture="resnet50")
    
    # Grad-CAM on binary logit
    cam = explainer.generate_gradcam(x)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() and cam.max() <= 1.0

    # Integrated Gradients on binary logit
    ig = explainer.generate_integrated_gradients(x, steps=5)
    assert ig.shape == (224, 224)
    assert 0.0 <= ig.min() and ig.max() <= 1.0
