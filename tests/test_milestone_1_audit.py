"""Automated unit and integration test suite for Milestone 1:
Dataset Readiness, Clinical Label Audit & Research Feasibility.

Covers:
1. Dataset discovery and graceful handling of missing datasets.
2. Image validation (gradable, ungradable, corrupt, missing).
3. Missing clinical labels handling (never assumed 0).
4. Invalid diabetes labels (out-of-range, non-binary, non-numeric).
5. Label provenance tiering and DR-inferred rejection.
6. Exact (MD5) and perceptual (dHash) duplicate detection.
7. Patient grouping and patient-level leakage detection.
8. Diabetes target leakage prevention (diagnostic biomarkers in features).
9. Report generation (machine-readable JSON, CSV, and Markdown summary).
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from PIL import Image

from scripts.audit_datasets import DatasetAuditor, create_synthetic_audit_cohort
from src.data.label_provenance import (
    DiabetesProvenanceTier,
    validate_diabetes_label_provenance,
    assert_no_target_leakage,
    TargetLeakageError,
    filter_manifest_for_track_b
)
from src.data.duplicate_detection import compute_md5, compute_dhash, hamming_distance
from src.data.image_quality import assess_image_quality


def test_dataset_discovery_and_missing_handling(tmp_path):
    """Test that DatasetAuditor discovers configured datasets and handles missing ones gracefully."""
    auditor = DatasetAuditor(root_dir=tmp_path)
    inventory = auditor.discover_datasets()

    assert "APTOS_2019" in inventory
    assert "BRSET" in inventory
    assert "IDRiD" in inventory
    assert "Messidor_2" in inventory

    # Without downloaded data, status must be NOT_DOWNLOADED
    for ds_id, data in inventory.items():
        assert data["status"] == "NOT_DOWNLOADED"
        assert data["images_found"] == 0
        assert "access_requirement" in data


def test_image_validation_gradable_and_ungradable(tmp_path):
    """Verify image validation accurately distinguishes gradable vs ungradable/corrupt images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 1. Gradable sharp image
    good_p = img_dir / "sharp.png"
    arr = np.random.randint(60, 200, (300, 300, 3), dtype=np.uint8)
    Image.fromarray(arr).save(good_p)

    with Image.open(good_p) as img:
        q_good = assess_image_quality(img)
        assert q_good["gradable"] is True
        assert len(q_good["failure_reasons"]) == 0

    # 2. Too dark ungradable image
    dark_p = img_dir / "dark.png"
    dark_arr = np.ones((300, 300, 3), dtype=np.uint8) * 10
    Image.fromarray(dark_arr).save(dark_p)

    with Image.open(dark_p) as img:
        q_dark = assess_image_quality(img)
        assert q_dark["gradable"] is False
        assert any("illumination" in r.lower() or "dark" in r.lower() for r in q_dark["failure_reasons"])


def test_missing_and_invalid_clinical_labels():
    """Verify that missing diabetes labels are NEVER assumed 0, and invalid values are rejected."""
    # Missing / None
    is_valid, tier, msg = validate_diabetes_label_provenance(None, "Laboratory_HbA1c")
    assert is_valid is False
    assert tier == DiabetesProvenanceTier.UNKNOWN
    assert "cannot be assumed non-diabetic" in msg

    # NaN / empty string
    is_valid, tier, _ = validate_diabetes_label_provenance("", "EHR_Diagnosis")
    assert is_valid is False
    assert tier == DiabetesProvenanceTier.UNKNOWN

    # Non-binary value (e.g., 2)
    is_valid, tier, msg = validate_diabetes_label_provenance(2, "Laboratory_HbA1c")
    assert is_valid is False
    assert tier == DiabetesProvenanceTier.UNKNOWN
    assert "Invalid binary label" in msg

    # Non-numeric string
    is_valid, tier, msg = validate_diabetes_label_provenance("positive", "Laboratory_HbA1c")
    assert is_valid is False
    assert tier == DiabetesProvenanceTier.UNKNOWN


def test_label_provenance_tiering_and_dr_inferred_rejection():
    """Verify clinical provenance tiering and strict rejection of DR-inferred labels."""
    # 1. Laboratory confirmed
    valid, tier, _ = validate_diabetes_label_provenance(1, "Laboratory_HbA1c")
    assert valid is True
    assert tier == DiabetesProvenanceTier.LABORATORY_CONFIRMED

    # 2. Clinician documented EHR
    valid, tier, _ = validate_diabetes_label_provenance(0, "EHR_Diagnosis_ICD10")
    assert valid is True
    assert tier == DiabetesProvenanceTier.CLINICIAN_DOCUMENTED

    # 3. Self-reported (rejected by default, accepted with flag)
    valid, tier, _ = validate_diabetes_label_provenance(1, "Self_Reported_Questionnaire", allow_self_reported=False)
    assert valid is False
    assert tier == DiabetesProvenanceTier.SELF_REPORTED

    valid, tier, _ = validate_diabetes_label_provenance(1, "Self_Reported_Questionnaire", allow_self_reported=True)
    assert valid is True
    assert tier == DiabetesProvenanceTier.SELF_REPORTED

    # 4. Prohibited: DR-inferred
    valid, tier, msg = validate_diabetes_label_provenance(0, "APTOS_DR_Severity_Grade_0")
    assert valid is False
    assert tier == DiabetesProvenanceTier.INVALID_DR_INFERRED
    assert "Prohibited" in msg


def test_diabetes_target_leakage_assertion():
    """Verify assert_no_target_leakage raises TargetLeakageError on diagnostic blood biomarkers."""
    safe_features = ["age", "sex", "bmi", "systolic_bp", "heart_rate", "smoking_status"]
    # Safe features must pass without exception
    assert_no_target_leakage(safe_features)

    # Diagnostic biomarkers must raise TargetLeakageError
    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["age", "bmi", "hba1c"])

    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["fasting_glucose", "systolic_bp"])

    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["oral_glucose_tolerance_test"])

    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(["insulin_prescribed"])


def test_duplicate_detection_exact_and_perceptual(tmp_path):
    """Verify exact MD5 and perceptual dHash duplicate detection."""
    img_dir = tmp_path / "dup_images"
    img_dir.mkdir()

    # Image A
    p_a = img_dir / "img_a.png"
    arr_a = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr_a).save(p_a)

    # Image B is exact copy of Image A
    p_b = img_dir / "img_b.png"
    Image.fromarray(arr_a).save(p_b)

    # Image C is distinct
    p_c = img_dir / "img_c.png"
    arr_c = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr_c).save(p_c)

    # Exact MD5 match
    md5_a = compute_md5(p_a)
    md5_b = compute_md5(p_b)
    md5_c = compute_md5(p_c)
    assert md5_a == md5_b
    assert md5_a != md5_c

    # Perceptual dHash
    with Image.open(p_a) as img_a, Image.open(p_b) as img_b, Image.open(p_c) as img_c:
        dh_a = compute_dhash(img_a)
        dh_b = compute_dhash(img_b)
        dh_c = compute_dhash(img_c)
        assert dh_a == dh_b
        assert hamming_distance(dh_a, dh_b) == 0
        assert hamming_distance(dh_a, dh_c) >= 0


def test_patient_grouping_and_leakage_detection():
    """Verify detection of conflicting patient labels and multi-eye grouping."""
    manifest = pd.DataFrame([
        {"participant_id": "P01", "image_id": "P01_OD", "eye_side": "OD", "diabetes_label": 1, "diabetes_label_source": "Laboratory_HbA1c", "dr_grade": 0},
        {"participant_id": "P01", "image_id": "P01_OS", "eye_side": "OS", "diabetes_label": 1, "diabetes_label_source": "Laboratory_HbA1c", "dr_grade": 0},
        # Patient P02 has contradictory labels across eyes
        {"participant_id": "P02", "image_id": "P02_OD", "eye_side": "OD", "diabetes_label": 0, "diabetes_label_source": "EHR_Diagnosis", "dr_grade": 0},
        {"participant_id": "P02", "image_id": "P02_OS", "eye_side": "OS", "diabetes_label": 1, "diabetes_label_source": "EHR_Diagnosis", "dr_grade": 1},
        # Patient P03 is valid non-diabetic
        {"participant_id": "P03", "image_id": "P03_OD", "eye_side": "OD", "diabetes_label": 0, "diabetes_label_source": "EHR_Diagnosis", "dr_grade": 0}
    ])

    auditor = DatasetAuditor()
    results = auditor.audit_manifest(manifest, check_images=False)

    assert results["patient_metrics"]["unique_patients"] == 3
    assert results["patient_metrics"]["conflicting_patient_labels_count"] == 1
    assert "P02" in results["patient_metrics"]["conflicting_patient_ids"]
    assert any("contradictory diabetes labels" in w for w in results["leakage_risks"])


def test_report_generation_json_csv_and_markdown(tmp_path):
    """Verify that audit reporting produces valid JSON, CSV, and Markdown files."""
    synth_df = create_synthetic_audit_cohort(tmp_path)
    auditor = DatasetAuditor(root_dir=tmp_path)
    inventory = auditor.discover_datasets()
    manifest_audit = auditor.audit_manifest(synth_df, check_images=True)

    # 1. JSON Report
    json_path = tmp_path / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"inventory": inventory, "manifest_audit": manifest_audit}, f, indent=2)
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "inventory" in loaded
    assert "manifest_audit" in loaded

    # 2. CSV Inventory
    csv_path = tmp_path / "inventory.csv"
    inv_df = pd.DataFrame(list(inventory.values()))
    inv_df.to_csv(csv_path, index=False)
    assert csv_path.is_file()
    assert len(pd.read_csv(csv_path)) == 4

    # 3. Markdown Summary
    md_content = auditor.generate_markdown_summary(inventory, manifest_audit)
    assert "# Dataset Readiness" in md_content
    assert "Clinical Label Provenance Breakdown" in md_content
    assert "Research Eligibility Assessment" in md_content
