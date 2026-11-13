"""Clinical Label Provenance and Target Leakage Safeguard Module for Track B.

Strictly enforces:
1. Categorization of diabetes ground-truth labels by evidentiary tier:
   - LABORATORY_CONFIRMED (HbA1c >= 6.5%, FPG >= 126 mg/dL, OGTT >= 200 mg/dL)
   - CLINICIAN_DOCUMENTED (EHR diagnosis with ICD-10 E11)
   - SELF_REPORTED (Questionnaire / antecedent interview)
   - UNKNOWN (Missing documentation - NEVER assumed non-diabetic)
   - INVALID_DR_INFERRED (Strictly prohibited)
2. Target Leakage Prevention:
   - Strictly prohibits using HbA1c, fasting glucose, or active hypoglycemic agents
     as input features when predicting diabetes risk.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DiabetesProvenanceTier(str, Enum):
    LABORATORY_CONFIRMED = "LABORATORY_CONFIRMED"
    CLINICIAN_DOCUMENTED = "CLINICIAN_DOCUMENTED"
    SELF_REPORTED = "SELF_REPORTED"
    UNKNOWN = "UNKNOWN"
    INVALID_DR_INFERRED = "INVALID_DR_INFERRED"


# Biomarkers that constitute target leakage if present in input features
LEAKAGE_BIOMARKER_COLUMNS = [
    "hba1c", "glycated_hemoglobin", "hba1c_pct", "hba1c_mmol_mol",
    "fpg", "fasting_plasma_glucose", "fasting_blood_glucose", "fasting_glucose", "glucose", "fbg",
    "ogtt", "oral_glucose_tolerance_test", "random_blood_glucose", "rbg",
    "insulin_prescribed", "metformin_prescribed", "oral_hypoglycemics"
]


class LabelProvenanceError(ValueError):
    """Raised when an invalid, inferred, or leaking label is detected."""
    pass


class TargetLeakageError(ValueError):
    """Raised when a diagnostic ground-truth measurement is used as an input feature."""
    pass


def validate_diabetes_label_provenance(
    label: Any,
    provenance_source: str,
    dr_grade: Optional[int] = None,
    allow_self_reported: bool = False
) -> Tuple[bool, DiabetesProvenanceTier, str]:
    """Validate a candidate diabetes label against clinical provenance rules.
    
    Returns:
        (is_valid_for_training, tier, message)
    """
    prov_clean = str(provenance_source).upper().strip()

    # Rule 1: DR grades must never be used to infer diabetes labels
    if "DR" in prov_clean or "RETINOPATHY" in prov_clean or "APTOS" in prov_clean:
        return False, DiabetesProvenanceTier.INVALID_DR_INFERRED, (
            "Prohibited: Diabetes label was derived from DR severity grades. "
            "A normal retina does not confirm non-diabetic status."
        )

    # Rule 2: Missing or None labels must NEVER be converted to non-diabetic (0)
    if label is None or pd.isna(label) or label == "" or str(label).lower() in ["none", "nan", "unknown"]:
        return False, DiabetesProvenanceTier.UNKNOWN, "Rejected: Missing diabetes status cannot be assumed non-diabetic."

    # Parse numeric binary label
    try:
        int_label = int(label)
        if int_label not in (0, 1):
            return False, DiabetesProvenanceTier.UNKNOWN, f"Invalid binary label value: {label}"
    except (ValueError, TypeError):
        return False, DiabetesProvenanceTier.UNKNOWN, f"Non-numeric diabetes label: {label}"

    # Rule 3: Tier identification
    if any(k in prov_clean for k in ["LAB", "HBA1C", "FPG", "OGTT", "VENOUS_BLOOD"]):
        return True, DiabetesProvenanceTier.LABORATORY_CONFIRMED, "Verified: Laboratory-confirmed clinical ground truth."

    elif any(k in prov_clean for k in ["EHR", "CLINICIAN", "DOCTOR", "HOSPITAL", "ICD"]):
        return True, DiabetesProvenanceTier.CLINICIAN_DOCUMENTED, "Verified: Clinician-documented EHR diagnosis."

    elif any(k in prov_clean for k in ["SELF", "SURVEY", "QUESTIONNAIRE", "INTERVIEW", "ANTECEDENT"]):
        if allow_self_reported:
            return True, DiabetesProvenanceTier.SELF_REPORTED, "Accepted: Self-reported label (explicitly authorized)."
        else:
            return False, DiabetesProvenanceTier.SELF_REPORTED, (
                "Rejected: Self-reported label excluded by strict laboratory validation protocol. "
                "Pass allow_self_reported=True to permit."
            )

    else:
        return False, DiabetesProvenanceTier.UNKNOWN, f"Unrecognized provenance source: '{provenance_source}'."


def assert_no_target_leakage(feature_names: List[str]) -> None:
    """Assert that no laboratory diagnostic biomarkers defining diabetes exist in the feature set."""
    normalized_features = [f.lower().strip().replace(" ", "_").replace("-", "_") for f in feature_names]
    leaking_cols = [f for f in normalized_features if any(leak in f for leak in LEAKAGE_BIOMARKER_COLUMNS)]

    if leaking_cols:
        raise TargetLeakageError(
            f"Target leakage detected! The input feature set contains diagnostic biomarkers: {leaking_cols}. "
            f"Using laboratory glucose or HbA1c to predict diabetes status is scientifically invalid."
        )


def filter_manifest_for_track_b(
    manifest_df: pd.DataFrame,
    allow_self_reported: bool = False,
    dr0_only: bool = False
) -> pd.DataFrame:
    """Filter and validate a patient manifest strictly for Track B training or evaluation.
    
    Guarantees:
    - Every retained sample has a valid, verified diabetes label.
    - Zero samples with DR-inferred labels.
    - Zero samples with unknown status assumed to be 0.
    - Optional isolation of the DR = 0 (no visible retinopathy) research cohort.
    """
    valid_rows = []
    
    for idx, row in manifest_df.iterrows():
        label = row.get("diabetes_label")
        prov = row.get("diabetes_label_source", "Unknown")
        dr_grade = row.get("dr_grade", None)

        is_valid, tier, msg = validate_diabetes_label_provenance(
            label=label,
            provenance_source=prov,
            dr_grade=dr_grade,
            allow_self_reported=allow_self_reported
        )

        if is_valid:
            if dr0_only and dr_grade is not None and int(dr_grade) != 0:
                continue
            r_dict = row.to_dict()
            r_dict["provenance_tier"] = tier.value
            valid_rows.append(r_dict)

    filtered_df = pd.DataFrame(valid_rows)
    return filtered_df
