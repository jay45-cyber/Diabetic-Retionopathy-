# Diabetes Ground-Truth Label Provenance & Data Inventory

**Document**: `docs/diabetes_label_provenance.md`  
**Date**: 2026-11-13  
**Lead AI Research Engineer**: Jay (`jay45-cyber`)  
**Purpose**: Rigorous accounting of label sources, evidentiary tiers, and suitability across all candidate datasets.

---

## 1. Provenance Inventory by Dataset

The project systematically distinguishes datasets suitable for Track A (DR Severity Staging) versus Track B (Systemic Diabetes Prediction):

| Dataset ID | Modality & Camera | Total Images | Available Diabetes Label | Label Provenance Source | Evidentiary Tier | Suitability for Track B (Diabetes Prediction) | Suitability for Track A (DR Grading) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **BRSET** | 45° Macula/Disc (Canon CR-2, Nikon) | 16,266 | Yes (`diabetes == 1 / 0`) | Electronic Health Records & Clinical Interview (Brazil Public Network) | `CLINICIAN_DOCUMENTED` / `SELF_REPORTED` | **SUITABLE** (Primary candidate; requires provenance separation) | Suitable |
| **APTOS 2019** | Macula-centered (Topcon Handheld) | 3,662 | **NO** | Re-labeled from Aravind Eye Hospital Diabetic Clinics (All patients have diabetes) | `UNKNOWN` (Zero non-diabetic controls) | **UNSUITABLE** (Grade 0 $\not\equiv$ Non-diabetic; using APTOS for diabetes causes inverted ground truth) | **PRIMARY BENCHMARK** (Track A) |
| **IDRiD** | 50° Field (Kowa VX-10 $\alpha$) | 516 | **NO** | Indian Eye Clinic DR grading protocol | `UNKNOWN` (Only DR and DME graded) | **UNSUITABLE** for diabetes prediction | Suitable for DR / DME |
| **Messidor / Messidor-2** | 45° Color Fundus (Topcon TRC NW6) | 1,200 / 1,748 | Partial | Diabetic eye screening cohorts | `UNKNOWN` / Mixed | **UNSUITABLE** for primary diabetes prediction without verified blood draw | Suitable for DR validation |
| **EyePACS** | Diverse clinical cameras | 88,702 | **NO** | Tele-ophthalmology screening for retinopathy | `UNKNOWN` | **UNSUITABLE** for diabetes prediction | Suitable for large-scale DR pre-training |

---

## 2. Detailed Dataset Audit Findings

### A. Brazilian Retinal Screening Dataset (BRSET)
- **Source**: Nature Scientific Data (2024), DOI: 10.1038/s41597-024-03096-9.
- **Patient Count**: 8,524 participants.
- **Image Count**: 16,266 bilateral photographs (OD and OS).
- **Available Covariates**: Age, biological sex, camera model, laterality, image quality score, diabetes status, diabetic retinopathy ICDR grade (0 to 4), macular edema presence.
- **Label Provenance Nuance**: While BRSET records `diabetes = 1` or `0`, the label represents clinical EHR intake and self-reported antecedent diabetes at community screening intake, rather than a prospective venous HbA1c test conducted on the same morning. Therefore, models trained on BRSET must be designated as predicting **documented systemic diabetes status**, with explicit recognition of potential under-diagnosis in rural participants who have never received blood testing.

### B. APTOS 2019 Blindness Detection (Aravind Eye Hospital)
- **Source**: Kaggle / Aravind Eye Hospital, Tamil Nadu, India.
- **Clinical Recuitment Context**: Patients were recruited specifically from specialized diabetic retinopathy examination clinics.
- **Critical Safety Guard**: A participant with Grade 0 DR in APTOS is a diabetic patient who has not yet developed visible microvascular retinopathy. If an engineer converts `diagnosis == 0` into a non-diabetic label and `diagnosis > 0` into a diabetic label, they introduce catastrophic target corruption, training the network to treat diabetic individuals as controls. The project strictly enforces that APTOS is restricted to Track A (5-class DR severity).

---

## 3. Label Provenance Enforcement Architecture

The system enforces label integrity through `src/data/label_provenance.py`:

```python
from src.data.label_provenance import validate_diabetes_label_provenance, DiabetesProvenanceTier

# Example 1: Prohibited DR inference attempt
valid, tier, msg = validate_diabetes_label_provenance(
    label=0, 
    provenance_source="APTOS_DR_Grade_0"
)
# Result: valid == False, tier == DiabetesProvenanceTier.INVALID_DR_INFERRED

# Example 2: Missing / unknown label attempt
valid, tier, msg = validate_diabetes_label_provenance(
    label=None,
    provenance_source="Unknown"
)
# Result: valid == False, tier == DiabetesProvenanceTier.UNKNOWN (Never converted to 0)

# Example 3: Verified EHR diagnosis
valid, tier, msg = validate_diabetes_label_provenance(
    label=1,
    provenance_source="EHR_Doctor_Diagnosis"
)
# Result: valid == True, tier == DiabetesProvenanceTier.CLINICIAN_DOCUMENTED
```

---

## 4. Ground-Truth Data Availability Notice

As established in the workspace audit, local hardware constraints and privacy boundaries prevent storing multi-gigabyte raw patient fundus archives in Git. When local raw datasets are not mounted:
1. The code pipeline, dataset adapters, quality checkers, model architectures, trainers, and evaluators are fully implemented and verified via automated synthetic unit and smoke tests.
2. The system documents that multi-epoch training on real multi-gigabyte clinical cohorts (BRSET) must be executed on cloud GPU instances (via `notebooks/colab_gpu_pipeline.ipynb`) or an authorized secure medical computing cluster.
3. No clinical training results or validation claims are fabricated.
