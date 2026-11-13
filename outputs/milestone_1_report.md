# Dataset Readiness, Clinical Label Audit & Research Feasibility Report

**Project**: Explainable AI for Early Type 2 Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Verification Engine**: `scripts/audit_datasets.py`  

---

## 1. Discoverable Datasets Inventory

| Dataset ID | Full Name | Status on Disk | Images Found | Track A (DR) | Track B (Diabetes) | Access & Governance Protocol |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **APTOS_2019** | APTOS 2019 Blindness Detection | `NOT_DOWNLOADED` | 0 | Eligible | Ineligible | Kaggle Account & Competition Rules Acceptance |
| **BRSET** | Brazilian Retinal Screening Dataset | `NOT_DOWNLOADED` | 0 | Eligible | Eligible | PhysioNet Credentialed Access (CITI Training, Data Use Agreement) |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset | `NOT_DOWNLOADED` | 0 | Eligible | Ineligible | IEEE Dataport Registration / Download |
| **Messidor_2** | Messidor-2 Retinal Image Dataset | `NOT_DOWNLOADED` | 0 | Eligible | Ineligible | Messidor Research Permission Agreement |

> [!NOTE]
> **Clinical Provenance & Partitioning Notice**:
> - **APTOS 2019 / IDRiD**: All recruited participants are diabetic. Grade 0 eyes are diabetic without retinopathy; they CANNOT serve as non-diabetic controls for Track B.
> - **BRSET (PhysioNet)**: Contains true EHR-confirmed systemic diabetes labels across both diabetic cases and verified non-diabetic controls. Primary candidate for Track B.

---

## 2. Active Manifest Audit Summary

- **Total Records Ingested**: 22
- **Unique Patient Count**: 12
- **Patient Identifier Completeness**: 100.0%
- **Average Images per Patient**: 1.83

### Clinical Label Provenance Breakdown
```json
{
  "LABORATORY_CONFIRMED": 10,
  "CLINICIAN_DOCUMENTED": 10,
  "SELF_REPORTED": 1,
  "UNKNOWN": 1,
  "INVALID_DR_INFERRED": 0
}
```

- **Valid Track B Screening Candidates**: 21
- **Early Research Target (DR = 0 & Diabetic)**: 5
- **Prohibited DR-Inferred Labels**: 0

### Research Eligibility Assessment
- **Track A (5-Class DR Severity Grading)**: ELIGIBLE
- **Track B (Systemic Diabetes Screening)**: ELIGIBLE
- **Isolated DR = 0 Pre-Clinical Remodeling Analysis**: FEASIBLE

### Quality & Leakage Audit Alerts
No patient leakage or target leakage detected in audited manifest.
