# Dataset Feasibility & Ground-Truth Provenance Audit

**Document**: `docs/dataset_feasibility.md` (See also `docs/dataset_feasibility_audit.md` and `docs/diabetes_label_provenance.md`)  
**Project Identity**: Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Date**: 2026-11-13  
**Status**: **ACTIVE STANDARD PROTOCOL**

---

## 1. Executive Summary & Critical Distinction

A fatal methodological flaw in naive medical AI implementations is assuming that public Diabetic Retinopathy (DR) datasets can be directly repurposed for Type 2 Diabetes detection by labeling Grade 0 as "No Diabetes" and Grades 1–4 as "Diabetes".

> [!WARNING]
> In clinical cohorts such as **APTOS 2019** and **IDRiD**, all participating patients were recruited from diabetic specialty clinics. A patient with Grade 0 DR in APTOS is **diabetic without visible retinopathy**. Treating Grade 0 as "Non-Diabetic" completely invalidates the clinical label and causes severe label contamination.

---

## 2. Systematic Candidate Dataset Audit

| Dataset | Origin & Population | Patients / Images | Diabetes Ground Truth | DR Grading | Suitability Classification | Data Access & License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BRSET** (Brazilian Retinal Screening) | Brazil (multi-ethnic public health system) | 8,524 patients / 16,266 images | **Yes** (EHR-confirmed Type 2 Diabetes; Yes/No) | Yes (ICDR 0–4) | **PRIMARY CANDIDATE**: Contains both diabetic and non-diabetic cohorts across all DR grades, enabling DR=0 analysis. | Open Access (PhysioNet Credentialed / CC-BY 4.0) |
| **mBRSET** (Multi-attribute BRSET) | Brazil | Subset of BRSET | **Yes** (Detailed demographic & systemic covariates) | Yes | **PRIMARY CANDIDATE**: Multimodal analysis with age, sex, and anatomical quality. | Open Access |
| **APTOS 2019** | India (Aravind Eye Hospital) | 3,662 images | **NO (All patients are diabetic)** | Yes (0–4) | **TRACK A BENCHMARK ONLY**: Indian demographic representation; strictly prohibited from serving as non-diabetic controls for Track B. | Open Access (Kaggle Research) |
| **IDRiD** | India (Nanded, Maharashtra) | 516 images | **NO (All patients are diabetic)** | Yes (0–4) + Lesion masks | **AUXILIARY ONLY**: Indian rural cohort; lesion segmentation and domain adaptation, but cannot train diabetes classifier. | Open Access (IEEE Dataport) |
| **Messidor / Messidor-2** | France (Hospital clinics) | 874 patients / 1,748 images | Partial / Inconsistent across cohorts | Yes (0–4) | **AUXILIARY / EXTERNAL TEST**: Useful for external domain generalization tests on DR detection. | Open Access / Research Agreement |
| **EyePACS** | USA (California Community Health) | 88,702 images | Limited to DR screening outcomes | Yes (0–4) | **REPRESENTATION LEARNING**: Pre-training retinal foundation encoders. | Research Access |

---

## 3. Dataset Ingestion Adapters Status

The codebase provides dedicated dataset adapters in `src/data/adapters/`:
- `brset_adapter.py`: Parses BRSET CSV, checks patient IDs, extracts bilateral eyes, camera types, and EHR diabetes labels.
- `aptos_adapter.py`: Ingests APTOS 2019, explicitly sets `diabetes_label = None` to enforce safety.
- `idrid_adapter.py`: Ingests IDRiD images and DR/DME severity.
- `messidor_adapter.py`: Ingests Messidor images and grades.
- `local_folder_adapter.py`: Ingests local clinical directories with customizable column mapping.
