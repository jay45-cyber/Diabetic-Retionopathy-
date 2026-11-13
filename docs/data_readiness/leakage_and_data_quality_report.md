# Patient-Level Safety, Data Quality & Leakage Audit Report

**Document**: `docs/data_readiness/leakage_and_data_quality_report.md`  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Date**: 2026-11-13  
**Auditor**: Senior Clinical ML Data-Validation Specialist & Software Quality Engineer  

---

## 1. Executive Summary

In medical machine learning, data leakage is the single largest cause of overly optimistic published performance that fails upon real-world clinical deployment. This report audits all potential leakage vectors and establishes automated verification mechanisms across:
1. Patient-level grouping and multi-eye isolation.
2. Multiple longitudinal visits.
3. Exact byte-level duplicates (MD5) and perceptual duplicates (dHash).
4. Cross-dataset duplicate risks (e.g., Kaggle / EyePACS / Messidor overlaps).
5. Automated image quality and gradability screening.

---

## 2. Patient-Level Grouping & Independence Rules

### The Danger of Image-Level Random Splitting
If a dataset with bilateral images (Right Eye = OD, Left Eye = OS) is split randomly at the image level:
- Patient X's right eye lands in the **training set**.
- Patient X's left eye lands in the **test set**.
- **Consequence**: The deep neural network learns patient-specific anatomical features (choroidal vasculature texture, unique optic nerve disc shape, pigmentation pattern) rather than generalizable pathology. The test evaluation becomes a memory test rather than an assessment of diagnostic generalization.

### Enforced Grouping Protocol
All splitting routines in the framework ([`src/data/grouped_split.py`](file:///c:/Users/Jay/projects/project_ML/src/data/grouped_split.py) and [`src/data/dr_dataset.py`](file:///c:/Users/Jay/projects/project_ML/src/data/dr_dataset.py)) enforce strict grouping on `participant_id`:
```python
# Guaranteed disjoint partition sets
assert len(train_pids.intersection(val_pids)) == 0
assert len(train_pids.intersection(test_pids)) == 0
assert len(val_pids.intersection(test_pids)) == 0
```
- **Bilateral Eye Grouping**: Both OD and OS eyes for any patient are assigned to the exact same partition.
- **Longitudinal Visit Grouping**: All follow-up examinations across multiple time points are assigned to the same partition.
- **Unidentified Patient Policy**: Datasets lacking verified patient IDs (such as APTOS 2019, where patient IDs were unreleased) are flagged explicitly. For APTOS, duplicate image filtering via perceptual hashing is applied to minimize cross-split patient overlap.

---

## 3. Duplicate Image Detection Protocol

Duplicate and near-duplicate images frequently contaminate public computer vision datasets due to multiple crops, re-compressions, or identical patient submissions across different challenges.

### Detection Hierarchy
1. **Exact Duplicate Detection (MD5 Checksum)**:
   - Evaluates 128-bit cryptographic hash of raw image byte streams via `compute_md5()`.
   - Identifies identical files regardless of file naming.
2. **Perceptual Duplicate Detection (Difference Hash - dHash)**:
   - Evaluates spatial gradient structure via `compute_dhash()`:
     1. Downsamples image to $9 \times 8$ grayscale.
     2. Computes relative horizontal intensity gradients ($P[x, y] > P[x+1, y]$).
     3. Produces a 64-bit binary perceptual signature.
   - Computes **Hamming Distance**: Number of differing bits between two 64-bit hashes.
     - $\text{Distance} = 0$: Identical perceptual structure.
     - $\text{Distance} \le 5$: Near-identical images (crops, contrast shifts, re-compressions).
     - $\text{Distance} > 10$: Visually distinct images.

### Cross-Dataset Duplicate Risks
- **EyePACS vs Kaggle DR vs APTOS**: A subset of public retinal datasets share images sourced from EyePACS telemedicine networks.
- **Safeguard**: When combining datasets (e.g., BRSET + local Indian cohorts), cross-dataset dHash indexing must be executed to prevent images from the training corpus leaking into the evaluation corpus.

---

## 4. Automated Image Quality & Gradability Assessment

In rural screening camps, ungradable photographs (caused by motion blur, poor pupil dilation, or lens smudges) can lead to erroneous algorithmic outputs if processed blindly.

The framework enforces automated image quality gating ([`src/data/image_quality.py`](file:///c:/Users/Jay/projects/project_ML/src/data/image_quality.py)):

| Quality Dimension | Quantitative Metric | Operational Threshold | Clinical Rationale |
| :--- | :--- | :---: | :--- |
| **Dimension Integrity** | Width & Height | $\ge 256 \times 256\text{ px}$ | Prevents extreme downsampling artifacts that obliterate microaneurysms. |
| **Sharpness / Focus** | 2D Discrete Laplacian Variance | $\text{Var}(\nabla^2 I) \ge 20.0$ | Detects severe optical and motion blur rendering fine vessels invisible. |
| **Mean Illumination** | Grayscale Mean Intensity | $15.0 \le \mu \le 240.0$ | Detects severely underexposed (black) or washed-out overexposed images. |

### Algorithmic Abstention Mechanism
In both the clinical inference pipeline ([`src/inference/risk_output.py`](file:///c:/Users/Jay/projects/project_ML/src/inference/risk_output.py)) and the Streamlit clinical dashboard ([`app/streamlit_app.py`](file:///c:/Users/Jay/projects/project_ML/app/streamlit_app.py)):
- If an image fails quality criteria, the system **abstains from generating a risk score**.
- A clear triage advisory is emitted: `"UNGRADABLE IMAGE: Poor optical quality / severe blur detected. Please re-capture retinal photograph before triage."`
