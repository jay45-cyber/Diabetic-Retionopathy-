# Track B Validation Protocol: Leakage-Safe Diabetes Screening

**Project Identity**: Explainable AI for Early Type 2 Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Date**: 2026-11-13  
**Status**: **ACTIVE STANDARD PROTOCOL**  

---

## 1. Core Validation Principles

Track B evaluates whether retinal microvascular morphology contains detectable signatures of systemic Type 2 Diabetes Mellitus. To prevent over-optimistic evaluation and ensure clinical generalizability, the protocol enforces five non-negotiable principles:

1. **Zero Patient Overlap**: No patient's data may span more than one partition.
2. **Bilateral Eye Grouping**: Left (OS) and right (OD) fundus photographs from the same patient must always remain in the same split.
3. **Longitudinal Visit Grouping**: Multi-visit scans from the same individual across time must reside in the same partition.
4. **Duplicate Removal**: Perceptual difference hashing (`dHash`) detects transformed or re-sampled duplicates across splits.
5. **Locked External Test Set**: The test split is evaluated exactly once after all hyperparameters and checkpoints are selected.

---

## 2. Partitioning Specifications

```
                       +-------------------------------------------------------+
                       |   Ingested Clinical Manifest (BRSET / mBRSET / Site)  |
                       +---------------------------+---------------------------+
                                                   |
                                                   v
                       +-------------------------------------------------------+
                       |    Duplicate & Integrity Filtering (dHash dist <= 4)   |
                       +---------------------------+---------------------------+
                                                   |
                                                   v
                       +-------------------------------------------------------+
                       |    Label Provenance Gateway: Filter Tier 1 & Tier 2   |
                       +---------------------------+---------------------------+
                                                   |
                                                   v
                       +-------------------------------------------------------+
                       |      Patient-Level Stratified Group Split (Seed=42)   |
                       +-------+-------------------+-------------------+-------+
                               |                   |                   |
                     70% Train v           15% Val v          15% Test v
                       +---------------+   +---------------+   +---------------+
                       | Model Training|   | Early Stopping|   | Locked Final  |
                       | (Weighted BCE)|   | (AUROC Tuning)|   | Benchmark     |
                       +---------------+   +---------------+   +---------------+
```

### Partitioning Ratios:
- **Training Set (70%)**: Used for parameter optimization with class-imbalance weighting.
- **Validation Set (15%)**: Used strictly for early stopping, learning rate decay, and Platt calibrator fitting.
- **Test Set (15%)**: Completely held out until final model reporting.

---

## 3. Mathematical Zero-Leakage Invariants

Let $\mathcal{P}_{\text{train}}$, $\mathcal{P}_{\text{val}}$, and $\mathcal{P}_{\text{test}}$ denote the sets of unique patient identifiers:
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset$$
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
$$\mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

Furthermore, for all image perceptual hashes $H(I)$:
$$\min_{\substack{I_a \in \text{Train} \\ I_b \in \text{Test}}} \text{HammingDistance}(H(I_a), H(I_b)) > 4$$

These assertions are verified automatically during test execution by `tests/test_patient_grouping.py`.

---

## 4. The DR = 0 Primary Research Subgroup Protocol

The core scientific hypothesis of Track B is that retinal microvasculature undergoes sub-clinical remodeling prior to visible retinopathy.
Therefore, all test evaluation runs must report two complementary evaluations:

1. **Full Test Cohort Performance**: Evaluates overall screening utility across the entire screened population.
2. **$\text{DR} = 0$ Subgroup Performance**:
   - Isolates participants with DR Grade 0 (no visible microaneurysms, hemorrhages, or exudates).
   - Compares diabetic individuals ($\text{DR} = 0, \text{Diabetes} = 1$) against healthy controls ($\text{DR} = 0, \text{Diabetes} = 0$).
   - Computes subgroup AUROC, AUPRC, sensitivity, specificity, and prevalence.

---

## 5. Domain Generalization & External Testing Protocols

To benchmark real-world deployment across rural screening centers:
1. **Camera-Held-Out Testing**: Evaluates models trained on desktop tabletop cameras (e.g. Canon CR-2) on low-cost handheld cameras (e.g. Topcon / Volk).
2. **Site-Held-Out Testing**: Evaluates models across different geographic screening networks (Latin America to India).
3. **Automated Abstention**: Evaluates whether the system safely abstains from predicting when an image has poor illumination ($<35$ mean luminance), severe blur (Laplacian variance $<100$), or significant out-of-distribution artifacts.
