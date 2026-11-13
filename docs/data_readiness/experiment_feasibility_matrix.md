# Experiment Feasibility Matrix & Dual-Model Design Specifications

**Document**: `docs/data_readiness/experiment_feasibility_matrix.md`  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Date**: 2026-11-13  
**Auditor**: Senior Deep Learning Architect & Medical ML Validation Lead  

---

## 1. Experimental Overview

The primary research objective is to assess whether retinal fundus photography contains discernible vascular signatures of systemic Type 2 diabetes, with particular focus on early pre-clinical detection in patients without visible retinopathy ($\text{DR} = 0$).

To test this scientifically, two independent experimental designs are specified:
1. **Experiment A (Retinal-Only)**: Measures the pure visual discrimination capacity of fundus imaging.
2. **Experiment B (Multimodal)**: Evaluates whether non-invasive screening covariates add synergistic predictive value to fundus imaging.

---

## 2. Experiment A: Retinal-Only Diabetes Risk Screening

```
[Preprocessed Fundus Photograph (384x384)]
                    |
      +-------------v-------------+
      |  Deep Retinal Backbone    |
      |  (ResNet-50 / ConvNeXt /  |
      |   RETFound ViT Adapter)   |
      +-------------+-------------+
                    | (256-dim embedding)
      +-------------v-------------+
      | Binary Classification Head|
      | Linear -> BN -> ReLU ->   |
      | Dropout -> Linear(1)      |
      +-------------+-------------+
                    |
      [Scalar Diabetes Logit z / Calibrated Risk Score]
```

### Technical Specification
- **Input Modality**: Color retinal fundus photograph.
  - Preprocessing: Circular field-of-view masking, aspect-ratio preserved crop, Luminance CLAHE ($\text{clip\_limit} = 2.0$), ImageNet normalization.
  - Resolution: Default $384 \times 384 \times 3$ (configurable to $224 \times 224$ for resource-constrained nodes).
- **Target Variable**: Binary systemic Type 2 diabetes status ($y \in \{0, 1\}$), sourced exclusively from `LABORATORY_CONFIRMED` or `CLINICIAN_DOCUMENTED` provenance tiers.
- **Candidate Architectures**:
  1. **ResNet-50 Baseline** (`torchvision.models.resnet50`): Established benchmark with ImageNet-1K initialization.
  2. **Retinal ConvNeXt** (`torchvision.models.convnext_tiny`): Modern convolutional architecture with $7 \times 7$ depthwise filters and LayerNorm.
  3. **EfficientNetV2-S / M** (`torchvision.models.efficientnet_v2_s`): Parameter-efficient architecture optimized for mobile screening.
  4. **Retinal Foundation Model (RETFound Adapter)**: Vision Transformer Base (`vit_b_16`) adapted for retinal self-supervised weights (Zhou et al., *Nature* 2023). Pretrained weights downloaded externally into `models/weights/retfound/`.
- **Loss Function**: `BCEWithLogitsLoss(pos_weight=w_{pos})` with dynamic positive weighting to address class imbalance.
- **Optimization**: AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$), Cosine Annealing learning rate decay, validation AUROC early stopping.

---

## 3. Experiment B: Multimodal Risk Screening (Retina + Non-Invasive Covariates)

```
[Preprocessed Fundus (384x384)]               [Eligible Clinical Covariates (M=4)]
               |                                              |
 +-------------v-------------+                  +-------------v-------------+
 | Visual Feature Backbone   |                  | Tabular Covariate Encoder |
 | (ResNet / ConvNeXt / ViT) |                  | Linear(M,64) -> BN ->     |
 +-------------+-------------+                  | ReLU -> Linear(64,32)     |
               |                                +-------------+-------------+
               | (256-dim)                                    | (32-dim)
               +----------------------+-----------------------+
                                      |
                        +-------------v-------------+
                        | Late Feature Concatenation|
                        | (288-dim fused vector)    |
                        +-------------+-------------+
                                      |
                        +-------------v-------------+
                        | Multimodal Fusion MLP     |
                        | Linear(288,128) -> ReLU ->|
                        | Dropout(0.2) -> Linear(1) |
                        +-------------+-------------+
                                      |
                        [Scalar Diabetes Logit z]
```

### Technical Specification
- **Visual Stream**: Any validated Experiment A backbone producing a 256-dimensional penultimate representation.
- **Tabular Clinical Stream**: 4 non-invasive clinical covariates available at triage time without laboratory testing:
  - `age`: Continuous (years).
  - `sex`: Binary categorical (0 = Male, 1 = Female).
  - `bmi`: Continuous Body Mass Index ($\text{kg/m}^2$).
  - `systolic_bp`: Continuous Systolic Blood Pressure ($\text{mmHg}$).
- **Target Leakage Prohibition**: Diagnostic blood biomarkers (`HbA1c`, `FPG`, `OGTT`, random glucose, or antidiabetic prescriptions) are strictly prohibited via `assert_no_target_leakage()`.
- **Multimodal Fusion Engine**: Late concatenation of visual embedding ($d_v = 256$) and clinical embedding ($d_c = 32$) $\to$ 2-layer MLP projection $\to$ single scalar logit.
- **Implementation**: [`src/models/diabetes_models.py::MultimodalRetinaModel`](file:///c:/Users/Jay/projects/project_ML/src/models/diabetes_models.py).

### Mandatory Clinical-Only Baseline
To scientifically prove that the retinal image contributes incremental value over demographic risk scores alone, the multimodal experiment must be benchmarked against a **Clinical-Only Baseline**:
- Architecture: 2-layer MLP / Logistic Regression trained exclusively on `[age, sex, bmi, systolic_bp]`.
- Scientific Decision Criterion:
  $$\Delta\text{AUROC} = \text{AUROC}_{\text{Multimodal}} - \max\left(\text{AUROC}_{\text{Retinal-Only}}, \, \text{AUROC}_{\text{Clinical-Only}}\right)$$
  A multimodal model is deemed genuinely superior only if $\Delta\text{AUROC} > 0$ with non-overlapping $95\%$ bootstrap confidence intervals. Claims of superiority prior to testing are scientifically prohibited.

---

## 4. Experiment Feasibility Matrix Across Datasets

| Dataset | Modality Support | Provenance Tier | Retinal-Only Feasibility | Multimodal Feasibility | Isolated $\text{DR}=0$ Feasibility | Primary Role in Project |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **APTOS 2019** | Image only | Inferred / Unknown | **Infeasible** (No controls) | **Infeasible** (No metadata) | **Infeasible** | Track A Benchmark (5-class DR) |
| **BRSET** | Image + Metadata | Clinician documented (EHR) | **High Feasibility** | **High Feasibility** (Age, sex, camera available) | **High Feasibility** | Track B Primary Training & Evaluation |
| **IDRiD** | Image only | Clinic cohort | **Infeasible** (No controls) | **Infeasible** | **Infeasible** | Track A Auxiliary Domain Adaptation |
| **Messidor-2** | Image only | Unknown | **Infeasible** (No diabetes label) | **Infeasible** | **Infeasible** | Track A External Validation Benchmark |
| **Prospective Rural Indian Cohort** | Image + POC HbA1c + Covariates | Laboratory confirmed | **Target Standard** | **Target Standard** | **Target Standard** | Future Clinical Validation (Milestone 2+) |
