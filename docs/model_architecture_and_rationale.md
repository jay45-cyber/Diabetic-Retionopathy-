# Model Architecture & Dual-Track Research Rationale

**Project Identity**: Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Auditor**: Lead Medical AI Research Architect  
**Status**: Comprehensive Specification for Track A (DR Benchmark) & Track B (Systemic Diabetes Screening)

---

## 1. Dual-Track Architectural Overview

The repository maintains two strictly separated research tracks to avoid confounding systemic disease screening with visible microvascular complication grading:

```
                           +------------------------------------------+
                           | Retinal Fundus Image Preprocessing       |
                           | (Circular ROI Crop, Luminance CLAHE)     |
                           +------------------------------------------+
                                                |
                 +------------------------------+------------------------------+
                 |                                                             |
                 v                                                             v
+------------------------------------+                       +------------------------------------+
| TRACK A: 5-Class DR Grading        |                       | TRACK B: Systemic Diabetes Screening|
| Benchmark (APTOS 2019)             |                       | Primary Research Track             |
+------------------------------------+                       +------------------------------------+
| - Vision Transformer Base (ViT-B)  |                       | - Level 1: ResNet-50 Baseline      |
| - ResNet-50 Benchmark              |                       | - Level 2: Retinal ConvNeXt        |
| - Swin Transformer                 |                       | - Level 3: RETFound ViT Adapter    |
| Target: 5 classes (Grade 0 to 4)   |                       | - Level 4: Multimodal Late Fusion  |
| Loss: Weighted Cross-Entropy       |                       | Target: Binary Logit (Risk Score)  |
| Metric: Quadratic Weighted Kappa   |                       | Loss: BCEWithLogits (pos_weight)   |
+------------------------------------+                       +------------------------------------+
```

---

## 2. Track A: Diabetic Retinopathy Severity Grading Architectures

Track A benchmarks standard and state-of-the-art vision architectures on the 5-class International Clinical Diabetic Retinopathy (ICDR) scale using the APTOS 2019 dataset.

### Vision Transformer Base (`ViT-B/16`)
- **Backbone**: Vision Transformer Base with $16 \times 16$ patch tokens (`timm.create_model('vit_base_patch16_224')` or `torchvision.models.vit_b_16`).
- **Feature Dimensionality**: 768-dimensional token embeddings.
- **Classification Head**: `Linear(768, 5)`.
- **Clinical Rationale**: Self-attention layers capture long-range spatial correlations across the entire retina, modeling the spatial co-occurrence of hard exudates, cotton-wool spots, and microaneurysms across superior, inferior, and macular arcades.

### Convolutional Benchmark (`ResNet-50`)
- **Backbone**: 50-layer deep residual network with bottleneck blocks (`torchvision.models.resnet50`).
- **Classification Head**: Global average pooling $\to$ `Linear(2048, 5)`.
- **Clinical Rationale**: Establishes a standard translational baseline widely accepted in ophthalmic computer vision benchmarks.

### Hierarchical Vision Transformer (`Swin Transformer`)
- **Backbone**: Shifted Window Transformer (`timm.create_model('swin_tiny_patch4_window7_224')`).
- **Classification Head**: `Linear(embed_dim, 5)`.
- **Clinical Rationale**: Combines local window self-attention with hierarchical shifted-window cross-attention, offering scale-invariant multi-resolution lesion analysis with linear computational complexity.

---

## 3. Track B: Systemic Type 2 Diabetes Risk Screening Architectures

Track B investigates whether retinal fundus images contain subtle sub-clinical vascular signatures (caliber alterations, tortuosity changes, arteriolar narrowing) indicative of systemic diabetes, including in eyes with zero visible retinopathy ($\text{DR} = 0$).

### Level 1: ResNet-50 Baseline (`ResNet50Diabetes`)
- **Backbone**: ResNet-50 pre-trained on ImageNet-1K (`IMAGENET1K_V2`).
- **Modifications**: Replaces the 1000-class head with:
  `Linear(2048, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.
- **Output**: Single scalar logit representing log-odds of diabetes risk.
- **Rationale**: Minimal complexity baseline ensuring that gains in advanced models reflect genuine signal rather than architecture capacity.

### Level 2: Modern ConvNet (`RetinalConvNeXt`)
- **Backbone**: `convnext_tiny` (`torchvision.models.convnext_tiny`).
- **Modifications**: 768-dim pooled feature $\to$ `Linear(768, 256) -> LayerNorm -> GELU -> Dropout(0.3) -> Linear(256, 1)`.
- **Rationale**: Incorporates modern Vision Transformer design choices ($7 \times 7$ depthwise convolutions, inverted bottlenecks, LayerNorm) while retaining convolutional inductive bias (spatial invariance and sample efficiency).

### Level 3: Retinal Foundation Model (`RetinalFoundationModel`)
- **Backbone**: Vision Transformer Base adapted for retinal foundation tasks (`torchvision.models.vit_b_16`).
- **Status & Weight Protocol**: Scaffolded with standard ViT-B/16 ImageNet weights, serving as a drop-in adapter for official RETFound MAE weights (Zhou et al., *Nature* 2023, 1.6M retinal fundus images) when downloaded into `models/weights/retfound/`.
- **Classification Head**: `Linear(768, 256) -> GELU -> Dropout(0.3) -> Linear(256, 1)`.
- **Rationale**: Retinal-specific self-supervised pre-training captures optical disc geometry and fine vascular networks that general ImageNet features overlook.

### Level 4: Multimodal Late Fusion (`MultimodalRetinaModel`)
- **Visual Backbone**: Any Track B visual backbone (`resnet50`, `convnext`, `foundation_vit`).
- **Tabular Covariate Stream**: 2-layer MLP (`Linear(M, 64) -> BatchNorm1d -> ReLU -> Linear(64, 32)`), where $M = 4$ for `[age, sex, bmi, systolic_bp]`.
- **Strict Leakage Prevention**: Diagnostic biochemical biomarkers (`HbA1c`, `FPG`, `OGTT`, antidiabetic medication) are strictly excluded from tabular inputs via `assert_no_target_leakage()`.
- **Fusion Layer**: Concatenated visual feature vector (256-dim) and clinical feature vector (32-dim) $\to$ `Linear(288, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 1)`.
- **Rationale**: Systemic diabetes manifests through both vascular end-organ changes and clinical demographic risk factors. Fusing non-invasive demographics with fundus imaging provides superior discrimination without requiring invasive phlebotomy.

---

## 4. Optimization & Hyperparameter Protocols

| Parameter | Track A (5-Class DR Grading) | Track B (Systemic Diabetes Screening) |
| :--- | :--- | :--- |
| **Objective** | Multi-class Classification | Binary Risk Prediction |
| **Loss Function** | Weighted Cross-Entropy ($\text{weight}_c = \frac{N}{C \cdot N_c}$) | `BCEWithLogitsLoss(pos_weight=w_{pos})` |
| **Primary Metric** | Macro-F1 / Quadratic Weighted Kappa | Validation AUROC |
| **Secondary Metric** | Referable DR Sens/Spec (Grade $\ge 2$) | AUPRC, Brier Score, ECE, DR=0 Subgroup AUROC |
| **Optimizer** | AdamW ($\text{lr} = 10^{-4}$, $\text{wd} = 10^{-4}$) | AdamW ($\text{lr} = 10^{-4}$, $\text{wd} = 10^{-4}$) |
| **LR Schedule** | Cosine Annealing (T_max=epochs, $\text{eta\_min}=10^{-6}$) | Cosine Annealing (T_max=epochs, $\text{eta\_min}=10^{-6}$) |
| **Early Stopping** | 5 epochs patience on Val Macro-F1 | 7 epochs patience on Val AUROC |
| **Mixed Precision**| `torch.cuda.amp.autocast(enabled=is_cuda)` | `torch.cuda.amp.autocast(enabled=is_cuda)` |
| **Bootstrap CIs** | 95% Patient-Level Bootstrap (1,000 resamples) | 95% Patient-Level Bootstrap (1,000 resamples) |
| **Post-Hoc Calibration**| Temperature Scaling | Platt Scaling (Logistic Sigmoid on Validation Logits) |
