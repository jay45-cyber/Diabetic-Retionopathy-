# Repository Inventory & Forensic Codebase Verification

**Document**: `docs/data_readiness/repository_inventory.md`  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Date**: 2026-11-13  
**Auditor**: Senior Medical AI Research Engineer & Clinical ML Data-Validation Specialist  

---

## 1. Git State & Working Tree Audit

- **Repository**: `https://github.com/jay45-cyber/Early-Diabetic_Retinopathi.git`
- **Active Branch**: `main`
- **Audited Commit SHA**: `5c5f62da072a80a9e637d8bf832adf1b2cffc429` (previously audited at `e161b31`)
- **Remote Synchronization**: Up to date with `origin/main`
- **Working Tree**: Clean (all changes tracked or governed by `.gitignore`)
- **PHS / PHI Protection**: `.gitignore` strictly guards raw imaging (`data/raw/`, `data/processed/`, `.zip`, `.tar.gz`, `.csv`), model binaries (`*.pt`, `*.pth`), and secret tokens.

---

## 2. Preserved Functional Components

In strict accordance with **RULE 1**, existing functional codebases have been preserved and verified:

### Track A: 5-Class Diabetic Retinopathy Severity Grading (Benchmark)
- **Data Ingestion**: `src/data/dr_dataset.py` (`DRGradingDataset` supporting 5 ICDR grades 0-4, circular masking, CLAHE contrast enhancement, dynamic inverse class weights).
- **Models**: `src/models/dr_models.py` (`ViTDR`, `ResNet50DR`, `SwinTransformerDR`, and `build_dr_model()` factory).
- **Trainer**: `src/training/dr_trainer.py` (`DRTrainer` with AdamW, Cosine LR scheduler, AMP with CPU fallback, early stopping on Validation Macro-F1).
- **Evaluation**: `src/evaluation/dr_metrics.py` (Accuracy, Macro-F1, Quadratic Weighted Kappa, Referable DR Sensitivity/Specificity for Grade $\ge 2$, AUROC, ECE, 95% bootstrap CIs).
- **Explainability**: `src/explainability/dr_explainer.py` (Grad-CAM for ResNet/Swin, Attention Rollout & Token Grad-CAM for ViT).
- **CLI Scripts**: `scripts/train_dr_model.py`, `scripts/evaluate_dr_model.py`, `scripts/generate_explanations.py`.

### Track B: Systemic Type 2 Diabetes Risk Screening (Original Research)
- **Clinical Label Provenance**: `src/data/label_provenance.py` (5-tier provenance model, strict rejection of DR-inferred labels, runtime assertion `assert_no_target_leakage()`).
- **Data Ingestion**: `src/data/diabetes_dataset.py` (`DiabetesScreeningDataset` with `dr0_only=True` filtering, circular masking, and non-leaking tabular clinical covariate integration).
- **Models**: `src/models/diabetes_models.py` (`build_diabetes_model()` factory: `resnet50`, `convnext`, `foundation_vit`, and `multimodal` late-fusion MLP).
- **Trainer**: `src/training/diabetes_trainer.py` (`DiabetesRiskTrainer` with `BCEWithLogitsLoss(pos_weight=w)`, AdamW, Cosine LR, early stopping on Validation AUROC).
- **Evaluation**: `scripts/evaluate_diabetes_model.py` (AUROC, AUPRC, Sens, Spec, PPV, NPV, Brier score, Platt calibration, ECE, 95% patient-level bootstrap CIs, and isolated $\text{DR} = 0$ subgroup analysis).
- **Explainability**: `src/explainability/diabetes_explainer.py` (Binary Grad-CAM and Integrated Gradients targeted directly at the scalar risk logit).
- **CLI Scripts**: `scripts/train_diabetes_model.py`, `scripts/evaluate_diabetes_model.py`.

### Interactive Clinical Web Application
- **Application**: `app/streamlit_app.py`
  - Tab 1: Track A — 5-Class DR Grading (APTOS Benchmark).
  - Tab 2: Track B — Early Type 2 Diabetes Risk Screening with input image quality gating (Laplacian blur and illumination audit), model abstention, Platt-calibrated risk, Grad-CAM overlays, and $\text{DR} = 0$ clinical triage disclaimers.

---

## 3. Analysis of Checkpoints & Stubs

1. **Demonstration & Verification Checkpoints**:
   - `models/checkpoints/resnet50_diabetes_best.pt`: Functional PyTorch checkpoint trained on the synthetic verification cohort. Used for local pipeline smoke testing.
   - `models/checkpoints/resnet50_demo.pt` / `vit_base_patch16_224_best.pt`: Architecture verification weights for Track A.
2. **RETFound Foundation Model Status**:
   - `src/models/retinal_foundation_model.py` and `src/models/diabetes_models.py` implement the exact architectural adapter for RETFound (Vision Transformer Base with 12 transformer encoder blocks and 768-dim embeddings).
   - Pretrained weights protocol: The adapter operates with ImageNet-1K initialization locally. Official masked autoencoder (MAE) weights trained on 1.6M retinal images (Zhou et al., *Nature* 2023) require an external ~1.2 GB download from official repositories into `models/weights/retfound/`. Documented honestly as an adapter awaiting external weights.

---

## 4. Test Suite Execution & Coverage

- **Total Automated Tests**: **42 / 42 PASSING (100%)**
- **Execution Time**: ~37.34 seconds on standard CPU
- **Coverage Breakdown**:
  - `tests/test_milestone_1_audit.py`: 8 tests (dataset discovery, image validation, missing labels, provenance tiering, target leakage, duplicate detection, patient grouping, reporting).
  - `tests/test_track_b_pipeline.py`: 7 tests (provenance, leakage, dataset, models, trainer, evaluation/calibration, XAI).
  - `tests/test_dr_pipeline.py`: 7 tests (DR dataset, splits, class weights, models, metrics, XAI, trainer).
  - `tests/test_image_preprocessing.py`: 3 tests (blur, dark ungradable, dimensions).
  - `tests/test_inference.py`: 3 tests (abstention, gradable inference, triage categories).
  - `tests/test_advanced_models.py`: 3 tests (ConvNeXt, Foundation stub, Multimodal fusion).
  - `tests/test_dataset_adapters.py`: 2 tests (BRSET EHR ingestion, APTOS safety).
  - `tests/test_patient_grouping.py`: 2 tests (zero-leakage grouping, leakage detector).
  - `tests/test_duplicate_detection.py`: 2 tests (identical and distinct dHash).
  - `tests/test_evaluation.py`: 2 tests (screening metrics, Platt calibrator/ECE).
  - `tests/test_explainability_methods.py`: 2 tests (Integrated Gradients, Occlusion).
  - `tests/test_bootstrap_ci.py`: 1 test (patient-level bootstrap CI computation).
