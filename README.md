# Dual-Track Retinal AI Research Framework: Diabetic Retinopathy Grading & Systemic Diabetes Risk Screening

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.15](https://img.shields.io/badge/pytorch-2.15.0-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-34%2F34%20passed-brightgreen.svg)](#7-testing--quality-assurance)
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](#)

A reproducible, research-grade medical computer vision framework engineered for resource-constrained community screening in rural India. The framework establishes a rigorous dual-track structure separating **Diabetic Retinopathy (DR) severity classification** from **systemic Type 2 diabetes risk prediction** from fundus photographs.

---

## 1. Dual-Track Research Architecture

```
                                  Retinal Fundus Photograph
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
   +-------------------------------+                     +-------------------------------+
   | TRACK A: DR Grading Benchmark |                     | TRACK B: Primary Research     |
   | (APTOS 2019 / ICDR Scale)     |                     | (Systemic Type 2 Diabetes)    |
   +-------------------------------+                     +-------------------------------+
   | - Vision Transformer (ViT-B)  |                     | - Level 1: ResNet-50 Baseline |
   | - ResNet-50 Benchmark         |                     | - Level 2: Retinal ConvNeXt   |
   | - Swin Transformer            |                     | - Level 3: RETFound Adapter   |
   |                               |                     | - Level 4: Multimodal Fusion  |
   | Target: 5 Classes (Grade 0-4) |                     | Target: Binary Diabetes Logit |
   | Metric: Quadratic Weighted    |                     | Metric: AUROC, AUPRC, Sens/Spec|
   |         Kappa, Macro-F1       |                     | Subgroup: Isolated DR=0 Cohort|
   | XAI: Class-specific Grad-CAM  |                     | XAI: Binary Grad-CAM & IG     |
   |      & Attention Rollout      |                     | Calibration: Platt Scaling    |
   +-------------------------------+                     +-------------------------------+
```

### Critical Clinical Principles Enforced by Design:
1. **Never Infer Diabetes from DR Grades**: An eye with $\text{DR} = 0$ in APTOS is a diabetic patient without microvascular complications, NOT a non-diabetic control. Track B requires dedicated non-diabetic controls and clinical provenance verification.
2. **Strict Provenance Hierarchy**: Labels are tiered into `LABORATORY_CONFIRMED`, `CLINICIAN_DOCUMENTED`, `SELF_REPORTED`, `UNKNOWN`, and `INVALID_DR_INFERRED`. Inferred labels are rejected at dataset ingestion.
3. **Zero Target Leakage**: Diagnostic biomarkers (`HbA1c`, `FPG`, `OGTT`, and antidiabetic medication) are barred from input features via runtime assertion `assert_no_target_leakage()`.
4. **Patient-Level Independence**: Bilateral fundus images (OD/OS) and multi-visit longitudinal images of each patient are strictly locked to the same partition.
5. **Pre-Clinical Hypothesis ($\text{DR} = 0$)**: Evaluates whether systemic hyperglycemia causes sub-clinical microvascular alterations detectable prior to visible retinopathy.

---

## 2. Track A: Diabetic Retinopathy Severity Grading (Benchmark)

Benchmarks vision architectures on the 5-class International Clinical Diabetic Retinopathy (ICDR) scale using the APTOS 2019 Blindness Detection dataset:

| Grade | Clinical Diagnosis | Retinal Pathophysiology | Screening Triage Action |
| :---: | :---: | :--- | :--- |
| **0** | **No DR** | No microaneurysms, hemorrhages, or exudates | Routine annual rescreening |
| **1** | **Mild NPDR** | Microaneurysms only | Follow-up within 6–12 months |
| **2** | **Moderate NPDR** | Microaneurysms, dot/blot hemorrhages, hard exudates | **Referable DR**: Ophthalmology referral (4–12 weeks) |
| **3** | **Severe NPDR** | 4-2-1 Rule: >20 hemorrhages in 4 quadrants, venous beading | **Referable DR**: Urgent referral (2–4 weeks) |
| **4** | **Proliferative DR** | Neovascularization (NVD/NVE), vitreous hemorrhage | **Referable DR**: Immediate treatment (panretinal laser / anti-VEGF) |

---

## 3. Track B: Systemic Type 2 Diabetes Risk Screening (Original Research)

Investigates systemic diabetes screening using non-mydriatic fundus images, focusing on early detection in rural areas without diagnostic phlebotomy laboratories:

| Component | Specification | Clinical Rationale |
| :--- | :--- | :--- |
| **Image Backbone** | ResNet-50, Retinal ConvNeXt, RETFound Adapter | Captures vascular caliber, arteriolar-to-venular ratio, and tortuosity. |
| **Multimodal Stream** | Non-invasive clinical covariates: `[age, sex, bmi, systolic_bp]` | Synergistic late fusion without requiring blood draws or lab tests. |
| **Optimization** | `BCEWithLogitsLoss(pos_weight=w)`, AdamW, Cosine LR | Balances class imbalance and prevents early-epoch gradient saturation. |
| **Evaluation Suite** | AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, Brier score | Full screening discrimination and operating safety profiling. |
| **Calibration** | Platt Scaling (Logistic Regression on validation logits) | Delivers clinically trustworthy risk probabilities with low ECE. |
| **Subgroup Analysis** | **Isolated $\text{DR} = 0$ Cohort** | Tests hypothesis of pre-clinical microvascular remodeling. |

---

## 4. Repository Structure

```
project_ML/
├── app/                            # Interactive clinical decision-support application
│   └── streamlit_app.py            # Dual-track dashboard (Tab 1: DR Grading, Tab 2: Diabetes Risk)
├── config/                         # Centralized YAML configuration files
│   ├── dr_grading_config.yaml      # Track A 5-class configuration
│   ├── project_config.yaml         # Project metadata and paths
│   └── training_config.yaml        # Legacy baseline training hyperparameters
├── data/                           # Data storage (git-ignored for clinical privacy)
│   ├── raw/                        # Raw fundus archives (APTOS, BRSET, IDRiD, Messidor)
│   └── splits/                     # Leakage-safe patient-stratified CSV manifests
├── docs/                           # Comprehensive research and clinical documentation
│   ├── clinical_reference_label_protocol.md
│   ├── dataset_feasibility.md
│   ├── diabetes_label_provenance.md
│   ├── evaluation_and_statistical_analysis.md
│   ├── explainability_protocol.md
│   ├── limitations_and_future_work.md
│   ├── model_architecture_and_rationale.md
│   ├── reproducibility_guide.md
│   └── track_b_validation_protocol.md
├── notebooks/                      # Research notebooks and Cloud GPU pipeline
│   ├── 01_dataset_exploration.ipynb
│   ├── 04_baseline_experiments.ipynb
│   └── colab_gpu_pipeline.ipynb    # End-to-end Google Colab GPU training pipeline
├── src/                            # Modular production-grade source code
│   ├── data/                       # Ingestion, provenance, dataset classes, quality audit
│   │   ├── diabetes_dataset.py     # Track B DiabetesScreeningDataset (dr0_only, multimodal)
│   │   ├── dr_dataset.py           # Track A DRGradingDataset (5 classes, CLAHE, splits)
│   │   ├── image_quality.py        # Automated Laplacian blur, luminosity, and contrast check
│   │   ├── label_provenance.py     # 5-tier provenance engine & target leakage assertion
│   │   └── preprocessing.py        # Circular fundus border cropping & ImageNet normalization
│   ├── models/                     # Model architectures
│   │   ├── diabetes_models.py      # ResNet50Diabetes, ConvNeXt, RETFound Adapter, Multimodal
│   │   └── dr_models.py            # ViTDR, ResNet50DR, SwinTransformerDR
│   ├── training/                   # Model trainers
│   │   ├── diabetes_trainer.py     # Track B DiabetesRiskTrainer (BCE, pos_weight, AUROC early stop)
│   │   └── dr_trainer.py           # Track A DRTrainer (Weighted CE, Macro-F1 early stop)
│   ├── evaluation/                 # Metrics & calibration
│   │   ├── calibration.py          # Platt scaling calibrator & Expected Calibration Error (ECE)
│   │   ├── classification_metrics.py # AUROC, AUPRC, Sens, Spec, PPV, NPV, Brier score
│   │   ├── confidence_intervals.py # 95% Patient-level bootstrap confidence intervals
│   │   └── dr_metrics.py           # Quadratic Weighted Kappa, Macro-F1, Referable DR metrics
│   └── explainability/             # Explainable AI (XAI)
│       ├── diabetes_explainer.py   # Track B Binary Grad-CAM & Integrated Gradients
│       ├── dr_explainer.py         # Track A Multi-class Grad-CAM & Attention Rollout
│       └── gradcam.py              # Grad-CAM++ core engine
├── scripts/                        # Standalone reproducible CLI tools
│   ├── train_dr_model.py           # Train Track A DR grading models
│   ├── evaluate_dr_model.py        # Evaluate Track A models (generates 4 artifacts)
│   ├── train_diabetes_model.py     # Train Track B diabetes models (with dry-run mode)
│   ├── evaluate_diabetes_model.py  # Evaluate Track B models (with synthetic mode & DR=0 subgroup)
│   └── generate_explanations.py    # Batch visual explainability generator
├── tests/                          # PyTest test suite (34 unit & integration tests)
│   ├── test_dr_pipeline.py         # Track A data, models, metrics, XAI tests
│   ├── test_track_b_pipeline.py    # Track B provenance, leakage, models, trainer, XAI tests
│   └── ...                         # Image quality, patient grouping, duplicate tests
├── pyproject.toml                  # Pytest configuration
└── README.md                       # Master research documentation
```

---

## 5. Quickstart & Installation

```bash
# 1. Clone repository
git clone https://github.com/jay45-cyber/Early-Diabetic_Retinopathi.git
cd Early-Diabetic_Retinopathi

# 2. Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 6. CLI Execution Guide

### Track A — Diabetic Retinopathy Severity Grading
```bash
# Fast dry-run smoke test (CPU-safe)
python scripts/train_dr_model.py --model_name resnet50 --dry_run

# Evaluate model on test set with 4 automated artifacts
python scripts/evaluate_dr_model.py --checkpoint models/checkpoints/resnet50_best.pt --synthetic

# Generate attention rollout and Grad-CAM visualizations
python scripts/generate_explanations.py --checkpoint models/checkpoints/vit_base_patch16_224_best.pt --num_samples 5
```

### Track B — Systemic Type 2 Diabetes Risk Screening
```bash
# Train Track B ResNet-50 screening model (dry-run mode)
python scripts/train_diabetes_model.py --model resnet50 --dry_run --epochs 1 --batch_size 4

# Evaluate Track B model with calibration, 95% Patient Bootstrap CIs, and DR=0 subgroup analysis
python scripts/evaluate_diabetes_model.py --synthetic --model resnet50 --checkpoint models/checkpoints/resnet50_diabetes_best.pt
```

### Launch Interactive Clinical Dashboard
```bash
streamlit run app/streamlit_app.py
```
- **Tab 1**: Track A — 5-Class DR Grading (APTOS Benchmark).
- **Tab 2**: Track B — Early Type 2 Diabetes Risk Screening with automated quality gate, Platt calibration, binary Grad-CAM, and DR=0 advisory.

---

## 7. Testing & Quality Assurance

The test suite validates data adapters, image processing, patient grouping, model forward passes, metrics, calibration, and explainability across both research tracks:

```bash
# Run complete test suite (34 tests)
pytest tests/ -v
```

```
============================= test session starts =============================
tests/test_advanced_models.py::test_convnext_forward_pass PASSED         [  2%]
tests/test_advanced_models.py::test_foundation_model_stub PASSED         [  5%]
tests/test_advanced_models.py::test_multimodal_fusion_forward_pass PASSED [  8%]
tests/test_bootstrap_ci.py::test_patient_bootstrap_ci_computation PASSED [ 11%]
tests/test_dataset_adapters.py::test_brset_adapter_ingestion PASSED      [ 14%]
tests/test_dataset_adapters.py::test_aptos_adapter_clinical_safety PASSED [ 17%]
tests/test_dr_pipeline.py::test_dr_dataset_and_clahe PASSED              [ 20%]
tests/test_dr_pipeline.py::test_stratified_dr_splits PASSED              [ 23%]
tests/test_dr_pipeline.py::test_inverse_class_weights PASSED             [ 26%]
tests/test_dr_pipeline.py::test_dr_models_forward PASSED                 [ 29%]
tests/test_dr_pipeline.py::test_dr_evaluation_metrics PASSED             [ 32%]
tests/test_dr_pipeline.py::test_dr_explainability_methods PASSED         [ 35%]
tests/test_dr_pipeline.py::test_dr_trainer_smoke PASSED                  [ 38%]
tests/test_duplicate_detection.py::test_perceptual_hash_identical_images PASSED [ 41%]
tests/test_duplicate_detection.py::test_perceptual_hash_distinct_images PASSED [ 44%]
tests/test_evaluation.py::test_compute_screening_metrics PASSED          [ 47%]
tests/test_evaluation.py::test_platt_calibrator_fitting_and_ece PASSED   [ 50%]
tests/test_explainability_methods.py::test_integrated_gradients PASSED   [ 52%]
tests/test_explainability_methods.py::test_occlusion_sensitivity PASSED  [ 55%]
tests/test_image_preprocessing.py::test_image_quality_sharp_gradable PASSED [ 58%]
tests/test_image_preprocessing.py::test_image_quality_ungradable_too_dark PASSED [ 61%]
tests/test_image_preprocessing.py::test_preprocessing_output_dimension_and_normalization PASSED [ 64%]
tests/test_inference.py::test_inference_abstention_on_ungradable_image PASSED [ 67%]
tests/test_inference.py::test_inference_success_on_gradable_image PASSED [ 70%]
tests/test_inference.py::test_format_screening_output_categories PASSED  [ 73%]
tests/test_patient_grouping.py::test_patient_grouping_zero_leakage PASSED [ 76%]
tests/test_patient_grouping.py::test_leakage_detector_fails_on_overlap PASSED [ 79%]
tests/test_track_b_pipeline.py::test_diabetes_label_provenance PASSED    [ 82%]
tests/test_track_b_pipeline.py::test_target_leakage_assertion PASSED     [ 85%]
tests/test_track_b_pipeline.py::test_diabetes_dataset_and_dr0_filtering PASSED [ 88%]
tests/test_track_b_pipeline.py::test_diabetes_models_forward PASSED      [ 91%]
tests/test_track_b_pipeline.py::test_diabetes_trainer_smoke PASSED       [ 94%]
tests/test_track_b_pipeline.py::test_diabetes_evaluation_and_calibration PASSED [ 97%]
tests/test_track_b_pipeline.py::test_diabetes_explainer PASSED           [100%]
============================= 34 passed in 36.98s =============================
```

---

## 8. Clinical Governance & Scientific Disclaimers

1. **Intended Use**: This software is designed strictly as an assistive clinical screening decision-support tool. It is **not** an FDA/CE-cleared diagnostic device.
2. **Clinical Confirmation Mandatory**: A high-risk screening output requires confirmatory venous blood testing ($\text{HbA1c}$) and clinical examination by a certified ophthalmologist or physician.
3. **Privacy & Data Security**: No patient-identifiable healthcare information (PHI) is tracked in version control.
