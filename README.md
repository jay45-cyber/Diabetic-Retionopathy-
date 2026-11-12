# Explainable ViT-Based Diabetic Retinopathy Severity Grading & Retinal Screening AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.15](https://img.shields.io/badge/pytorch-2.15.0-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-27%2F27%20passed-brightgreen.svg)](#7-testing--quality-assurance)
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](#)

A reproducible, research-grade deep learning framework comparing **Vision Transformers (ViT-B/16)**, **Swin Transformers (Shifted Window Attention)**, and **ResNet-50** for multi-stage Diabetic Retinopathy (DR) severity classification on color fundus photographs. Designed for clinical decision support and assistive screening in resource-constrained community healthcare settings (e.g., rural India), incorporating multi-method Explainable AI (Grad-CAM, Attention Rollout, Integrated Gradients) and strict patient-level separation.

---

## 1. Clinical Context & Benchmark Objectives

Diabetic Retinopathy is a microvascular complication of diabetes and the leading cause of preventable blindness in working-age adults worldwide. This project provides a robust, clinically calibrated 5-class grading system complying with the International Clinical Diabetic Retinopathy (ICDR) scale and the **APTOS 2019 Blindness Detection** benchmark:

| Severity Grade | Clinical Diagnosis | Key Diagnostic Retinal Findings | Actionable Triage Recommendation |
| :---: | :---: | :--- | :--- |
| **Grade 0** | **No DR** | Absence of visible microaneurysms, hemorrhages, or exudates | Routine annual screening |
| **Grade 1** | **Mild NPDR** | Microaneurysms only | Re-screen within 6–12 months |
| **Grade 2** | **Moderate NPDR** | More than microaneurysms but less than severe (dot/blot hemorrhages, hard exudates, cotton wool spots) | **Referable DR**: Specialist ophthalmology referral within 4–12 weeks |
| **Grade 3** | **Severe NPDR** | 4-2-1 Rule: >20 intraretinal hemorrhages in all 4 quadrants, venous beading in ≥2 quadrants, or IRMA in ≥1 quadrant | **Referable DR**: Urgent ophthalmology referral within 2–4 weeks |
| **Grade 4** | **Proliferative DR (PDR)** | Neovascularization (NVD/NVE), vitreous/preretinal hemorrhage, tractional retinal detachment | **Referable DR**: Immediate ophthalmology referral (prompt panretinal photocoagulation / anti-VEGF therapy) |

> [!IMPORTANT]
> **Clinical Referable DR Definition**:
> Patients with **Grade ≥ 2** (Moderate, Severe, or Proliferative DR) require specialized ophthalmic intervention. Sensitivity and specificity for Referable DR are computed and prioritized alongside Quadratic Weighted Kappa (QWK) and Macro-F1.

---

## 2. Architecture Comparison Matrix

The framework benchmarks three diverse computer vision paradigms on fundus photographs:

| Model Architecture | Parameter Count | Receptive Field & Mechanism | Best Suited For | Explainability Method |
| :--- | :---: | :--- | :--- | :--- |
| **Vision Transformer (ViT-B/16)** | ~86.6M | Global self-attention across non-overlapping $16 \times 16$ spatial patches | Long-range retinal morphological dependencies and diffuse lesion patterns | Attention Rollout & ViT Token Grad-CAM |
| **Swin Transformer (Base / Tiny)** | ~88M / ~28M | Hierarchical shifted-window self-attention ($M=7$) with linear complexity | Multi-scale retinal lesions (tiny microaneurysms to large vitreous hemorrhages) | Swin Window Grad-CAM |
| **ResNet-50 (Baseline)** | ~25.6M | Deep convolutional residual bottleneck blocks ($3 \times 3$ and $1 \times 1$) | Local microvascular edge and texture extraction | Convolutional Grad-CAM / Grad-CAM++ |

---

## 3. Repository Structure

```
project_ML/
├── config/                         # Centralized YAML configuration files
│   ├── dr_grading_config.yaml      # 5-Class DR grading configuration (ViT, ResNet, Swin)
│   ├── project_config.yaml         # Research project metadata and directory paths
│   └── training_config.yaml        # Legacy baseline training hyperparameters
├── data/                           # Data storage (git-ignored for clinical privacy)
│   ├── raw/                        # Raw fundus datasets (APTOS 2019, BRSET, IDRiD, EyePACS)
│   ├── interim/                    # Normalized intermediate fundus crops
│   └── processed/splits/           # Reproducible 70/15/15 stratified train/val/test CSVs
├── notebooks/                      # Jupyter research notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_quality_analysis.ipynb
│   ├── 03_patient_distribution_analysis.ipynb
│   ├── 04_baseline_experiments.ipynb
│   ├── 05_model_analysis.ipynb
│   └── colab_gpu_pipeline.ipynb    # End-to-end Google Colab cloud GPU training notebook
├── src/                            # Core production-grade source code
│   ├── data/                       # DR dataset loading, circular cropping, CLAHE, stratified splitting
│   │   ├── dr_dataset.py           # DRGradingDataset (5 classes, circular mask, CLAHE, inverse class weights)
│   │   ├── image_quality.py        # Automated gradability check (brightness, blur, contrast)
│   │   ├── preprocessing.py        # Circular fundus border cropping & ImageNet normalization
│   │   └── duplicate_detection.py  # Perceptual dHash duplicate detection
│   ├── models/                     # Deep learning architectures
│   │   └── dr_models.py            # ViTDR, ResNet50DR, SwinTransformerDR & build_dr_model factory
│   ├── training/                   # Model training and optimization
│   │   └── dr_trainer.py           # DRTrainer (AdamW, Cosine LR, AMP, Macro-F1 early stopping)
│   ├── evaluation/                 # Clinical evaluation metrics
│   │   └── dr_metrics.py           # Macro-F1, QWK, Referable DR Sens/Spec, AUROC, ECE, 95% Bootstrap CI
│   ├── explainability/             # Explainable AI (XAI)
│   │   ├── dr_explainer.py         # DRModelExplainer (Grad-CAM, Attention Rollout, Integrated Gradients)
│   │   ├── gradcam.py              # Grad-CAM++ implementation
│   │   └── integrated_gradients.py # Path-integral feature attribution
│   ├── inference/                  # Clinical inference & risk triage
│   │   └── risk_output.py          # Calibrated screening score and clinical referral categorization
│   └── utils/                      # Logging, reproducibility, config parsing
├── scripts/                        # Standalone CLI tools
│   ├── train_dr_model.py           # Model training CLI supporting all backbones & config overrides
│   ├── evaluate_dr_model.py        # Model evaluation CLI outputting metrics.json, plots, and CI
│   └── generate_explanations.py    # Batch visual explainability generator (Grad-CAM, Rollout, IG)
├── tests/                          # Automated PyTest suite (27 unit & integration tests)
│   ├── test_dr_pipeline.py         # 5-class DR dataset, models, metrics, XAI, and trainer smoke tests
│   └── ...                         # Data leakage, image quality, grouping, and inference tests
├── app/                            # Interactive Streamlit screening prototype
│   └── streamlit_app.py            # Clinical upload, gradability audit, triage risk & CAM heatmap
├── models/checkpoints/             # Saved model weights (*_best.pt)
├── outputs/                        # Research artifacts
│   ├── dr_experiments/             # Metrics, confusion matrices, classification reports
│   └── explanations/               # Visual heatmaps ({model}/{method}/{image_id}.png)
├── pyproject.toml                  # Pytest configuration and project metadata
└── README.md                       # Main research documentation
```

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.11.x
- Virtual environment (`venv`)

```powershell
# 1. Clone repository
git clone https://github.com/jay45-cyber/Early-Diabetic_Retinopathi.git
cd Early-Diabetic_Retinopathi

# 2. Activate Python 3.11 virtual environment
.\venv\Scripts\Activate.ps1

# 3. Verify installed dependencies
pip install -r requirements.txt
```

---

## 5. CLI Execution Guide

### A. Model Training
Train any of the supported architectures using the centralized YAML configuration or CLI overrides:

```powershell
# Train Vision Transformer (ViT-B/16) with default configuration
python scripts/train_dr_model.py --config config/dr_grading_config.yaml

# Train ResNet-50 for 15 epochs with batch size 16
python scripts/train_dr_model.py --model_name resnet50 --epochs 15 --batch_size 16

# Train Swin Transformer with CLAHE preprocessing enabled
python scripts/train_dr_model.py --model_name swin_base_patch4_window7_224 --use_clahe

# Fast dry-run smoke test (1 epoch on 32 samples)
python scripts/train_dr_model.py --model_name resnet50 --dry_run
```

### B. Comprehensive Evaluation
Evaluate trained checkpoints on the independent test set, producing all mandatory research artifacts:

```powershell
# Evaluate trained ViT checkpoint on test split
python scripts/evaluate_dr_model.py `
  --checkpoint models/checkpoints/vit_base_patch16_224_best.pt `
  --test_csv data/processed/splits/dr_test.csv `
  --image_dir data/raw/aptos2019/train_images `
  --output_dir outputs/dr_experiments/vit_results

# Evaluate with automated synthetic test generation (pipeline smoke test)
python scripts/evaluate_dr_model.py `
  --checkpoint models/checkpoints/resnet50_demo.pt `
  --synthetic `
  --output_dir outputs/dr_experiments/eval_test
```

**Artifacts automatically saved to `--output_dir`:**
1. `metrics.json`: Full numeric results (Accuracy, Macro-F1, QWK, Referable DR Sens/Spec, AUROC, ECE, 95% Bootstrap CIs).
2. `classification_report.csv`: Per-class precision, recall, F1-score, and support.
3. `confusion_matrix.png`: High-resolution normalized $5 \times 5$ confusion matrix heatmap.
4. `evaluation_summary.txt`: Plain-text clinical report ready for publication or dissertation inclusion.

### C. Visual Explainability Generation
Generate visual attribution heatmaps overlaid on retinal fundus photographs:

```powershell
# Generate Grad-CAM, Attention Rollout, and Integrated Gradients for test images
python scripts/generate_explanations.py `
  --checkpoint models/checkpoints/vit_base_patch16_224_best.pt `
  --model_name vit_base_patch16_224 `
  --num_samples 5 `
  --output_dir outputs/explanations
```

Heatmaps are saved to:
`outputs/explanations/{model_name}/{method_name}/{image_id}.png`

### D. Interactive Web Demonstration Prototype
Launch the interactive Streamlit clinical decision-support application:

```powershell
streamlit run app/streamlit_app.py
```

---

## 6. Reproducible Splitting & Class Imbalance Strategy

1. **Stratified Splitting (70% Train / 15% Val / 15% Test)**:
   - Partitioning is stratified across all five severity grades.
   - Fixed random seed (`seed = 42`) guarantees exact reproducibility.
   - Duplicate prevention prevents retinal images from crossing split boundaries.
   - The test set is isolated and never observed during model selection or hyperparameter tuning.

2. **Handling Severe Class Imbalance**:
   - APTOS 2019 contains strong class disparity (No DR has ~1,805 samples vs Severe DR with ~193 samples).
   - The loss function dynamically applies normalized inverse-frequency weights:
     $$w_c = \frac{N}{C \cdot N_c}$$
   - Model selection is governed strictly by **Validation Macro-F1**, preventing the model from degenerating into majority-class trivial prediction.

---

## 7. Testing & Quality Assurance

The test suite validates data ingestion, patient grouping, image preprocessing, model forward passes, clinical evaluation metrics, explainability modules, and trainer execution loops:

```powershell
# Run the complete test suite
.\venv\Scripts\pytest.exe tests/ -v
```

**Test Suite Coverage (27 / 27 Tests Passing)**:
- `test_dr_pipeline.py`:
  - `test_dr_dataset_and_clahe`: Dataset indexing, circular crop, CLAHE, tensor shapes.
  - `test_stratified_dr_splits`: 70/15/15 ratio, zero split overlap, class preservation.
  - `test_inverse_class_weights`: Inverse weighting logic for imbalanced classes.
  - `test_dr_models_forward`: Forward pass shape `[B, 5]` for ViT-B/16, ResNet-50, and Swin.
  - `test_dr_evaluation_metrics`: Accuracy, Macro-F1, QWK, Referable DR Sens/Spec, ECE, Bootstrap CI.
  - `test_dr_explainability_methods`: ResNet Grad-CAM, ViT Rollout, Integrated Gradients, PIL overlay.
  - `test_dr_trainer_smoke`: Mini-batch training step, loss reduction, checkpoint persistence.
- `test_patient_grouping.py`: Strict patient-level grouping and zero-leakage verification.
- `test_image_preprocessing.py`: Circular cropping, image illumination, gradability audit.
- `test_duplicate_detection.py`: Perceptual dHash detection of identical and transformed images.
- `test_inference.py`: Automated abstention on ungradable images and clinical triage formatting.
- `test_advanced_models.py`: ConvNeXt, multimodal tabular-imaging fusion, foundation models.

---

## 8. Academic Integrity & Scientific Disclaimer

1. **Academic Context**: This codebase is developed for research engineering, comparative benchmarking, and academic dissertation purposes (aligning with MSc / M.Tech medical computer vision standards).
2. **Clinical Disclaimer**: This software is an assistive screening research prototype. It is **not** an FDA/CE-cleared medical device and must **not** be used as a primary diagnostic tool. Clinical management decisions must always be made by a licensed ophthalmologist or qualified medical professional.
3. **Data Protection**: No patient-identifiable healthcare information (PHI) or private clinical records are stored or tracked within this repository.
