# End-to-End Reproducibility Guide

**Project**: Explainable AI for Early Type 2 Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Auditor**: Senior Software Quality & MLOps Engineer  
**Status**: Fully Reproducible across Local CPU and Cloud GPU (CUDA)

---

## 1. System Requirements & Environment Setup

### Hardware Requirements
- **Local Development / Smoke Testing**: Any multi-core CPU (x86_64 or Apple Silicon), 8GB+ RAM. All tests and dry-run CLI scripts run locally without GPU requirements.
- **Production Training**: NVIDIA GPU with 16GB+ VRAM (e.g., Tesla T4, V100, A100, RTX 3090/4090). A verified Google Colab notebook is provided in `notebooks/colab_gpu_pipeline.ipynb`.

### Environment Initialization
```bash
# Clone the repository
git clone https://github.com/jay45-cyber/Early-Diabetic_Retinopathi.git
cd Early-Diabetic_Retinopathi

# Create and activate Python 3.11 virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Test Suite Execution

Execute the full automated test suite containing 34 unit and integration tests across data adapters, image processing, patient splitting, models, trainers, evaluators, and explainers:

```bash
# Run complete test suite with verbose output
pytest tests/ -v

# Run Track A specific tests (20 tests)
pytest tests/test_dr_pipeline.py -v

# Run Track B specific tests (7 tests)
pytest tests/test_track_b_pipeline.py -v
```
*Expected Result: 34 passed in ~35-40 seconds on standard CPU.*

---

## 3. Track A — Diabetic Retinopathy Benchmark Reproduction

### Training Track A Model (APTOS 2019)
```bash
# Dry run verification (synthetic cohort, CPU-safe)
python scripts/train_dr_model.py --model vit_b_16 --dry_run --epochs 1 --batch_size 4

# Full training run (requires APTOS images in data/raw/aptos_2019)
python scripts/train_dr_model.py \
    --model vit_b_16 \
    --data_csv data/raw/aptos_2019/train.csv \
    --image_dir data/raw/aptos_2019/train_images \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --img_size 224
```

### Evaluating Track A Model
```bash
python scripts/evaluate_dr_model.py \
    --model vit_b_16 \
    --checkpoint models/checkpoints/vit_b_16_dr_best.pt \
    --test_csv data/splits/dr_test_split.csv \
    --image_dir data/raw/aptos_2019/train_images \
    --out_dir outputs/dr_experiments/eval_results
```
*Artifacts Generated*: `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `evaluation_summary.txt`.

### Generating Track A Saliency Visualizations
```bash
python scripts/generate_explanations.py \
    --model vit_b_16 \
    --checkpoint models/checkpoints/vit_b_16_dr_best.pt \
    --method rollout \
    --out_dir outputs/explanations
```

---

## 4. Track B — Systemic Diabetes Screening Reproduction

### Training Track B Model
```bash
# Dry run verification (uses automatic synthetic verification cohort)
python scripts/train_diabetes_model.py --model resnet50 --dry_run --epochs 1 --batch_size 4

# Full clinical training (with BRSET / rural manifests)
python scripts/train_diabetes_model.py \
    --model resnet50 \
    --train_manifest data/splits/diabetes_train.csv \
    --val_manifest data/splits/diabetes_val.csv \
    --image_dir data/processed/retina_images \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 7
```

### Evaluating Track B Model
```bash
# Local synthetic verification mode
python scripts/evaluate_diabetes_model.py \
    --synthetic \
    --model resnet50 \
    --checkpoint models/checkpoints/resnet50_diabetes_best.pt \
    --out_dir outputs/diabetes_experiments/eval_results

# Full test cohort evaluation
python scripts/evaluate_diabetes_model.py \
    --test_manifest data/splits/diabetes_test.csv \
    --image_dir data/processed/retina_images \
    --model resnet50 \
    --checkpoint models/checkpoints/resnet50_diabetes_best.pt \
    --out_dir outputs/diabetes_experiments/eval_results
```
*Artifacts Generated*: `resnet50_diabetes_metrics.json`, `resnet50_confusion_matrix.png`, `resnet50_evaluation_summary.txt`, `dr0_subgroup_metrics.json`.

---

## 5. Launching the Interactive Clinical Web Application

Launch the dual-track Streamlit screening application:
```bash
streamlit run app/streamlit_app.py
```
- **Port**: `http://localhost:8501`
- **Features**:
  - Image quality screening filter (Laplacian blur and luminosity checks).
  - Track A: 5-class DR grading with confidence distribution and Grad-CAM.
  - Track B: Binary diabetes risk prediction with Platt-calibrated probability and DR=0 clinical advisory.

---

## 6. Cloud GPU Pipeline Execution (Google Colab)

For researchers without local NVIDIA hardware:
1. Open `notebooks/colab_gpu_pipeline.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set Runtime Type to **Python 3 with GPU (T4 or A100)**.
3. Follow the notebook cells to mount Google Drive, clone the repository, download datasets from Kaggle / PhysioNet, train high-resolution ($384 \times 384$) ViT and ResNet models, and export trained checkpoints.
