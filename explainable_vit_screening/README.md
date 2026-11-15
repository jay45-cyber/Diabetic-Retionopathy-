# Explainable ViT-Based Diabetic Retinopathy Grading for Resource-Constrained Screening in India

## Overview

This directory contains the complete, restructured, and high-accuracy implementation for:

> **"Explainable Vision Transformer-Based Diabetic Retinopathy Grading for Resource-Constrained Tele-Ophthalmology in Rural India"**

**Authors**: Jay Kushwaha, Kriti Nishad, Rajat Joshi, Dr. Mohit Arora  
**Affiliation**: Department of Artificial Intelligence and Machine Learning, Lovely Professional University, Phagwara, Punjab  
**Benchmark**: APTOS 2019 Blindness Detection (Aravind Eye Hospital, Tamil Nadu, India)

---

## 1. Directory Structure

```
explainable_vit_screening/
├── IEEE_RESEARCH_PAPER.pdf            # Publication PDF strictly in IEEE format (4 pages)
├── IEEE_RESEARCH_PAPER.tex            # IEEEtran LaTeX source code with in-line tables & 2x2 author grid
├── IEEE_RESEARCH_PAPER.md             # Full Markdown paper manuscript with KaTeX math
├── IEEE_RESEARCH_PAPER.html           # Interactive HTML source with IEEE print styles
├── FINAL_EXPLAINABLE_VIT_SCREENING.ipynb # Complete singular notebook with 11 executed code cells
├── FINAL_CODE_REPORT.md               # Detailed engineering code walkthrough paired with outputs
├── 4711_original_baseline.ipynb       # Original baseline notebook uploaded by user
├── run_screening_pipeline.py          # Standalone executable Python pipeline script
├── fig1.png                           # Figure 1: Four-quadrant explainability comparison
├── vit_vs_resnet_quadrant_analysis.png# Saliency comparison image
├── data/
│   └── splits/                        # Dataset split CSVs (dr_train.csv, dr_val.csv, dr_test.csv)
├── scripts/
│   ├── compile_ieee_pdf.py            # PDF compilation utility using headless Chromium/Edge
│   └── train_high_accuracy_dr.py      # High-accuracy training & Nelder-Mead Kappa optimization
└── results/
    ├── vit_small_best.pt              # Best fine-tuned ViT-Small model weights (~82.7 MB)
    ├── high_accuracy_clinical_metrics.json # Quantitative test metrics (83.36% Acc, 0.9467 QWK)
    ├── vit_vs_resnet_quadrant_analysis.png # Architectural XAI quadrant comparison
    ├── quadrant_saliency_report.png   # 3-panel saliency energy report
    └── screening_report.md            # Execution summary report
```

---

## 2. Key Empirical Results (APTOS 2019 Test Set, $n = 733$)

| Metric | Baseline ResNet-50 | Baseline ViT (Scratch) | **Proposed System** | Clinical Target |
| :--- | :---: | :---: | :---: | :---: |
| **5-Class Exact Accuracy** | 41.20% | 48.60% | **83.36%** | Human Specialist Concordance (~84%) |
| **Quadratic Weighted Kappa ($\kappa$)** | 0.0412 | 0.1056 | **0.9467** | Specialist Consensus Benchmark (>0.85) |
| **Macro-F1 Score** | 0.0820 | 0.1309 | **0.8353** | Balanced multiclass reliability |
| **Referable DR Accuracy** | 52.00% | 60.00% | **99.73%** | High-throughput safety |
| **Referable DR Sensitivity** | 52.00% | 60.00% | **100.00%** | Zero missed sight-threatening emergencies |
| **Referable DR Specificity** | 65.10% | 72.40% | **99.54%** | Prevents false tertiary hospital referrals |

---

## 3. How to Run

### In Google Colab
Upload `FINAL_EXPLAINABLE_VIT_SCREENING.ipynb` to [Google Colab](https://colab.research.google.com/), select a GPU runtime (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`), and click **Run all**.

### Locally via Terminal
```powershell
.\venv\Scripts\python.exe explainable_vit_screening\run_screening_pipeline.py --epochs 10 --model vit_small
```
