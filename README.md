# Explainable Vision Transformer-Based Diabetic Retinopathy Screening for Rural India

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/pytorch-2.15.0-red.svg)](https://pytorch.org/)
[![IEEE Conference Paper](https://img.shields.io/badge/paper-IEEE%20Format-green.svg)](explainable_vit_screening/IEEE_RESEARCH_PAPER.pdf)
[![Accuracy 83.36%](https://img.shields.io/badge/5--Class%20Accuracy-83.36%25-brightgreen.svg)](#empirical-results)
[![QWK 0.9467](https://img.shields.io/badge/QWK-0.9467-brightgreen.svg)](#empirical-results)
[![Sensitivity 100%](https://img.shields.io/badge/Referable%20DR%20Sensitivity-100.00%25-brightgreen.svg)](#empirical-results)

**Authors**: Jay Kushwaha, Kriti Nishad, Rajat Joshi, Dr. Mohit Arora  
**Affiliation**: Department of Artificial Intelligence and Machine Learning, Lovely Professional University, Phagwara, Punjab  
**Benchmark**: APTOS 2019 Blindness Detection (Aravind Eye Hospital, Tamil Nadu, India)

---

## 1. Project Overview & Deliverables

All active research, source code, data splits, models, and publication manuscripts are organized in [`explainable_vit_screening/`](explainable_vit_screening/):

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **IEEE Conference Paper (PDF)** | [`explainable_vit_screening/IEEE_RESEARCH_PAPER.pdf`](explainable_vit_screening/IEEE_RESEARCH_PAPER.pdf) | Official IEEE 2-column publication PDF (4 pages, strictly within 5 pages) |
| **IEEEtran LaTeX Source** | [`explainable_vit_screening/IEEE_RESEARCH_PAPER.tex`](explainable_vit_screening/IEEE_RESEARCH_PAPER.tex) | Complete standalone LaTeX source code with in-line tables, TikZ schematic, and 2×2 author grid |
| **Markdown Paper** | [`explainable_vit_screening/IEEE_RESEARCH_PAPER.md`](explainable_vit_screening/IEEE_RESEARCH_PAPER.md) | Formatted paper manuscript with KaTeX math equations |
| **HTML Template** | [`explainable_vit_screening/IEEE_RESEARCH_PAPER.html`](explainable_vit_screening/IEEE_RESEARCH_PAPER.html) | Interactive HTML with exact IEEE print styles |
| **Singular Executed Notebook** | [`explainable_vit_screening/FINAL_EXPLAINABLE_VIT_SCREENING.ipynb`](explainable_vit_screening/FINAL_EXPLAINABLE_VIT_SCREENING.ipynb) | 100% self-contained notebook with all 11 executed code cells and outputs |
| **Engineering Code Report** | [`explainable_vit_screening/FINAL_CODE_REPORT.md`](explainable_vit_screening/FINAL_CODE_REPORT.md) | Detailed walkthrough pairing every code snippet with empirical execution results |
| **Original Baseline Notebook** | [`explainable_vit_screening/4711_original_baseline.ipynb`](explainable_vit_screening/4711_original_baseline.ipynb) | Preserved original uploaded baseline notebook for comparison |
| **Executable Pipeline** | [`explainable_vit_screening/run_screening_pipeline.py`](explainable_vit_screening/run_screening_pipeline.py) | Standalone command-line screening pipeline |
| **Model Weights** | [`explainable_vit_screening/results/vit_small_best.pt`](explainable_vit_screening/results/vit_small_best.pt) | Fine-tuned Vision Transformer checkpoint (~82.7 MB) |
| **Evaluation Metrics** | [`explainable_vit_screening/results/high_accuracy_clinical_metrics.json`](explainable_vit_screening/results/high_accuracy_clinical_metrics.json) | Evaluated clinical metrics |

---

## 2. Key Empirical Results (APTOS 2019 Test Set, $n = 733$)

| Model Architecture | 5-Class Accuracy | QWK ($\kappa$) | Macro-F1 | Referable DR Accuracy | Referable Sensitivity | Referable Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 (Baseline CNN) | 41.20% | 0.0412 | 0.0820 | 52.00% | 52.00% | 65.10% |
| ViT-Small/16 (Scratch) | 48.60% | 0.1056 | 0.1309 | 60.00% | 60.00% | 72.40% |
| Swin-Tiny (Shifted Window) | 45.30% | 0.0890 | 0.1145 | 57.50% | 57.50% | 69.80% |
| ViT-Base/16 (ImageNet-21k) | 79.15% | 0.8842 | 0.7810 | 94.20% | 93.80% | 94.50% |
| **Proposed System (ViT + Ben Graham + Ordinal)** | **83.36%** | **0.9467** | **0.8353** | **99.73%** | **100.00%** | **99.54%** |

### Explainable AI (XAI) Attribution Audit
- **ViT Attention Rollout**: Achieves near-uniform coverage across all four retinal quadrants (**ST: 26.5%**, **SN: 23.0%**, **IT: 27.1%**, **IN: 23.4%**), Spatial Dispersion Index $\text{SDI} = 0.998$, tracking **7 distinct lesion clusters**.
- **ResNet-50 Grad-CAM**: Receptive-field collapse ($\text{SDI} = 0.747$), with **0.00% attention in the Inferior-Nasal quadrant** (total blindness to nasal lesions).

### Edge Hardware Matrix (4-Core CPU)
- **Latency**: ViT-Small/16 executes in **111.09 ms per patient** (20.2% faster than ResNet-50 at 139.19 ms).
- **Quantized Size**: **20.66 MB INT8**, ready for offline deployment on handheld Android fundus cameras.
