# Milestone 4: Locked Test Evaluation Report

## 1. Executive Summary & Protocol Conformance

This document records the locked test set evaluation of the machine learning models developed under the project:
**"Explainable AI for Early Type 2 Diabetes Risk Screening Using Retinal Fundus Images in Rural India."**

In accordance with strict clinical AI evaluation principles:
- The evaluation was conducted once on a pre-designated, locked test partition ($N = 100$ fundus images across $50$ unique patients).
- **Zero data leakage**: No hyperparameter tuning, feature selection, threshold calibration, or architecture iteration was conducted using the locked test partition.
- Calibration parameters (Platt scaling sigmoid) were fitted strictly on the validation partition and transferred frozen to the locked test cohort.
- Metric uncertainties are quantified via $1,000$ patient-level stratified bootstrap resamples ($95\%$ empirical confidence intervals).

---

## 2. Locked Test Cohort Demographics

| Attribute | Value |
|---|---|
| **Total Test Images ($N$)** | $100$ |
| **Unique Patients ($M$)** | $50$ (bilateral OD/OS pairs) |
| **Ground-Truth Diabetes Prevalence** | $44.0\%$ ($n=44$ diabetic, $n=56$ control) |
| **Pre-clinical Subgroup ($\text{DR} = 0$)** | $60$ images ($30$ diabetic, $30$ control) |
| **Visible DR Subgroup ($\text{DR} \ge 1$)** | $40$ images ($14$ diabetic, $26$ control) |
| **Age Distribution** | $< 55$ years: $40$ images; $\ge 55$ years: $60$ images |
| **Sex Distribution** | Female: $54$ images; Male: $46$ images |
| **Image Resolution & Preprocessing** | $224 \times 224 \times 3$, circle-cropped, local contrast normalized |

---

## 3. Tri-Model Comparative Performance Benchmark

All models were evaluated under identical frozen inference conditions:

| Metric | Clinical-Only Baseline | Retinal-Only Model (ResNet-50) | Multimodal Late Fusion |
|---|---|---|---|
| **AUROC** | **0.5731** [0.4088, 0.7295] | 0.4594 [0.3164, 0.6032] | **0.5731** [0.4088, 0.7295] |
| **AUPRC** | **0.5224** [0.3513, 0.7444] | 0.4160 [0.2760, 0.6095] | 0.5109 [0.3361, 0.7366] |
| **Brier Score** | 0.2896 [0.2155, 0.3697] | 0.2480 [0.2201, 0.2760] | **0.2458** [0.2087, 0.2844] |
| **Sensitivity** | 0.5000 (22 / 44) | 0.0000 (0 / 44)* | 0.4091 (18 / 44) |
| **Specificity** | 0.6071 (34 / 56) | 1.0000 (56 / 56)* | **0.6786** (38 / 56) |
| **Precision (PPV)**| 0.5000 (22 / 44) | 0.0000 (0 / 0)* | **0.5000** (18 / 36) |
| **NPV** | 0.6071 (34 / 56) | 0.5600 (56 / 100)* | 0.5938 (38 / 64) |
| **F1-Score** | 0.5000 | 0.0000* | 0.4500 |
| **TP / FP / TN / FN**| 22 / 22 / 34 / 22 | 0 / 0 / 56 / 44* | 18 / 18 / 38 / 26 |

*\*Note: Retinal-only model operated at raw default logit threshold $\tau = 0.5$; uncalibrated probabilities were tightly clustered in the $[0.40, 0.48]$ range, necessitating threshold calibration or Platt scaling to select an optimal screening operating point.*

---

## 4. Incremental Value Analysis ($\Delta\text{AUROC}$)

A primary scientific hypothesis in medical computer vision is whether imaging features provide incremental diagnostic value over simple, low-cost clinical covariates (age, sex, BMI, blood pressure):

$$\Delta\text{AUROC} = \text{AUROC}_{\text{Multimodal}} - \max\left(\text{AUROC}_{\text{Retinal}}, \text{AUROC}_{\text{Clinical}}\right) = 0.5731 - 0.5731 = \mathbf{+0.0000}$$

### Scientific Interpretation
1. **No Evidence of Incremental Discrimination**: On the locked test cohort, the multimodal late-fusion network achieved an AUROC identical to the standalone clinical baseline ($0.5731$).
2. **Confidence Interval Overlap**: The $95\%$ bootstrap confidence interval for multimodal ($[0.4088, 0.7295]$) completely overlaps the clinical baseline ($[0.4088, 0.7295]$) and the retinal baseline ($[0.3164, 0.6032]$).
3. **Brier Score Improvement**: The multimodal model achieved a lower Brier score ($0.2458$ vs $0.2896$), indicating improved probabilistic reliability despite neutral discrimination.
4. **Conclusion**: In strict accordance with research integrity, **no claim of multimodal superiority or autonomous retinal diagnostic replacement is supported** by these empirical findings. Retinal imaging alone cannot currently replace clinical risk profiling without larger prospective cohorts with laboratory HbA1c reference standards.

---

## 5. Artifact References

- Machine-readable results: `outputs/metrics/milestone_4_results.json`
- Locked test predictions: `outputs/metrics/locked_test_predictions.csv`
- Publication figures:
  - ROC curves: `outputs/figures/roc_curves_comparison.png`
  - PR curves: `outputs/figures/pr_curves_comparison.png`
  - Confusion matrices: `outputs/figures/locked_test_confusion_matrices.png`
