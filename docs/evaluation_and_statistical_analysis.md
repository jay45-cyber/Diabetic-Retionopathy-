# Evaluation Metrics & Statistical Analysis Protocol

**Project Identity**: Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Auditor**: Senior Biostatistician & AI Evaluation Lead  
**Scope**: Rigorous dual-track statistical protocol for Track A (DR Benchmark) and Track B (Systemic Diabetes Screening)

---

## 1. Dual-Track Evaluation Philosophy

Medical computer vision models deployed in screening settings face two distinct clinical tasks with different statistical requirements:

1. **Track A (Ordinal DR Staging)**: Multi-class ordinal problem where distance between classes matters (e.g., misclassifying Grade 0 as Grade 1 is minor; misclassifying Grade 0 as Grade 4 is clinically disastrous).
2. **Track B (Binary Diabetes Screening)**: Binary population risk screening under class imbalance ($10\% - 20\%$ prevalence), where high sensitivity is paramount to prevent false reassurance, high NPV is required for safety, and calibration is essential for clinical decision-making.

---

## 2. Track A: Diabetic Retinopathy Severity Grading Protocol

### Primary Metrics
- **Accuracy**: Overall fraction of correctly classified images across 5 grades.
- **Macro-Averaged F1-Score**: Unweighted arithmetic mean of F1-scores across all 5 classes, penalizing models that collapse onto majority classes (Grade 0).
- **Quadratic Weighted Kappa (QWK / Cohen's Kappa with Quadratic Weights)**:
  $$\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(C - 1)^2}$$
  Penalizes severe grade misclassifications quadratically. Gold standard benchmark for ICDR classification.
- **Referable DR Sensitivity & Specificity**:
  - Classifies images with Grade $\ge 2$ (Moderate, Severe, PDR) as *Referable DR*.
  - Sensitivity: True Positive Rate for requiring ophthalmological referral.
  - Specificity: True Negative Rate for non-referable cases (Grade 0 and 1).

### Track A Automated Artifacts
Every evaluation run of `scripts/evaluate_dr_model.py` automatically generates:
1. `metrics.json`: Full quantitative metric breakdown.
2. `classification_report.csv`: Per-class precision, recall, F1-score, and support.
3. `confusion_matrix.png`: $5 \times 5$ normalized and raw contingency matrix.
4. `evaluation_summary.txt`: Human-readable clinical validation summary.

---

## 3. Track B: Systemic Diabetes Risk Screening Protocol

### Quantitative Screening Suite
| Metric Name | Mathematical Formulation | Screening Interpretation |
| :--- | :--- | :--- |
| **AUROC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Threshold-independent discrimination across the complete operating curve. |
| **AUPRC** | $\int_0^1 \text{Precision}(\text{Recall}^{-1}(r)) \, dr$ | Primary metric under class imbalance; reflects positive predictive power across recall tiers. |
| **Sensitivity (Recall)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **Primary clinical safety metric**: Percentage of diabetic individuals correctly identified for diagnostic workup. |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Percentage of non-diabetic controls spared unnecessary secondary blood tests. |
| **Negative Predictive Value (NPV)** | $\frac{\text{TN}}{\text{TN} + \text{FN}}$ | Confidence that an individual screened negative is truly non-diabetic. |
| **Positive Predictive Value (PPV)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Probability that a screened-positive individual is confirmed diabetic upon lab diagnostic testing. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing positive prediction and coverage. |
| **Brier Score** | $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ | Strict proper scoring rule quantifying probability calibration and accuracy (0.0 = perfect). |

### Calibration & Reliability Analysis
Neural networks frequently yield uncalibrated confidence estimates. To ensure clinical safety:
- **Platt Scaling**: A univariate logistic model is trained on locked validation logits $z$:
  $$P(Y=1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
- **Expected Calibration Error (ECE)**: Measures the weighted absolute discrepancy between mean predicted confidence $\text{conf}(B_m)$ and observed empirical accuracy $\text{acc}(B_m)$ across $M = 10$ quantile bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 4. Patient-Level Bootstrap Confidence Intervals (95% CI)

To account for inter-patient clustering and variance without assuming normal error distributions:
1. **Resampling Unit**: Unique `participant_id` (or patient ID), NOT individual fundus images.
2. **Multi-Eye Grouping**: When a patient is sampled, all available images (OD, OS, and longitudinal visits) are bundled into that bootstrap sample.
3. **Resampling Iterations**: $B = 1,000$ empirical bootstrap iterations with replacement.
4. **Interval Calculation**: Percentile method computing the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles:
   $$\text{CI}_{95\%} = \left[ \hat{\theta}^*_{(0.025)}, \, \hat{\theta}^*_{(0.975)} \right]$$
   Reported for AUROC, AUPRC, Sensitivity, Specificity, and Brier score.

---

## 5. Isolated $\text{DR} = 0$ Subgroup Evaluation Protocol

The central scientific premise of Track B is whether retinal imaging detects systemic diabetes **before** visible diabetic retinopathy develops.

### Evaluation Protocol
1. Subselect the subset of the test partition where clinician-graded DR is strictly 0:
   $$\mathcal{D}_{\text{DR}=0} = \{ (x_i, y_i) \in \mathcal{D}_{\text{test}} \mid \text{DR\_grade}_i = 0 \}$$
2. Re-compute AUROC, AUPRC, Sensitivity, and Specificity exclusively within $\mathcal{D}_{\text{DR}=0}$.
3. Benchmark against non-diabetic controls ($\text{DR} = 0, \text{Diabetes} = 0$).
4. If $\text{AUROC}_{\text{DR}=0} > 0.65$ with $95\%$ CI lower bound exceeding $0.50$, empirical evidence supports sub-clinical microvascular remodeling prior to clinical retinopathy.
