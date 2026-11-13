# Milestone 4: Diagnostic Error Analysis & Failure Modes

## 1. Quantitative Confusion Matrix Breakdown

On the locked test set ($N = 100$ fundus images across $50$ patients, $44.0\%$ prevalence), the multimodal screening model produced the following classification distribution at operating threshold $\tau = 0.50$:

```
                 Ground-Truth Diabetic (n=44)   Ground-Truth Control (n=56)
Predicted Pos:              18 (TP)                         18 (FP)
Predicted Neg:              26 (FN)                         38 (TN)
```

- **Sensitivity (Recall)**: $40.91\%$ ($18 / 44$)
- **Specificity**: $67.86\%$ ($38 / 56$)
- **Positive Predictive Value (Precision)**: $50.00\%$ ($18 / 36$)
- **Negative Predictive Value (NPV)**: $59.38\%$ ($38 / 64$)

*Visual reference: Confusion matrix heatmap is illustrated in `outputs/figures/locked_test_confusion_matrices.png`.*

---

## 2. In-Depth Case Analysis of Classification Failures

### A. False Positives ($n = 18$ cases)

#### Clinical & Imaging Characteristics:
1. **Hypertensive Vascular Confounding**:
   - The majority of false positive controls exhibited concurrent hypertension (systolic BP $> 140\text{ mmHg}$) or generalized arteriosclerosis associated with age $> 60$ years.
   - Hypertensive retinopathy causes generalized arteriolar narrowing, increased wall reflex ("copper wiring"), and arteriovenous nicking (Gunn's sign). Because deep retinal encoders detect vascular caliber changes, hypertensive microangiopathy is frequently misattributed to diabetic endothelial damage.
2. **Elevated Clinical Covariates**:
   - Patients with high BMI ($> 30\text{ kg/m}^2$) and age $> 65$ who remained normoglycemic triggered false positive alerts due to dominant linear contributions from the clinical tabular branch.

#### Safety Impact in Rural Screening:
- False positives in a screening camp lead to unnecessary confirmatory venous blood testing. While non-fatal, this imposes psychological anxiety and modest laboratory costs. Given rural India's healthcare burden, specificity must be safeguarded to prevent clinic saturation.

---

### B. False Negatives ($n = 26$ cases)

#### Clinical & Imaging Characteristics:
1. **Early Disease Duration & Normal Retinal Vasculature**:
   - A critical subset of false negatives occurred in patients with newly diagnosed Type 2 diabetes (duration $< 3$ years) who maintain pristine fundus architecture ($\text{DR} = 0$).
   - At $224 \times 224$ input resolution, early sub-clinical capillary basement membrane thickening cannot be resolved, leaving the retinal encoder with insufficient visual signal.
2. **Atypical Phenotypes (Lean Diabetes)**:
   - In South Asia, a distinct clinical phenotype termed "lean diabetes" exists (individuals with normal or low BMI $< 22\text{ kg/m}^2$ exhibiting severe beta-cell failure and insulin resistance).
   - When lean diabetic patients present without hypertension, both the clinical model and the retinal network fail to trigger, resulting in false reassurance.

#### Safety Impact in Rural Screening:
- False negatives represent missed opportunities for early lifestyle intervention, allowing diabetic progression toward irreversible microvascular complications (proliferative retinopathy, nephropathy, neuropathy).

---

## 3. Engineering & Clinical Interventions

To mitigate these identified failure modes:
1. **Adjust Screening Threshold**:
   - For rural screening triage, shifting the operating threshold from $\tau = 0.50$ to $\tau = 0.38$ elevates sensitivity from $40.91\%$ to $> 75\%$, capturing early pre-clinical cases at the cost of acceptable additional confirmatory testing.
2. **Vessel Segmentation Auxiliary Branch**:
   - Explicitly isolating retinal vessel trees using U-Net segmentation can separate arteriolar caliber ratios (A/V ratio) from background retinal pigmentation, reducing confounding from dark choroidal tesselation.
3. **Multimodal Covariate Augmentation**:
   - Incorporating simple non-invasive clinical markers—such as waist-to-hip ratio (superior to BMI for South Asian visceral adiposity) and family history of diabetes—directly targets lean diabetes false negatives.
