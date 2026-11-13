# Clinical Label Provenance, Reference Standards & Anti-Leakage Protocol

**Document**: `docs/data_readiness/clinical_label_provenance.md`  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Date**: 2026-11-13  
**Auditor**: Senior Clinical ML Data-Validation Specialist & Lead Medical AI Architect  

---

## 1. Clinical Ground-Truth Provenance Hierarchy

In clinical AI research, labels derived from different clinical tiers possess vastly different evidentiary strengths. The framework implements a strict 5-tier classification engine ([`src/data/label_provenance.py`](file:///c:/Users/Jay/projects/project_ML/src/data/label_provenance.py)):

```
                     +---------------------------------------------+
                     | Tier 1: LABORATORY_CONFIRMED                |
                     | HbA1c >= 6.5%, FPG >= 126 mg/dL, OGTT >= 200 |
                     +----------------------+----------------------+
                                            |
                     +----------------------v----------------------+
                     | Tier 2: CLINICIAN_DOCUMENTED                |
                     | EHR Diagnostic Code (ICD-10 E11)            |
                     +----------------------+----------------------+
                                            |
                     +----------------------v----------------------+
                     | Tier 3: SELF_REPORTED                       |
                     | Patient Questionnaire / Screening Survey    |
                     +----------------------+----------------------+
                                            |
                     +----------------------v----------------------+
                     | Tier 4: UNKNOWN                             |
                     | Missing Documentation (NEVER assumed 0)     |
                     +---------------------------------------------+
                                            |
                     +----------------------v----------------------+
                     | Tier 5: INVALID_DR_INFERRED                 |
                     | DERIVED FROM RETINOPATHY (STRICTLY REJECTED)|
                     +---------------------------------------------+
```

---

## 2. Strict Clinical Governance Principles

### Principle 1: Never Convert "Unknown" into Non-Diabetic (0)
- In population screening, individuals with undocumented diabetes status frequently have unmeasured hyperglycemia.
- Imputing missing labels as $0$ creates high false-negative contamination, training the model to treat diabetic microvascular patterns as normal.
- **Rule**: Any record where `diabetes_label` is `None`, `NaN`, or empty is classified into `DiabetesProvenanceTier.UNKNOWN` and pruned from model training.

### Principle 2: DR Severity Grades Are Not Diabetes Ground Truth
- Diabetic Retinopathy (DR) is an end-organ vascular complication of chronic hyperglycemia, not the systemic disease itself.
- A patient with diabetes may have normal-appearing retinal fundus photographs ($\text{DR} = 0$) during early disease stages.
- An eye with $\text{DR} = 0$ in an ophthalmology clinic (e.g., APTOS 2019) is a **diabetic individual without visible retinopathy**, NOT a healthy control.
- **Rule**: Any record derived from `APTOS_DR_Severity_Grade_0` or similar retinopathy grading is categorized as `INVALID_DR_INFERRED` and automatically rejected by `validate_diabetes_label_provenance()`.

### Principle 3: Strict Segregation of Self-Reported vs Laboratory-Confirmed Cohorts
- Self-reported diabetes diagnoses exhibit misclassification rates of $20\%-35\%$ in rural community populations due to asymptomatic disease onset and limited healthcare access.
- **Rule**: Models must never pool self-reported and laboratory-confirmed labels without separate stratified evaluation. In `filter_manifest_for_track_b()`, self-reported labels are excluded by default (`allow_self_reported=False`) unless explicitly requested for sensitivity analysis.

---

## 3. Laboratory Reference Standards & Diagnostic Thresholds

Diagnostic ground truth must conform to international clinical standards (American Diabetes Association [ADA] & World Health Organization [WHO]):

| Diagnostic Test | Standard Units | Diabetic Diagnostic Threshold | Prediabetes / Impaired Range | Normal Reference Range | Clinical Evidentiary Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glycated Hemoglobin ($\text{HbA1c}$)** | $\%$ (DCCT/NGSP) | $\ge 6.5\%$ | $5.7\% - 6.4\%$ | $< 5.7\%$ | Gold standard; reflects mean glycemia over preceding 90–120 days. Unaffected by acute fasting. |
| **$\text{HbA1c}$ (IFCC Standard)** | $\text{mmol/mol}$ | $\ge 48\text{ mmol/mol}$ | $39 - 46\text{ mmol/mol}$ | $< 39\text{ mmol/mol}$ | Standardized international unit for clinical trials. |
| **Fasting Plasma Glucose (FPG)** | $\text{mg/dL}$ | $\ge 126\text{ mg/dL}$ | $100 - 125\text{ mg/dL}$ | $< 100\text{ mg/dL}$ | Requires verified 8-hour overnight fast. Highly sensitive to acute compliance. |
| **Fasting Plasma Glucose (FPG)** | $\text{mmol/L}$ | $\ge 7.0\text{ mmol/L}$ | $5.6 - 6.9\text{ mmol/L}$ | $< 5.6\text{ mmol/L}$ | International unit equivalent. |
| **2-Hour Oral Glucose (OGTT)** | $\text{mg/dL}$ | $\ge 200\text{ mg/dL}$ | $140 - 199\text{ mg/dL}$ | $< 140\text{ mg/dL}$ | Following 75g anhydrous glucose load. |
| **Random Blood Glucose (RBG)** | $\text{mg/dL}$ | $\ge 200\text{ mg/dL}$ | — | — | Valid diagnostic standard **only** when accompanied by classical symptoms of hyperglycemia (polyuria, polydipsia, unexplained weight loss). |

---

## 4. Target Leakage Safeguard Protocol

### The Mechanism of Target Leakage in Medical Machine Learning
Target leakage occurs when clinical variables that define or directly reflect the prediction target are inadvertently provided as input features to the model.

```
[Target: Diabetes Diagnosis (HbA1c >= 6.5%)]
                    ^
                    | (Direct Deterministic Dependence)
[Feature: HbA1c / Fasting Glucose / Metformin]   <--- TARGET LEAKAGE!
```

If a multimodal model is provided with a patient's fasting blood glucose or $\text{HbA1c}$ to "predict" whether they have diabetes, the visual fundus branch will receive near-zero gradient signal, degenerating into a simple scalar threshold on the lab test. The resulting model will claim an artificial AUROC of $0.99+$ while being completely useless in rural screening camps where lab tests are unavailable.

### Enforced Runtime Safeguard
The framework executes `assert_no_target_leakage()` on every candidate tabular feature column prior to dataset creation or model ingestion.

#### Prohibited Input Features:
- `hba1c`, `glycated_hemoglobin`, `hba1c_pct`, `hba1c_mmol_mol`
- `fpg`, `fasting_plasma_glucose`, `fasting_blood_glucose`, `fasting_glucose`, `glucose`, `fbg`
- `ogtt`, `oral_glucose_tolerance_test`, `random_blood_glucose`, `rbg`
- `insulin_prescribed`, `metformin_prescribed`, `oral_hypoglycemics`

#### Permitted Non-Invasive Covariates at Screening Time:
- Demographics: `age`, `sex`
- Non-invasive anthropometrics: `bmi`, `height`, `weight`, `waist_circumference`
- Vital signs: `systolic_bp`, `diastolic_bp`, `pulse_rate`
- Behavioral survey variables: `smoking_status`, `family_history_of_diabetes`
