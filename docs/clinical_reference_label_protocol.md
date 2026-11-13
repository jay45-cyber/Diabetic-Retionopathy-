# Clinical Reference Label Protocol & Diagnostic Standards

**Project Identity**: Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Clinical Reference Guidelines**: ICMR (Indian Council of Medical Research), WHO, & ADA (American Diabetes Association) Standards of Medical Care  
**Date**: 2026-11-13  
**Status**: **ACTIVE STANDARD PROTOCOL**

---

## 1. Laboratory Diagnostic Reference Thresholds

To establish scientifically rigorous and clinically defensible ground-truth labels for systemic Type 2 diabetes, the project adopts international consensus thresholds:

| Diagnostic Biomarker | Normal (Non-Diabetic Control) | Prediabetes (Impaired Glucose) | Diabetes Mellitus (Screening Target) | Standard Units & Assay |
| :--- | :--- | :--- | :--- | :--- |
| **HbA1c (Glycated Hemoglobin)** | $< 5.7\%$ ($< 39$ mmol/mol) | $5.7\% - 6.4\%$ ($39 - 47$ mmol/mol) | $\ge 6.5\%$ ($\ge 48$ mmol/mol) | NGSP certified / IFCC reference method |
| **Fasting Plasma Glucose (FPG)** | $< 100$ mg/dL ($< 5.6$ mmol/L) | $100 - 125$ mg/dL ($5.6 - 6.9$ mmol/L) | $\ge 126$ mg/dL ($\ge 7.0$ mmol/L) | Hexokinase / Glucose oxidase, 8hr fast |
| **2-Hour Post-Load Glucose (OGTT)** | $< 140$ mg/dL ($< 7.8$ mmol/L) | $140 - 199$ mg/dL ($7.8 - 11.0$ mmol/L) | $\ge 200$ mg/dL ($\ge 11.1$ mmol/L) | 75g anhydrous glucose dissolved in water |
| **Random Plasma Glucose** | N/A | N/A | $\ge 200$ mg/dL ($11.1$ mmol/L) | Verified only in presence of classic symptoms |

---

## 2. Provenance Tier Hierarchy

All candidate ground truth labels ingested by the system are tagged into one of five evidentiary tiers:

1. **Tier 1: `LABORATORY_CONFIRMED`**
   - High-grade scientific ground truth.
   - Patient has documented venous plasma testing meeting ADA/ICMR thresholds within $\pm 90$ days of the retinal photograph.
   - Permitted for primary training, validation, and external testing.
2. **Tier 2: `CLINICIAN_DOCUMENTED`**
   - Medium-high grade clinical ground truth.
   - Physician-verified diagnosis documented in electronic health records (EHR) with confirmed ICD-10 code (`E11: Type 2 Diabetes Mellitus`).
   - Permitted for training and validation with explicit provenance tracking.
3. **Tier 3: `SELF_REPORTED`**
   - Low-medium grade subjective ground truth.
   - Patient self-reports antecedent diabetes diagnosis during screening camp intake or demographic survey without laboratory record verification.
   - **Protocol Safeguard**: Excluded from primary benchmark evaluation unless explicitly allowed via `--allow_self_reported`. Never merged silently with Tier 1.
4. **Tier 4: `UNKNOWN`**
   - Incomplete or unverified status.
   - **Protocol Safeguard**: Never converted to non-diabetic (0). Must be rejected from binary classification datasets to prevent negative-class label corruption.
5. **Tier 5: `INVALID_DR_INFERRED`**
   - **Strictly Prohibited**: Any label derived by equating DR Grade 0 with non-diabetic status, or DR Grade $\ge 1$ with diabetes.
   - Automatically flagged, logged, and rejected by `src/data/label_provenance.py`.

---

## 3. Prevention of Target Leakage

> [!CAUTION]
> **Scientific Target Leakage Rule**:
> When training a machine learning model to predict diabetes risk, laboratory biomarkers that define the disease (HbA1c, fasting plasma glucose, oral glucose tolerance test, or active prescriptions for insulin or metformin) must **NEVER** be provided as input features. Providing laboratory HbA1c to predict diabetes status constitutes trivial tautological target leakage.
>
> In multimodal models, only non-leaking pre-diagnostic clinical covariates realistically available at primary community triage (e.g. age, biological sex, BMI, waist-to-hip ratio, systolic/diastolic blood pressure, family history) may be included.

---

## 4. Disconnection from Diabetic Retinopathy (DR) Staging

Under no circumstances will the presence or absence of diabetic retinopathy lesions dictate the ground truth label:
$$\text{Label}(P) = f(\text{Laboratory Glycemia}(P)) \quad \not\equiv \quad f(\text{Retinal Lesion Score}(P))$$

The relationship between diabetes status and retinal manifestations forms four distinct cohorts:

| Cohort | Retinal Severity | Systemic Diabetes | Research Role & Clinical Meaning |
| :---: | :---: | :---: | :--- |
| **Cohort A** | **$\text{DR} = 0$** | **Diabetes = 1** | **Primary Research Cohort**: True diabetic individuals prior to visible microvascular retinopathy. Focus of early screening discovery. |
| **Cohort B** | **$\text{DR} = 0$** | **Diabetes = 0** | **True Control Cohort**: Healthy non-diabetic individuals with normal fundus appearance. |
| **Cohort C** | **$\text{DR} \ge 1$** | **Diabetes = 1** | **Established Complication**: Established diabetic retinopathy; validates that vascular feature extractors capture classic microvascular lesions. |
| **Cohort D** | **$\text{DR} \ge 1$** | **Diabetes = 0** | **Anomalous Cohort**: Non-diabetic retinopathy (e.g. hypertensive retinopathy, retinal vein occlusion). Requires clinical differential diagnosis. |

---

## 5. Temporal Consistency & Conflicting Labels Policy

1. **Temporal Horizon**: Retinal imaging and laboratory blood tests must have been acquired within a 90-day window. If the elapsed time exceeds 90 days, the record is flagged as `TEMPORAL_DISCORDANCE` and isolated.
2. **Conflicting Laboratory Measurements**: If FPG is elevated ($\ge 126\text{ mg/dL}$) but HbA1c is normal ($< 5.7\%$), or vice versa:
   - The label is classified as `DISCORDANT_GLYCEMIA` and excluded from primary training partitions.
   - Such cases are held in a separate sensitivity-analysis test split to evaluate model calibration on borderline pathology.
