# Clinical Data Acquisition Protocol & Research Governance Framework

**Document**: `docs/data_readiness/clinical_data_acquisition_plan.md`  
**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  
**Date**: 2026-11-13  
**Auditor**: Senior Medical AI Research Engineer & Clinical Protocol Architect  

---

## 1. Context & Governance Notice

> [!IMPORTANT]
> **Regulatory & Ethical Compliance Notice**:
> This document specifies a prospective research protocol outline for a future clinical collaboration with rural eye hospitals and community screening camps in India.
> - **Zero Autonomous Data Collection**: No actual clinical data collection or patient contact has occurred or will be initiated autonomously.
> - **No Institutional Claims**: No approval from an Institutional Ethics Committee (IEC) or Institutional Review Board (IRB) is fabricated or claimed. All execution is contingent upon future formal academic/clinical partnership agreements and formal IEC approval.

---

## 2. Study Design & Participant Selection

### Study Setting & Target Population
- **Setting**: Community diabetic retinopathy and general health screening camps in rural and semi-urban primary health centers (PHCs) across high-prevalence Indian states (e.g., Tamil Nadu, Maharashtra, Karnataka).
- **Design**: Prospective, cross-sectional, multi-center observational screening study.
- **Target Population**: Adults aged $\ge 35$ years presenting for vision screening or routine community health check-ups.

### Inclusion Criteria
1. Adults aged $\ge 35$ years.
2. Willing and able to provide informed consent (written signature or vernacular witnessed thumbprint).
3. Ability to sit upright for non-mydriatic fundus photography.
4. Agreeable to concurrent point-of-care venous or capillary blood testing for $\text{HbA1c}$.

### Exclusion Criteria
1. Media opacities preventing fundus visualization (dense cataract LOCS III grade $\ge 3$, corneal scars, vitreous hemorrhage).
2. Prior bilateral retinal surgery, vitrectomy, or extensive panretinal photocoagulation (PRP).
3. Co-existing non-diabetic retinal vascular occlusions, severe age-related macular degeneration (AMD), or high myopia ($> -6.00\text{ D}$) with chorioretinal atrophy.
4. Active pregnancy (gestational diabetes is etiologically distinct from Type 2 diabetes).
5. Unstable systemic condition requiring emergency medical referral.

---

## 3. Informed Consent & Language Protocol

- **Vernacular Consent**: Patient information sheets and informed consent forms must be provided in local Indian languages (Tamil, Marathi, Kannada, Hindi) alongside English.
- **Witnessed Thumbprint**: In low-literacy rural populations, witnessed verbal consent accompanied by an ink thumbprint and independent impartial witness signature is required under Indian Council of Medical Research (ICMR) ethical guidelines.
- **Explicit Scope of Consent**:
  1. De-identified retinal photographs and non-invasive demographic parameters used for research.
  2. Point-of-care $\text{HbA1c}$ blood testing.
  3. Direct return of results to the participant with clinical counseling and referral letter if hyperglycemia or referable retinopathy is detected.

---

## 4. Retinal Fundus Image Acquisition Protocol

- **Camera Hardware**: Handheld non-mydriatic digital fundus cameras suitable for rural deployment (e.g., Remidio Non-Mydriatic Fundus on Phone [FOP-NM], Volk VistaView, or Forus 3nethra).
- **Capture Protocol**:
  - Examination conducted in a dimly lit, enclosed mobile screening booth to promote physiological pupil dilation without pharmacologic mydriasis.
  - Two fields per eye:
    - **Field 1 (Macula-Centered)**: $45^\circ$ field centered on the fovea.
    - **Field 2 (Disc-Centered)**: $45^\circ$ field centered on the optic nerve head.
  - Bilateral Imaging: Both eyes (OD and OS) captured for every participant.
- **Gradability Check**: Real-time quality verification using the automated `ImageQualityAssessor` (screening for Laplacian blur, dark vignetting, and pupil alignment) before the participant leaves the screening station.

---

## 5. Synchronized Clinical & Laboratory Reference Standards

To achieve **Tier 1 (`LABORATORY_CONFIRMED`)** ground truth:
- **Primary Laboratory Measure**: Point-of-Care (POC) Glycated Hemoglobin ($\text{HbA1c}$) analyzer certified by the National Glycohemoglobin Standardization Program (NGSP) and International Federation of Clinical Chemistry (IFCC) (e.g., DCA Vantage or Abbott Afinion).
- **Diagnostic Cutoffs**:
  - $\text{HbA1c} \ge 6.5\%$ ($48\text{ mmol/mol}$): **Confirmed Diabetes**.
  - $\text{HbA1c} = 5.7\% - 6.4\%$ ($39 - 46\text{ mmol/mol}$): **Prediabetes** (stratified analysis cohort).
  - $\text{HbA1c} < 5.7\%$ ($< 39\text{ mmol/mol}$): **Confirmed Non-Diabetic Control**.
- **Secondary Reference (Subset)**: Fasting plasma glucose (FPG) with verified 8-hour fast where feasible.
- **Ophthalmic Reference**: Adjudicated diabetic retinopathy severity grading (ICDR grades 0 to 4) conducted independently by two certified retina specialists, with a senior consultant arbitrating discrepancies.

---

## 6. Patient Linkage, De-Identification & Privacy Governance

```
[Participant at Screening Camp]
              |
              | (Encrypted Hash Salt)
              v
[Pseudonymous ID: IND-RUR-2026-XXXX]
              |
      +-------+-------+
      |               |
      v               v
[De-Identified   [Locked Clinical
 Fundus Images]   Research Database]
 (OD/OS Linked)   (HbA1c, Age, Sex, BMI)
```

- **De-Identification Protocol**: All Personally Identifiable Information (name, Aadhaar number, phone number, address) is decoupled at the point of capture.
- **Patient Identifier Hash**: A cryptographic salted hash generates `participant_id` (e.g., `IND-TN-2026-0042`). Both eyes (`_OD`, `_OS`) share this exact prefix to guarantee zero leakage during dataset splitting.
- **Regulatory Framework Compliance**:
  - Digital Personal Data Protection Act (DPDPA 2023, India).
  - Digital Information Security in Healthcare Act (DISHA guidelines).
  - Indian Council of Medical Research (ICMR) Ethical Guidelines for Biomedical Research (2017).

---

## 7. Institutional Ethics Approval & Multi-Center Validation Roadmap

1. **IEC / IRB Protocol Submission**: Joint submission between academic research lead and clinical collaborating institutions (e.g., tertiary eye hospital network in India).
2. **Clinical Trial Registry**: Prospective registration on the Clinical Trials Registry - India (CTRI).
3. **Multi-Center External Validation**:
   - Training & Development: Center 1 (e.g., rural Tamil Nadu).
   - Prospective External Validation: Center 2 (e.g., rural Maharashtra) using different fundus camera hardware to evaluate geographic, optical, and ethnic domain robustness.
