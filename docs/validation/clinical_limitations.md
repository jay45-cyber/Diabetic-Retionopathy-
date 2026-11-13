# Milestone 4: Clinical Limitations, Governance & Deployment Guardrails

## 1. Clinical Ground-Truth & Label Provenance Limitations

A primary scientific risk in ophthalmic AI is the confusion of **retinopathy grading** with **systemic diabetes screening**:

1. **Proxy vs Confirmatory Reference Standards**:
   - Public retinal datasets (APTOS 2019, IDRiD, Messidor-2) are annotated exclusively for Diabetic Retinopathy severity (ICDR grades 0–4), not systemic glycemia.
   - For Milestone 4 research, systemic diabetes experiments utilized controlled proxy cohorts with calibrated clinical metadata (age, sex, BMI, blood pressure) and known diabetic status.
   - **Mandatory Reference Standard for Clinical Use**: In real-world deployment, systemic diabetes ground truth must be established by venous laboratory testing:
     - Glycated Hemoglobin ($\text{HbA1c} \ge 6.5\%$ / $48\text{ mmol/mol}$) via NGSP-certified, DCCT-standardized laboratory HPLC analyzers.
     - Fasting Plasma Glucose ($\text{FPG} \ge 126\text{ mg/dL}$ / $7.0\text{ mmol/L}$) following an 8-hour overnight fast.
   - Self-reported diabetes or fingerstick capillary glucometer tests exhibit substantial variability ($\pm 15\%$) and cannot serve as definitive clinical reference standards.

---

## 2. Real-World Field Conditions in Rural India

Screening camps conducted in rural Primary Healthcare Centers (PHCs) and Community Healthcare Centers (CHCs) face severe operational constraints:

### A. Non-Mydriatic Ungradability Rates
- Routine dilation (mydriasis with tropicamide) is clinically unfeasible in community screening camps due to stinging discomfort, prolonged photophobia ($4$–$6$ hours), and risk of precipitating acute angle-closure glaucoma in predisposed eyes.
- Non-mydriatic fundus photography under physiological miosis ($< 3.0\text{ mm}$ pupil) in geriatric patients with senile cataracts leads to ungradability rates of **$12\%$ to $28\%$**.
- **System Safeguard**: The automated quality gateway (`src/data/image_quality.py`) rigorously detects underexposure, blur, and lens artifact, enforcing algorithmic abstention. The system instructs health workers to recapture the image rather than generating an unreliable prediction.

### B. Computational & Environmental Edge Constraints
- Mobile screening vans often operate in remote geographies lacking stable internet connectivity and reliable air conditioning.
- The inference pipeline must execute offline on resource-constrained hardware (laptop CPU or embedded edge accelerators such as NVIDIA Jetson Orin Nano).
- While ResNet-50 operates efficiently on CPU ($< 150\text{ ms}$ per image), scaling to multi-view ensemble ViT architectures at $512 \times 512$ resolution requires edge quantization (INT8) to prevent latency bottlenecks.

---

## 3. Regulatory Governance & Software as a Medical Device (SaMD)

1. **Intended Use Definition**:
   - This system is an **academic research prototype designed for risk stratification and screening triage assistance**.
   - It is **not an autonomous diagnostic device** and must never replace clinical evaluation by an ophthalmologist, diabetologist, or general physician.
2. **Regulatory Classification**:
   - Under the Central Drugs Standard Control Organization (CDSCO) Medical Device Rules in India, AI diagnostic software is categorized as **Class C (Moderate-High Risk) Software as a Medical Device (SaMD)**.
   - Clinical deployment requires formal prospective investigational device clinical trials (IDCT) across multi-center Indian cohorts, Good Clinical Practice (GCP) compliance, and CDSCO registration.
3. **Data Privacy & Ethics (DPDPA 2023)**:
   - In accordance with India's Digital Personal Data Protection Act (DPDPA 2023), all fundus photographs and demographic metadata must be stripped of Personally Identifiable Information (PII) at the edge before storage or synchronization.
   - Patient consent must be obtained in vernacular regional languages prior to imaging.
