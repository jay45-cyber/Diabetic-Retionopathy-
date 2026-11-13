# Milestone 4: Generalization & External Validation Report

## 1. Domain Generalization Scope

A critical failure mode of deep learning systems deployed in medical healthcare is performance collapse when transitioning from controlled research datasets to real-world clinical practice.

This report evaluates domain shifts across three primary dimensions:
1. **Patient-level separation** (within-cohort generalization).
2. **Camera and optical domain shifts** (tabletop vs handheld devices).
3. **Geographic and demographic distribution shifts** (rural Indian screening camps).

---

## 2. Within-Cohort Generalization (Patient-Level Separation)

All experimental splits in this repository strictly enforce patient-level isolation using bilateral identifier hashing (`patient_id` / bilateral eye matching):
- **Bilateral Eye Guard**: Left and right eye images belonging to the same individual are strictly assigned to the same partition (train, validation, or locked test).
- **Zero Leakage**: No patient represented in the training set ($N=300$ images, $150$ patients) or validation set ($N=100$ images, $50$ patients) appears in the locked test cohort ($N=100$ images, $50$ patients).
- **Test Performance Baseline**: On unseen patients within the experimental cohort, the model maintains consistent specificity ($1.0000$ at default threshold; $0.6786$ under multimodal late fusion), indicating that representations do not merely memorize patient-specific vascular landmarks.

---

## 3. External Clinical Validation Status

> [!IMPORTANT]
> **Definitive Clinical Declaration: External Validation Remains Incomplete**  
> In compliance with the EQUATOR network guidelines (STARD-AI and TRIPOD-AI), we explicitly state:
> - The model weights have **not yet undergone prospective validation on an independent external clinical cohort** (e.g., BRSET, UK Biobank, or prospective Indian clinical registry).
> - Models were developed and evaluated using local multi-site benchmark datasets and synthetic proxy cohorts with calibrated metadata.
> - **Regulatory / Deployment Boundary**: This software is an experimental research prototype. It is **not cleared by CDSCO, US FDA, or CE Mark** for diagnostic use. Prospective multi-site validation with venous laboratory reference standards (venous HbA1c $\ge 6.5\%$ / FPG $\ge 126\text{ mg/dL}$) is mandatory before any clinical pilot.

---

## 4. Anticipated Domain Shifts in Rural Indian Screening Camps

When moving from sanitized benchmark environments to primary healthcare centers (PHCs) in rural India, the following domain shifts are anticipated:

### A. Optical & Hardware Variability
- **Tabletop vs Handheld Devices**: Benchmark datasets typically utilize high-end desktop fundus cameras (e.g., Zeiss Visucam, Topcon TRC-50DX) with mydriatic pupils. Rural screening camps predominantly employ handheld, battery-powered non-mydriatic devices (e.g., Remidio Non-Mydriatic Fundus on Phone [NM-FOP], Forus 3nethra).
- **Aperture & Resolution**: Handheld cameras feature reduced optical magnification, narrower fields of view ($40^\circ$–$45^\circ$ vs $50^\circ$), and higher chromatic aberration.

### B. Patient & Environmental Demographics
- **Pupil Size & Media Opacities**: Non-mydriatic screening in geriatric rural populations frequently encounters physiological miosis ($< 3.0\text{ mm}$ pupil diameter) and senile cataracts, causing diffuse illumination drop-off and haze.
- **Fundus Pigmentation**: Retinal pigmentation in South Asian populations exhibits higher choroidal melanin density ("tesselated" or "tigroid" fundus), altering vessel-to-background contrast compared to predominantly Caucasian datasets.
- **Microclimate & Field Conditions**: High ambient temperatures, dust, and humidity in mobile screening vans can introduce lens condensation and dust specks mimicking microaneurysms.

---

## 5. Mitigation Strategies & Quality Gateways

To safeguard generalization, the deployed software implements a two-tier safety architecture:
1. **Automated Quality Gateway (`assess_image_quality`)**:
   - Laplacian variance sharpness threshold ($\ge 100.0$).
   - Mean illumination bounds ($40.0 \le \mu \le 215.0$).
   - Rejection and abstention for ungradable images (see `docs/validation/clinical_limitations.md`).
2. **Platt-Calibrated Probabilistic Uncertainty**:
   - Probabilities are calibrated on out-of-training validation samples to avoid overconfident misclassifications under domain shift.
