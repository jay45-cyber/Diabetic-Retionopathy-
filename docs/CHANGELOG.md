# Changelog

All notable technical and research changes to the **Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India** project are documented contemporaneously in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [0.2.0] - 2026-11-12
### Added
- **5-Class DR Grading Pipeline**: Implemented `src/data/dr_dataset.py` supporting APTOS 2019 five-class severity standard (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative), circular fundus cropping, luminance CLAHE, and normalized inverse-frequency class weighting.
- **Stratified Splitting**: Built deterministic 70% Train / 15% Val / 15% Test split (`create_stratified_dr_splits`) with fixed random seed (42) and duplicate perceptual hash prevention.
- **Model Architecture Zoo**: Added `src/models/dr_models.py` implementing `ViTDR` (ViT-B/16), `ResNet50DR`, `SwinTransformerDR`, and model factory `build_dr_model()`.
- **Upgraded Trainer**: Added `src/training/dr_trainer.py` (`DRTrainer`) featuring AdamW optimizer, Cosine Annealing learning rate schedule, mixed-precision `torch.amp.autocast` with CPU fallback, gradient clipping, early stopping tracking Validation Macro-F1 (patience = 5), and checkpoint saving to `models/checkpoints/`.
- **Clinical Evaluation Suite**: Implemented `src/evaluation/dr_metrics.py` computing Accuracy, Macro/Weighted F1, Quadratic Weighted Kappa (QWK), Referable DR Sensitivity/Specificity (Grade >= 2), One-vs-Rest AUROC, Multi-Class Expected Calibration Error (ECE), and 95% bootstrap confidence intervals.
- **Automated Artifacts Generator**: Automated export of `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, and `evaluation_summary.txt`.
- **Explainable AI Suite**: Implemented `src/explainability/dr_explainer.py` supporting ResNet-50 Grad-CAM, ViT Attention Rollout across 12 transformer blocks, ViT Token Grad-CAM ($14 \times 14$ grid), Swin Window Grad-CAM, and class-targeted Integrated Gradients.
- **CLI Scripts**: Added `scripts/train_dr_model.py`, `scripts/evaluate_dr_model.py`, and `scripts/generate_explanations.py`.
- **Automated Tests**: Added `tests/test_dr_pipeline.py` with 7 comprehensive unit/integration tests; entire repository test suite expanded to 27 tests passing with 100% success rate.
- **Comprehensive Documentation**: Published 21-section final research handover report at `docs/project_management/final_project_report.md` and upgraded `README.md`.

---

## [0.1.0] - 2026-11-01
### Added
- **Phase 0 Baseline**: Initialized workspace, completed hardware and software audit (`docs/WORKSPACE_AUDIT.md`).
- **Git & GitHub Integration**: Configured Git credentials for `jay45-cyber`, synchronized remote `origin/main`, created initial commit `cc8a192`.
- **Runtime Environment**: Installed Python 3.11.9, created project virtual environment (`venv`), established `requirements.txt` and `pyproject.toml`.
- **Directory Hierarchy**: Established complete research scaffolding (`config/`, `docs/`, `data/`, `notebooks/`, `src/`, `tests/`, `app/`, `outputs/`).
- **Configuration Management**: Created modular YAML configs (`project_config.yaml`, `dataset_config.yaml`, `training_config.yaml`, `evaluation_config.yaml`).
- **Living Documentation**: Published full suite of foundational documentation:
  - `IMPLEMENTATION_PLAN.md` (WP1–WP12 end-to-end plan)
  - `PROJECT_STATUS.md` and `PROGRESS_TRACKER.md`
  - `TRACEABILITY_MATRIX.md`
  - `research_scope.md` and `research_questions.md`
  - `dataset_feasibility_audit.md` (distinguishing DR grading from diabetes)
  - `clinical_reference_label_protocol.md` (HbA1c / FPG standards)
  - `patient_level_splitting_and_leakage_audit.md`
  - `preprocessing_specification.md`
  - `model_architecture_and_rationale.md`
  - `evaluation_and_statistical_analysis.md`
  - `explainability_protocol.md`
  - `application_and_inference_specification.md`
  - `security_privacy_and_safety.md`
  - `model_card.md`
  - `risk_register.md`
  - `scientific_assumptions_and_decisions.md` (ADR-001 through ADR-004)
  - `errors_and_resolutions.md` (ERR-001 through ERR-003)
  - `development_logs/2026-11-01_phase0_workspace_audit.md`

