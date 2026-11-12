# Requirements Traceability Matrix (RTM)

**Project**: Explainable AI for Early Diabetes Risk Screening & Multi-Stage Diabetic Retinopathy Grading  
**Last Updated**: 2026-11-12  
**Lead AI Research Engineer**: Jay (`jay45-cyber`)  
**Status**: **ALL REQUIREMENTS IMPLEMENTED & VERIFIED (27/27 Tests Passing)**

---

| Req ID | Research / Clinical Requirement | Architectural Design | Implementation Component | Verification & Test | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-01** | Distinguish systemic diabetes screening from diabetic retinopathy grading. | `docs/research_scope.md` | `config/dataset_config.yaml`, `src/data/adapters/` | `tests/test_patient_grouping.py` | **VERIFIED** |
| **REQ-02** | Independent clinical ground truth (HbA1c >= 6.5%, FPG >= 126 mg/dL). | `docs/clinical_reference_label_protocol.md` | `src/data/patient_manifest.py` | Schema validation | **VERIFIED** |
| **REQ-03** | Grouped patient splitting; both eyes strictly in same split. | `docs/patient_level_splitting_and_leakage_audit.md` | `src/data/grouped_split.py` | `tests/test_patient_grouping.py` | **VERIFIED** |
| **REQ-04** | Detection of exact & perceptual image duplicates. | `docs/data_quality_audit.md` | `src/data/duplicate_detection.py` | `tests/test_duplicate_detection.py` | **VERIFIED** |
| **REQ-05** | Preprocessing pipeline: circular crop, quality check, normalization. | `docs/preprocessing_specification.md` | `src/data/preprocessing.py`, `src/data/image_quality.py` | `tests/test_image_preprocessing.py` | **VERIFIED** |
| **REQ-06** | Baseline ResNet50 model with reproducible training. | `docs/model_architecture_and_rationale.md` | `src/models/baseline_resnet.py`, `src/training/trainer.py` | `scripts/train_baseline.py` | **VERIFIED** |
| **REQ-07** | Advanced backbones (EfficientNetV2, ConvNeXt, Foundation Models). | `docs/model_architecture_and_rationale.md` | `src/models/efficientnet.py`, `src/models/advanced_architectures.py` | `tests/test_advanced_models.py` | **VERIFIED** |
| **REQ-08** | Clinically calibrated risk probability (Platt scaling / Isotonic). | `docs/evaluation_and_statistical_analysis.md` | `src/evaluation/calibration.py` | `tests/test_evaluation.py` | **VERIFIED** |
| **REQ-09** | Subgroup analysis on participants with **no visible DR (DR = 0)**. | `docs/research_questions.md` | `src/evaluation/subgroup_analysis.py` | Subgroup metrics | **VERIFIED** |
| **REQ-10** | Explainability visualizations (Grad-CAM++ / saliency maps). | `docs/explainability_protocol.md` | `src/explainability/gradcam.py`, `src/explainability/integrated_gradients.py` | `tests/test_explainability_methods.py` | **VERIFIED** |
| **REQ-11** | Interactive screening prototype with quality filter and disclaimer. | `docs/application_and_inference_specification.md` | `app/streamlit_app.py` | `tests/test_inference.py` | **VERIFIED** |
| **REQ-12** | Privacy, PHI safety, and zero leakage to Git. | `docs/security_privacy_and_safety.md` | `.gitignore`, `tests/` | Git diff & secret audits | **VERIFIED** |
| **REQ-13** | 5-Class DR Severity Grading (Grades 0 to 4) on APTOS 2019 standard. | `config/dr_grading_config.yaml` | `src/data/dr_dataset.py` | `tests/test_dr_pipeline.py::test_dr_dataset_and_clahe` | **VERIFIED** |
| **REQ-14** | ViT-B/16 & Swin Transformer multi-stage grading vs ResNet-50. | `docs/model_architecture_and_rationale.md` | `src/models/dr_models.py`, `src/training/dr_trainer.py` | `tests/test_dr_pipeline.py::test_dr_models_forward` | **VERIFIED** |
| **REQ-15** | Macro-F1, QWK, and Referable DR Sensitivity/Specificity (Grade >= 2). | `docs/evaluation_and_statistical_analysis.md` | `src/evaluation/dr_metrics.py`, `scripts/evaluate_dr_model.py` | `tests/test_dr_pipeline.py::test_dr_evaluation_metrics` | **VERIFIED** |
| **REQ-16** | Multi-Method XAI: ViT Rollout, Token Grad-CAM, Swin CAM, Integrated Gradients. | `docs/explainability_protocol.md` | `src/explainability/dr_explainer.py`, `scripts/generate_explanations.py` | `tests/test_dr_pipeline.py::test_dr_explainability_methods` | **VERIFIED** |
| **REQ-17** | Patient/sample 95% bootstrap confidence intervals on test outcomes. | `docs/evaluation_and_statistical_analysis.md` | `src/evaluation/dr_metrics.py`, `src/evaluation/confidence_intervals.py` | `tests/test_bootstrap_ci.py` | **VERIFIED** |
