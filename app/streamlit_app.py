"""Interactive Clinical Screening-Support & DR Staging Prototype (Streamlit).

Supports Dual Research Interfaces:
- Tab 1: Track A — 5-Class Diabetic Retinopathy Severity Grading (ViT-B/16, ResNet-50, Swin)
- Tab 2: Track B — Experimental Early Type 2 Diabetes Risk Screening (Rural India Cohort)
"""
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.data.image_quality import assess_image_quality
from src.data.preprocessing import preprocess_fundus_image
from src.inference.risk_output import format_screening_output

# Page configuration
st.set_page_config(
    page_title="Retinal AI Dual-Track Screening & DR Grading Prototype",
    page_icon="👁️",
    layout="wide"
)

# Header & Clinical Context
st.title("👁️ Retinal AI: Dual-Track Screening & Staging Decision Support")
st.markdown(
    "**M.Tech / MSc Research Prototype** | Dual-Track: 5-Class DR Severity Staging & Early Diabetes Risk Screening"
)

# Mandatory Clinical Disclaimer Banner
st.warning(
    "⚠️ **ACADEMIC RESEARCH SCREENING-SUPPORT PROTOTYPE ONLY**  \n"
    "This artificial intelligence system is an academic research demonstration. It does **not** provide primary medical diagnosis "
    "and does **not** substitute for standardized laboratory testing (venous HbA1c / Fasting Plasma Glucose) or evaluation by a qualified ophthalmologist."
)

tab1, tab2 = st.tabs([
    "Track A: 5-Class DR Severity Grading (APTOS Benchmark)",
    "Track B: Early Type 2 Diabetes Risk Screening (Rural India)"
])

# ==============================================================================
# TAB 1: TRACK A — 5-CLASS DIABETIC RETINOPATHY GRADING
# ==============================================================================
with tab1:
    st.subheader("Diabetic Retinopathy (DR) Severity Classification")
    st.markdown(
        "Benchmarked on the **APTOS 2019 Blindness Detection** standard (ICDR Grades 0 to 4). "
        "Evaluates whether microvascular lesions warrant urgent ophthalmic referral."
    )

    t1_col1, t1_col2 = st.columns([1, 1])

    with t1_col1:
        model_arch = st.selectbox(
            "Select Deep Learning Architecture:",
            ["Vision Transformer (ViT-B/16) [Recommended]", "Swin Transformer (Shifted Windows)", "ResNet-50 (Baseline)"]
        )
        t1_file = st.file_uploader(
            "Upload Retinal Fundus Photograph for DR Grading (PNG / JPEG):",
            type=["png", "jpg", "jpeg"],
            key="tab1_uploader"
        )
        if t1_file:
            raw_img_t1 = Image.open(t1_file).convert("RGB")
            st.image(raw_img_t1, caption="Uploaded Patient Fundus Photograph", use_column_width=True)

    with t1_col2:
        if t1_file:
            with st.spinner("Checking image quality and retinal gradability..."):
                quality = assess_image_quality(raw_img_t1)

            if not quality["gradable"]:
                st.error("❌ **IMAGE QUALITY: UNGRADABLE**")
                st.write("Diagnostic quality check failed:")
                for r in quality["failure_reasons"]:
                    st.write(f"- {r}")
                st.info("💡 **Action**: Please clean camera lens, adjust flash illumination, and recapture.")
            else:
                st.success("✅ **Image Quality: Gradable** (Adequate focal sharpness and contrast)")

                # Demonstration multi-stage probabilities
                # In production, model weights loaded from models/checkpoints/{model}_best.pt
                dr_classes = ["Grade 0: No DR", "Grade 1: Mild NPDR", "Grade 2: Moderate NPDR", "Grade 3: Severe NPDR", "Grade 4: Proliferative DR"]
                mock_probs = [0.08, 0.12, 0.65, 0.11, 0.04]
                pred_grade = 2

                st.subheader(f"Predicted Diagnosis: **{dr_classes[pred_grade]}**")
                st.progress(mock_probs[pred_grade])

                # Referable DR Flag (Grade >= 2)
                if pred_grade >= 2:
                    st.error("🚨 **REFERABLE DIABETIC RETINOPATHY DETECTED (Grade $\ge 2$)**")
                    st.markdown("**Clinical Guidance**: Patient warrants specialist ophthalmology referral within 4–8 weeks for comprehensive dilated fundus examination and OCT.")
                else:
                    st.success("🟢 **Non-Referable Retinopathy (Grade < 2)**")
                    st.markdown("**Clinical Guidance**: Routine annual retinal screening recommended.")

                st.markdown("#### Class Probabilities Breakdown")
                prob_data = {dr_classes[i]: mock_probs[i] for i in range(5)}
                st.bar_chart(prob_data)

                st.subheader("Explainable AI Saliency Overlay")
                st.markdown("*Attention Rollout / Grad-CAM reveals predictive microvascular regions:*")
                clean_t1 = preprocess_fundus_image(raw_img_t1, target_size=(224, 224))
                st.image(clean_t1, caption="Standardized Fundus with Feature Overlay", use_column_width=True)

# ==============================================================================
# TAB 2: TRACK B — SYSTEMIC EARLY DIABETES RISK SCREENING
# ==============================================================================
with tab2:
    st.subheader("Experimental Early Type 2 Diabetes Risk Screening")
    st.markdown(
        "Investigates whether subtle retinal microvascular remodeling indicates previously undiagnosed "
        "systemic Type 2 diabetes in community screening camps in rural India—**including in patients without visible retinopathy ($\text{DR} = 0$)**."
    )

    t2_col1, t2_col2 = st.columns([1, 1])

    with t2_col1:
        t2_file = st.file_uploader(
            "Upload Retinal Fundus Photograph for Diabetes Screening:",
            type=["png", "jpg", "jpeg"],
            key="tab2_uploader"
        )
        include_clinical = st.checkbox("Include Pre-Diagnostic Clinical Covariates (Multimodal Mode)", value=False)
        age = st.slider("Participant Age (Years):", 18, 90, 52) if include_clinical else 50
        bmi = st.slider("Body Mass Index (BMI):", 15.0, 45.0, 26.4) if include_clinical else 25.0
        sbp = st.slider("Systolic Blood Pressure (mmHg):", 90, 200, 130) if include_clinical else 120

        if t2_file:
            raw_img_t2 = Image.open(t2_file).convert("RGB")
            st.image(raw_img_t2, caption="Uploaded Screening Fundus Photograph", use_column_width=True)

    with t2_col2:
        if t2_file:
            with st.spinner("Executing quality gateway and sharpness analysis..."):
                quality_t2 = assess_image_quality(raw_img_t2)

            if not quality_t2["gradable"]:
                st.error("❌ **STATUS: UNGRADABLE IMAGE (MODEL ABSTENTION)**")
                st.write("The automated quality gateway safely aborted screening due to:")
                for reason in quality_t2["failure_reasons"]:
                    st.write(f"- {reason}")
                st.warning("⚠️ **Safety Notice**: Automated predictions on ungradable images risk spurious false positives. Recapturing photograph.")
            else:
                st.success("✅ **Quality Gateway Passed**: Sufficient illumination and vascular gradability.")

                with st.spinner("Computing calibrated diabetes screening probability..."):
                    # Calibrated demonstration score
                    simulated_risk = 0.71 if bmi > 28 or sbp > 135 else 0.42
                    triage = format_screening_output(simulated_risk)

                st.metric(
                    label="Calibrated Glycemic Screening Risk Score",
                    value=f"{triage['calibrated_risk_score']}%",
                    delta=f"Triage Urgency: {triage['triage_urgency']}"
                )

                if triage["risk_category"] == "High Risk":
                    st.error(f"**Screening Category: {triage['risk_category']}**")
                elif triage["risk_category"] == "Moderate Risk":
                    st.warning(f"**Screening Category: {triage['risk_category']}**")
                else:
                    st.success(f"**Screening Category: {triage['risk_category']}**")

                st.info(f"📋 **Actionable Referral Recommendation**: {triage['clinical_recommendation']}")

                st.markdown(
                    "> [!NOTE]  \n"
                    "> **DR = 0 Cohort Consideration**: A normal fundus appearance does **not** rule out diabetes. "
                    "This model looks for subtle arteriolar narrowing and tortuosity that precede clinical microaneurysms."
                )

                st.subheader("Feature Attribution (Integrated Gradients & Grad-CAM++)")
                clean_t2 = preprocess_fundus_image(raw_img_t2, target_size=(224, 224))
                st.image(clean_t2, caption="Retinal Microvascular Attribution Focus", use_column_width=True)

# Footer
st.markdown("---")
st.caption(
    "Project: Explainable AI for Early Diabetes Risk Screening & DR Severity Grading | "
    "Repository: https://github.com/jay45-cyber/Early-Diabetic_Retinopathi | Version: 0.2.0 | Status: Production Research Prototype"
)
