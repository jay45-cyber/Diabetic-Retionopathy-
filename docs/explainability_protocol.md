# Explainable AI (XAI) Protocol & Clinical Plausibility Review

**Project Identity**: Explainable AI for Early Diabetes Risk Screening Using Retinal Fundus Images in Rural India  
**Auditor**: Lead AI Research Engineer & Medical Imaging Specialist  
**Status**: Comprehensive Protocol for Track A (Multi-Class DR) and Track B (Systemic Diabetes Logit)

---

## 1. Clinical Motivation & Safety Grounding

In medical imaging, post-hoc explainability serves two distinct clinical imperatives:
1. **Shortcut & Artifact Detection**: Verifying that decisions are driven by physiological retinal anatomy (vessels, optic disc, macula) rather than non-biological confounders (camera dust, edge vignetting, illumination gradients, or border compression).
2. **Pre-Clinical Hypothesis Generation**: Inspecting whether systemic risk correlates with subtle vascular features (arteriolar narrowing, venular dilation, tortuosity changes) in eyes without visible retinopathy ($\text{DR} = 0$).

> [!WARNING]
> **Clinical Governance Notice**:
> Saliency maps and gradient attributions represent mathematical sensitivity of the network's output to input activations. They do **NOT** constitute medical proof of biological causality, histological lesions, or independent biomarker discovery. All visualizations must be accompanied by clinical confidence indicators and quality screening.

---

## 2. Dual-Track Explainability Architecture

```
                                Trained Neural Architecture
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
   +-------------------------------+                     +-------------------------------+
   | TRACK A: Multi-Class DR Head  |                     | TRACK B: Scalar Diabetes Logit|
   +-------------------------------+                     +-------------------------------+
   | - Target: Predicted Class c   |                     | - Target: Risk Logit z        |
   | - ResNet: Layer4 Grad-CAM     |                     | - ResNet/ConvNeXt: Grad-CAM   |
   | - ViT: Attention Rollout      |                     | - Integrated Gradients:       |
   | - ViT: Token Grad-CAM         |                     |   Path from black baseline    |
   | - Swin: Norm-Layer Grad-CAM   |                     | - Retinal Heatmap Overlay     |
   +-------------------------------+                     +-------------------------------+
                   |                                                     |
                   +--------------------------+--------------------------+
                                              |
                                              v
                              +-------------------------------+
                              | Clinical Plausibility Audit   |
                              | - Vascular Arcade Alignment   |
                              | - Optic Disc & Macula Check   |
                              | - Weight Randomization Sanity |
                              +-------------------------------+
```

---

## 3. Track A Explainability Methods (Diabetic Retinopathy)

Track A visualizes class-discriminative evidence for the predicted ordinal DR severity grade $c \in \{0, 1, 2, 3, 4\}$.

### Convolutional Architectures (ResNet-50)
- **Method**: Standard Grad-CAM (Selvaraju et al., 2017) applied to `model.layer4[-1]`.
- **Target Gradient**:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
- **Lesion Alignment**: Highlights focal microaneurysms, hard exudates, and intraretinal hemorrhages.

### Vision Transformer (ViT-B/16)
- **Method 1: Attention Rollout (Abnar & Zuidema, 2020)**:
  - Recursively multiplies self-attention matrices across all 12 transformer layers, accounting for residual skip connections:
    $$\hat{A}^{(l)} = 0.5 A^{(l)} + 0.5 I, \quad R^{(L)} = \prod_{l=1}^L \hat{A}^{(l)}$$
  - Extracts the `[CLS]` token's cumulative attention map across spatial patch tokens, reshaped to $14 \times 14$ and upsampled to input resolution.
- **Method 2: Token Grad-CAM**:
  - Computes gradient of predicted class logit $y_c$ with respect to the output patch tokens of the final Transformer encoder block.

### Shifted Window Transformer (Swin)
- **Method**: Grad-CAM targeted at the penultimate layer normalization layer `model.norm`.

---

## 4. Track B Explainability Methods (Systemic Diabetes Screening)

Track B investigates subtle global microvascular variations by attributing directly to the scalar diabetes logit $z$ (or probability $p = \sigma(z)$).

### Binary Grad-CAM (`DiabetesRiskExplainer.generate_gradcam`)
- **Target**: Scalar logit $z$.
- **Target Layer**: `layer4[-1]` (ResNet-50) or final convolutional stage (ConvNeXt).
- **Processing**:
  1. Computes $\frac{\partial z}{\partial A^k}$.
  2. Weights feature maps by global-average-pooled gradients.
  3. Applies $\text{ReLU}$ to isolate features increasing diabetes risk.
  4. Resizes to $224 \times 224$ (or $384 \times 384$) and normalizes to $[0, 1]$.

### Integrated Gradients (`DiabetesRiskExplainer.generate_integrated_gradients`)
- **Mathematical Definition** (Sundararajan et al., 2017):
  $$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} \, d\alpha$$
  where $x'$ is a non-informative black baseline ($0$-tensor) and $M = 25$ Riemann sum interpolation steps.
- **Axiomatic Properties**: Satisfies *Completeness* ($\sum_i \text{IG}_i(x) = F(x) - F(x')$) and *Implementation Invariance*.
- **Retinal Sensitivity**: Identifies localized perivascular attributions and macular vascular densities without spatial downsampling artifacts.

---

## 5. Clinical Plausibility Checklist & Sanity Auditing

Visualizations must undergo verification against known physical fundus landmarks:

| Landmark | Biological Role | Expected Attribution in Early Diabetes |
| :--- | :--- | :--- |
| **Superior / Inferior Arcades** | Major arterial and venous trunks | Caliber shifts, generalized arteriolar narrowing, venular dilation. |
| **Optic Disc (Papilla)** | Convergence of nerve fibers & central vessels | Peripapillary capillary network changes; absent of glare artifacts. |
| **Macular Region** | Foveal avascular zone (FAZ) & capillary plexus | Subtle FAZ irregularities, microvascular loss (pre-clinical drop-out). |
| **Peripheral Retina** | Distal capillary beds | Microaneurysm clustering (if transitioning to Grade 1 DR). |

### Sanity Checks (Adebayo et al., NeurIPS 2018)
- **Cascading Parameter Randomization**: Saliency methods are evaluated on randomized model weights. If heatmaps remain unchanged when network weights are scrambled, the method is rejected as an edge-detector rather than a model explanation.
- **Data Randomization**: Evaluating whether explanations degenerate when trained on randomly permuted patient labels.
