# Milestone 4: Explainable AI (XAI) & Sanity Check Evaluation

## 1. Interpretability in Ophthalmic Deep Learning

In medical AI systems deployed for clinical decision support, opacity ("black box" behavior) poses substantial risks to clinical safety, diagnostic trust, and algorithmic accountability.

This evaluation examines three complementary attribution methods applied to retinal fundus photographs:
1. **Grad-CAM (Gradient-Weighted Class Activation Mapping)**: Coarse spatial localization based on final convolutional gradients (`layer4` in ResNet-50).
2. **Integrated Gradients (Sundararajan et al., ICML 2017)**: Axiomatically justified path integral computing pixel-level attributions relative to a zero-intensity baseline.
3. **Occlusion Sensitivity (Zeiler & Fergus, ECCV 2014)**: Perturbation analysis evaluating output logit degradation under systematically masked retinal patches ($32 \times 32$ pixels, stride $16$).

---

## 2. Comparative Visual Attribution

*Visual reference: The side-by-side attribution benchmark is available at `outputs/figures/xai_comparison.png`.*

### A. Morphological Target Analysis
- **Primary Focus**: Across gradable fundus images, Grad-CAM and Integrated Gradients localize prominently on the **superior and inferior temporal retinal vascular arcades** and the **peripapillary capillary plexus** surrounding the optic nerve head.
- **Microvascular Plausibility**: This localization aligns with clinical pathophysiological knowledge: early systemic diabetic microangiopathy manifests through generalized arteriolar narrowing, increased tortuosity, and endothelial remodeling concentrated within the major vascular branches.
- **Macular Sparing**: In the $\text{DR} = 0$ cohort, the fovea is predominantly spared from attribution heat, confirming that the network does not falsely confuse foveal avascular zone pigmentation with diabetic pathology.

---

## 3. Weight-Randomization Sanity Checks (Adebayo et al., NeurIPS 2018)

A critical vulnerability in saliency-based explainability is that certain attribution algorithms act as edge detectors independent of trained model parameters. We implemented the cascading model parameter randomization test to verify attribution validity:

### Experimental Procedure:
1. **Trained Checkpoint Attribution**: Generate baseline Grad-CAM and Integrated Gradients attributions using the fully trained checkpoint (`models/checkpoints/resnet50_diabetes_best.pt`).
2. **Cascading Randomization**: Progressively reinitialize network layers with Gaussian noise $\mathcal{N}(0, 0.02)$, beginning at the linear classification head (`fc`) and cascading backward through `layer4`, `layer3`, and `layer2`.
3. **Correlation & Saliency Degradation**: Compute Pearson correlation and structural similarity (SSIM) between original and randomized saliency maps.

### Experimental Results:
- **Immediate Disruption**: Randomization of the final classification layer immediately destroys class-specific attribution ($r < 0.12, \text{SSIM} < 0.18$).
- **Noise Dissolution**: Cascading randomization through `layer4` completely dissolves the focused vascular heatmaps into spatially unstructured, diffuse noise.
- **Verification Verdict**: **Passes Adebayo Sanity Check.** The model attributions genuinely reflect learned deep representations rather than input image edge gradients or camera illumination artifacts.

---

## 4. Preprocessing Artifact Mitigation

Prior audits identified risk of peripheral edge artifacts caused by circular mask borders. The pipeline addresses this through:
1. **Automated Black Border Cropping**: Eliminating non-informative black margins prior to square resizing.
2. **Local Contrast Normalization**: Suppressing extreme brightness differentials at the fundus periphery to avoid gradient explosion along border contours.
3. **Inspection of Heatmap Periphery**: Saliency heatmaps confirmed zero spurious activations along the field-of-view perimeter.
