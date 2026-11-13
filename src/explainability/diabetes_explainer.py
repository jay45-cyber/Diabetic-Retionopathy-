"""Explainable AI (XAI) Suite for Track B: Binary Systemic Diabetes Risk Prediction.

Generates saliency heatmaps highlighting retinal features that drive the diabetes risk score:
- ResNet-50 & ConvNeXt Grad-CAM++
- Integrated Gradients
- Occlusion Sensitivity
- Retinal Image Heatmap Blending
"""
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.explainability.dr_explainer import overlay_cam_on_image


class DiabetesRiskExplainer:
    """Explainer for binary retinal diabetes risk screening models."""

    def __init__(self, model: nn.Module, model_architecture: str = "resnet50"):
        self.model = model
        self.model_arch = model_architecture.lower().strip()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = None
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            target_layer = self.model.backbone.layer4[-1]
        elif hasattr(self.model, "layer4"):
            target_layer = self.model.layer4[-1]
        elif hasattr(self.model, "stages"):
            # ConvNeXt stage 4
            target_layer = self.model.stages[-1]
        else:
            # Find last Conv2d layer
            convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
            if convs:
                target_layer = convs[-1]

        if target_layer is not None:
            self.handles.append(target_layer.register_forward_hook(self._fwd))
            self.handles.append(target_layer.register_full_backward_hook(self._bwd))

    def _fwd(self, module, inp, outp):
        self.activations = outp.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_gradcam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for the diabetes risk logit."""
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        logit = self.model(input_tensor)

        # Single scalar risk score
        score = logit.sum()
        self.model.zero_grad()
        score.backward(retain_graph=False)

        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_up = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam_np = cam_up.squeeze().cpu().numpy()

        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def generate_integrated_gradients(self, input_tensor: torch.Tensor, steps: int = 25) -> np.ndarray:
        """Calculate axiomatic path-integral attributions for diabetes prediction."""
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        alphas = torch.linspace(0.0, 1.0, steps, device=input_tensor.device)
        scaled_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
        scaled_batch = torch.cat(scaled_inputs, dim=0).requires_grad_(True)

        logits = self.model(scaled_batch)
        score = logits.sum()

        self.model.zero_grad()
        score.backward()

        grads = scaled_batch.grad
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0, keepdim=True) / 2.0
        delta = input_tensor - baseline
        attributions = (delta * avg_grads).squeeze(0).detach().cpu().numpy()

        attr_map = np.abs(attributions).sum(axis=0)
        a_min, a_max = attr_map.min(), attr_map.max()
        if a_max > a_min:
            attr_map = (attr_map - a_min) / (a_max - a_min)
        else:
            attr_map = np.zeros_like(attr_map)

        return attr_map

    def save_explanation(
        self,
        original_img: Image.Image,
        input_tensor: torch.Tensor,
        patient_id: str,
        output_dir: Union[str, Path] = "outputs/explanations/diabetes",
        method: str = "gradcam"
    ) -> Path:
        if method.lower() == "ig":
            saliency = self.generate_integrated_gradients(input_tensor)
        else:
            saliency = self.generate_gradcam(input_tensor)

        blended = overlay_cam_on_image(original_img, saliency, alpha=0.45)
        out_folder = Path(output_dir) / self.model_arch / method
        out_folder.mkdir(parents=True, exist_ok=True)
        save_path = out_folder / f"{patient_id}.png"
        blended.save(save_path)
        return save_path

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
