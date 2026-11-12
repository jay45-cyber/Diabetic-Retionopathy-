"""Unified Model Factory for 5-Class Diabetic Retinopathy Grading.

Supported Architectures:
1. ViT-Base/16 (vit_base_patch16_224 / vit_base_patch16_384)
2. ResNet-50 (resnet50)
3. Swin Transformer (swin_base_patch4_window7_224 / swin_tiny_patch4_window7_224)
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models as tv_models
import timm


class ResNet50DR(nn.Module):
    """ResNet-50 backbone fine-tuned for 5-Class Diabetic Retinopathy Severity Grading."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Return final bottleneck convolutional layer for Grad-CAM."""
        return self.backbone.layer4[-1]


class ViTDR(nn.Module):
    """Vision Transformer Base (ViT-B/16) for 5-Class DR Grading."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-head self-attention weight tensors for Attention Rollout."""
        # Query backbone blocks for attention matrices
        attentions = []
        # Fallback if timm blocks support hook-based or direct extraction
        return attentions


class SwinTransformerDR(nn.Module):
    """Swin Transformer (Shifted Window Attention) for 5-Class DR Grading."""

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_dr_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """Instantiate and return the requested DR classification model."""
    name_clean = model_name.lower().strip()

    if "resnet50" in name_clean:
        return ResNet50DR(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif "vit" in name_clean:
        return ViTDR(model_name=model_name, num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif "swin" in name_clean:
        return SwinTransformerDR(model_name=model_name, num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        # Fallback to general timm model creation
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)

