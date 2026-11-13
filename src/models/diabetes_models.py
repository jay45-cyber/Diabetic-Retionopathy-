"""Unified Model Factory for Track B: Binary Systemic Diabetes Risk Prediction.

Supported Architectures:
1. ResNet-50 Baseline ('resnet50', 'baseline_resnet50')
2. ConvNeXt ('convnext', 'convnext_tiny')
3. EfficientNetV2 ('efficientnet', 'efficientnet_v2')
4. Retinal Foundation Model Adapter ('retfound', 'foundation_vit')
5. Multimodal Vision + Clinical Covariates Network ('multimodal')
"""
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from src.models.convnext import RetinalConvNeXt
from src.models.retinal_foundation_model import RetinalFoundationModel
from src.models.multimodal_model import MultimodalRetinaModel


class ResNet50Diabetes(nn.Module):
    """ResNet-50 backbone fine-tuned for binary systemic diabetes risk screening."""

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)  # Single raw scalar logit output
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding vector [B, 2048]."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits.squeeze(-1)

    def get_last_conv_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


def build_diabetes_model(
    model_name: str = "resnet50",
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    num_clinical_features: int = 4
) -> nn.Module:
    """Instantiate and return the requested Track B diabetes risk prediction model."""
    name_clean = model_name.lower().strip()

    if "resnet" in name_clean:
        return ResNet50Diabetes(pretrained=pretrained, dropout_rate=dropout_rate)

    elif "convnext" in name_clean:
        return RetinalConvNeXt(pretrained=pretrained, dropout_rate=dropout_rate)

    elif "foundation" in name_clean or "retfound" in name_clean:
        return RetinalFoundationModel(pretrained_backbone=pretrained, dropout_rate=dropout_rate)

    elif "multimodal" in name_clean:
        return MultimodalRetinaModel(
            num_clinical_features=num_clinical_features,
            pretrained_vision=pretrained,
            dropout_rate=dropout_rate
        )

    else:
        # Default fallback to ResNet-50
        return ResNet50Diabetes(pretrained=pretrained, dropout_rate=dropout_rate)
