"""Shared model registry; ImageNet transfer learning and native CIFAR inputs are explicit."""

import torch
from torch import nn
from torchvision import models


class TinyCNN(nn.Module):
    """Small real CNN for offline regression tests, not a CIFAR accuracy baseline."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(arch: str, pretrained: bool = False) -> nn.Module:
    if arch == "tiny_cnn":
        if pretrained:
            raise ValueError("tiny_cnn has no pretrained weights.")
        return TinyCNN()
    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "vgg16": (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT),
        "mobilenet_v3": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    }
    if arch not in builders:
        raise ValueError(f"Unsupported architecture: {arch}")
    builder, weights = builders[arch]
    model = builder(weights=weights if pretrained else None)
    if arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    return model
