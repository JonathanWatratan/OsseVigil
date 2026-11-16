from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def build_classifier(num_classes: int, architecture: str = "resnet18", pretrained: bool = True):
    from torchvision.models import (
        EfficientNet_B0_Weights,
        ResNet18_Weights,
        efficientnet_b0,
        resnet18,
    )

    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def load_checkpoint(path: str | Path, device: str):
    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = build_classifier(
        num_classes=len(class_to_idx),
        architecture=checkpoint.get("architecture", "resnet18"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
