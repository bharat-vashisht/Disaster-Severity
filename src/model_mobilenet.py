"""
model_mobilenet.py
MobileNetV2-based classifier for disaster severity assessment.
Used as the second model in the architecture comparison against ResNet18.

MobileNetV2 is ~3x smaller and faster than ResNet18 at the cost
of a small accuracy trade-off — ideal for deployment on edge devices
(drones, field tablets) where ResNet18 may be too heavy.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_mobilenet(num_classes=4, freeze_backbone=True):
    """
    Builds a MobileNetV2 classifier.

    Args:
        num_classes:     number of output classes (4 for AIDERv2)
        freeze_backbone: freeze all layers except the final classifier.
                         Keeps CPU training time reasonable.

    Returns:
        model: torch.nn.Module ready for training
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # MobileNetV2's classifier is model.classifier[1] (a Linear layer)
    # last_channel = 1280 for the default MobileNetV2
    in_features = model.last_channel
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = get_mobilenet(num_classes=4, freeze_backbone=True)
    trainable, total = count_trainable_params(model)
    print(f"MobileNetV2 — Trainable: {trainable:,} / {total:,} total")

    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print("Output shape:", output.shape)   # [2, 4]
