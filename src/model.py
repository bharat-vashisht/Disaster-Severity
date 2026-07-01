"""
model.py
Defines the ResNet18-based classifier for disaster severity classification.
Uses transfer learning: pretrained ImageNet weights, only the final layer is retrained
(plus optionally the last block), which keeps CPU training time reasonable.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=4, freeze_backbone=True):
    """
    Builds a ResNet18 model for classification.

    Args:
        num_classes: number of output classes (4 for AIDERv2: Earthquake, Fire, Flood, Normal)
        freeze_backbone: if True, freezes all conv layers and only trains the final
                          fully connected layer. This is MUCH faster on CPU.
                          Set to False later if you have time/GPU and want higher accuracy.

    Returns:
        model: torch.nn.Module ready for training
    """
    # Load ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last conv block (layer4) so the model can adapt
        # mid-level features to aerial/disaster imagery — improves accuracy
        # without the full cost of training every layer.
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the final fully connected layer to match our number of classes.
    # This new layer is always trainable regardless of freeze_backbone.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_params(model):
    """Utility to check how many parameters will actually be updated during training."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Quick sanity check: python src/model.py
    model = get_model(num_classes=4, freeze_backbone=True)
    trainable, total = count_trainable_params(model)

    print(model.fc)
    print(f"Trainable parameters: {trainable:,} / {total:,} total")

    # Test a forward pass with dummy data matching our image size (224x224)
    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2 fake images
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [2, 4]