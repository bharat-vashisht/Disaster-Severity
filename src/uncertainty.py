"""
uncertainty.py  (v2)
Monte Carlo Dropout for uncertainty quantification.

ResNet18 has no built-in Dropout layers, so we patch one in at runtime
by wrapping the FC layer. This requires no retraining — the dropout only
fires at inference time during the MC sampling passes, then gets removed.

Usage:
    from uncertainty import mc_dropout_predict
    mean_probs, std_probs, pred_idx, confidence, uncertainty = \
        mc_dropout_predict(model, tensor, n_passes=30, device=device)
"""

import torch
import torch.nn as nn
import numpy as np


def mc_dropout_predict(model, input_tensor, n_passes=30, device="cpu"):
    """
    Runs N stochastic forward passes by temporarily patching a Dropout
    layer onto the model's FC head (works for any ResNet variant).

    Args:
        model:        trained PyTorch ResNet model
        input_tensor: [1, 3, H, W] preprocessed image tensor (CPU)
        n_passes:     MC samples (20-30 is a good CPU-friendly range)
        device:       torch device

    Returns:
        mean_probs:  np.array [num_classes]  — averaged softmax
        std_probs:   np.array [num_classes]  — std dev (uncertainty)
        pred_idx:    int                     — argmax of mean_probs
        confidence:  float                  — mean_probs[pred_idx]
        uncertainty: float                  — std_probs[pred_idx]
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # ── Patch: wrap the existing FC with Dropout ──────────────────────
    original_fc  = model.fc
    patched_fc   = nn.Sequential(nn.Dropout(p=0.3), original_fc)
    model.fc     = patched_fc

    # Enable the dropout layer for stochastic inference
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # ── MC sampling ──────────────────────────────────────────────────
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(input_tensor)
            probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    # ── Restore original FC (important — don't leave patched) ─────────
    model.fc = original_fc
    model.eval()

    # ── Aggregate ─────────────────────────────────────────────────────
    all_probs   = np.array(all_probs)          # [n_passes, num_classes]
    mean_probs  = all_probs.mean(axis=0)
    std_probs   = all_probs.std(axis=0)

    pred_idx    = int(np.argmax(mean_probs))
    confidence  = float(mean_probs[pred_idx])
    uncertainty = float(std_probs[pred_idx])

    return mean_probs, std_probs, pred_idx, confidence, uncertainty


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from model import get_model

    device     = torch.device("cpu")
    checkpoint = torch.load("models/best_model.pth", map_location=device)
    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test on a dummy image
    dummy = torch.randn(1, 3, 224, 224)
    mean_p, std_p, pred, conf, unc = mc_dropout_predict(
        model, dummy, n_passes=30, device=device
    )

    print("Mean probs:",
          {class_names[i]: f"{mean_p[i]:.4f}" for i in range(len(class_names))})
    print("Std  probs:",
          {class_names[i]: f"{std_p[i]:.4f}" for i in range(len(class_names))})
    print(f"Predicted:  {class_names[pred]}")
    print(f"Confidence: {conf:.4f}")
    print(f"Uncertainty:{unc:.4f}")
    print()
    print("Uncertainty is non-zero:", any(s > 0 for s in std_p))