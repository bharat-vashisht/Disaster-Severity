"""
explain.py  (upgraded)
Generates dual explainability visualizations:
  - Grad-CAM  (original)
  - Grad-CAM++ (improved gradient aggregation, better for multiple instances)

Side-by-side output: Original | Grad-CAM | Grad-CAM++
Saves to results/gradcam/

Run from project root:
    python src/explain.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_model
from severity import compute_severity_score

# ---- Config ----
MODEL_PATH    = "models/best_model.pth"
TEST_DIR      = "data/Test"
OUTPUT_DIR    = "results/gradcam"
NUM_PER_CLASS = 3
IMG_SIZE      = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_model(device):
    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    model       = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model — epoch {checkpoint['epoch']}, val_acc={checkpoint['val_acc']:.4f}")
    print(f"Classes: {class_names}\n")
    return model, class_names


def preprocess(image_path):
    img      = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb      = np.array(img).astype(np.float32) / 255.0
    tensor   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])(img).unsqueeze(0)
    return tensor, rgb


def run_cam(cam_obj, input_tensor, pred_idx):
    targets      = [ClassifierOutputTarget(pred_idx)]
    grayscale    = cam_obj(input_tensor=input_tensor, targets=targets)[0]
    return grayscale


def save_comparison(rgb, cam1, cam2, title, save_path, sev_result):
    """Saves a 3-panel figure: Original | Grad-CAM | Grad-CAM++."""
    vis1 = show_cam_on_image(rgb, cam1, use_rgb=True)
    vis2 = show_cam_on_image(rgb, cam2, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0f1117")

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image", color="white", fontsize=11, pad=8)

    axes[1].imshow(vis1)
    axes[1].set_title("Grad-CAM", color="#f39c12", fontsize=11, pad=8)

    axes[2].imshow(vis2)
    axes[2].set_title("Grad-CAM++", color="#3498db", fontsize=11, pad=8)

    tier_color = sev_result["color"]
    fig.suptitle(
        f"{title}  |  Severity Score: {sev_result['score']}/100  "
        f"({sev_result['tier']})  |  Confidence: {sev_result['confidence']}%",
        color=tier_color, fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, class_names = load_model(device)
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam, \
         GradCAMPlusPlus(model=model, target_layers=target_layers) as cam_pp:

        for class_name in class_names:
            class_dir = os.path.join(TEST_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue

            images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))][:NUM_PER_CLASS]

            print(f"Class '{class_name}' — {len(images)} images")

            for img_file in images:
                path = os.path.join(class_dir, img_file)
                tensor, rgb = preprocess(path)
                tensor = tensor.to(device)

                # Predict
                with torch.no_grad():
                    out   = model(tensor)
                    probs = torch.softmax(out, dim=1)[0]
                    conf, pred_idx = torch.max(probs, 0)
                    pred_idx = pred_idx.item()
                    conf     = conf.item()

                predicted = class_names[pred_idx]
                correct   = "✓" if predicted == class_name else "✗"

                # Both CAMs
                g1 = run_cam(cam,    tensor, pred_idx)
                g2 = run_cam(cam_pp, tensor, pred_idx)

                # Severity
                sev = compute_severity_score(predicted, conf)

                title     = f"True: {class_name} | Pred: {predicted} {correct}"
                out_name  = f"{class_name}_{img_file.split('.')[0]}_comparison.png"
                out_path  = os.path.join(OUTPUT_DIR, out_name)
                save_comparison(rgb, g1, g2, title, out_path, sev)

                print(f"  {img_file} -> {predicted} ({conf*100:.1f}%) "
                      f"[{correct}] score={sev['score']} {sev['tier']}")

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()