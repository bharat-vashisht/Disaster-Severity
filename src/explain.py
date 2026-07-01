"""
explain.py
Generates Grad-CAM heatmaps for the trained disaster severity model.
This is the "explainable AI" part of the project: it shows WHICH regions
of a satellite/aerial image the model focused on to make its prediction.

Run from project root:
    python src/explain.py

Output: saves heatmap overlay images to results/gradcam/
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_model

# ---- Config ----
MODEL_PATH = "models/best_model.pth"
TEST_DIR = "data/Test"
OUTPUT_DIR = "results/gradcam"
NUM_IMAGES_PER_CLASS = 3   # how many example images to explain per class
IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model_and_classes(device):
    """Loads the trained model + class names from the saved checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model (trained to epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']:.4f})")
    print(f"Classes: {class_names}")

    return model, class_names


def preprocess_image(image_path):
    """
    Loads an image and returns:
      - input_tensor: normalized tensor ready for the model, shape [1, 3, 224, 224]
      - rgb_image: float32 numpy array in [0,1] range, shape [224, 224, 3]
                    (used as the background for the heatmap overlay)
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    rgb_image = np.array(img_resized).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    input_tensor = transform(img_resized).unsqueeze(0)  # add batch dimension

    return input_tensor, rgb_image


def generate_gradcam_for_image(model, cam, input_tensor, rgb_image, true_label_idx=None):
    """
    Runs Grad-CAM on a single image.
    Returns: visualization (heatmap overlay), predicted_class_idx, confidence
    """
    # Get model prediction first
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

    # Generate Grad-CAM for the PREDICTED class (explains "why the model said this")
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # first (only) image in batch

    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return visualization, predicted_idx, confidence


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, class_names = load_model_and_classes(device)

    # Target the last conv block of ResNet18 — standard choice for Grad-CAM
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for class_name in class_names:
            class_dir = os.path.join(TEST_DIR, class_name)
            if not os.path.isdir(class_dir):
                print(f"  Skipping missing folder: {class_dir}")
                continue

            image_files = [f for f in os.listdir(class_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            image_files = image_files[:NUM_IMAGES_PER_CLASS]

            print(f"\nProcessing class '{class_name}' ({len(image_files)} images)...")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                input_tensor, rgb_image = preprocess_image(img_path)
                input_tensor = input_tensor.to(device)

                visualization, pred_idx, confidence = generate_gradcam_for_image(
                    model, cam, input_tensor, rgb_image
                )

                predicted_class = class_names[pred_idx]
                correct = "CORRECT" if predicted_class == class_name else "WRONG"

                # ---- Save side-by-side: original | heatmap overlay ----
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(rgb_image)
                axes[0].set_title(f"Original\nTrue: {class_name}")
                axes[0].axis("off")

                axes[1].imshow(visualization)
                axes[1].set_title(f"Grad-CAM\nPred: {predicted_class} "
                                   f"({confidence*100:.1f}%) [{correct}]")
                axes[1].axis("off")

                plt.tight_layout()

                out_name = f"{class_name}_{img_file.split('.')[0]}_gradcam.png"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                print(f"  {img_file} -> predicted: {predicted_class} "
                      f"({confidence*100:.1f}%) [{correct}] -> saved {out_name}")

    print(f"\nAll Grad-CAM visualizations saved to: {OUTPUT_DIR}/")
    print("Use these images directly in your presentation slides.")


if __name__ == "__main__":
    main()