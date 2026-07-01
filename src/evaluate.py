"""
evaluate.py
Generates a full evaluation report on the test set:
  - Confusion matrix (saved as an image)
  - Per-class precision, recall, F1-score
  - Overall accuracy

Run from project root:
    python src/evaluate.py

Output: results/confusion_matrix.png, prints classification report to console
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders
from model import get_model

MODEL_PATH = "models/best_model.pth"
OUTPUT_DIR = "results"


def get_all_predictions(model, loader, device):
    """Runs the model on the full loader and collects all predictions + true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path):
    """Saves a heatmap visualization of the confusion matrix."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Disaster Severity Classification")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    _, _, test_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Test images: {len(test_loader.dataset)}")

    # ---- Load model ----
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})\n")

    # ---- Run predictions on full test set ----
    print("Running predictions on test set...")
    true_labels, predictions = get_all_predictions(model, test_loader, device)

    # ---- Overall accuracy ----
    accuracy = (true_labels == predictions).mean()
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%\n")

    # ---- Per-class metrics ----
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names, digits=4))

    # ---- Confusion matrix ----
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(cm)

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path)

    # ---- Save text report too, for your README/appendix ----
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, predictions, target_names=class_names, digits=4))
        f.write("\n\nConfusion Matrix (rows=true, cols=predicted):\n")
        f.write(str(cm))
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()