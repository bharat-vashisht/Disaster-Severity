"""
streamlit_app.py
Interactive web app for the Explainable Disaster Severity Assessment project.

User uploads a satellite/aerial image -> model predicts the disaster class ->
Grad-CAM heatmap shows WHY the model made that prediction.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Allow importing from src/ regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import get_model

# ---- Config ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Map classes to a severity level + color, for the "severity assessment" framing
SEVERITY_MAP = {
    "Normal":     {"severity": "No Disaster",  "color": "#2ecc71"},
    "Fire":       {"severity": "High Severity", "color": "#e74c3c"},
    "Flood":      {"severity": "High Severity", "color": "#3498db"},
    "Earthquake": {"severity": "High Severity", "color": "#e67e22"},
}


@st.cache_resource
def load_model():
    """Loads the trained model once and caches it across reruns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(pil_image):
    """Resizes + normalizes the uploaded image, returns (input_tensor, rgb_array)."""
    img_resized = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_image = np.array(img_resized).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    input_tensor = transform(img_resized).unsqueeze(0)

    return input_tensor, rgb_image


def predict_and_explain(model, input_tensor, rgb_image, device):
    """Runs prediction + Grad-CAM, returns (predicted_idx, confidence, all_probs, heatmap)."""
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, dim=0)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(predicted_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return predicted_idx, confidence, probs.cpu().numpy(), heatmap


def main():
    st.set_page_config(page_title="Disaster Severity Assessment", page_icon="🛰️", layout="wide")

    st.title("🛰️ Explainable Disaster Severity Assessment")
    st.markdown(
        "Upload a satellite or aerial image. The model classifies the disaster type "
        "and **Grad-CAM** highlights the exact regions that drove its decision — "
        "making the prediction explainable rather than a black box."
    )

    # Load model (cached, so this only runs once)
    try:
        model, class_names, device = load_model()
    except FileNotFoundError:
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Make sure you've run `python src/train.py` first to generate it."
        )
        return

    st.sidebar.header("About")
    st.sidebar.markdown(
        "**Classes:** " + ", ".join(class_names) + "\n\n"
        "**Model:** ResNet18 (transfer learning)\n\n"
        "**Explainability:** Grad-CAM\n\n"
        "**Dataset:** AIDERv2 (Aerial Image Dataset for Emergency Response)"
    )

    uploaded_file = st.file_uploader(
        "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        input_tensor, rgb_image = preprocess_image(pil_image)

        with st.spinner("Analyzing image..."):
            pred_idx, confidence, all_probs, heatmap = predict_and_explain(
                model, input_tensor, rgb_image, device
            )

        predicted_class = class_names[pred_idx]
        severity_info = SEVERITY_MAP.get(predicted_class, {"severity": "Unknown", "color": "#95a5a6"})

        # ---- Result banner ----
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Class", predicted_class)
        col_b.metric("Confidence", f"{confidence*100:.1f}%")
        col_c.metric("Severity Level", severity_info["severity"])

        # ---- Side-by-side images ----
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(rgb_image, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Explanation")
            st.image(heatmap, use_container_width=True)
            st.caption(
                "Red/yellow regions show where the model focused most heavily "
                "to make this prediction."
            )

        # ---- Full probability breakdown ----
        st.markdown("---")
        st.subheader("Confidence Across All Classes")
        prob_dict = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
        st.bar_chart(prob_dict)

    else:
        st.info("Upload an image above to get started. Try a few from your `data/Test` folder.")


if __name__ == "__main__":
    main()