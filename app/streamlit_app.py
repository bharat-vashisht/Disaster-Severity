"""
streamlit_app.py  (v3 — with Llama 3.2 report generation)
Run from project root:
    streamlit run app/streamlit_app.py
"""

import os, sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import get_model
from severity import compute_severity_score
from uncertainty import mc_dropout_predict
from llm_report import generate_report, check_ollama_running

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

DISASTER_META = {
    "Earthquake": {"icon": "🏚️", "description": "Structural collapse detected. Prioritise search & rescue.", "action": "Deploy structural engineers and SAR teams immediately."},
    "Fire":       {"icon": "🔥", "description": "Active fire or burn scar detected in the area.",            "action": "Alert fire suppression units and evacuate downwind zones."},
    "Flood":      {"icon": "🌊", "description": "Flood inundation detected. Water levels may be rising.",    "action": "Issue flood warnings and mobilise water rescue assets."},
    "Normal":     {"icon": "✅", "description": "No disaster features detected in this image.",               "action": "No immediate action required. Continue monitoring."},
}


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #080C14; color: #E2E8F0; }
    .main { background: #080C14; }
    section[data-testid="stSidebar"] { background: #0D1321; border-right: 1px solid #1E293B; }
    .app-header { background: linear-gradient(135deg, #0D1321 0%, #111827 100%); border: 1px solid #1E293B; border-radius: 12px; padding: 28px 32px 20px; margin-bottom: 24px; position: relative; overflow: hidden; }
    .app-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #EF4444, #F97316, #EAB308, #22C55E); }
    .app-header h1 { font-family: 'Space Mono', monospace; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.5px; color: #F1F5F9; margin: 0 0 6px 0; }
    .app-header p { color: #94A3B8; font-size: 0.88rem; margin: 0; line-height: 1.6; }
    [data-testid="stFileUploader"] { background: #0D1321; border: 2px dashed #1E293B; border-radius: 10px; padding: 12px; }
    .card { background: #0D1321; border: 1px solid #1E293B; border-radius: 10px; padding: 20px 22px; margin-bottom: 16px; }
    .card-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #64748B; margin-bottom: 8px; }
    .card-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; line-height: 1; }
    .card-sub { font-size: 0.8rem; color: #94A3B8; margin-top: 4px; }
    .alert-box { border-radius: 10px; padding: 16px 20px; margin: 16px 0; border-left: 4px solid; }
    .alert-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
    .alert-body { font-size: 0.85rem; color: #CBD5E1; line-height: 1.6; }
    .section-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; border-bottom: 1px solid #1E293B; padding-bottom: 8px; margin: 24px 0 16px 0; }
    .report-box { background: #0D1321; border: 1px solid #1E293B; border-radius: 10px; padding: 24px 28px; margin-top: 8px; font-size: 0.88rem; line-height: 1.8; color: #CBD5E1; white-space: pre-wrap; }
    .report-box strong, .report-header { color: #F1F5F9; font-weight: 600; }
    .llm-badge { display: inline-flex; align-items: center; gap: 6px; background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 4px 12px; font-size: 0.75rem; color: #94A3B8; margin-bottom: 12px; }
    .unc-bar-wrap { background: #1E293B; border-radius: 99px; height: 8px; margin: 8px 0; overflow: hidden; }
    .unc-bar-fill { height: 100%; border-radius: 99px; }
    .model-tag { display: inline-block; background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 3px 10px; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #94A3B8; margin: 2px 3px; }
    h1, h2, h3 { color: #F1F5F9 !important; }
    .stButton>button { background: #1D4ED8; color: white; border: none; border-radius: 8px; font-weight: 600; padding: 8px 20px; }
    div[data-testid="metric-container"] { background: #0D1321; border: 1px solid #1E293B; border-radius: 10px; padding: 14px 16px; }
    [data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; font-size: 1.5rem !important; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    model       = get_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, device, checkpoint


def preprocess(pil_image):
    img    = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb    = np.array(img).astype(np.float32) / 255.0
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])(img).unsqueeze(0)
    return tensor, rgb


def run_gradcam(model, tensor, pred_idx, device):
    t = tensor.to(device)
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        g1 = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred_idx)])[0]
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        g2 = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred_idx)])[0]
    return g1, g2


def make_severity_gauge(score, color):
    fig, ax = plt.subplots(figsize=(5, 0.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(0, 100, height=0.5, color="#1E293B", zorder=1)
    ax.barh(0, score, height=0.5, color=color, zorder=2)
    ax.text(min(score + 2, 95), 0, f"{score}/100",
            va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def make_confidence_bars(class_names, probs):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor("#0D1321")
    ax.set_facecolor("#0D1321")
    bar_colors = ["#F97316" if i == np.argmax(probs) else "#3B82F6"
                  for i in range(len(class_names))]
    bars = ax.barh(class_names, [p * 100 for p in probs],
                   color=bar_colors, height=0.5)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="#94A3B8", fontsize=8)
    ax.set_xlim(0, 115)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.spines["left"].set_color("#1E293B")
    ax.set_xlabel("Confidence (%)", color="#64748B", fontsize=8)
    ax.tick_params(axis='x', colors='#475569')
    plt.tight_layout(pad=0.5)
    return fig


def format_report(report_text):
    """Bolds the section headers in the report for better readability."""
    import re
    formatted = re.sub(
        r'(1\. SITUATION SUMMARY|2\. RISK ASSESSMENT|3\. RECOMMENDED ACTIONS|4\. ANALYST NOTE)',
        r'**\1**', report_text
    )
    return formatted


def main():
    st.set_page_config(
        page_title="DisasterSight — Satellite Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown("""
    <div class="app-header">
        <h1>🛰️ DisasterSight</h1>
        <p>Explainable AI-powered disaster severity assessment from aerial and satellite imagery.
        Combines ResNet-18 classification, Grad-CAM visual explanations, MC Dropout uncertainty
        quantification, and Llama 3.2 natural language reporting.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, class_names, device, checkpoint = load_model()
    except FileNotFoundError:
        st.error("Model not found. Run `python src/train.py` first.")
        return

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔬 Model Info")
        st.markdown(
            '<span class="model-tag">ResNet-18</span>'
            '<span class="model-tag">Transfer Learning</span>'
            '<span class="model-tag">AIDERv2</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div style="margin-top:14px;font-size:0.82rem;color:#64748B">
            Epoch {checkpoint['epoch']} &nbsp;·&nbsp;
            Val Acc {checkpoint['val_acc']*100:.2f}% &nbsp;·&nbsp;
            Test Acc 97.10%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Options")
        use_mc      = st.toggle("MC Dropout uncertainty", value=True)
        n_passes    = st.slider("Dropout passes", 10, 50, 20, 5, disabled=not use_mc)
        show_pp     = st.toggle("Show Grad-CAM++ comparison", value=True)
        use_llm     = st.toggle("Generate Llama 3.2 report", value=True)

        # Ollama status indicator
        if use_llm:
            ollama_ok = check_ollama_running()
            status_color = "#22C55E" if ollama_ok else "#EF4444"
            status_text  = "Ollama running" if ollama_ok else "Ollama offline"
            st.markdown(
                f'<div style="font-size:0.78rem;color:{status_color};margin-top:6px">'
                f'● {status_text}</div>',
                unsafe_allow_html=True
            )
            if not ollama_ok:
                st.caption("Run `ollama serve` in a terminal to enable reports.")

        st.markdown("---")
        st.markdown("### 📋 Classes")
        for cls in class_names:
            meta = DISASTER_META.get(cls, {})
            st.markdown(f"{meta.get('icon','•')} **{cls}**")
            st.caption(meta.get('description',''))

    # ── Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a satellite or aerial image here",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
            <div style="font-size:3.5rem;margin-bottom:16px">🛰️</div>
            <div style="font-size:1.05rem;font-weight:500;color:#475569">
                Upload a satellite image to begin analysis
            </div>
            <div style="font-size:0.82rem;margin-top:8px;color:#334155">
                Supports aerial imagery · Earthquake · Fire · Flood · Normal terrain
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    pil_image        = Image.open(uploaded)
    tensor, rgb      = preprocess(pil_image)

    with st.spinner("Analysing image…"):
        t = tensor.to(device)

        with torch.no_grad():
            out   = model(t)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        if use_mc:
            mean_probs, std_probs, pred_idx, confidence, uncertainty = \
                mc_dropout_predict(model, tensor, n_passes=n_passes, device=device)
            probs = mean_probs
        else:
            pred_idx    = int(np.argmax(probs))
            confidence  = float(probs[pred_idx])
            uncertainty = None
            std_probs   = None

        predicted_class = class_names[pred_idx]
        sev  = compute_severity_score(predicted_class, confidence, uncertainty)
        meta = DISASTER_META.get(predicted_class, {})

        gcam, gcam_pp  = run_gradcam(model, tensor, pred_idx, device)
        overlay_cam    = show_cam_on_image(rgb, gcam,    use_rgb=True)
        overlay_pp     = show_cam_on_image(rgb, gcam_pp, use_rgb=True)

    # ── Metric cards ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    unc_pct   = round(uncertainty * 100, 1) if uncertainty is not None else None
    unc_color = "#22C55E" if unc_pct is not None and unc_pct < 5 \
                else "#EAB308" if unc_pct is not None and unc_pct < 15 else "#EF4444"

    with c1:
        st.markdown(f"""<div class="card">
            <div class="card-title">Detected Class</div>
            <div class="card-value" style="color:{sev['color']}">{meta.get('icon','')} {predicted_class}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card">
            <div class="card-title">Confidence</div>
            <div class="card-value" style="color:#F1F5F9">{confidence*100:.1f}%</div>
            <div class="card-sub">{'MC avg · ' + str(n_passes) + ' passes' if use_mc else 'Single pass'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <div class="card-title">Severity Score</div>
            <div class="card-value" style="color:{sev['color']}">{sev['score']}<span style="font-size:1rem;color:#64748B">/100</span></div>
            <div class="card-sub">{sev['tier']}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card">
            <div class="card-title">Uncertainty</div>
            <div class="card-value" style="color:{unc_color}">{f'{unc_pct}%' if unc_pct is not None else 'N/A'}</div>
            <div class="card-sub">{'Reliable' if unc_pct is not None and unc_pct < 5 else 'Verify manually' if unc_pct is not None and unc_pct < 15 else 'Elevated'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Alert ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="alert-box" style="background:{sev['color']}18;border-color:{sev['color']}">
        <div class="alert-title" style="color:{sev['color']}">{meta.get('icon','')} {sev['tier']} — {predicted_class}</div>
        <div class="alert-body">{meta.get('description','')} &nbsp;·&nbsp;
            <strong style="color:#E2E8F0">Action:</strong> {meta.get('action','')}
        </div>
        {"<div style='margin-top:8px;font-size:0.8rem;color:#F59E0B'>⚠️ " + sev['reliability_note'] + "</div>" if sev.get('reliability_note') else ""}
    </div>""", unsafe_allow_html=True)

    # ── Severity gauge ───────────────────────────────────────
    st.markdown('<div class="section-label">Severity gauge</div>', unsafe_allow_html=True)
    fig_gauge = make_severity_gauge(sev["score"], sev["color"])
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

    # ── Visual explanation ───────────────────────────────────
    st.markdown('<div class="section-label">Visual explanation (Grad-CAM)</div>',
                unsafe_allow_html=True)

    if show_pp:
        cols   = st.columns(3)
        imgs   = [rgb, overlay_cam, overlay_pp]
        labels = ["Original Image", "Grad-CAM", "Grad-CAM++"]
        caps   = ["Input to the model",
                  "Regions driving the prediction",
                  "Improved gradient aggregation"]
    else:
        cols   = st.columns(2)
        imgs   = [rgb, overlay_cam]
        labels = ["Original Image", "Grad-CAM"]
        caps   = ["Input to the model", "Regions driving the prediction"]

    for col, img, lbl, cap in zip(cols, imgs, labels, caps):
        with col:
            st.markdown(f"**{lbl}**")
            st.image(img, use_container_width=True, caption=cap)

    # ── Confidence + uncertainty breakdown ───────────────────
    st.markdown('<div class="section-label">Model output breakdown</div>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        st.markdown("**Class confidence distribution**")
        fig_bars = make_confidence_bars(class_names, probs)
        st.pyplot(fig_bars, use_container_width=True)
        plt.close()

    with col_r:
        if use_mc and std_probs is not None:
            st.markdown("**Per-class uncertainty**")
            for i, cls in enumerate(class_names):
                uv = float(std_probs[i]) * 100
                bc = "#22C55E" if uv < 3 else "#EAB308" if uv < 8 else "#EF4444"
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94A3B8;margin-bottom:3px">
                        <span>{cls}</span><span>{uv:.2f}%</span>
                    </div>
                    <div class="unc-bar-wrap">
                        <div class="unc-bar-fill" style="width:{min(uv*5,100)}%;background:{bc}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:0.85rem;padding:20px 0">Enable MC Dropout in the sidebar for uncertainty estimates.</div>',
                        unsafe_allow_html=True)

    # ── LLM Report ───────────────────────────────────────────
    st.markdown('<div class="section-label">AI Disaster Assessment Report</div>',
                unsafe_allow_html=True)

    if use_llm:
        st.markdown(
            '<div class="llm-badge">🦙 Generated by Llama 3.2 · Local inference · Ollama</div>',
            unsafe_allow_html=True
        )

        prediction_data = {
            "predicted_class": predicted_class,
            "confidence":      confidence * 100,
            "severity_score":  sev["score"],
            "severity_tier":   sev["tier"],
            "uncertainty":     unc_pct,
            "class_probs":     {class_names[i]: float(probs[i]) * 100
                                 for i in range(len(class_names))},
        }

        with st.spinner("Llama 3.2 is writing the assessment report…"):
            report = generate_report(prediction_data)

        st.markdown(
            f'<div class="report-box">{format_report(report)}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color:#475569;font-size:0.85rem;padding:16px 0">'
            'Enable "Generate Llama 3.2 report" in the sidebar to see the AI assessment.</div>',
            unsafe_allow_html=True
        )

    # ── Footer ───────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:48px;padding-top:20px;border-top:1px solid #1E293B;
                text-align:center;color:#334155;font-size:0.75rem">
        DisasterSight &nbsp;·&nbsp; ResNet-18 + Grad-CAM &nbsp;·&nbsp;
        Llama 3.2 via Ollama &nbsp;·&nbsp; AIDERv2 Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()