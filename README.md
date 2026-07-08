# 🛰️ DisasterSight

> **Explainable AI-powered disaster severity assessment from satellite and aerial imagery**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.10%25-brightgreen?style=flat)

---

## 📌 Overview

DisasterSight is an end-to-end explainable AI system that:

1. **Classifies** aerial/satellite images into 4 disaster categories (Earthquake, Fire, Flood, Normal)
2. **Scores** severity on a 0–100 scale using model confidence and domain knowledge
3. **Explains** predictions visually using Grad-CAM and Grad-CAM++ heatmaps
4. **Quantifies** prediction uncertainty using Monte Carlo Dropout
5. **Reports** a professional disaster assessment in natural language via **Llama 3.2** (local LLM)

Most disaster AI is a black box. DisasterSight shows emergency responders **what** was detected, **how severe** it is, **where** in the image the evidence is, and **how confident** the model is — making every prediction actionable and auditable.

---

## 🎯 Results

| Metric | Value |
|---|---|
| Test Accuracy | **97.10%** |
| Best Epoch | 4 / 8 |
| Training Time | ~152 min (CPU only) |
| Parameters Trained | 8.4M / 11.2M total |

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 🏚️ Earthquake | 97.76% | 91.21% | 94.37% |
| 🔥 Fire | 99.08% | 98.62% | **98.85%** |
| 🌊 Flood | 97.43% | 98.01% | 97.72% |
| ✅ Normal | 94.72% | 97.69% | 96.18% |

---

## 🏗️ Architecture

```
Satellite Image
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ dataset.py  │───▶│   model.py   │───▶│ explain.py │
│ Preprocess  │    │  ResNet-18   │    │  Grad-CAM  │
│  Augment    │    │ (Transfer    │    │ Grad-CAM++ │
└─────────────┘    │  Learning)   │    └─────┬──────┘
                   └──────────────┘          │
                                             ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ severity.py  │◀───│uncertainty.py│
                   │  0-100 Score │    │ MC Dropout  │
                   └──────┬───────┘    └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │llm_report.py │
                   │  Llama 3.2  │
                   │ via Ollama  │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Streamlit    │
                   │   Web App   │
                   └──────────────┘
```

---

## 📁 Project Structure

```
disaster-severity-xai/
├── app/
│   └── streamlit_app.py       # Web dashboard
├── src/
│   ├── dataset.py             # Data loading + augmentation
│   ├── model.py               # ResNet-18 classifier
│   ├── model_mobilenet.py     # MobileNetV2 (comparison)
│   ├── train.py               # Training loop
│   ├── train_mobilenet.py     # MobileNetV2 training
│   ├── evaluate.py            # Confusion matrix + metrics
│   ├── explain.py             # Grad-CAM + Grad-CAM++
│   ├── uncertainty.py         # MC Dropout
│   ├── severity.py            # Severity scoring engine
│   └── llm_report.py         # Llama 3.2 report generation
├── models/                    # Saved checkpoints (gitignored)
├── results/
│   ├── confusion_matrix.png
│   ├── evaluation_report.txt
│   └── gradcam/               # Heatmap visualizations
├── notebooks/                 # EDA and experiments
├── data/                      # AIDERv2 dataset (gitignored)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/disaster-severity-xai.git
cd disaster-severity-xai
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install grad-cam streamlit pillow matplotlib seaborn scikit-learn tqdm
```

### 4. Download the dataset

Download **AIDERv2** from [Zenodo](https://zenodo.org/records/10891054) and extract into:

```
data/
├── Train/   → Earthquake, Fire, Flood, Normal
├── Val/     → Earthquake, Fire, Flood, Normal
└── Test/    → Earthquake, Fire, Flood, Normal
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Generate Grad-CAM explanations

```bash
python src/explain.py
```

### 7. Run the web app

```bash
# Optional: start Ollama for LLM reports
ollama serve   # in a separate terminal

streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Key Technologies

| Component | Technology |
|---|---|
| Deep Learning | PyTorch + torchvision |
| Model | ResNet-18 (transfer learning) |
| Explainability | Grad-CAM, Grad-CAM++ (`pytorch-grad-cam`) |
| Uncertainty | Monte Carlo Dropout |
| Severity Scoring | Custom domain-weighted scoring engine |
| LLM Report | Llama 3.2 3B via Ollama (local inference) |
| Web App | Streamlit + custom CSS (dark theme) |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Dataset | AIDERv2 — 16,723 aerial images, 4 classes |

---

## 🔬 Explainability

DisasterSight uses two complementary XAI methods:

- **Grad-CAM** — Highlights the single strongest activation region in the last conv layer
- **Grad-CAM++** — Improved gradient aggregation; better for images with multiple damage zones

Both are shown side-by-side in the web app, giving users two independent visual explanations for every prediction.

---

## 🎲 Uncertainty Quantification

Using **Monte Carlo Dropout** (30 forward passes at inference time):

- Runs the model 30 times with dropout noise enabled
- Measures variance (std-dev) across predictions
- Low std-dev → reliable prediction → act on it
- High std-dev → uncertain → verify manually first

This is a Bayesian deep learning technique that turns a point prediction into a distribution.

---

## 🦙 Llama 3.2 Integration

After classification, structured outputs are sent to Llama 3.2 (running locally via Ollama) which generates a 4-section professional disaster assessment report:

1. **Situation Summary** — What was detected and overall severity
2. **Risk Assessment** — Specific risks and affected populations
3. **Recommended Actions** — Actionable steps for response teams
4. **Analyst Note** — Confidence/uncertainty interpretation

Everything runs **locally** — no internet, no API keys, no cloud.

---

## ⚠️ Limitations

- Trained on aerial imagery only — won't generalise to ground-level photos
- Earthquake has lowest recall (91.2%) — collapsed structures can resemble normal urban terrain
- Severity base weights are manually set, not learned from disaster response data
- MC Dropout is applied as a runtime patch (ResNet-18 has no built-in dropout layers)

---

## 🔭 Future Work

- GPS integration — plot predictions on a live map
- Sentinel-2 multispectral data (13 bands) for richer feature extraction
- REST API deployment for integration with drone/satellite systems
- Before/after temporal analysis to measure damage progression
- Fine-tune Llama 3.2 on real disaster response reports

---

## 📊 Dataset

**AIDERv2** (Aerial Image Dataset for Emergency Response)
- 16,723 labelled aerial images
- 4 classes: Earthquake, Fire, Flood, Normal
- Pre-split: Train (13,399) / Val (1,670) / Test (1,654)
- Source: [Zenodo DOI: 10891054](https://zenodo.org/records/10891054)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a college project demonstrating explainable AI for disaster response.*