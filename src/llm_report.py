"""
llm_report.py
Generates a professional disaster assessment report using Llama 3.2
running locally via Ollama.

Takes structured prediction data (class, confidence, severity, uncertainty)
and produces a natural-language report formatted like an emergency analyst
would write — suitable for display in the app and inclusion in presentations.

Usage:
    from llm_report import generate_report
    report = generate_report(prediction_data)
"""

import requests
import json

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3.2"
TIMEOUT_SEC = 120   # Llama 3.2 3B should respond well within this on CPU


def generate_report(prediction_data: dict) -> str:
    """
    Generates a structured disaster assessment report.

    Args:
        prediction_data: dict with keys:
            - predicted_class: str  (e.g. "Flood")
            - confidence:      float (0-100, percentage)
            - severity_score:  float (0-100)
            - severity_tier:   str  (e.g. "High Severity")
            - uncertainty:     float or None (0-100, percentage)
            - class_probs:     dict {class_name: probability_pct}

    Returns:
        str: formatted report text, or error message if Ollama unreachable
    """

    cls        = prediction_data["predicted_class"]
    conf       = prediction_data["confidence"]
    score      = prediction_data["severity_score"]
    tier       = prediction_data["severity_tier"]
    unc        = prediction_data.get("uncertainty")
    probs      = prediction_data.get("class_probs", {})

    unc_line = (
        f"- Prediction uncertainty (MC Dropout): {unc:.1f}% "
        f"({'low — reliable' if unc < 5 else 'moderate — verify' if unc < 15 else 'elevated — treat with caution'})"
        if unc is not None else "- Uncertainty estimation: not available"
    )

    prob_lines = "\n".join(
        f"  · {k}: {v:.1f}%" for k, v in probs.items()
    )

    prompt = f"""You are a professional disaster response analyst writing a formal satellite imagery assessment report.

SATELLITE IMAGE ANALYSIS DATA:
- Detected disaster type: {cls}
- Model confidence: {conf:.1f}%
- Severity score: {score}/100
- Severity tier: {tier}
{unc_line}
- Full class probability breakdown:
{prob_lines}

Write a concise, professional disaster assessment report with exactly these four sections:

1. SITUATION SUMMARY (2-3 sentences): What was detected, confidence level, and overall severity.

2. RISK ASSESSMENT (3-4 sentences): Specific risks associated with this disaster type, what areas/populations may be affected, and how the severity score informs response prioritisation.

3. RECOMMENDED ACTIONS (3-5 bullet points): Specific, actionable steps for emergency response teams based on the detected disaster type and severity tier.

4. ANALYST NOTE (1-2 sentences): Comment on model confidence and uncertainty — whether the prediction is reliable enough to act on, or whether ground-truth verification is advised.

Use formal language appropriate for an emergency operations centre. Do not include disclaimers about being an AI. Write as if this is a real field report.
"""

    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # low temp = more consistent, factual output
            "num_predict": 400,    # enough for a full report without being verbose
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        response.raise_for_status()
        data   = response.json()
        report = data.get("response", "").strip()
        return report

    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Ollama is not running. Start it with `ollama serve` "
            "in a separate terminal, then reload the app."
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️ Llama 3.2 took too long to respond. "
            "Try reducing the image or restarting Ollama."
        )
    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"


def check_ollama_running() -> bool:
    """Quick health check — returns True if Ollama API is reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    # Quick test: python src/llm_report.py
    print("Checking Ollama...")
    if not check_ollama_running():
        print("Ollama is not running. Start with: ollama serve")
    else:
        print("Ollama is running. Generating test report...\n")
        test_data = {
            "predicted_class": "Flood",
            "confidence":      94.5,
            "severity_score":  61.4,
            "severity_tier":   "Moderate Severity",
            "uncertainty":     3.2,
            "class_probs": {
                "Earthquake": 1.2,
                "Fire":        2.1,
                "Flood":      94.5,
                "Normal":      2.2,
            }
        }
        report = generate_report(test_data)
        print(report)
