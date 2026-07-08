"""
severity.py
Converts a raw model prediction (class + confidence) into a calibrated
disaster severity score (0-100) with an interpretable tier label.

Why this exists: a classifier alone tells you WHAT was detected.
A severity assessment tool needs to tell you HOW URGENT it is.
This module encodes domain knowledge (base severity per disaster type)
combined with model confidence to produce a single actionable score.
"""

# ---- Base severity weight per class ----
# These represent the inherent risk-to-life/property of each category,
# independent of model confidence. Calibrated against typical disaster
# response triage practices (not from training data — this is domain knowledge
# layered on top of the classifier output).
BASE_SEVERITY = {
    "Earthquake": 85,   # structural collapse risk, often most life-threatening
    "Flood": 65,        # significant but more localized / slower-evolving
    "Fire": 75,          # fast-spreading, high property + life risk
    "Normal": 0,         # no disaster detected
}

SEVERITY_TIERS = [
    (0, 15,  "No Disaster",      "#2ecc71"),
    (15, 40, "Low Severity",     "#f1c40f"),
    (40, 65, "Moderate Severity", "#d87925"),
    (65, 85, "High Severity",    "#e74c3c"),
    (85, 101, "Critical Severity", "#8e0000"),
]


def get_severity_tier(score):
    """Maps a 0-100 score to a (label, color) tier."""
    for low, high, label, color in SEVERITY_TIERS:
        if low <= score < high:
            return label, color
    return "Unknown", "#95a5a6"


def compute_severity_score(predicted_class, confidence, uncertainty=None):
    """
    Computes a 0-100 severity score.

    Args:
        predicted_class: str, e.g. "Fire"
        confidence: float in [0, 1], model's softmax confidence for predicted_class
        uncertainty: optional float in [0, 1], e.g. std-dev from MC Dropout.
                      If provided, high uncertainty slightly REDUCES the
                      reported severity confidence (we don't want to cry wolf
                      on an uncertain prediction) and this should be surfaced
                      to the user as "score may be unreliable".

    Returns:
        dict with score, tier label, tier color, and explanation breakdown
    """
    base = BASE_SEVERITY.get(predicted_class, 50)

    # Severity score = base severity scaled by how confident the model is.
    # A low-confidence "Fire" prediction shouldn't scream "Critical" —
    # confidence acts as a multiplier/dampener on the base severity.
    raw_score = base * confidence

    # If uncertainty estimate is available, penalize the score slightly
    # when uncertainty is high (the model itself isn't sure).
    reliability_note = None
    if uncertainty is not None:
        # uncertainty in [0,1], where higher = less reliable
        penalty = uncertainty * 15  # max 15-point penalty
        raw_score = max(0, raw_score - penalty)
        if uncertainty > 0.15:
            reliability_note = (
                "Model uncertainty is elevated for this prediction — "
                "treat this score as indicative, not definitive."
            )

    score = round(min(100, max(0, raw_score)), 1)
    tier_label, tier_color = get_severity_tier(score)

    return {
        "score": score,
        "tier": tier_label,
        "color": tier_color,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 1),
        "base_severity": base,
        "reliability_note": reliability_note,
    }


if __name__ == "__main__":
    # Quick sanity checks: python src/severity.py
    test_cases = [
        ("Earthquake", 0.99),
        ("Fire", 0.74),
        ("Flood", 0.55),
        ("Normal", 0.98),
        ("Fire", 0.40),
    ]

    for cls, conf in test_cases:
        result = compute_severity_score(cls, conf)
        print(f"{cls:12s} conf={conf:.2f} -> score={result['score']:5.1f} "
              f"tier={result['tier']}")
