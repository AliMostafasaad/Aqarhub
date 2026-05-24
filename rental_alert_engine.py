"""Aqar Hub — Rental Alert Engine"""
from typing import Dict, Any


def make_rental_decision(user_rent: float, expected_rent: float,
                         confidence: str, feature_score: int = 0) -> Dict[str, Any]:
    ratio = user_rent / expected_rent if expected_rent > 0 else 1.0

    under_t, over_t = 0.80, 1.25

    if confidence == "LOW":
        under_t, over_t = 0.70, 1.35
    elif confidence == "HIGH":
        under_t, over_t = 0.85, 1.20

    if ratio < 0.50:
        alert, confidence = "UNDERPRICED", "LOW"
    elif ratio > 2.00:
        alert, confidence = "OVERPRICED", "LOW"
    elif ratio < under_t:
        alert = "UNDERPRICED"
    elif ratio > over_t:
        alert = "OVERPRICED"
    else:
        alert = "FAIR"

    return {
        "alert": alert,
        "ratio": float(ratio),
        "confidence": confidence,
        "threshold_lower": float(under_t),
        "threshold_upper": float(over_t),
        "feature_score": feature_score,
    }
