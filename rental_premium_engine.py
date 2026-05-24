"""Aqar Hub — Rental Premium Engine"""
from typing import Tuple

from feature_weights import compute_feature_score


def compute_premium(features: dict) -> Tuple[float, int, str]:
    score = compute_feature_score(features)

    if score <= 10:
        premium, tier = 0.0, "base"
    elif score <= 20:
        premium, tier = 0.05, "standard"
    elif score <= 35:
        premium, tier = 0.12, "enhanced"
    elif score <= 50:
        premium, tier = 0.20, "premium"
    else:
        premium, tier = 0.30, "luxury"

    return min(premium, 0.45), score, tier
