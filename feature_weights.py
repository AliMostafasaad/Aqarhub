"""Aqar Hub — Rental Feature Weights"""
from typing import Dict

FEATURE_WEIGHTS: Dict[str, int] = {
    "furnished": 20,
    "transport": 10,
    "ac": 8,
    "parking": 6,
    "pool": 5,
    "elevator": 5,
    "security": 5,
    "garden": 4,
    "gym": 4,
    "balcony": 3,
    "generator": 3,
    "cameras": 3,
    "storage": 2,
    "wifi": 2,
    "school": 2,
    "hospital": 2,
    "mosque": 1,
}


def compute_feature_score(features: Dict[str, bool]) -> int:
    return sum(FEATURE_WEIGHTS.get(k, 0) for k, v in features.items() if v)


def get_feature_breakdown(features: Dict[str, bool]) -> Dict[str, int]:
    return {k: FEATURE_WEIGHTS.get(k, 0) for k, v in features.items() if v}
