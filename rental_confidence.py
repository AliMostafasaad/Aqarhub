"""Aqar Hub — Rental Confidence System"""
from typing import Dict, Any, Optional


def compute_rental_confidence(features: Dict[str, bool], feature_sources: Dict[str, str],
                              area: str, property_type: str, description: str,
                              bundle: Optional[dict] = None) -> tuple[str, int]:
    score = 50

    for src in feature_sources.values():
        if src == "checkbox":
            score += 5
        elif src == "nlp":
            score += 2

    score = min(score, 80)

    area_l = str(area).lower()
    if area_l in ("unknown", "", "na", "n/a"):
        score -= 15
    elif any(x in area_l for x in ["new cairo", "sheikh zayed", "6th october",
                                     "nasr city", "maadi", "zamalek", "heliopolis"]):
        score += 10
    else:
        score += 5

    desc_len = len(str(description).strip())
    if desc_len < 20:
        score -= 10
    elif desc_len > 100:
        score += 5

    if bundle:
        rarity = bundle.get("rarity_map", {}).get((area, property_type), 0.0)
        if rarity > 0.5:
            score -= 15
        elif rarity > 0.3:
            score -= 8

    score = max(20, min(95, score))

    if score >= 75:
        return "HIGH", score
    elif score >= 50:
        return "MEDIUM", score
    return "LOW", score
