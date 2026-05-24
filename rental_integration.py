"""Aqar Hub — Rental Integration Layer"""
from typing import Dict, Any

from feature_extractor import extract_features, get_feature_sources
from feature_weights import compute_feature_score
from rental_estimator import estimate_rent
from rental_premium_engine import compute_premium
from rental_alert_engine import make_rental_decision
from rental_reason_builder import build_rental_reason
from rental_confidence import compute_rental_confidence


def analyze_rental(user_rent: float, df_proc, X, bundle: dict,
                   checkbox_features: Dict[str, Any], desc_text: str) -> Dict[str, Any]:
    model = bundle["model"]
    lo_price, med_price, hi_price = model.predict_price_interval(X)
    sale_price = float(med_price[0])

    features = extract_features(desc_text, checkboxes=checkbox_features)
    sources = get_feature_sources(desc_text, checkboxes=checkbox_features)

    premium_pct, feat_score, tier = compute_premium(features)

    area = str(df_proc.iloc[0].get("city", "Unknown"))
    compound = str(df_proc.iloc[0].get("compound", ""))
    prop_type = str(df_proc.iloc[0].get("type_clean", "Apartment"))

    rent_est = estimate_rent(sale_price, prop_type, area, compound, desc_text)

    adj_min = rent_est["min_rent"] * (1 + premium_pct)
    adj_mid = rent_est["mid_rent"] * (1 + premium_pct)
    adj_max = rent_est["max_rent"] * (1 + premium_pct)

    conf_lvl, conf_score = compute_rental_confidence(
        features, sources, area, prop_type, desc_text, bundle
    )
    if rent_est["confidence"] == "LOW" or conf_lvl == "LOW":
        final_conf = "LOW"
    elif rent_est["confidence"] == "MEDIUM" or conf_lvl == "MEDIUM":
        final_conf = "MEDIUM"
    else:
        final_conf = "HIGH"

    decision = make_rental_decision(user_rent, adj_mid, final_conf, feat_score)

    reason = build_rental_reason(
        decision["alert"], adj_min, adj_max, final_conf, feat_score, rent_est["zone"]
    )

    return {
        "mode": "rental",
        "alert": decision["alert"],
        "reason": reason,
        "confidence": final_conf,
        "confidence_score": conf_score,
        "ratio": decision["ratio"],
        "user_rent": user_rent,
        "expected_rent_range": {"min": adj_min, "mid": adj_mid, "max": adj_max},
        "sale_price_estimate": sale_price,
        "yield": {
            "min": rent_est["yield_min"],
            "typical": rent_est["yield_typical"],
            "max": rent_est["yield_max"],
        },
        "zone": rent_est["zone"],
        "premium_pct": premium_pct,
        "feature_score": feat_score,
        "feature_tier": tier,
        "active_features": [k for k, v in features.items() if v],
        "thresholds": {
            "lower": decision["threshold_lower"],
            "upper": decision["threshold_upper"],
        },
    }
