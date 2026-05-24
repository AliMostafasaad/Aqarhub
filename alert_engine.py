"""Aqar Hub — Decision Engine v3.2 (Safety Layers with Extreme Deviation Override)

v3.2 Changes:
- Safety layers NO LONGER force FAIR blindly
- Extreme ratios (<0.4 or >2.0) override safety → UNDER/OVER with LOW confidence
- Arabic explanations updated to reflect uncertainty when data is weak
- _build_arabic_reason() handles all cases cleanly
"""
import re
import numpy as np
from typing import Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAPE_BY_TYPE = {
    "Penthouse": 0.162,
    "Apartment": 0.175,
    "Chalet":    0.194,
    "Duplex":    0.211,
    "Villa":     0.254,
}

BASE_CONFIDENCE = {
    "Penthouse": 82,
    "Apartment": 85,
    "Chalet":    75,
    "Duplex":    68,
    "Villa":     55,
}

TYPE_RELIABILITY = {
    "Penthouse": 1.05,
    "Apartment": 1.00,
    "Chalet":    0.95,
    "Villa":     0.85,
    "Duplex":    0.80,
}

BIAS_FACTOR = {
    "compound":   1.041,
    "standalone": 1.051,
}

COASTAL_PATTERNS = r"north\s*coast|sahel|ras\s*el?\s*hekma|matrouh|marsa\s*matrouh|el\s*alamein|hacienda|marassi|amwaj|bo\s*island|jefaira"
RED_SEA_PATTERNS = r"sokhna|ain\s*sokhna|red\s*sea|ghardaqa|hurghada|sharm\s*el\s*sheikh|dahab|porto\s*sokhna|mountain\s*view\s*sokhna"

PREMIUM_CITIES = {
    "new cairo", "5th settlement", "fifth settlement",
    "sheikh zayed", "6th october", "new capital",
    "tagamoa", "el tagamoa", "التجمع",
}

STANDALONE_CALIBRATION = {
    "new cairo":      0.72,
    "tagamoa":        0.72,
    "sheikh zayed":   0.78,
    "6th october":    0.80,
    "new capital":    0.75,
}
DEFAULT_STANDALONE_DISCOUNT = 0.82

VILLA_CLUSTER_MEDIAN_P50 = {
    "urban":    18_500_000,
    "coastal":   6_200_000,
    "red_sea":   8_900_000,
}

VILLA_CLUSTER_OVERALL_MEDIAN = 15_000_000

VILLA_CLUSTER_FACTOR = {
    zone: median / VILLA_CLUSTER_OVERALL_MEDIAN
    for zone, median in VILLA_CLUSTER_MEDIAN_P50.items()
}

VILLA_CLUSTER_MAPE = {
    "urban":    0.18,
    "coastal":  0.35,
    "red_sea":  0.28,
}

VILLA_ZONE_THRESHOLDS = {
    "urban":   (0.65, 1.50),
    "coastal": (0.55, 1.70),
    "red_sea": (0.60, 1.60),
}

EXTREME_RATIO_LOW  = 0.40
EXTREME_RATIO_HIGH = 2.00

# ── SAFETY LAYER THRESHOLDS ─────────────────────────────────────────────────
RARITY_HIGH_THRESHOLD = 0.50
RARITY_MED_THRESHOLD  = 0.30
BENCHMARK_UPPER       = 1.80
BENCHMARK_LOWER       = 0.40
RARITY_PENALTY_HIGH   = 15
RARITY_PENALTY_MED    = 8
BENCHMARK_PENALTY     = 25


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def format_price_egp(price: float) -> str:
    if price >= 1_000_000:
        val = price / 1_000_000
        return f"{val:.2f}".rstrip("0").rstrip(".") + "M"
    elif price >= 1_000:
        return f"{price / 1_000:.0f}K"
    else:
        return f"{price:,.0f}"


def compute_coverage_score(coverage: Dict[str, bool]) -> float:
    if not coverage:
        return 50.0
    weights = {
        "building_age": 15,
        "floor_level": 10,
        "amenities": 20,
        "condition_score": 15,
        "view_score": 10,
        "financial_info": 15,
    }
    score = sum(weights.get(k, 5) for k, v in coverage.items() if v)
    return float(min(100, score))


def detect_villa_market_zone(area: str, desc: str, compound: str = "") -> str:
    blob = f"{area} {desc} {compound}".lower()
    if re.search(COASTAL_PATTERNS, blob):
        return "coastal"
    if re.search(RED_SEA_PATTERNS, blob):
        return "red_sea"
    return "urban"


def check_down_payment(asking: float, predicted: float) -> Optional[str]:
    ratio = asking / predicted if predicted > 0 else 1.0
    if ratio < 0.35:
        dp_estimate = predicted * 0.20
        return (
            f"⚠️ Alert: The entered price ({format_price_egp(asking)}) equals "
            f"{ratio*100:.0f}% of the expected price. "
            f"Did you enter the down payment (≈{format_price_egp(dp_estimate)}) "
            f"instead of the full price?"
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def _get_standalone_discount(area: str, features: Dict[str, Any]) -> float:
    in_compound = features.get("in_compound") == "yes"
    if in_compound:
        return 1.0
    area_lower = str(area).lower()
    is_premium = any(city in area_lower for city in PREMIUM_CITIES)
    if not is_premium:
        return 1.0
    for city_key, discount in STANDALONE_CALIBRATION.items():
        if city_key in area_lower:
            return discount
    return DEFAULT_STANDALONE_DISCOUNT


def _apply_standalone_calibration(
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    area: str,
    features: Dict[str, Any],
) -> tuple[float, float, float]:
    discount = _get_standalone_discount(area, features)
    if discount >= 0.99:
        return predicted_price, lo_price, hi_price
    pred_log = np.log1p(predicted_price)
    lo_log = np.log1p(lo_price)
    hi_log = np.log1p(hi_price)
    calibrated_pred = np.expm1(pred_log + np.log(discount))
    calibrated_lo = np.expm1(lo_log + np.log(discount))
    calibrated_hi = np.expm1(hi_log + np.log(discount))
    return float(calibrated_pred), float(calibrated_lo), float(calibrated_hi)


def _apply_bias_correction(
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    features: Dict[str, Any],
) -> tuple[float, float, float]:
    in_compound = features.get("in_compound") == "yes"
    factor = BIAS_FACTOR["compound"] if in_compound else BIAS_FACTOR["standalone"]
    if factor <= 1.0:
        return predicted_price, lo_price, hi_price
    return predicted_price / factor, lo_price / factor, hi_price / factor


def _correct_villa_prediction(
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    zone: str,
) -> tuple[float, float, float]:
    factor = VILLA_CLUSTER_FACTOR.get(zone, 1.0)
    return predicted_price * factor, lo_price * factor, hi_price * factor


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY LAYERS (v3.2)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_rarity_score(area: str, prop_type: str, bundle: Optional[dict]) -> float:
    if bundle is None:
        return 0.0
    rarity_map = bundle.get("rarity_map", {})
    key = (str(area).strip(), str(prop_type).strip())
    if key in rarity_map:
        return float(rarity_map[key])
    pps_map = bundle.get("pps_map", {})
    if key not in pps_map:
        return 0.7
    return 0.0


def _compute_benchmark_deviation(
    predicted_price: float,
    area: str,
    prop_type: str,
    size_sqm: Optional[float],
    bundle: Optional[dict],
) -> tuple[float, bool]:
    if bundle is None or size_sqm is None or size_sqm <= 0:
        return 1.0, False
    pps_map = bundle.get("pps_map", {})
    pps_gm = bundle.get("pps_gm", 0.0)
    key = (str(area).strip(), str(prop_type).strip())
    log_pps = pps_map.get(key, pps_gm)
    median_pps = np.exp(float(log_pps))
    expected_price = median_pps * float(size_sqm)
    if expected_price <= 0 or not np.isfinite(expected_price):
        return 1.0, False
    deviation = predicted_price / expected_price
    is_unrealistic = deviation > BENCHMARK_UPPER or deviation < BENCHMARK_LOWER
    return deviation, is_unrealistic


def _apply_safety_layers(
    alert: str,
    confidence_score: int,
    confidence_level: str,
    area: str,
    property_type: str,
    predicted_price: float,
    size_sqm: Optional[float],
    features: Dict[str, Any],
    bundle: Optional[dict],
    ratio: float,
) -> tuple[str, int, str, bool, bool]:
    """
    v3.2: Extreme ratios (<0.4 or >2.0) override safety → keep alert with LOW confidence.
    Returns (alert, score, level, is_rare, is_unrealistic).
    """
    # ── 1. Rarity Check ──
    rarity_score = _compute_rarity_score(area, property_type, bundle)
    is_rare = False
    rarity_penalty = 0
    if rarity_score >= RARITY_HIGH_THRESHOLD:
        rarity_penalty = RARITY_PENALTY_HIGH
        is_rare = True
    elif rarity_score >= RARITY_MED_THRESHOLD:
        rarity_penalty = RARITY_PENALTY_MED
        is_rare = True

    # ── 2. Benchmark Check ──
    _, is_unrealistic = _compute_benchmark_deviation(
        predicted_price, area, property_type, size_sqm, bundle
    )
    benchmark_penalty = BENCHMARK_PENALTY if is_unrealistic else 0

    # ── 3. Extreme Override (v3.2 CRITICAL FIX) ──
    # If ratio is extreme, NEVER hide the signal behind FAIR
    is_extreme = (ratio < EXTREME_RATIO_LOW) or (ratio > EXTREME_RATIO_HIGH)

    # ── 4. Apply penalties ──
    adjusted_score = max(20, confidence_score - rarity_penalty - benchmark_penalty)

    if is_extreme:
        # Keep the alert (UNDER/OVER) but force LOW confidence
        adjusted_score = min(adjusted_score, 35)
        confidence_level = "LOW"
    elif is_rare or is_unrealistic:
        # Non-extreme + abnormal data → FAIR with LOW confidence
        alert = "FAIR"
        adjusted_score = min(adjusted_score, 35)
        confidence_level = "LOW"
    else:
        # Normal path
        if adjusted_score < 50:
            confidence_level = "LOW"
        elif adjusted_score < 75:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "HIGH"

    return alert, adjusted_score, confidence_level, is_rare, is_unrealistic


# ─────────────────────────────────────────────────────────────────────────────
# ARABIC REASON BUILDER (v3.2)
# ─────────────────────────────────────────────────────────────────────────────
def _build_arabic_reason(
    alert: str,
    ratio: float,
    confidence_level: str,
    is_rare: bool,
    is_unrealistic: bool,
    low_price: float,
    high_price: float,
) -> str:
    """
    Build 2-line Arabic explanation.
    Line 1: Deviation description
    Line 2: Uncertainty note (if data is weak)
    """
    low_str = format_price_egp(low_price)
    high_str = format_price_egp(high_price)
    weak_data = is_rare or is_unrealistic

    if alert == "UNDERPRICED":
        if ratio < EXTREME_RATIO_LOW:
            line1 = f"السعر أقل بكثير من النطاق المتوقع ({low_str}–{high_str})، مما يشير إلى فرصة قوية."
        else:
            line1 = f"السعر أقل من النطاق المتوقع ({low_str}–{high_str}) للعقارات المشابهة."
    elif alert == "OVERPRICED":
        if ratio > EXTREME_RATIO_HIGH:
            line1 = f"السعر أعلى بكثير من النطاق المتوقع ({low_str}–{high_str}) مقارنة بالسوق."
        else:
            line1 = f"السعر أعلى من النطاق المتوقع ({low_str}–{high_str}) للعقارات المشابهة."
    else:
        line1 = f"السعر ضمن النطاق المتوقع ({low_str}–{high_str}) للعقارات المشابهة."

    if weak_data and confidence_level == "LOW":
        line2 = "لكن دقة التقدير منخفضة بسبب قلة البيانات في هذه المنطقة."
    elif confidence_level == "LOW":
        line2 = "لكن مستوى الثقة منخفض بسبب محدودية المعلومات المتاحة."
    elif confidence_level == "MEDIUM":
        line2 = "التقدير متوسط الثقة بناءً على البيانات المتاحة."
    else:
        line2 = "التقدير عالي الثقة بناءً على بيانات السوق المتاحة."

    return f"{line1}\n{line2}"


def _build_reason(
    alert: str,
    property_type: str,
    area: str,
    lo_price: float,
    hi_price: float,
    features: Dict[str, Any],
    zone: str = "",
    calibrated: bool = False,
    ratio: float = 1.0,
    confidence_level: str = "MEDIUM",
    is_rare: bool = False,
    is_unrealistic: bool = False,
) -> str:
    """
    Build English reason + append Arabic reason.
    """
    area_str = area if area and area != "Unknown" else "similar locations"
    low_str = format_price_egp(lo_price)
    high_str = format_price_egp(hi_price)

    deviation_pct = abs(ratio - 1.0) * 100
    if ratio < 1.0:
        direction = "below"
        percentile = max(1, min(50, int((1.0 - ratio) * 100)))
        positioning = f"among the lowest {percentile}%"
    else:
        direction = "above"
        percentile = max(1, min(50, int((ratio - 1.0) * 100)))
        positioning = f"among the highest {percentile}%"

    if alert == "UNDERPRICED":
        if deviation_pct >= 25:
            interpret = (
                f"significantly lower than the market range ({low_str}–{high_str}). "
                f"Your price is ~{deviation_pct:.0f}% {direction} expected value, "
                f"placing it {positioning} of comparable listings."
            )
        else:
            interpret = f"lower than most comparable listings in the {low_str}–{high_str} range."
    elif alert == "OVERPRICED":
        if deviation_pct >= 25:
            interpret = (
                f"significantly higher than the market range ({low_str}–{high_str}). "
                f"Your price is ~{deviation_pct:.0f}% {direction} expected value, "
                f"placing it {positioning} of comparable listings."
            )
        else:
            interpret = f"higher than most comparable listings in the {low_str}–{high_str} range."
    elif alert == "ESTIMATE ONLY":
        interpret = "within a wide valuation range with higher uncertainty."
    else:
        interpret = f"generally in line with the market range of {low_str}–{high_str}."

    zone_prefix = ""
    if zone == "coastal":
        zone_prefix = "Coastal "
    elif zone == "red_sea":
        zone_prefix = "Red Sea "

    reason = (
        f"Similar {zone_prefix}{property_type.lower()}s in {area_str} typically sell between "
        f"{low_str} and {high_str}. Your price is {interpret}"
    )

    if calibrated:
        reason += " (Standalone area pricing applied.)"

    # ── Append Arabic explanation (v3.2) ──
    arabic_reason = _build_arabic_reason(
        alert, ratio, confidence_level, is_rare, is_unrealistic, lo_price, hi_price
    )
    reason += f"\n\n{arabic_reason}"

    opt = []
    if features.get("has_pool") == "yes":
        opt.append("A pool supports a higher valuation.")
    if features.get("developer_level") == "premium":
        opt.append("Premium developers typically command higher prices.")
    if opt:
        reason += " " + " ".join(opt)

    return reason


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
def _compute_confidence(
    ratio: float,
    lower: float,
    upper: float,
    base: int,
    property_type: str,
    features: Dict[str, Any],
    uncertainty_ratio: float,
) -> tuple[int, str]:
    penalty = 0

    if features.get("land_size") == "missing":
        penalty += 15 if property_type == "Villa" else 5
    if features.get("in_compound") == "no":
        penalty += 5 if property_type == "Villa" else 10
    if features.get("developer_level") == "unknown":
        penalty += 5
    if uncertainty_ratio > 0.40:
        penalty += 10

    if ratio < lower:
        margin = (lower - ratio) / lower if lower > 0 else 0
        signal_boost = int(min(margin * 60, 15))
    elif ratio > upper:
        margin = (ratio - upper) / upper if upper > 0 else 0
        signal_boost = int(min(margin * 60, 15))
    else:
        center = (lower + upper) / 2
        band_half = (upper - lower) / 2
        if band_half > 0:
            distance_pct = abs(ratio - center) / band_half
            distance_pct = max(0, min(1, distance_pct))
            signal_boost = -int(distance_pct * 15)
        else:
            signal_boost = 0

    score = max(20, min(95, base - penalty + signal_boost))

    reliability = TYPE_RELIABILITY.get(property_type, 1.0)
    score = int(score * reliability)
    score = max(20, min(95, score))

    if score >= 75:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID ALERT
# ─────────────────────────────────────────────────────────────────────────────
def _hybrid_alert(
    actual_price: float,
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    lower_ratio: float,
    upper_ratio: float,
) -> tuple[str, float]:
    ratio = actual_price / predicted_price if predicted_price > 0 else 1.0

    if actual_price < lo_price:
        hard = "UNDERPRICED"
    elif actual_price > hi_price:
        hard = "OVERPRICED"
    else:
        hard = "FAIR"

    if ratio < lower_ratio:
        soft = "UNDERPRICED"
    elif ratio > upper_ratio:
        soft = "OVERPRICED"
    else:
        soft = "FAIR"

    if hard == soft:
        return hard, ratio
    if hard == "FAIR" and soft in ("UNDERPRICED", "OVERPRICED"):
        return soft, ratio
    if hard in ("UNDERPRICED", "OVERPRICED") and soft == "FAIR":
        return hard, ratio
    return hard, ratio


# ─────────────────────────────────────────────────────────────────────────────
# VILLA HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def _handle_villa(
    actual_price: float,
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    area: str,
    features: Dict[str, Any],
    size_sqm: Optional[float] = None,
    bundle: Optional[dict] = None,
) -> Dict[str, Any]:
    zone = detect_villa_market_zone(
        area,
        features.get("desc", ""),
        features.get("compound", "")
    )

    pred_bias, lo_bias, hi_bias = _apply_bias_correction(
        predicted_price, lo_price, hi_price, features
    )
    pred_cal, lo_cal, hi_cal = _apply_standalone_calibration(
        pred_bias, lo_bias, hi_bias, area, features
    )
    corr_pred, corr_lo, corr_hi = _correct_villa_prediction(
        pred_cal, lo_cal, hi_cal, zone
    )

    ratio = actual_price / corr_pred if corr_pred > 0 else 1.0
    uncertainty_ratio = (corr_hi - corr_lo) / corr_pred if corr_pred > 0 else 0.0

    zone_lower, zone_upper = VILLA_ZONE_THRESHOLDS.get(zone, (0.60, 1.60))
    if uncertainty_ratio > 0.40:
        zone_lower *= 0.90
        zone_upper *= 1.10
    elif uncertainty_ratio < 0.20:
        zone_lower *= 0.97
        zone_upper *= 0.97

    alert, _ = _hybrid_alert(
        actual_price, corr_pred, corr_lo, corr_hi,
        zone_lower, zone_upper
    )

    base = BASE_CONFIDENCE["Villa"]
    if zone == "urban" and features.get("developer_level") == "premium":
        base = min(65, base + 10)
    if zone in ("coastal", "red_sea"):
        base = min(base, 50)

    confidence_score, confidence_level = _compute_confidence(
        ratio=ratio,
        lower=zone_lower,
        upper=zone_upper,
        base=base,
        property_type="Villa",
        features=features,
        uncertainty_ratio=uncertainty_ratio,
    )

    # ── SAFETY LAYERS (v3.2) ──
    alert, confidence_score, confidence_level, is_rare, is_unrealistic = _apply_safety_layers(
        alert, confidence_score, confidence_level,
        area, "Villa", corr_pred, size_sqm, features, bundle, ratio
    )

    calibrated = (pred_cal != pred_bias) or (pred_bias != predicted_price)
    reason = _build_reason(
        alert, "Villa", area, corr_lo, corr_hi, features,
        zone=zone, calibrated=calibrated, ratio=ratio,
        confidence_level=confidence_level, is_rare=is_rare, is_unrealistic=is_unrealistic,
    )

    return {
        "alert": alert,
        "reason": reason,
        "confidence": confidence_level,
        "confidence_score": confidence_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DECISION
# ─────────────────────────────────────────────────────────────────────────────
def make_decision(
    actual_price: float,
    predicted_price: float,
    lo_price: float,
    hi_price: float,
    property_type: str,
    area: str,
    features: Dict[str, Any],
    size_sqm: Optional[float] = None,
    bundle: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Decision engine v3.2 — Alert, Reason, Confidence + Safety Layers + Extreme Override.
    """
    if property_type == "Villa":
        return _handle_villa(
            actual_price, predicted_price, lo_price, hi_price,
            area, features, size_sqm, bundle
        )

    # ── NON-VILLA ──
    pred_bias, lo_bias, hi_bias = _apply_bias_correction(
        predicted_price, lo_price, hi_price, features
    )
    pred_cal, lo_cal, hi_cal = _apply_standalone_calibration(
        pred_bias, lo_bias, hi_bias, area, features
    )
    calibrated = (pred_cal != pred_bias) or (pred_bias != predicted_price)

    mape = MAPE_BY_TYPE.get(property_type, 0.20)
    margin = mape * 0.90
    lower = 1.0 - margin
    upper = 1.0 + margin

    uncertainty_ratio = (hi_cal - lo_cal) / pred_cal if pred_cal > 0 else 0.0
    if uncertainty_ratio > 0.40:
        lower *= 0.90
        upper *= 1.10
    elif uncertainty_ratio < 0.20:
        lower *= 0.97
        upper *= 0.97

    alert, ratio = _hybrid_alert(
        actual_price, pred_cal, lo_cal, hi_cal, lower, upper
    )

    base = BASE_CONFIDENCE.get(property_type, 75)
    if property_type == "Duplex":
        base = min(base, 65)

    confidence_score, confidence_level = _compute_confidence(
        ratio=ratio,
        lower=lower,
        upper=upper,
        base=base,
        property_type=property_type,
        features=features,
        uncertainty_ratio=uncertainty_ratio,
    )

    # ── SAFETY LAYERS (v3.2) ──
    alert, confidence_score, confidence_level, is_rare, is_unrealistic = _apply_safety_layers(
        alert, confidence_score, confidence_level,
        area, property_type, pred_cal, size_sqm, features, bundle, ratio
    )

    reason = _build_reason(
        alert, property_type, area, lo_cal, hi_cal, features,
        calibrated=calibrated, ratio=ratio,
        confidence_level=confidence_level, is_rare=is_rare, is_unrealistic=is_unrealistic,
    )

    return {
        "alert": alert,
        "reason": reason,
        "confidence": confidence_level,
        "confidence_score": confidence_score,
    }


def get_alert(
    user_price: float,
    predicted_price: float,
    prop_type: str,
    width: float,
    coverage_score: float,
    p10: float,
    p90: float,
    size_sqm: Optional[float] = None,
    bundle: Optional[dict] = None,
    area: str = "Unknown",
) -> Dict[str, Any]:
    """API-friendly wrapper around make_decision."""
    features = {
        "land_size": "present",
        "in_compound": "yes" if coverage_score >= 40 else "no",
        "has_pool": "no",
        "developer_level": "unknown",
    }
    return make_decision(
        actual_price=user_price,
        predicted_price=predicted_price,
        lo_price=p10,
        hi_price=p90,
        property_type=prop_type,
        area=area,
        features=features,
        size_sqm=size_sqm,
        bundle=bundle,
    )