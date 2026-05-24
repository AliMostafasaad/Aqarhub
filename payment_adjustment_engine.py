"""Aqar Hub — Payment Adjustment Engine v2.0
Estimates financing premium using Present Value (PV) discounting.
Deterministic, explainable, production-safe.
"""
from typing import Dict, Any, Tuple
import math

# ── BASELINE INFLATION/OPPORTUNITY COST FACTOR ──
# Conservative annual rate for Egyptian real estate market (2026 baseline)
ANNUAL_DISCOUNT_RATE = 0.14  # 14% — reflects inflation + opportunity cost

# ── FINANCING MULTIPLIERS (legacy fallback for short installments) ──
INSTALLMENT_MULTIPLIERS = {
    "cash":          1.00,
    "short":         1.05,   # 1–4 years: minimal premium
    "medium":        1.10,   # 5–7 years
    "long":          1.18,   # 8–12 years
    "very_long":     1.25,   # 13+ years
}

# Down payment impact (additional adjustment)
DOWN_PAYMENT_IMPACT = {
    "high":    0.00,   # ≥ 30%
    "medium":  0.03,   # 15–29%
    "low":     0.05,   # 5–14%
    "very_low": 0.08,  # < 5%
}


def classify_installment(installment_years: float) -> str:
    """Classify installment duration into tier."""
    if installment_years <= 0 or installment_years != installment_years:
        return "cash"
    if installment_years <= 4:
        return "short"
    if installment_years <= 7:
        return "medium"
    if installment_years <= 12:
        return "long"
    return "very_long"


def classify_down_payment(down_payment_ratio: float) -> str:
    """Classify down payment ratio into tier."""
    if down_payment_ratio != down_payment_ratio or down_payment_ratio <= 0:
        return "medium"
    if down_payment_ratio >= 0.30:
        return "high"
    if down_payment_ratio >= 0.15:
        return "medium"
    if down_payment_ratio >= 0.05:
        return "low"
    return "very_low"


def compute_pv_adjustment(
    asking_price: float,
    installment_years: float,
    down_payment_ratio: float,
) -> Tuple[float, str, str, float]:
    """
    Compute Present Value adjustment for long-term installment plans.

    For installments > 7 years, applies PV discounting:
        PV = Future_Value / ((1 + r) ** n)

    For shorter terms, uses legacy linear multiplier.

    Returns:
        adjusted_price: cash-equivalent price
        installment_tier: str
        dp_tier: str
        discount_rate_used: float
    """
    inst_tier = classify_installment(installment_years)
    dp_tier = classify_down_payment(down_payment_ratio)

    if inst_tier == "cash":
        return asking_price, "cash", dp_tier, 0.0

    # For long/very_long installments: use PV discounting
    if inst_tier in ("long", "very_long") and installment_years > 0:
        # Present Value of future payments
        pv_multiplier = 1.0 / ((1.0 + ANNUAL_DISCOUNT_RATE) ** installment_years)
        # Blend PV with down payment impact
        dp_add = DOWN_PAYMENT_IMPACT.get(dp_tier, 0.03)
        blended_mult = min(pv_multiplier + dp_add, 1.0)  # Cap at 1.0 (no inflation)
        adjusted = asking_price * blended_mult
        return adjusted, inst_tier, dp_tier, ANNUAL_DISCOUNT_RATE

    # For short/medium: legacy linear model
    base_mult = INSTALLMENT_MULTIPLIERS.get(inst_tier, 1.10)
    dp_add = DOWN_PAYMENT_IMPACT.get(dp_tier, 0.03)
    multiplier = min(base_mult + dp_add, 1.35)
    adjusted = asking_price / multiplier

    return adjusted, inst_tier, dp_tier, 0.0


def adjust_price_for_comparison(
    asking_price: float,
    payment_type: str,
    installment_years: float,
    down_payment_ratio: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Convert installment asking price to cash-equivalent for fair comparison.
    Uses PV discounting for long-term installments, legacy linear for short-term.
    """
    # Normalize payment type
    ptype = str(payment_type).lower().strip()
    is_installment = any(kw in ptype for kw in ["install", "قسط", "تقسيط", "installment"])

    if not is_installment:
        return asking_price, {
            "original_price": asking_price,
            "adjusted_price": asking_price,
            "multiplier": 1.00,
            "installment_tier": "cash",
            "down_payment_tier": "high",
            "payment_type": payment_type,
            "installment_years": 0,
            "down_payment_ratio": down_payment_ratio,
            "is_adjusted": False,
            "discount_rate": 0.0,
            "method": "cash",
        }

    adjusted, inst_tier, dp_tier, discount_rate = compute_pv_adjustment(
        asking_price, installment_years, down_payment_ratio
    )

    # Calculate effective multiplier for reporting
    effective_mult = asking_price / adjusted if adjusted > 0 else 1.0

    meta = {
        "original_price": asking_price,
        "adjusted_price": adjusted,
        "multiplier": effective_mult,
        "installment_tier": inst_tier,
        "down_payment_tier": dp_tier,
        "payment_type": payment_type,
        "installment_years": installment_years,
        "down_payment_ratio": down_payment_ratio,
        "is_adjusted": effective_mult > 1.01,
        "discount_rate": discount_rate,
        "method": "pv" if discount_rate > 0 else "linear",
    }

    return adjusted, meta


def build_payment_explanation(meta: Dict[str, Any]) -> str:
    """Build short Arabic explanation about financing impact."""
    if not meta.get("is_adjusted"):
        return ""

    inst_tier = meta["installment_tier"]
    method = meta.get("method", "linear")

    if method == "pv":
        return "تم احتساب تأثير القيمة الحالية للتقسيط الطويل عند تحليل السعر."
    elif inst_tier == "short":
        return "تم احتساب تأثير التقسيط القصير عند تحليل السعر."
    elif inst_tier in ("long", "very_long"):
        return "السعر يتضمن نظام سداد طويل الأمد قد يرفع القيمة الإجمالية للعقار."
    else:
        return "تم احتساب تأثير نظام التقسيط عند تحليل السعر."