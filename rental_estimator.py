"""Aqar Hub — Rental Estimator (Yield Layer)"""
from typing import Dict, Any

from rental_config import RENTAL_YIELDS, DEFAULT_YIELD

# ── SALE MODEL MAPE BY TYPE (for error propagation guard) ──
# Source: alert_engine.py MAPE_BY_TYPE — used to expand rent boundaries
SALE_MAPE_BY_TYPE = {
    "Penthouse": 0.162,
    "Apartment": 0.175,
    "Chalet":    0.194,
    "Duplex":    0.211,
    "Villa":     0.254,
}


def _resolve_zone(area: str, compound: str = "", desc: str = "") -> str:
    blob = f"{str(area).lower()} {str(compound).lower()} {str(desc).lower()}"
    if any(x in blob for x in ["north coast", "sahel", "ras el hekma", "matrouh",
                                 "hacienda", "marassi", "amwaj", "bo island", "jefaira"]):
        return "coastal"
    if any(x in blob for x in ["sokhna", "red sea", "ghardaqa", "hurghada",
                                 "sharm", "dahab", "porto sokhna"]):
        return "red_sea"
    if any(x in blob for x in ["new cairo", "5th settlement", "fifth settlement",
                                 "tagamoa", "التجمع"]):
        return "new_cairo"
    return "urban"


def estimate_rent(sale_price: float, property_type: str, area: str,
                  compound: str = "", desc: str = "") -> Dict[str, Any]:
    """
    Estimate monthly rent with MAPE-based error propagation guard.
    Expands min/max boundaries proportionally to sale model uncertainty.
    """
    zone = _resolve_zone(area, compound, desc)
    key = (zone, property_type)

    if key in RENTAL_YIELDS:
        min_y, max_y, typ_y = RENTAL_YIELDS[key]
        conf = "HIGH"
    else:
        fallback = (zone, "Apartment")
        if fallback in RENTAL_YIELDS:
            min_y, max_y, typ_y = RENTAL_YIELDS[fallback]
            conf = "MEDIUM"
        else:
            min_y, max_y, typ_y = DEFAULT_YIELD
            conf = "LOW"

    # ── MAPE ERROR PROPAGATION GUARD ──
    # Expand rent boundaries by sale model MAPE to prevent false precision
    mape = SALE_MAPE_BY_TYPE.get(property_type, 0.20)
    mape_buffer = 1.0 + mape  # e.g., Villa: 1.254

    base_min = sale_price * min_y / 12.0
    base_mid = sale_price * typ_y / 12.0
    base_max = sale_price * max_y / 12.0

    # Widen boundaries: min shrinks, max expands to account for sale uncertainty
    guarded_min = base_min / mape_buffer
    guarded_max = base_max * mape_buffer

    return {
        "zone": zone,
        "min_rent": guarded_min,
        "mid_rent": base_mid,
        "max_rent": guarded_max,
        "yield_min": min_y,
        "yield_typical": typ_y,
        "yield_max": max_y,
        "confidence": conf,
        "sale_mape": mape,
        "mape_buffer_applied": mape_buffer,
    }