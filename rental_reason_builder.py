"""Aqar Hub — Rental Arabic Reason Builder"""
from alert_engine import format_price_egp


def build_rental_reason(alert: str, expected_min: float, expected_max: float,
                        confidence: str, feature_score: int, zone: str = "") -> str:
    low = format_price_egp(expected_min)
    high = format_price_egp(expected_max)

    zone_map = {
        "coastal": "الساحلية",
        "red_sea": "البحر الأحمر",
        "new_cairo": "القاهرة الجديدة",
        "urban": "المشابهة",
    }
    zlabel = zone_map.get(zone, "المشابهة")

    if alert == "UNDERPRICED":
        line1 = f"السعر أقل من المتوقع ({low}–{high}) مقارنة بالعقارات {zlabel}."
    elif alert == "OVERPRICED":
        line1 = f"السعر المطلوب أعلى من المعتاد ({low}–{high}) للعقارات {zlabel}."
    else:
        line1 = f"السعر ضمن النطاق المتوقع ({low}–{high}) للعقارات {zlabel}."

    if confidence == "LOW":
        line2 = "لكن مستوى الثقة منخفض بسبب محدودية البيانات المتاحة."
    elif feature_score >= 30:
        line2 = "التقدير يعتمد على المميزات المتعددة المتوفرة في العقار."
    elif feature_score >= 15:
        line2 = "التقدير يعتمد على المميزات المتوفرة والموقع."
    else:
        line2 = "التقدير يعتمد على المنطقة والمميزات الأساسية المتوفرة."

    return f"{line1}\n{line2}"
