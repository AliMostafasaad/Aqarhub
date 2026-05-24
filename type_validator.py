"""Aqar Hub — Property Type Validation & Normalization Layer
This is the SINGLE source of truth for type enforcement.
All other modules must call these functions — never check types manually.
"""
from typing import Tuple, Optional
from config import ALLOWED_TYPES, TYPE_NORMALIZATION_MAP, UNSUPPORTED_RAW_TYPES


class TypeValidationError(Exception):
    """Raised when an unsupported property type is encountered."""
    pass


def normalize_property_type(raw_type):
    # 1. تنظيف النص
    t = str(raw_type).strip().lower()
    
    # 2. قاموس الترجمة من عربي لإنجليزي
    arabic_map = {
        "شقة": "Apartment",
        "شقه": "Apartment",
        "شقا": "Apartment",
        "فيلا": "Villa",
        "فيله": "Villa",
        "فيله": "Villa",
        "شاليه": "Chalet",
        "شليه": "Chalet",
        "دوبلكس": "Duplex",
        "دوبليكس": "Duplex",
        "بنتهاوس": "Penthouse",
        "رووف": "Penthouse",
        "روف": "Penthouse"
    }
    
    # 3. لو الكلمة عربي، استبدلها بالترجمة الإنجليزية
    if t in arabic_map:
        t = arabic_map[t]

    # 4. تظهير الكلمة عشان تتطابق مع الموديل (أول حرف كابيتال)
    t = t.title()

    # 5. تجميع أنواع الفيلات تحت فئة واحدة
    villa_variants = ["Ivilla", "Twin House", "Townhouse", "Twin Villa", "Standalone Villa"]
    if t in villa_variants:
        t = "Villa"

    # 6. التأكد إن النوع مدعوم في الموديل
    if t not in ALLOWED_TYPES:
        raise TypeValidationError(f"'{raw_type}' is not supported.")

    # 7. أهم سطر: إرجاع النتيجة للواجهة
    return t
   

def is_supported_type(raw_type: str) -> bool:
    """Soft check — returns True/False without raising."""
    try:
        normalize_property_type(raw_type)
        return True
    except TypeValidationError:
        return False


def get_unsupported_response(raw_type: str) -> dict:
    """
    Standardized rejection response for unsupported types.
    Used by API/GUI when validation fails.
    """
    return {
        "alert": "UNSUPPORTED",
        "predicted_price": None,
        "user_price": None,
        "confidence_score": 0,
        "confidence_level": "NONE",
        "reason": (
            f"Property type '{raw_type}' is not supported. "
            f"Aqar Hub currently supports: {', '.join(sorted(ALLOWED_TYPES))}."
        ),
        "uncertainty": "N/A",
        "data_quality": "N/A",
        "data_quality_score": 0,
        "ratio": None,
        "threshold_lower": None,
        "threshold_upper": None,
    }


def filter_supported_types(df, type_col="type_clean"):
    """
    DataFrame filter — drops unsupported types before training/inference.
    
    Returns: (filtered_df, dropped_count)
    """
    import pandas as pd
    
    original_count = len(df)
    
    # Use canonical normalize_property_type for both filter AND transform
    def _normalize_safe(t):
        try:
            return normalize_property_type(str(t))
        except TypeValidationError:
            return None
    
    normalized = df[type_col].apply(_normalize_safe)
    mask = normalized.notna()
    filtered = df[mask].copy()
    
    # Apply normalization in-place using the canonical function
    filtered[type_col] = normalized[mask].values
    
    dropped = original_count - len(filtered)
    return filtered, dropped