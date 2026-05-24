"""Aqar Hub — Structured Feature Extractor (Rental Layer)"""
import re
from typing import Dict, Any, Optional

from rental_config import FEATURE_ALIASES, ARABIC_VARIANTS


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    for variant, canonical in ARABIC_VARIANTS.items():
        text = text.replace(variant.lower(), canonical.lower())
    return text


def extract_features(description: str, checkboxes: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    checkboxes = checkboxes or {}
    features: Dict[str, bool] = {}
    desc_norm = _normalize_text(description)

    for feature, aliases in FEATURE_ALIASES.items():
        if feature in checkboxes and checkboxes[feature] is not None:
            val = checkboxes[feature]
            if isinstance(val, str):
                features[feature] = val.lower() in ("yes", "true", "1", "on")
            else:
                features[feature] = bool(val)
            continue

        found = False
        for alias in aliases:
            pat = re.escape(alias.lower())
            if len(alias) <= 4:
                pat = r'(?:^|\s)' + pat + r'(?:$|\s|\.|,|!)'
            if re.search(pat, desc_norm):
                found = True
                break
        features[feature] = found

    return features


def get_feature_sources(description: str, checkboxes: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    checkboxes = checkboxes or {}
    sources: Dict[str, str] = {}
    desc_norm = _normalize_text(description)

    for feature in FEATURE_ALIASES:
        if feature in checkboxes and checkboxes[feature] is not None:
            sources[feature] = "checkbox"
            continue

        found = False
        for alias in FEATURE_ALIASES[feature]:
            pat = re.escape(alias.lower())
            if len(alias) <= 4:
                pat = r'(?:^|\s)' + pat + r'(?:$|\s|\.|,|!)'
            if re.search(pat, desc_norm):
                found = True
                break
        sources[feature] = "nlp" if found else "none"

    return sources
