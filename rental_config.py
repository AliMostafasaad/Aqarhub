"""Aqar Hub — Rental Intelligence Configuration"""
from typing import Dict, Tuple

RENTAL_YIELDS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("urban", "Apartment"):     (0.055, 0.075, 0.065),
    ("urban", "Villa"):         (0.045, 0.065, 0.055),
    ("urban", "Duplex"):        (0.050, 0.070, 0.060),
    ("urban", "Penthouse"):     (0.050, 0.070, 0.060),
    ("urban", "Chalet"):        (0.040, 0.060, 0.050),
    ("new_cairo", "Apartment"): (0.055, 0.070, 0.062),
    ("new_cairo", "Villa"):     (0.045, 0.060, 0.052),
    ("new_cairo", "Duplex"):    (0.050, 0.065, 0.057),
    ("new_cairo", "Penthouse"): (0.050, 0.065, 0.057),
    ("coastal", "Apartment"):   (0.040, 0.060, 0.050),
    ("coastal", "Villa"):       (0.035, 0.055, 0.045),
    ("coastal", "Chalet"):      (0.045, 0.065, 0.055),
    ("coastal", "Duplex"):      (0.040, 0.060, 0.050),
    ("red_sea", "Apartment"):   (0.045, 0.065, 0.055),
    ("red_sea", "Villa"):       (0.040, 0.060, 0.050),
    ("red_sea", "Chalet"):      (0.050, 0.070, 0.060),
}

DEFAULT_YIELD: Tuple[float, float, float] = (0.050, 0.070, 0.060)

FEATURE_ALIASES: Dict[str, list] = {
    "furnished": ["مفروش", "مفروشة", "مفروشه", "فرش كامل", "فرش", "تأثيث", "fully furnished", "furnished", "semi furnished"],
    "ac": ["تكييف", "تكييفات", "تكيف", "مكيف", "مكيفة", "مكيفه", "مكيفات", "ac", "air condition", "air conditioning", "conditioning"],
    "parking": ["جراج", "موقف", "باركينج", "جراجات", "مواقف", "parking", "garage", "car park", "موقف سيارات", "موقف خاص"],
    "elevator": ["مصعد", "اسانسير", "أسانسير", "اسانسيرات", "مصاعد", "elevator", "lift"],
    "security": ["امن", "حارس", "حرس", "بوابة", "بوابات", "سيكيوريتي", "security", "guard", "gated", "secured"],
    "gym": ["جيم", "نادي رياضي", "رياضة", "فتنس", "gym", "fitness", "health club", "sports club"],
    "pool": ["حمام سباحة", "بيسين", "مسطح مائي", "سباحة", "pool", "swimming pool", "swimming"],
    "balcony": ["بلكونة", "بلكونه", "تراس", "لانش", "شرفة", "balcony", "terrace", "lanai", "veranda", "patio"],
    "garden": ["حديقة", "جاردن", "جاردنات", "حدائق", "garden", "private garden", "green area", "yard"],
    "storage": ["مخزن", "مخازن", "مستودع", "تخزين", "storage", "store room", "storeroom"],
    "generator": ["مولد", "مولدات", "كهرباء احتياطية", "باور", "generator", "power backup", "backup power"],
    "cameras": ["كاميرات", "كاميرا", "مراقبة", "cameras", "cctv", "surveillance", "security cameras"],
    "mosque": ["مسجد", "مساجد", "مصلى", "صلاة", "mosque", "prayer area", "masjid"],
    "school": ["مدرسة", "مدارس", "تعليم", "دولي", "school", "international school", "academy", "nursery"],
    "hospital": ["مستشفى", "مستشفيات", "عيادة", "مركز صحي", "hospital", "clinic", "medical center"],
    "transport": ["مترو", "مواصلات", "مواصلات عامة", "حافلة", "اتوبيس", "ترام", "transport", "metro", "subway", "bus", "station", "near station"],
    "wifi": ["واي فاي", "واى فاى", "انترنت", "نت", "dsl", "فايبر", "wifi", "wi-fi", "internet", "broadband", "fiber"],
}

ARABIC_VARIANTS: Dict[str, str] = {
    "اسانسير": "مصعد",
    "أسانسير": "مصعد",
    "اسانسيرات": "مصاعد",
    "واى فاى": "واي فاي",
    "تكيف": "تكييف",
    "مكيفه": "مكيفة",
    "مفروشه": "مفروشة",
    "بلكونه": "بلكونة",
    "جراجات": "جراج",
    "مواقف": "موقف",
    "مصاعد": "مصعد",
    "مولدات": "مولد",
    "كاميرا": "كاميرات",
    "مساجد": "مسجد",
    "مدارس": "مدرسة",
    "مستشفيات": "مستشفى",
    "عيادة": "مستشفى",
}
