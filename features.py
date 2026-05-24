"""Aqar Hub v22.0 — Feature Engineering, NLP & Target Encoding"""
import re
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
KWS = [
    ("north_coast",    r"north\s*coast|ras\s*el?\s*hekma|sahel"),
    ("penthouse",      r"\bpenthouse\b"),
    ("lagoon",         r"lagoon"),
    ("fully_finished", r"fully.?finish"),
    ("sea_view",       r"sea.?view|panoramic.*sea|ocean.?view"),
    ("furnished",      r"\bfurnish"),
    ("new_cairo",      r"new\s*cairo|5th\s*settlement|fifth\s*settlement"),
    ("premium_dev",    r"palm hills|sodic|orascom|emaar|mountain view|la vista|hyde park|tatweer"),
    ("garden",         r"private.?garden|\bgarden\b"),
    ("pool",           r"\bpool\b|swimming"),
    ("compound_kw",    r"\bcompound\b"),
    ("ready_now",      r"immediate.?delivery|ready.?to.?move|استلام\s*فوري"),
    ("sheikh_zayed",   r"sheikh.?zayed|6th.?october"),
    ("duplex",         r"\bduplex\b"),
    ("prime_loc",      r"prime.?location|prime"),
    ("core_shell",     r"core.?shell|bare.?shell|unfinished"),
    ("installment_kw", r"install|قسط|تقسيط"),
    ("resort",         r"resort|chalet"),
]

LUXURY_TERMS = r"sea.?view|lagoon|prime|fully.?finish|super.?lux|luxury|penthouse|panoramic|golf.?view"
BUDGET_TERMS = r"core.?shell|bare.?shell|unfinished|semi.?finish|economy|affordable"
INVEST_TERMS = r"install|قسط|تقسيط|compound|resort|chalet|north.?coast|sahel|delivery\s*\d{4}"

NUM_COLS = [
    "size_sqm", "bedrooms_n", "bathrooms_n",
    "has_maid", "is_installment", "is_cash",
    "room_ratio", "install_years", "log_comp_size",
    "finish_ord", "view_score",
    "down_payment_ratio", "log_down_payment",
    "market_tier", "size_per_bedroom", "size_x_finish",
    "price_per_sqm_estimate",
    "luxury_score", "budget_score", "investment_signal",
    "building_age", "under_construction",
    "floor_level", "has_explicit_floor",
    "legal_status", "condition_score",
    "has_elevator", "has_garage", "has_security", "has_balcony",
    "has_ac", "has_gas", "has_storage",
    "near_metro", "near_mall", "near_highway", "main_street",
    "corner_unit", "quiet_zone",
    "motivation_urgent", "motivation_negotiable",
    "motivation_cash_only", "motivation_below_market", "motivation_score",
    "north_facing", "south_facing", "east_west", "street_view",
    "has_master_bedroom", "has_dressing_room", "has_driver_room", "has_nanny_room",
    "has_phone_in_desc", "caps_ratio", "excl_count", "sentence_count", "num_count",
    "area_market_zone_ord",
    "is_compound", "compound_tier", "standalone_premium", "compound_premium",
    "log_size_sqm", "size_sqm_inv",
]

TE_COLS = [
    "type_clean_te", "governorate_te", "city_te", "compound_te",
    "comp_type_te", "bed_type_te", "nc_type_te", "finish_type_te",
    "type_tier_te", "city_comp_te", "governorate_comp_te",
]


# ─────────────────────────────────────────────────────────────────────────────
# NLP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _kw_features(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    d = d.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    out = {}
    for name, pat in KWS:
        out["nlp_" + name] = d.str.contains(pat, regex=True).astype(np.int8).values
    out["desc_len"] = (d.str.len() / 500).clip(0, 4).astype(np.float32).values
    out["nlp_score"] = np.column_stack(list(out.values())).sum(axis=1).astype(np.float32)
    return pd.DataFrame(out, index=desc_series.index)


def _semantic_scores(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    d = d.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    return pd.DataFrame({
        "luxury_score":      d.str.count(LUXURY_TERMS).astype(np.float32).values,
        "budget_score":      d.str.count(BUDGET_TERMS).astype(np.float32).values,
        "investment_signal": d.str.count(INVEST_TERMS).astype(np.float32).values,
    }, index=desc_series.index)


# ─────────────────────────────────────────────────────────────────────────────
# v20 CONTENT FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────
def _extract_building_age(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    yr1 = d.str.extract(r"(?:deliver|delivery|استلام|تسليم|سنة|عام)\s*(?:in|في|by)?\s*(\d{4})").iloc[:, 0]
    yr2 = d.str.extract(r"(\d{4})\s*(?:deliver|delivery|استلام|تسليم)").iloc[:, 0]
    delivery_year = pd.to_numeric(yr1.fillna(yr2), errors="coerce")
    age_text = d.str.extract(r"(\d+)\s*(?:years?\s*old|سنين?\s*عمر|سنة\s*عمر)").iloc[:, 0]
    age_from_text = pd.to_numeric(age_text, errors="coerce")
    ANCHOR_YEAR = 2026  # Frozen model baseline year — prevents silent feature drift
    building_age = (ANCHOR_YEAR - delivery_year).fillna(age_from_text)
    building_age = building_age.where((building_age >= 0) & (building_age <= 100))
    under_construction = d.str.contains(
        r"under\s*construction|قيد\s*الانشاء|unfinished\s*building|لم\s*يتم\s*البناء", regex=True
    ).astype(np.int8)
    return pd.DataFrame({
        "building_age": building_age.astype(np.float32).values,
        "under_construction": under_construction.values,
    }, index=desc_series.index)


def _extract_floor_level(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    floor = pd.Series(np.nan, index=desc_series.index)
    floor[d.str.contains(r"\bground\s*floor\b|الدور\s*الارضي|دور\s*ارضي", regex=True)] = 0
    floor[d.str.contains(r"\bfirst\s*floor\b|الدور\s*الاول|1st\s*floor", regex=True)] = 1
    floor[d.str.contains(r"\bsecond\s*floor\b|الدور\s*الثاني|2nd\s*floor", regex=True)] = 2
    floor[d.str.contains(r"\bthird\s*floor\b|الدور\s*الثالث|3rd\s*floor", regex=True)] = 3
    floor[d.str.contains(r"\bhigh\s*floor\b|دور\s*عالي|علوي", regex=True)] = 4
    floor[d.str.contains(r"\btop\s*floor\b|roof\s*top|اخر\s*دور|الدور\s*الاخير", regex=True)] = 5
    num_floor = d.str.extract(r"(?:floor|دور)\s*(\d+)|(\d+)(?:st|nd|rd|th)\s*(?:floor|دور)").bfill(axis=1).iloc[:, 0]
    num_floor = pd.to_numeric(num_floor, errors="coerce")
    floor = floor.fillna(num_floor)
    return pd.DataFrame({
        "floor_level": floor.clip(0, 50).astype(np.float32).values,
        "has_explicit_floor": (~floor.isna()).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_legal_status(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    status = pd.Series(0, index=desc_series.index)
    status[d.str.contains(r"registered|مسجل|tabo|طابو|سجل\s*عقاري", regex=True)] = 3
    status[d.str.contains(r"equity|حصة|equal\s*share|شراكة", regex=True)] = 2
    status[d.str.contains(r"contract|عقد|عقود", regex=True)] = 1
    status[d.str.contains(r"not\s*registered|غير\s*مسجل|بدون\s*عقد|without\s*papers", regex=True)] = -1
    return pd.DataFrame({"legal_status": status.astype(np.int8).values}, index=desc_series.index)


def _extract_condition(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    cond = pd.Series(np.nan, index=desc_series.index)
    cond[d.str.contains(r"needs\s*renovation|يحتاج\s*تجديد|needs\s*work|يحتاج\s*صيانة", regex=True)] = 0.0
    cond[d.str.contains(r"old\s*building|عمارة\s*قديمة|used|مستعمل|قديم", regex=True)] = 1.0
    cond[d.str.contains(r"good\s*condition|حالة\s*جيدة|average|متوسط", regex=True)] = 2.0
    cond[d.str.contains(r"renovated|متشطب|renewed|مجدد|تم\s*التجديد", regex=True)] = 3.0
    cond[d.str.contains(r"brand\s*new|جديد\s*بالكامل|first\s*use|اول\s*استخدام|never\s*lived|لم\s*يسكن", regex=True)] = 4.0
    return pd.DataFrame({"condition_score": cond.astype(np.float32).values}, index=desc_series.index)


def _extract_amenity_flags(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    return pd.DataFrame({
        "has_elevator":  d.str.contains(r"\belevator\b|أسانسير|مصعد", regex=True).astype(np.int8).values,
        "has_garage":    d.str.contains(r"\bgarage\b|جراج|موقف\s*سيارات|parking", regex=True).astype(np.int8).values,
        "has_security":  d.str.contains(r"\bsecurity\b|أمن|حارس|guard|gated|بوابة", regex=True).astype(np.int8).values,
        "has_balcony":   d.str.contains(r"\bbalcony\b|بلكونة|تراس|terrace|لانش", regex=True).astype(np.int8).values,
        "has_ac":        d.str.contains(r"\b(ac|air\s*condition|conditioning|تكييف|مكيف)\b", regex=True).astype(np.int8).values,
        "has_gas":       d.str.contains(r"\bgas\b|غاز\s*طبيعي|natural\s*gas", regex=True).astype(np.int8).values,
        "has_storage":   d.str.contains(r"\bstorage\b|مخزن|store\s*room|مستودع", regex=True).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_micro_location(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    return pd.DataFrame({
        "near_metro":    d.str.contains(r"metro|مترو|subway|station|محطة", regex=True).astype(np.int8).values,
        "near_mall":     d.str.contains(r"mall|مول|shopping\s*center|سوق\s*تجاري", regex=True).astype(np.int8).values,
        "near_highway":  d.str.contains(r"highway|طريق\s*سريع|ring\s*road|محور|autostrad", regex=True).astype(np.int8).values,
        "main_street":   d.str.contains(r"main\s*street|شارع\s*رئيسي|على\s*الشارع|facing\s*street", regex=True).astype(np.int8).values,
        "corner_unit":   d.str.contains(r"corner|زاوية|ناصية|زاويه", regex=True).astype(np.int8).values,
        "quiet_zone":    d.str.contains(r"quiet|هادئ|سكني|calm|residential", regex=True).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_motivation(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    urgent       = d.str.contains(r"urgent|عاجل|فوري|سريع|للبيع\s*السريع", regex=True).astype(np.int8)
    negotiable   = d.str.contains(r"negotiable|قابل\s*للتفاوض|تفاوض|سوم", regex=True).astype(np.int8)
    cash_only    = d.str.contains(r"cash\s*only|كاش\s*فقط", regex=True).astype(np.int8)
    below_market = d.str.contains(r"below\s*market|أقل\s*من\s*السوق|فرصة|opportunity|لقطة", regex=True).astype(np.int8)
    return pd.DataFrame({
        "motivation_urgent":       urgent.values,
        "motivation_negotiable":   negotiable.values,
        "motivation_cash_only":    cash_only.values,
        "motivation_below_market": below_market.values,
        "motivation_score":        (urgent + negotiable + below_market).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_exposure(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    return pd.DataFrame({
        "north_facing": d.str.contains(r"\bnorth\s*facing\b|شمالي|اتجاه\s*شمال", regex=True).astype(np.int8).values,
        "south_facing": d.str.contains(r"\bsouth\s*facing\b|جنوبي|اتجاه\s*جنوب", regex=True).astype(np.int8).values,
        "east_west":    d.str.contains(r"\beast\b|\bwest\b|شرقي|غربي|شرق|غرب", regex=True).astype(np.int8).values,
        "street_view":  d.str.contains(r"street\s*view|اطلالة\s*شارع|على\s*الشارع", regex=True).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_room_config(desc_series):
    d = desc_series.fillna("").astype(str).str.lower()
    return pd.DataFrame({
        "has_master_bedroom": d.str.contains(r"master\s*bedroom|ماستر|غرفة\s*رئيسية", regex=True).astype(np.int8).values,
        "has_dressing_room":  d.str.contains(r"dressing\s*room|دريسينج|غرفة\s*ملابس", regex=True).astype(np.int8).values,
        "has_driver_room":    d.str.contains(r"driver\s*room|غرفة\s*سائق", regex=True).astype(np.int8).values,
        "has_nanny_room":     d.str.contains(r"nanny|مربية|babysitter|غرفة\s*مربية", regex=True).astype(np.int8).values,
    }, index=desc_series.index)


def _extract_listing_meta(desc_series):
    d = desc_series.fillna("").astype(str)
    # Tightened: look for phone-like patterns, not bare 3+ digit sequences
    has_phone = d.str.contains(
        r"(?:\+\d{1,3}[\s-]?)?(?:\(\d+\)|\d{2,})[\s.-]?\d{3,}[\s.-]?\d{3,}|\b\d{7,15}\b",
        regex=True,
    ).astype(np.int8)
    caps_ratio = (d.str.count(r'[A-Z]') / d.str.len().clip(lower=1)).fillna(0).astype(np.float32)
    excl_count = d.str.count(r'!').clip(0, 10).astype(np.int8)
    sent_count = d.str.count(r'[.!?۔]').clip(1, 50).astype(np.int8)
    num_count = d.str.count(r'\d').clip(0, 100).astype(np.int8)
    return pd.DataFrame({
        "has_phone_in_desc": has_phone.values,
        "caps_ratio":        caps_ratio.values,
        "excl_count":        excl_count.values,
        "sentence_count":    sent_count.values,
        "num_count":         num_count.values,
    }, index=desc_series.index)


def _extract_all_content(desc_series):
    """Run all v20 content extractors and return combined DataFrame."""
    return pd.concat([
        _extract_building_age(desc_series),
        _extract_floor_level(desc_series),
        _extract_legal_status(desc_series),
        _extract_condition(desc_series),
        _extract_amenity_flags(desc_series),
        _extract_micro_location(desc_series),
        _extract_motivation(desc_series),
        _extract_exposure(desc_series),
        _extract_room_config(desc_series),
        _extract_listing_meta(desc_series),
    ], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING & BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def kfold_te(tr, te, col, target="log_price", k=20, n_splits=5, seed=42):
    enc = np.zeros(len(tr), dtype=np.float32)
    gm = float(tr[target].mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for ti, vi in kf.split(tr):
        s = tr.iloc[ti].groupby(col)[target].agg(["mean", "count"])
        s["e"] = (s["count"] * s["mean"] + k * gm) / (s["count"] + k)
        enc[vi] = tr.iloc[vi][col].map(s["e"].to_dict()).fillna(gm).values.astype(np.float32)
    s_all = tr.groupby(col)[target].agg(["mean", "count"])
    s_all["e"] = (s_all["count"] * s_all["mean"] + k * gm) / (s_all["count"] + k)
    return enc, te[col].map(s_all["e"].to_dict()).fillna(gm).values.astype(np.float32), s_all["e"].to_dict()


def kfold_price_per_sqm(tr, te, keys=("city", "type_clean"), n_splits=5, seed=42):
    tr = tr.copy()
    tr["_lpps"] = tr["log_price"] - np.log1p(tr["size_sqm"].clip(lower=1))
    gm = float(tr["_lpps"].median())
    enc = np.full(len(tr), gm, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    key_cols = list(keys)
    for ti, vi in kf.split(tr):
        mp = tr.iloc[ti].groupby(key_cols)["_lpps"].median().to_dict()
        vals = tr.iloc[vi][key_cols].apply(lambda r: mp.get(tuple(r), gm), axis=1)
        enc[vi] = vals.astype(np.float32).values
    s_all = tr.groupby(key_cols)["_lpps"].median().to_dict()
    te_enc = te[key_cols].apply(lambda r: s_all.get(tuple(r), gm), axis=1).astype(np.float32).values
    return enc, te_enc, s_all, gm


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def assemble_X(frame, kw, lsa, train_medians=None, lo=None, hi=None):
    available_num = [c for c in NUM_COLS if c in frame.columns]
    available_te = [c for c in TE_COLS if c in frame.columns]
    X = pd.concat([
        frame[available_num + available_te].reset_index(drop=True),
        kw.reset_index(drop=True),
        lsa.reset_index(drop=True),
    ], axis=1)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)
    if train_medians is None:
        medians = X.median(numeric_only=True)
        X.fillna(medians, inplace=True)
        lo = X.quantile(0.01)
        hi = X.quantile(0.99)
        X = X.clip(lower=lo, upper=hi, axis=1)
        return X, medians, lo, hi
    X.fillna(train_medians, inplace=True)
    return X.clip(lower=lo, upper=hi, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_sets(all_cols):
    te_cols = [c for c in all_cols if c.endswith("_te")]
    ppsqm = "price_per_sqm_estimate"
    col_xgb = [c for c in all_cols if c not in te_cols and c != ppsqm]
    col_cat = list(all_cols)
    if len(col_xgb) < 12:
        extras = [c for c in all_cols if c not in col_xgb and c not in te_cols and c != ppsqm]
        col_xgb.extend(extras[:12 - len(col_xgb)])
    return col_xgb, col_cat