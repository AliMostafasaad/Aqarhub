"""Aqar Hub v22.4 — Data Loading & Cleaning 
(FIXED: Extreme Location Bias — Compound/Standalone Segmentation,
        Standalone Premium Signals, Area Market Zone)
"""
import re
import pandas as pd
import numpy as np
from config import ALLOWED_TYPES
from type_validator import filter_supported_types


# ── COMPOUND KNOWLEDGE BASE ──────────────────────────────────────────────
KNOWN_LUXURY_COMPOUNDS = {
    "taj city", "mivida", "hyde park", "mountain view", "palm hills",
    "sodic", "la vista", "el gouna", "sahl hasheesh", "marassi", "hacienda",
    "amwaj", "bo island", "jefaira", "mountain view icity", "palm hills katameya",
    "village gate", "fleur de ville", "the waterway", "eastown", "westown",
    "allegria", "katameya dunes", "katameya plaza", "uptown cairo",
}

KNOWN_MID_COMPOUNDS = {
    "madinaty", "beit el watan", "arkan", "zed west",
    "rehab", "el rehab", "fifth square", "woodville",
    "golf extension", "golf views", "el patio", "patio", "casa", "alma", 
    "hayah", "mena garden city", "village gardens", "galleria",
}

STANDALONE_KEYWORDS = [
    "standalone", "al yasmine", "el yasmin", "yasmin", "yasmine",
    "el yasmine", "yasmeen", "el yasmeen",
    "عمارة", "منطقة سكنية", "حي", "الحي", "المنطقة",
    "بدون كمبوند", "بدون مجمع", "شقة في عمارة",
    "residential area", "neighborhood", "district", "off compound",
    "عمارات", "منطقة", "حى", "الحى", "بيت", "منزل",
]


def _to_float(s):
    try:
        return float(re.sub(r"[^\d.]", "", str(s)))
    except:
        return np.nan


def _parse_sqm(v):
    s = str(v).lower().strip()
    s = re.sub(r'[‌‍‎‏  ⁠᠎ -   　]', ' ', s)
    s = s.strip()

    for pat, mult in [(r"([\d,]+(?:\.\d+)?)\s*sqm", 1.0),
                      (r"([\d,]+(?:\.\d+)?)\s*sqft", 0.0929)]:
        m = re.search(pat, s)
        if m:
            val = float(m.group(1).replace(",", "")) * mult
            return round(val, 1) if 5 <= val <= 5000 else np.nan
    m = re.search(r"([\d,]+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(1).replace(",", ""))
        return val if 5 <= val <= 5000 else np.nan
    return np.nan


def _parse_payment(v):
    if pd.isna(v):
        return np.nan
    s = str(v).lower().replace(",", "")
    mult = 1e6 if re.search(r"\bm\b|million|مليون", s) else \
           1e3 if re.search(r"\bk\b|thousand|الف|ألف", s) else 1.0
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) * mult if m else np.nan


def _parse_loc(loc):
    if pd.isna(loc):
        return {"compound": "Unknown", "city": "Unknown", "governorate": "Unknown"}
    parts = [p.strip() for p in str(loc).split(",")]
    n = len(parts)
    return {
        "compound": parts[0] if n >= 1 else "Unknown",
        "city": parts[-2] if n >= 3 else parts[0],
        "governorate": parts[-1] if n >= 2 else parts[0],
    }


def _size_bucket(s):
    try:
        v = float(s)
        if v < 80:
            return "small"
        elif v < 150:
            return "medium"
        elif v < 250:
            return "large"
        else:
            return "xlarge"
    except:
        return "medium"


def _market_tier(gov, city, compound):
    blob = f"{str(gov).lower()} {str(city).lower()} {str(compound).lower()}"
    if re.search(r"north\s*coast|sahel|ras\s*el?\s*hekma|matrouh|gouna|sokhna", blob):
        return 3.0
    if re.search(r"new\s*cairo|5th\s*settlement|fifth\s*settlement|sheikh.?zayed|6th.?october|new\s*capital", blob):
        return 2.0
    if re.search(r"giza|maadi|zamalek|heliopolis|nasr\s*city|mohandessin", blob):
        return 1.0
    return 0.0


def _area_market_zone(city, compound, desc):
    """Classify area into urban/coastal/red_sea for ALL property types."""
    blob = f"{str(city).lower()} {str(compound).lower()} {str(desc).lower()}"
    if re.search(r"north\s*coast|sahel|ras\s*el?\s*hekma|matrouh|marsa\s*matrouh|el\s*alamein|hacienda|marassi|amwaj|bo\s*island|jefaira", blob):
        return "coastal"
    if re.search(r"sokhna|ain\s*sokhna|red\s*sea|ghardaqa|hurghada|sharm\s*el\s*sheikh|dahab|porto\s*sokhna|mountain\s*view\s*sokhna", blob):
        return "red_sea"
    return "urban"


def _extract_compound_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compound vs standalone signals:
      - is_compound   : 1 if inside a known compound, 0 if standalone
      - compound_tier : 0=standalone/unknown, 1=mid-market, 2=premium
    """
    compound_lower = df["compound"].fillna("").str.lower().str.strip()
    desc_lower = df["desc_clean"].fillna("").str.lower()

    # Known compound lists
    all_known = KNOWN_LUXURY_COMPOUNDS | KNOWN_MID_COMPOUNDS
    is_known = compound_lower.isin(all_known)

    has_compound_kw = compound_lower.str.contains(
        r"\bcompound\b|\bكمبوند\b|\bمجمع\b", regex=True, na=False
    )
    desc_has_compound = desc_lower.str.contains(
        r"\bcompound\b|\bكمبوند\b|\bمجمع\b", regex=True, na=False
    )

    # Standalone signals
    is_standalone = (
        compound_lower.isin(["", "-", "unknown", "na", "n/a", "other", "none"])
        | df["compound"].isna()
        | (compound_lower.str.len() < 3)
    )
    for kw in STANDALONE_KEYWORDS:
        is_standalone |= compound_lower.str.contains(kw, regex=False, na=False)
        is_standalone |= desc_lower.str.contains(kw, regex=False, na=False)

    # Final is_compound: must have compound signal AND not be standalone
    df["is_compound"] = (
        (is_known | has_compound_kw | desc_has_compound) & ~is_standalone
    ).astype(np.int8)

    # Tier: 0=standalone, 1=mid, 2=premium
    tier = pd.Series(0, index=df.index, dtype=np.float32)
    mid_mask = compound_lower.isin(KNOWN_MID_COMPOUNDS) | desc_lower.str.contains(
        "|".join(KNOWN_MID_COMPOUNDS), regex=True, na=False
    )
    luxury_mask = compound_lower.isin(KNOWN_LUXURY_COMPOUNDS) | desc_lower.str.contains(
        "|".join(KNOWN_LUXURY_COMPOUNDS), regex=True, na=False
    )
    tier[mid_mask] = 1.0
    tier[luxury_mask] = 2.0
    df["compound_tier"] = tier

    # ── NEW: Explicit standalone signals to fight location bias ──
    df["standalone_premium"] = (
        (df["is_compound"] == 0) & (df["market_tier"] >= 2.0)
    ).astype(np.int8)
    
    df["compound_premium"] = (
        (df["is_compound"] == 1) & (df["compound_tier"] == 2.0)
    ).astype(np.int8)

    return df


def _fix_mislabeled_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-correct obviously mislabeled types based on description keywords + size.
    Conservative: only fixes when description explicitly mentions the true type.
    """
    desc = df["desc_clean"].fillna("").str.lower()
    apt_mask = df["type_clean"] == "Apartment"

    # Penthouse keyword
    penthouse_kw = desc.str.contains(r"\bpenthouse\b|بنتهاوس|بنتهاوس", regex=True, na=False)
    df.loc[apt_mask & penthouse_kw, "type_clean"] = "Penthouse"

    # Duplex keyword
    duplex_kw = desc.str.contains(r"\bduplex\b|دوبلكس|دوبليكس|دوبليكس", regex=True, na=False)
    df.loc[apt_mask & duplex_kw, "type_clean"] = "Duplex"

    # Villa keyword + very large size (> 400 sqm)
    villa_kw = desc.str.contains(r"\bvilla\b|فيلا|توين هاوس|تاون هاوس|i?villa", regex=True, na=False)
    villa_size_mask = (df["size_sqm"] > 400) & apt_mask
    df.loc[apt_mask & villa_kw & villa_size_mask, "type_clean"] = "Villa"

    n_fixed = (apt_mask & (penthouse_kw | duplex_kw | (villa_kw & villa_size_mask))).sum()
    if n_fixed > 0:
        print(f"     Auto-corrected {int(n_fixed)} mislabeled listings")

    return df


def _detect_statistical_mislabels(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW: Statistical outlier detection for mislabeled apartments.
    Identifies apartments with extreme ppsm per city×size_bucket using IQR fence.
    These are likely villas/penthouses mislabeled as apartments, skewing small-area predictions.
    """
    apt_mask = df["type_clean"] == "Apartment"
    if apt_mask.sum() < 100:
        return df

    apt = df[apt_mask].copy()
    apt["ppsm"] = apt["price_egp"] / apt["size_sqm"].clip(lower=1)

    # Size buckets
    apt["size_bucket"] = pd.cut(
        apt["size_sqm"],
        bins=[0, 80, 150, 250, 9999],
        labels=["small", "medium", "large", "xlarge"]
    )

    # IQR per city × size_bucket
    stats = (
        apt.groupby(["city", "size_bucket"])["ppsm"]
        .agg(q25=lambda x: x.quantile(0.25),
             q75=lambda x: x.quantile(0.75),
             count="count")
        .reset_index()
    )
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats["upper_fence"] = stats["q75"] + 3.0 * stats["iqr"]
    # Only apply fence where we have enough samples
    stats = stats[stats["count"] >= 10]

    if len(stats) == 0:
        return df

    apt = apt.merge(
        stats[["city", "size_bucket", "upper_fence"]],
        on=["city", "size_bucket"],
        how="left"
    )

    # Flag suspicious: ppsm > upper_fence AND either large size or penthouse keyword
    suspicious = apt[
        (apt["ppsm"] > apt["upper_fence"]) &
        ((apt["size_sqm"] > 300) | apt["desc_clean"].str.lower().str.contains("penthouse", na=False))
    ].copy()

    n_suspicious = len(suspicious)
    if n_suspicious > 0:
        # Auto-correct: large size → Villa, penthouse keyword → Penthouse
        for idx, row in suspicious.iterrows():
            desc_lower = str(row.get("desc_clean", "")).lower()
            if "penthouse" in desc_lower:
                df.loc[idx, "type_clean"] = "Penthouse"
            elif row["size_sqm"] > 350:
                df.loc[idx, "type_clean"] = "Villa"
            elif row["size_sqm"] > 250:
                df.loc[idx, "type_clean"] = "Duplex"

        print(f"     Statistical mislabel guard: corrected {n_suspicious} outlier apartments")

    return df


def load_and_clean(csv_path, seed=42):
    """Load raw CSV and apply all cleaning / parsing."""
    df = pd.read_csv(csv_path)
    df["listing_type"] = df["url"].apply(
        lambda x: "rent" if "/plp/rent/" in str(x).lower() else "buy"
    )
    df["price_egp"] = df["price"].apply(_to_float)
    df.dropna(subset=["price_egp"], inplace=True)

    # Conservative price clip (5%-95%)
    q1, q3 = df["price_egp"].quantile(0.05), df["price_egp"].quantile(0.95)
    df = df[(df["price_egp"] >= q1) & (df["price_egp"] <= q3)].copy()
    df["log_price"] = np.log1p(df["price_egp"]).astype(np.float32)

    # Core property attributes
    df["size_sqm"] = df["size"].apply(_parse_sqm)
    df["bedrooms_n"] = df["bedrooms"].apply(
        lambda v: float(re.search(r"(\d+)", str(v)).group(1))
        if re.search(r"(\d+)", str(v)) else np.nan
    )
    df["bathrooms_n"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    df["has_maid"] = (
        df["bedrooms"].fillna("").astype(str).str.lower().str.contains("maid").astype(np.int8)
    )

    # Payment method
    pay = df["payment_method"].fillna("").astype(str).str.lower()
    df["is_installment"] = pay.str.contains(r"install|قسط|تقسيط", regex=True).astype(np.int8)
    df["is_cash"] = pay.str.contains(r"\bcash\b|كاش", regex=True).astype(np.int8)

    # Down payment
    dp = df["down_payment"].apply(_parse_payment) if "down_payment" in df.columns \
         else pd.Series(np.nan, index=df.index)
    dp_ratio = (dp / df["price_egp"]).replace([np.inf, -np.inf], np.nan)
    dp_ratio = dp_ratio.where((dp_ratio > 0) & (dp_ratio < 1.0))
    df["down_payment_ratio"] = dp_ratio.astype(np.float32)
    df["log_down_payment"] = np.log1p(dp.clip(lower=0)).astype(np.float32)

    # Type & description
    df["type_clean"] = df["type"].fillna("Other")
    df["desc_clean"] = df["description"].fillna("").astype(str)
    df["desc_clean"] = df["desc_clean"].str.replace(r"http\S+|www\.\S+", " ", regex=True)
    df["room_ratio"] = df["bathrooms_n"] / df["bedrooms_n"].replace(0.0, np.nan)

    # Location parsing
    loc_df = df["location"].apply(_parse_loc).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), loc_df.reset_index(drop=True)], axis=1)

    # Description-derived simple features
    desc = df["desc_clean"].str.lower()
    yr = desc.str.extract(r"(\d+)\s*year").iloc[:, 0].astype(float)
    df["install_years"] = yr.where((yr >= 2) & (yr <= 25))

    # Compound size proxy
    comp_counts = df["compound"].value_counts()
    df["log_comp_size"] = np.log1p(df["compound"].map(comp_counts)).astype(np.float32)

    # Finish ordinal
    finish = pd.Series(np.nan, index=df.index)
    finish[desc.str.contains(r"core.?shell|bare.?shell|unfinished", regex=True)] = 0.0
    finish[desc.str.contains(r"semi.?finish", regex=True)] = 1.0
    finish[desc.str.contains(r"fully.?finish|full.?finish", regex=True)] = 2.0
    finish[desc.str.contains(r"luxury.?finish|super.?lux", regex=True)] = 3.0
    df["finish_ord"] = finish.values.astype(np.float32)

    # View score
    view = np.zeros(len(df), dtype=np.float32)
    view[desc.str.contains(r"pool.?view", regex=True)] = 1
    view[desc.str.contains(r"garden.?view", regex=True)] = 2
    view[desc.str.contains(r"lagoon", regex=True)] = 3
    view[desc.str.contains(r"sea.?view|ocean.?view|panoramic.*sea", regex=True)] = 4
    view[desc.str.contains(r"golf.?view|golf.?course", regex=True)] = 5
    df["view_score"] = view

    # Market tier & derived size features
    df["market_tier"] = df.apply(
        lambda r: _market_tier(r["governorate"], r["city"], r["compound"]), axis=1
    ).astype(np.float32)

    # ── NEW: Area market zone for all types (not just Villa) ──
    df["area_market_zone"] = df.apply(
        lambda r: _area_market_zone(r["city"], r["compound"], r["desc_clean"]), axis=1
    )

    # ── NEW: Ordinal encoding for area_market_zone (tree-safe, backward-compatible) ──
    _zone_ord_map = {"urban": 0.0, "coastal": 1.0, "red_sea": 2.0}
    df["area_market_zone_ord"] = df["area_market_zone"].map(_zone_ord_map).fillna(0.0).astype(np.float32)

    # ── Compound vs Standalone ──
    df = _extract_compound_features(df)

    # Area elasticity features (critical for inverse PPSM fix)
    df["log_size_sqm"] = np.log1p(df["size_sqm"].clip(lower=1)).astype(np.float32)
    df["size_sqm_inv"] = (1.0 / df["size_sqm"].clip(lower=1)).astype(np.float32)

    df["size_per_bedroom"] = (df["size_sqm"] / df["bedrooms_n"].replace(0.0, np.nan)).astype(np.float32)
    df["size_x_finish"] = (df["size_sqm"].fillna(0) * df["finish_ord"].fillna(0)).astype(np.float32)
    df["size_bucket"] = df["size_sqm"].apply(_size_bucket)

    # ── Fix mislabeled types (keyword-based) ──
    df = _fix_mislabeled_types(df)

    # ── NEW: Statistical mislabel guard (IQR-based) ──
    df = _detect_statistical_mislabels(df)

    # ── NEW: city × compound interaction for target encoding ──
    df["city_comp"] = df["city"] + "_comp" + df["is_compound"].astype(str)
    
    # ── NEW: governorate × compound interaction ──
    df["governorate_comp"] = df["governorate"] + "_comp" + df["is_compound"].astype(str)

    # ── STRICT TYPE FILTER ──
    df, dropped = filter_supported_types(df, type_col="type_clean")
    if dropped > 0:
        print(f"     Dropped {dropped:,} unsupported property types")
        print(f"     Supported: {', '.join(sorted(ALLOWED_TYPES))}")
        print(f"     Dropped include: iVilla, Twin House, Townhouse, Hotel Apartment, Cabin, Land, Bulk Sale")

    # Safety check
    assert set(df["type_clean"].unique()).issubset(ALLOWED_TYPES), \
        f"Unexpected types leaked: {set(df['type_clean'].unique()) - ALLOWED_TYPES}"

    return df, comp_counts.to_dict()