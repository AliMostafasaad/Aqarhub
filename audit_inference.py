"""Aqar Hub — Inference Pipeline Audit Tool
Traces every feature from GUI input to model prediction.
"""
import numpy as np
import pandas as pd
import joblib
import sys

from config import MODEL_PATH
from data import _parse_loc, _market_tier, _extract_compound_features, _area_market_zone
from features import _extract_all_content, _kw_features, _semantic_scores, assemble_X


def audit_al_yasmine():
    """Full trace for the problematic Al Yasmine standalone case."""
    
    print("=" * 72)
    print("  INFERENCE AUDIT — Al Yasmine Standalone")
    print("=" * 72)
    
    # 1. Simulate GUI input
    raw_type = "Apartment"
    price_egp = 5_800_000
    size_sqm = 120.0
    bedrooms = 2.0
    bathrooms = 1.0
    location = "Al Yasmine, New Cairo"
    desc = "Standalone apartment in Al Yasmine residential area, fully finished."
    finish = "Fully Finish"
    view = "None"
    payment = "Cash"
    
    print(f"\n[INPUT]")
    print(f"  Type: {raw_type}")
    print(f"  Location: {location}")
    print(f"  Size: {size_sqm} sqm")
    print(f"  Price: {price_egp:,} EGP")
    print(f"  Desc: {desc[:60]}...")
    
    # 2. Parse location
    loc = _parse_loc(location)
    print(f"\n[LOCATION PARSE]")
    print(f"  compound: '{loc['compound']}'")
    print(f"  city: '{loc['city']}'")
    print(f"  governorate: '{loc['governorate']}'")
    
    # 3. Build minimal dataframe (mimicking gui.py build_dataframe)
    row = {
        "url": "https://example.com/plp/buy/",
        "price": str(price_egp), "price_egp": price_egp,
        "log_price": np.log1p(price_egp),
        "description": desc, "desc_clean": desc,
        "location": location, "type": raw_type, "type_clean": raw_type,
        "size": str(size_sqm), "size_sqm": size_sqm,
        "bedrooms": str(bedrooms), "bedrooms_n": bedrooms,
        "bathrooms": str(bathrooms), "bathrooms_n": bathrooms,
        "has_maid": 0,
        "payment_method": payment.lower(),
        "is_installment": 0, "is_cash": 1,
        "down_payment": "", "down_payment_ratio": np.nan,
        "log_down_payment": np.nan,
        "room_ratio": bathrooms / bedrooms,
        "compound": loc["compound"], "city": loc["city"], "governorate": loc["governorate"],
        "install_years": np.nan,
        "log_comp_size": np.log1p(10),
        "finish_ord": 2.0, "view_score": 0,
        "market_tier": _market_tier(loc["governorate"], loc["city"], loc["compound"]),
        "size_per_bedroom": size_sqm / bedrooms,
        "size_x_finish": size_sqm * 2.0,
        "log_size_sqm": np.log1p(size_sqm),
        "size_sqm_inv": 1.0 / size_sqm,
        "area_market_zone": _area_market_zone(loc["city"], loc["compound"], desc),
    }
    
    df = pd.DataFrame([row])
    
    # 4. Extract compound features (CRITICAL STEP)
    df = _extract_compound_features(df)
    df["city_comp"] = df["city"] + "_comp" + df["is_compound"].astype(str)
    df["governorate_comp"] = df["governorate"] + "_comp" + df["is_compound"].astype(str)
    
    print(f"\n[COMPOUND FEATURES]")
    print(f"  is_compound: {df['is_compound'].iloc[0]}  (0=standalone, 1=compound)")
    print(f"  compound_tier: {df['compound_tier'].iloc[0]}  (0=standalone, 1=mid, 2=premium)")
    print(f"  standalone_premium: {df['standalone_premium'].iloc[0]}  (1=standalone in premium city)")
    print(f"  compound_premium: {df['compound_premium'].iloc[0]}")
    print(f"  city_comp: '{df['city_comp'].iloc[0]}'")
    
    # 5. Load model and trace
    if not sys.path[0].endswith("Aqar hub k"):
        sys.path.insert(0, r"C:\Users\Admin\Downloads\Aqar hub k")
    
    bundle = joblib.load(MODEL_PATH)
    te_maps = bundle["te_maps"]
    
    print(f"\n[TARGET ENCODINGS]")
    for col in ["city", "compound", "city_comp"]:
        if col in te_maps:
            key = df[col].iloc[0]
            val = te_maps[col]["map"].get(key, te_maps[col]["gm"])
            print(f"  {col}_te for '{key}': {val:.4f}")
    
    # 6. Preprocess and predict
    from gui import AqarHubGUI  # Import the GUI class to use its preprocess
    
    # We can't easily instantiate GUI without tkinter, so let's do manual preprocess
    tfidf = bundle["tfidf"]
    svd = bundle["svd"]
    top_feats = bundle["top_feats"]
    train_medians = bundle["train_medians"]
    lo = bundle["clip_lo"]
    hi = bundle["clip_hi"]
    
    v20 = _extract_all_content(df["desc_clean"])
    df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)
    
    # Check if critical features exist
    print(f"\n[FEATURE MATRIX CHECK]")
    for feat in ["is_compound", "compound_tier", "standalone_premium", "city_comp_te", "log_size_sqm"]:
        status = "✅" if feat in df.columns else "❌ MISSING"
        val = df[feat].iloc[0] if feat in df.columns else "N/A"
        print(f"  {status} {feat}: {val}")
    
    print(f"\n{'='*72}")
    print("  AUDIT COMPLETE")
    print(f"{'='*72}")
    print("\n  If 'is_compound' shows 0 and 'standalone_premium' shows 1,")
    print("  the signal is reaching the model correctly.")
    print("  If prediction is still too high, use alert_engine calibration.")


if __name__ == "__main__":
    audit_al_yasmine()