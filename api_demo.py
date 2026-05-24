"""Aqar Hub — API Demo (Exact Production Output)"""
import numpy as np
import pandas as pd
import joblib
import json

from config import CSV_PATH, MODEL_PATH, SEED
from data import load_and_clean
from features import (
    _extract_all_content, _kw_features, _semantic_scores, assemble_X
)
from alert_engine import get_alert, compute_coverage_score


def extract_coverage_from_row(row):
    """Build coverage dict from processed dataframe row."""
    return {
        "building_age": pd.notna(row.get("building_age")),
        "floor_level": pd.notna(row.get("floor_level")),
        "amenities": any(row.get(c, 0) for c in [
            "has_elevator", "has_garage", "has_security",
            "has_balcony", "has_ac", "has_gas", "has_storage"
        ]),
        "condition_score": pd.notna(row.get("condition_score")),
        "view_score": row.get("view_score", 0) > 0,
        "financial_info": pd.notna(row.get("down_payment_ratio")) or
                          row.get("is_installment", 0) == 1 or
                          row.get("is_cash", 0) == 1,
    }


def preprocess_for_inference(df, bundle):
    """
    Transform raw dataframe into model-ready feature matrix.
    Mirrors training pipeline (v20 content + NLP + TE + assembly).
    """
    tfidf = bundle["tfidf"]
    svd = bundle["svd"]
    te_maps = bundle["te_maps"]
    pps_map = bundle["pps_map"]
    pps_gm = bundle["pps_gm"]
    train_medians = bundle["train_medians"]
    lo = bundle["clip_lo"]
    hi = bundle["clip_hi"]
    top_feats = bundle["top_feats"]

    # v20 content extraction
    v20 = _extract_all_content(df["desc_clean"])
    df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)

    # Price-per-sqm benchmark
    df["price_per_sqm_estimate"] = df[["city", "type_clean"]].apply(
        lambda r: pps_map.get((r["city"], r["type_clean"]), pps_gm), axis=1
    )

    # Target encodings
    for col in ["type_clean", "governorate", "city", "compound"]:
        m = te_maps[col]["map"]
        gm = te_maps[col]["gm"]
        df[col + "_te"] = df[col].map(m).fillna(gm)

    # Combo target encodings
    df["comp_type"] = df["compound"] + "_" + df["type_clean"]
    df["bed_type"] = df["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df["type_clean"]
    d2 = df["desc_clean"].str.lower()
    df["is_nc"] = d2.str.contains(
        r"north\s*coast|ras\s*el?\s*hekma|sahel", regex=True
    ).astype(np.int8)
    df["nc_type"] = df["is_nc"].astype(str) + "_" + df["type_clean"]
    df["finish_type"] = df["finish_ord"].fillna(-1).astype(int).astype(str) + "_" + df["type_clean"]
    df["type_tier"] = df["type_clean"] + "_" + df["market_tier"].astype(int).astype(str)

    for combo in ["comp_type", "bed_type", "nc_type", "finish_type", "type_tier"]:
        m = te_maps[combo]["map"]
        gm = te_maps[combo]["gm"]
        df[combo + "_te"] = df[combo].map(m).fillna(gm)

    # NLP features
    kw = _kw_features(df["desc_clean"])
    sem = _semantic_scores(df["desc_clean"])
    for c in ["luxury_score", "budget_score", "investment_signal"]:
        df[c] = sem[c].values

    # TF-IDF + SVD
    mat = tfidf.transform(df["desc_clean"])
    lsa = svd.transform(mat).astype(np.float32)
    lsa_df = pd.DataFrame(
        lsa, columns=[f"lsa_{i}" for i in range(lsa.shape[1])], index=df.index
    )

    # Assemble final matrix
    X = assemble_X(df, kw, lsa_df, train_medians=train_medians, lo=lo, hi=hi)
    return X[top_feats], df


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    df, _ = load_and_clean(CSV_PATH)
    samples = df.sample(5, random_state=SEED)
    true_prices = samples["price_egp"].values

    X, df_proc = preprocess_for_inference(samples, bundle)
    lo_price, med_price, hi_price = model.predict_price_interval(X)

    print("\n" + "=" * 80)
    print("  AQAR HUB v22.0 — PRODUCTION OUTPUT DEMO")
    print("=" * 80)

    for i, (idx, row) in enumerate(samples.iterrows()):
        width = (hi_price[i] - lo_price[i]) / med_price[i] if med_price[i] > 0 else 0.0
        coverage = compute_coverage_score(extract_coverage_from_row(row))
        result = get_alert(
            user_price=true_prices[i],
            predicted_price=med_price[i],
            prop_type=row["type_clean"],
            width=width,
            coverage_score=coverage,
            p10=lo_price[i],
            p90=hi_price[i],
        )

        print(f"\n  Sample {i+1}: {row['type_clean']} | {row['compound']}")
        print("  " + "-" * 76)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()