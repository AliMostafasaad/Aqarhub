"""Aqar Hub v22.4c — Training Pipeline (CatBoost Only)
(CLEAN: Added bias-fix features, NO monotonic constraints, NO sample weights)
"""
import warnings
warnings.filterwarnings("ignore")

import gc
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from catboost import CatBoostRegressor

from config import CSV_PATH, MODEL_PATH, TARGET, TEST_SIZE, N_SPLITS, SEED, ALLOWED_TYPES
from data import load_and_clean
from features import (
    _kw_features, _semantic_scores, _extract_all_content,
    kfold_te, kfold_price_per_sqm, assemble_X, build_feature_sets,
)
from models import CatBoostEnsemble


def _catboost_params_v22_4d(seed=42):
    return dict(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,        # ↑ penalize location overfitting
        random_strength=2.0,    # ↑ more split randomness
        bagging_temperature=0.3,
        min_data_in_leaf=40,    # ↑ prevent micro-location buckets
        rsm=0.65,               # ↓ each tree sees fewer features
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )

def main():
    print("=" * 72)
    print("  Aqar Hub v22.4c - CatBoost (Bias Features Only, No Constraints)")
    print("=" * 72)

    # -- 1. Load & clean ---------------------------------------------------
    print("\n[1] Loading data...")
    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: '{CSV_PATH}' not found.")
    df, comp_counts_dict = load_and_clean(CSV_PATH, seed=SEED)
    print(f"     Clean rows: {len(df):,}")
    
    premium = df[df["market_tier"] >= 2.0]
    print(f"     Premium-city listings: {len(premium):,}")
    print(f"       Compound: {(premium['is_compound']==1).sum():,}")
    print(f"       Standalone: {(premium['is_compound']==0).sum():,}")

    # -- 2. Content features -----------------------------------------------
    print("[2] Extracting content signals...")
    v20 = _extract_all_content(df["desc_clean"])
    df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)
    print(f"     Content features extracted: {v20.shape[1]}")

    # -- 3. Train / test split ---------------------------------------------
    print("[3] Train / test split...")
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"     Train={len(df_tr):,}  Test={len(df_te):,}")

    # -- 4. Price-per-sqm benchmark ----------------------------------------
    print("[4] Price-per-sqm benchmark...")
    pps_tr, pps_te, pps_map, pps_gm = kfold_price_per_sqm(df_tr, df_te, seed=SEED)
    df_tr["price_per_sqm_estimate"] = pps_tr
    df_te["price_per_sqm_estimate"] = pps_te

    # -- 5. Target encodings -----------------------------------------------
    print("[5] KFold target encodings...")
    te_maps = {}
    for col, k in [("type_clean", 20), ("governorate", 20), ("city", 20), ("compound", 30)]:
        a, b, m = kfold_te(df_tr, df_te, col, TARGET, k=k, seed=SEED)
        df_tr[col + "_te"] = a
        df_te[col + "_te"] = b
        te_maps[col] = {"map": m, "gm": float(df_tr[TARGET].mean())}

    # city_comp interaction TE (compound-aware location)
    a, b, m = kfold_te(df_tr, df_te, "city_comp", TARGET, k=20, seed=SEED)
    df_tr["city_comp_te"] = a
    df_te["city_comp_te"] = b
    te_maps["city_comp"] = {"map": m, "gm": float(df_tr[TARGET].mean())}

    for combo, k in [("comp_type", 30), ("bed_type", 15), ("nc_type", 10),
                     ("finish_type", 15), ("type_tier", 15)]:
        if combo == "comp_type":
            df_tr["comp_type"] = df_tr["compound"] + "_" + df_tr["type_clean"]
            df_te["comp_type"] = df_te["compound"] + "_" + df_te["type_clean"]
        elif combo == "bed_type":
            df_tr["bed_type"] = df_tr["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df_tr["type_clean"]
            df_te["bed_type"] = df_te["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df_te["type_clean"]
        elif combo == "nc_type":
            for frame in (df_tr, df_te):
                d2 = frame["desc_clean"].str.lower()
                frame["is_nc"] = d2.str.contains(r"north\s*coast|ras\s*el?\s*hekma|sahel", regex=True).astype(np.int8)
                frame["nc_type"] = frame["is_nc"].astype(str) + "_" + frame["type_clean"]
        elif combo == "finish_type":
            df_tr["finish_type"] = df_tr["finish_ord"].fillna(-1).astype(int).astype(str) + "_" + df_tr["type_clean"]
            df_te["finish_type"] = df_te["finish_ord"].fillna(-1).astype(int).astype(str) + "_" + df_te["type_clean"]
        elif combo == "type_tier":
            df_tr["type_tier"] = df_tr["type_clean"] + "_" + df_tr["market_tier"].astype(int).astype(str)
            df_te["type_tier"] = df_te["type_clean"] + "_" + df_te["market_tier"].astype(int).astype(str)
        a, b, m = kfold_te(df_tr, df_te, combo, TARGET, k=k, seed=SEED)
        df_tr[combo + "_te"] = a
        df_te[combo + "_te"] = b
        te_maps[combo] = {"map": m, "gm": float(df_tr[TARGET].mean())}

    # -- 6. NLP ------------------------------------------------------------
    print("[6] NLP pipeline...")
    kw_tr = _kw_features(df_tr["desc_clean"])
    kw_te = _kw_features(df_te["desc_clean"])
    sem_tr = _semantic_scores(df_tr["desc_clean"])
    sem_te = _semantic_scores(df_te["desc_clean"])
    for c in ["luxury_score", "budget_score", "investment_signal"]:
        df_tr[c] = sem_tr[c].values
        df_te[c] = sem_te[c].values

    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2),
                            min_df=5, max_df=0.85, sublinear_tf=True)
    mat_tr = tfidf.fit_transform(df_tr["desc_clean"])
    mat_te = tfidf.transform(df_te["desc_clean"])
    svd = TruncatedSVD(n_components=25, random_state=SEED)
    lsa_tr = svd.fit_transform(mat_tr).astype(np.float32)
    lsa_te = svd.transform(mat_te).astype(np.float32)
    del mat_tr, mat_te
    gc.collect()

    LSA_COLS = [f"lsa_{i}" for i in range(25)]
    lsa_tr_df = pd.DataFrame(lsa_tr, columns=LSA_COLS, index=df_tr.index)
    lsa_te_df = pd.DataFrame(lsa_te, columns=LSA_COLS, index=df_te.index)
    del lsa_tr, lsa_te
    gc.collect()

    # -- 7. Assemble matrices (per-fold to prevent data leakage) ------------
    print("[7] Assembling feature matrices...")
    # NOTE: Global assembly for feature selection ONLY. 
    # Per-fold medians/clips applied inside CV loop to prevent leakage.
    X_train_all, train_medians, lo, hi = assemble_X(df_tr, kw_tr, lsa_tr_df)
    X_test_all = assemble_X(df_te, kw_te, lsa_te_df, train_medians=train_medians, lo=lo, hi=hi)
    y_train = df_tr[TARGET].values.astype(np.float32)
    y_test = df_te[TARGET].values.astype(np.float32)
    del lsa_tr_df, lsa_te_df, kw_tr, kw_te
    gc.collect()

    # -- 8. Feature selection ----------------------------------------------
    print("[8] Feature selection...")
    quick = CatBoostRegressor(
        iterations=300, depth=5, learning_rate=0.08,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    quick.fit(X_train_all, y_train)
    fi = pd.Series(quick.feature_importances_, index=X_train_all.columns)

    MUST_KEEP = [
        "size_sqm", "bedrooms_n", "bathrooms_n",
        "compound_te", "city_te", "governorate_te",
        "type_clean_te", "finish_ord", "view_score",
        "price_per_sqm_estimate", "log_comp_size",
        "market_tier", "size_per_bedroom",
        # Bias-fix features
        "is_compound", "compound_tier", "log_size_sqm", "city_comp_te",
        "standalone_premium", "compound_premium",
        "size_sqm_inv",
    ]
    must_present = [c for c in MUST_KEEP if c in X_train_all.columns]
    top_feats = list(dict.fromkeys(must_present + fi.nlargest(50).index.tolist()))[:55]
    X_train_all = X_train_all[top_feats]
    X_test_all = X_test_all[top_feats]
    print(f"     Selected: {len(top_feats)} features")

    # -- 9. Feature views --------------------------------------------------
    print("[9] Building feature views...")
    _, col_cat = build_feature_sets(top_feats)
    print(f"     CAT (full): {len(col_cat)}")

    # -- 10. CatBoost primary model ----------------------------------------
    print("[10] Training CatBoost (5-fold)...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cat_oof = np.zeros(len(X_train_all), dtype=np.float32)
    test_pred = np.zeros(len(X_test_all), dtype=np.float32)
    cat_models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), 1):
        # Per-fold assembly: calculate medians/clips from training split only
        X_tr_fold = X_train_all.iloc[tr_idx]
        X_va_raw = X_train_all.iloc[va_idx]

        fold_medians = X_tr_fold.median(numeric_only=True)
        fold_lo = X_tr_fold.quantile(0.01)
        fold_hi = X_tr_fold.quantile(0.99)

        X_tr_fold = X_tr_fold.clip(lower=fold_lo, upper=fold_hi, axis=1)
        X_va_fold = X_va_raw.clip(lower=fold_lo, upper=fold_hi, axis=1)

        Xt = X_tr_fold[col_cat]
        Xv = X_va_fold[col_cat]
        yt = y_train[tr_idx]
        yv = y_train[va_idx]

        m = CatBoostRegressor(**_catboost_params_v22_4d(SEED), early_stopping_rounds=100)
        m.fit(Xt, yt, eval_set=(Xv, yv), verbose=False)
        cat_oof[va_idx] = m.predict(Xv).astype(np.float32)
        test_pred += m.predict(X_test_all[col_cat]).astype(np.float32) / N_SPLITS
        cat_models.append(m)
        print(f"     Fold {fold} CatBoost R2 = {r2_score(yv, cat_oof[va_idx]):.4f}")

    print(f"     CatBoost OOF R2 = {r2_score(y_train, cat_oof):.4f}")

    # -- 11. Final evaluation ----------------------------------------------
    print("[11] Final evaluation...")
    ensemble = CatBoostEnsemble(cat_models, col_cat, top_feats)

    pred_log = ensemble.predict(X_test_all)
    pred_price = ensemble.predict_price(X_test_all)
    true_price = np.expm1(y_test)

    te_r2 = r2_score(y_test, pred_log)
    tr_r2 = r2_score(y_train, cat_oof)
    mape = np.mean(np.abs(true_price - pred_price) / true_price) * 100

    print(f"\n{'='*72}")
    print(f"  RESULTS")
    print(f"{'='*72}")
    print(f"  Train R2 = {tr_r2:.4f}")
    print(f"  Test R2  = {te_r2:.4f}")
    print(f"  Gap      = {tr_r2 - te_r2:.4f}")
    print(f"  MAPE     = {mape:.2f}%")
    print(f"{'='*72}")

    # Per-type breakdown
    df_te2 = df_te.copy()
    df_te2["abs_pct"] = np.abs(true_price - pred_price) / true_price * 100
    print("\n  Per-type MAPE:")
    mape_by_type = {}
    for tp, g in df_te2.groupby("type_clean"):
        if len(g) >= 10 and tp in ALLOWED_TYPES:
            m = g["abs_pct"].mean()
            mape_by_type[tp] = m
            print(f"    {tp:22s}  MAPE={m:.1f}%  n={len(g)}")

    unexpected = set(mape_by_type.keys()) - ALLOWED_TYPES
    assert not unexpected, f"Unsupported types leaked to metrics: {unexpected}"

    # Feature importance
    fi_raw = pd.Series(cat_models[0].get_feature_importance(), index=col_cat)
    fi_normalized = fi_raw / fi_raw.sum()
    loc_dominance = fi_normalized[fi_normalized.index.str.contains("te|city|compound|governorate", regex=True)].sum()
    print(f"\n  Location features dominance: {loc_dominance:.1%}")
    if loc_dominance > 0.40:
        print("  WARNING: Location over-dominant")
    else:
        print("  OK Location dominance within acceptable range")

    # Bias Audit
    print("\n  BIAS AUDIT — Standalone vs Compound in Premium Cities:")
    premium_mask = (df_te["market_tier"] >= 2.0)
    for status, label in [(1, "Compound"), (0, "Standalone")]:
        mask = premium_mask & (df_te["is_compound"] == status)
        if mask.sum() >= 5:
            sub_mape = df_te2.loc[mask, "abs_pct"].mean()
            sub_bias = ((pred_price[mask] - true_price[mask]) / true_price[mask]).mean() * 100
            print(f"    {label:12s}  n={mask.sum():4d}  MAPE={sub_mape:.1f}%  Bias={sub_bias:+.1f}%")
            if abs(sub_bias) > 20:
                print(f"      ⚠️ SEVERE BIAS: {label} properties are {'over' if sub_bias > 0 else 'under'}priced by {abs(sub_bias):.0f}%")

    # -- 12. Save ----------------------------------------------------------
    print(f"\n[12] Saving -> {MODEL_PATH}")
    # ── NEW: Rarity map for safety layer ──
    rarity_map = (
        df_tr.groupby(["city", "type_clean"])
        .size()
        .pipe(lambda s: 1.0 / (s + 1.0))
        .to_dict()
    )

    bundle = {
        "model": ensemble,
        "tfidf": tfidf,
        "svd": svd,
        "top_feats": top_feats,
        "col_cat": col_cat,
        "train_medians": train_medians,
        "clip_lo": lo,
        "clip_hi": hi,
        "comp_counts": comp_counts_dict,
        "te_maps": te_maps,
        "pps_map": pps_map,
        "pps_gm": pps_gm,
        "train_median_ppsm": float(np.expm1(pps_gm)),
        "rarity_map": rarity_map,
        "test_r2": float(te_r2),
        "test_mape": float(mape),
        "mape_by_type": mape_by_type,
        "location_dominance": float(loc_dominance),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    sz = os.path.getsize(MODEL_PATH) / 1e6
    print(f"  Saved: {MODEL_PATH} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()