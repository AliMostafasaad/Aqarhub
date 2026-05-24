"""Aqar Hub - Per-Type Inference Engine v22.3

Routes each property type to its dedicated model.
Usage:
    from per_type_inference import PerTypeInferenceEngine
    engine = PerTypeInferenceEngine("models_per_type/manifest.joblib")
    result = engine.predict(raw_type="Villa", ...)
"""
import os
import re
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Tuple

from config import ALLOWED_TYPES
from data import (
    _parse_sqm, _parse_payment, _parse_loc, _market_tier,
    _extract_compound_features, _size_bucket
)
from features import (
    _extract_all_content, _kw_features, _semantic_scores, assemble_X
)
from alert_engine import make_decision, check_down_payment
from type_validator import normalize_property_type, TypeValidationError


class PerTypeInferenceEngine:
    """Routes predictions to type-specific models."""

    def __init__(self, manifest_path: str = "models_per_type/manifest.joblib"):
        self.manifest = joblib.load(manifest_path)
        self.models = {}
        for prop_type, model_path in self.manifest["models"].items():
            self.models[prop_type] = joblib.load(model_path)
        print(f"Loaded {len(self.models)} type-specific models: {list(self.models.keys())}")

    def build_dataframe(self, raw_type: str, price_egp: float, size_sqm: float,
                        bedrooms: float, bathrooms: float, location: str,
                        desc: str, finish: str = "Unknown", view: str = "None",
                        payment: str = "Cash", down_payment: Optional[float] = None) -> pd.DataFrame:
        """Same as InferenceEngine.build_dataframe - builds full feature row."""
        prop_type = normalize_property_type(raw_type)
        loc = _parse_loc(location)
        compound, city, governorate = loc["compound"], loc["city"], loc["governorate"]

        finish_map = {"Unknown": np.nan, "Core & Shell": 0.0, "Semi-Finish": 1.0,
                      "Fully Finish": 2.0, "Luxury": 3.0}
        finish_ord = finish_map.get(finish, np.nan)
        view_map = {"None": 0, "Pool": 1, "Garden": 2, "Lagoon": 3, "Sea": 4, "Golf": 5}
        view_score = view_map.get(view, 0)

        payment_lower = payment.lower()
        is_installment = 1 if "install" in payment_lower else 0
        is_cash = 1 if "cash" in payment_lower else 0

        market_tier = _market_tier(governorate, city, compound)
        comp_counts = {}  # Will be loaded from model bundle
        log_comp_size = np.log1p(10)
        room_ratio = bathrooms / bedrooms if bedrooms > 0 else np.nan
        size_per_bedroom = size_sqm / bedrooms if bedrooms > 0 else np.nan
        size_x_finish = (size_sqm if pd.notna(size_sqm) else 0) * (finish_ord if pd.notna(finish_ord) else 0)
        log_size_sqm = np.log1p(max(size_sqm, 1))
        size_sqm_inv = 1.0 / max(size_sqm, 1)
        size_bucket = _size_bucket(size_sqm)

        down_payment_ratio = np.nan
        log_down_payment = np.nan
        if down_payment is not None and price_egp > 0:
            dpr = down_payment / price_egp
            if 0 < dpr < 1.0:
                down_payment_ratio = dpr
                log_down_payment = np.log1p(down_payment)

        desc_lower = desc.lower()
        yr_match = re.search(r"(\d+)\s*year", desc_lower)
        install_years = float(yr_match.group(1)) if yr_match else np.nan

        row = {
            "url": "https://example.com/plp/buy/",
            "price": str(price_egp), "price_egp": price_egp,
            "log_price": np.log1p(price_egp),
            "description": desc, "desc_clean": desc,
            "location": location, "type": raw_type, "type_clean": prop_type,
            "size": str(size_sqm), "size_sqm": size_sqm,
            "bedrooms": str(bedrooms), "bedrooms_n": bedrooms,
            "bathrooms": str(bathrooms), "bathrooms_n": bathrooms,
            "has_maid": 1 if "maid" in desc_lower else 0,
            "payment_method": payment_lower,
            "is_installment": is_installment, "is_cash": is_cash,
            "down_payment": str(down_payment) if down_payment else "",
            "down_payment_ratio": down_payment_ratio,
            "log_down_payment": log_down_payment,
            "room_ratio": room_ratio,
            "compound": compound, "city": city, "governorate": governorate,
            "install_years": install_years,
            "log_comp_size": log_comp_size,
            "finish_ord": finish_ord, "view_score": view_score,
            "market_tier": market_tier,
            "size_per_bedroom": size_per_bedroom,
            "size_x_finish": size_x_finish,
            "log_size_sqm": log_size_sqm,
            "size_sqm_inv": size_sqm_inv,
            "size_bucket": size_bucket,
        }

        df = pd.DataFrame([row])
        df = _extract_compound_features(df)
        df["city_comp"] = df["city"] + "_comp" + df["is_compound"].astype(str)
        return df

    def preprocess(self, df: pd.DataFrame, bundle: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess using a specific model's bundle."""
        tfidf = bundle["tfidf"]
        svd = bundle["svd"]
        te_maps = bundle["te_maps"]
        pps_map = bundle["pps_map"]
        pps_gm = bundle["pps_gm"]
        train_medians = bundle["train_medians"]
        lo = bundle["clip_lo"]
        hi = bundle["clip_hi"]
        top_feats = bundle["top_feats"]

        v20 = _extract_all_content(df["desc_clean"])
        df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)

        df["price_per_sqm_estimate"] = df[["city", "type_clean"]].apply(
            lambda r: pps_map.get((r["city"], r["type_clean"]), pps_gm), axis=1
        )

        for col in ["type_clean", "governorate", "city", "compound", "city_comp"]:
            if col + "_te" in top_feats and col in te_maps:
                m = te_maps[col]["map"]
                gm = te_maps[col]["gm"]
                df[col + "_te"] = df[col].map(m).fillna(gm)

        df["comp_type"] = df["compound"] + "_" + df["type_clean"]
        df["bed_type"] = df["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df["type_clean"]
        d2 = df["desc_clean"].str.lower()
        df["is_nc"] = d2.str.contains(r"north\s*coast|ras\s*el?\s*hekma|sahel", regex=True).astype(np.int8)
        df["nc_type"] = df["is_nc"].astype(str) + "_" + df["type_clean"]
        df["finish_type"] = df["finish_ord"].fillna(-1).astype(int).astype(str) + "_" + df["type_clean"]
        df["type_tier"] = df["type_clean"] + "_" + df["market_tier"].astype(int).astype(str)

        for combo in ["comp_type", "bed_type", "nc_type", "finish_type", "type_tier"]:
            if combo + "_te" in top_feats and combo in te_maps:
                m = te_maps[combo]["map"]
                gm = te_maps[combo]["gm"]
                df[combo + "_te"] = df[combo].map(m).fillna(gm)

        kw = _kw_features(df["desc_clean"])
        sem = _semantic_scores(df["desc_clean"])
        for c in ["luxury_score", "budget_score", "investment_signal"]:
            df[c] = sem[c].values

        mat = tfidf.transform(df["desc_clean"])
        lsa = svd.transform(mat).astype(np.float32)
        lsa_df = pd.DataFrame(lsa, columns=[f"lsa_{i}" for i in range(lsa.shape[1])], index=df.index)

        X = assemble_X(df, kw, lsa_df, train_medians=train_medians, lo=lo, hi=hi)
        return X[top_feats], df

    def predict(self, raw_type: str, price_egp: float, size_sqm: float,
                bedrooms: float, bathrooms: float, location: str,
                desc: str, finish: str = "Unknown", view: str = "None",
                payment: str = "Cash", down_payment: Optional[float] = None) -> Dict[str, Any]:
        """Route to type-specific model and return prediction + decision."""
        prop_type = normalize_property_type(raw_type)

        if prop_type not in self.models:
            raise TypeValidationError(f"No dedicated model for type: {prop_type}")

        bundle = self.models[prop_type]
        df = self.build_dataframe(raw_type, price_egp, size_sqm, bedrooms, bathrooms,
                                   location, desc, finish, view, payment, down_payment)
        X, df_proc = self.preprocess(df, bundle)

        model = bundle["model"]
        lo_price, med_price, hi_price = model.predict_price_interval(X)

        area = df_proc.iloc[0].get("city", "Unknown")
        if area == "Unknown":
            area = location.split(",")[-1].strip() or "Unknown"

        desc_lower = desc.lower()
        has_pool = "pool" in desc_lower or "swimming" in desc_lower
        premium_devs = ["palm hills", "sodic", "orascom", "emaar",
                       "mountain view", "la vista", "hyde park"]
        developer_level = "premium" if any(d in desc_lower for d in premium_devs) else "unknown"

        features = {
            "land_size": "present" if pd.notna(df.iloc[0].get("size_sqm")) else "missing",
            "in_compound": "yes" if df_proc.iloc[0].get("is_compound", 0) == 1 else "no",
            "has_pool": "yes" if has_pool else "no",
            "developer_level": developer_level,
            "desc": desc,
            "compound": df_proc.iloc[0].get("compound", ""),
        }

        decision = make_decision(
            actual_price=price_egp,
            predicted_price=float(med_price[0]),
            lo_price=float(lo_price[0]),
            hi_price=float(hi_price[0]),
            property_type=prop_type,
            area=area,
            features=features,
        )

        dp_warning = check_down_payment(price_egp, float(med_price[0]))

        return {
            "predicted_price": float(med_price[0]),
            "price_interval": {"low": float(lo_price[0]), "high": float(hi_price[0])},
            "alert": decision["alert"],
            "reason": decision["reason"],
            "confidence": decision["confidence"],
            "confidence_score": decision["confidence_score"],
            "down_payment_warning": dp_warning,
            "property_type": prop_type,
            "model_type": "per_type",
            "model_mape": bundle.get("test_mape", None),
        }