"""Aqar Hub v22.4c — FastAPI Production Inference Server
Tailored for Aqar Hub Mobile App Flow with Pure Arabic UI Formatting.
"""
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Flat imports (same directory) ──
from config import MODEL_PATH, ALLOWED_TYPES
from data import _parse_loc, _market_tier, _area_market_zone, _extract_compound_features
from features import _extract_all_content, _kw_features, _semantic_scores, assemble_X
from alert_engine import make_decision
from type_validator import normalize_property_type, TypeValidationError

# Global Model Bundle
model_bundle: dict = {}

# قواميس الترجمة والربط بين واجهة التطبيق والموديل
PROPERTY_TYPE_MAPPING = {
    "شقة": "Apartment",
    "فيلا": "Villa",
    "ستوديو": "Apartment",  
    "بنتهاوس": "Penthouse",
    "دوبلكس": "Duplex",
    "شاليه": "Chalet"
}

AMENITIES_TRANSLATION = {
    "مصعد": "elevator",
    "أمن وحراسة": "security",
    "صالة رياضية": "gym",
    "حمام سباحة": "pool",
    "تكييف هواء": "air conditioning",
    "شرفة": "balcony",
    "حديقة": "garden",
    "غرفة تخزين": "storage"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        script_dir = Path(__file__).parent
        alt_path = script_dir / MODEL_PATH
        if alt_path.exists():
            model_path = alt_path
        else:
            raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    model_bundle = joblib.load(model_path)
    print(f"✅ Model loaded successfully: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    yield
    model_bundle.clear()


app = FastAPI(
    title="Aqar Hub — Mobile Production API",
    version="22.4c",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MobileValuationRequest(BaseModel):
    property_type: str = Field(..., examples=["شقة"]) 
    governorate: str = Field(..., examples=["الجيزة"])
    city: str = Field(..., examples=["الدقي"])
    detailed_address: Optional[str] = Field(default="", examples=["الدقي خلف نادي الصيد"])
    bedrooms: str = Field(..., examples=["3"])  
    bathrooms: str = Field(..., examples=["1"]) 
    size_sqm: float = Field(..., gt=0, examples=[150.0])
    amenities: List[str] = Field(default=[], examples=[["مصعد", "حمام سباحة", "أمن وحراسة"]])
    asking_price: float = Field(..., gt=0, examples=[12000.0])
    description: Optional[str] = Field(default="")


class ValuationResponse(BaseModel):
    alert: str
    reason: str


def _build_dataframe(property_type_en: str, location_combined: str, size_sqm: float, 
                     beds: int, baths: int, full_desc: str, asking_price: float) -> pd.DataFrame:
    loc = _parse_loc(location_combined)
    payment = "cash"
    
    market_tier = _market_tier(loc["governorate"], loc["city"], loc["compound"])
    comp_counts = model_bundle.get("comp_counts", {})
    log_comp_size = np.log1p(comp_counts.get(loc["compound"], 10))

    row = {
        "url": "https://example.com/", "price": str(asking_price), "price_egp": asking_price, "log_price": np.log1p(asking_price),
        "description": full_desc, "desc_clean": full_desc, "location": location_combined, "type": property_type_en,
        "type_clean": property_type_en, "size": str(size_sqm), "size_sqm": size_sqm,
        "bedrooms": str(beds), "bedrooms_n": float(beds), "bathrooms": str(baths), "bathrooms_n": float(baths),
        "has_maid": 1 if "maid" in full_desc.lower() else 0, "payment_method": payment, "is_installment": 0, "is_cash": 1,
        "down_payment_ratio": np.nan, "log_down_payment": np.nan, "room_ratio": baths / beds if beds > 0 else np.nan,
        "compound": loc["compound"], "city": loc["city"], "governorate": loc["governorate"], "install_years": np.nan,
        "log_comp_size": log_comp_size, "finish_ord": 2.0, "view_score": 2, "market_tier": market_tier,
        "size_per_bedroom": size_sqm / beds if beds > 0 else np.nan, "size_x_finish": size_sqm * 2.0, "log_size_sqm": np.log1p(size_sqm),
        "size_sqm_inv": 1.0 / size_sqm, "area_market_zone": 0.0
    }

    df = pd.DataFrame([row])
    df = _extract_compound_features(df)
    df["city_comp"] = df["city"] + "_comp" + df["is_compound"].astype(str)
    df["governorate_comp"] = df["governorate"] + "_comp" + df["is_compound"].astype(str)
    return df


def _preprocess(df: pd.DataFrame) -> tuple:
    bundle = model_bundle
    te_maps = bundle["te_maps"]
    v20 = _extract_all_content(df["desc_clean"])
    df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)

    df["price_per_sqm_estimate"] = df[["city", "type_clean"]].apply(
        lambda r: bundle["pps_map"].get((r["city"], r["type_clean"]), bundle["pps_gm"]), axis=1
    )

    for col in ["type_clean", "governorate", "city", "compound"]:
        df[col + "_te"] = df[col].map(te_maps[col]["map"]).fillna(te_maps[col]["gm"])

    if "city_comp" in te_maps:
        df["city_comp_te"] = df["city_comp"].map(te_maps["city_comp"]["map"]).fillna(te_maps["city_comp"]["gm"])
    else:
        df["city_comp_te"] = te_maps.get("type_clean", {}).get("gm", 0.0)

    if "governorate_comp" in te_maps:
        df["governorate_comp_te"] = df["governorate_comp"].map(te_maps["governorate_comp"]["map"]).fillna(te_maps["governorate_comp"]["gm"])
    else:
        df["governorate_comp_te"] = te_maps.get("type_clean", {}).get("gm", 0.0)

    df["comp_type"] = df["compound"] + "_" + df["type_clean"]
    df["bed_type"] = df["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df["type_clean"]
    df["is_nc"] = df["desc_clean"].str.lower().str.contains(r"north\s*coast|sahel", regex=True).astype(np.int8)
    df["nc_type"] = df["is_nc"].astype(str) + "_" + df["type_clean"]
    df["finish_type"] = "2_" + df["type_clean"]
    df["type_tier"] = df["type_clean"] + "_" + df["market_tier"].astype(int).astype(str)

    for combo in ["comp_type", "bed_type", "nc_type", "finish_type", "type_tier"]:
        df[combo + "_te"] = df[combo].map(te_maps[combo]["map"]).fillna(te_maps[combo]["gm"])

    kw = _kw_features(df["desc_clean"])
    sem = _semantic_scores(df["desc_clean"])
    for c in ["luxury_score", "budget_score", "investment_signal"]:
        df[c] = sem[c].values

    mat = bundle["tfidf"].transform(df["desc_clean"])
    lsa = bundle["svd"].transform(mat).astype(np.float32)
    lsa_df = pd.DataFrame(lsa, columns=[f"lsa_{i}" for i in range(lsa.shape[1])], index=df.index)

    X = assemble_X(df, kw, lsa_df, train_medians=bundle["train_medians"], lo=bundle["clip_lo"], hi=bundle["clip_hi"])
    return X[bundle["top_feats"]], df


@app.post("/api/v1/valuation/analyze", response_model=ValuationResponse)
def analyze_property(req: MobileValuationRequest):
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model bundle not initialized.")

    property_type_en = PROPERTY_TYPE_MAPPING.get(req.property_type, "Apartment")
    location_combined = f"{req.city}, {req.governorate}"
    if req.detailed_address:
        location_combined = f"{req.detailed_address}, {location_combined}"

    try:
        beds = int(req.bedrooms) if req.bedrooms.isdigit() else 2
        baths = int(req.bathrooms) if req.bathrooms.isdigit() else 1
    except Exception:
        beds, baths = 2, 1

    english_amenities = [AMENITIES_TRANSLATION[a] for a in req.amenities if a in AMENITIES_TRANSLATION]
    amenities_text = " ".join(english_amenities)
    full_description = f"{req.description or ''} {amenities_text}".strip()

    try:
        df_input = _build_dataframe(property_type_en, location_combined, req.size_sqm, beds, baths, full_description, req.asking_price)
        X, df_proc = _preprocess(df_input)

        model = model_bundle["model"]
        lo_price, med_price, hi_price = model.predict_price_interval(X)

        decision = make_decision(
            actual_price=req.asking_price, predicted_price=med_price[0], lo_price=lo_price[0],
            hi_price=hi_price[0], property_type=property_type_en, area=req.city,
            features={"desc": full_description, "in_compound": "no"}, size_sqm=req.size_sqm, bundle=model_bundle
        )

        # ── 🔥 الفلتر القاطع لعزل النص الإنجليزي تماماً بناءً على آخر حرف إنجليزي ──
        raw_reason = decision.get("reason", "")
        
        # إيجاد مواضع كل الحروف الإنجليزية في السطر
        eng_letters = [m.end() for m in re.finditer(r"[a-zA-Z]", raw_reason)]
        
        if eng_letters:
            last_eng_idx = max(eng_letters)
            # قطع النص الإنجليزي بالكامل وتنظيف الرموز الشاردة من اليسار لليمين
            clean_arabic_reason = raw_reason[last_eng_idx:].lstrip(". -–\t\n")
        else:
            clean_arabic_reason = raw_reason

        # تنظيف نهائي لعلامات السطور لضمان استقامة الـ RTL داخل واجهة التطبيق
        clean_arabic_reason = clean_arabic_reason.replace("\n", " ").strip()

        return ValuationResponse(
            alert=decision["alert"],
            reason=clean_arabic_reason
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


try:
    from a2wsgi import ASGIMiddleware
    wsgi_app = ASGIMiddleware(app)
except ImportError:
    wsgi_app = app

if __name__ == "__main__":
    import uvicorn
    import os
    #Railway يقص الـ بورت الديناميكي من السيرفر، ولو مش موجود يشغل 8000 محلياً
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    