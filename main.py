import os
import re
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Union

# 1. إعداد تطبيق FastAPI
app = FastAPI(
    title="Aqar Hub Valuation API",
    description="Production-grade API for automated real estate valuation in Egypt (v22)",
    version="22.0.0"
)

# تفعيل الـ CORS عشان الأبليكيشن يقدر يكلم السيرفر بدون قيود
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تحميل موديل الـ Machine Learning (CatBoost Engine v22)
MODEL_PATH = "aqar_hub_v22_model.joblib"

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully: {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Using dummy predictor for fallback.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None


# 3. دالة تنظيف المعالجة اللغوية (تطبق على المدخلات فقط وليس المخرجات)
def clean_arabic_description(text: str) -> str:
    if not text:
        return ""
    # إزالة الروابط، الإيموجيز، وحروف الإنجليزي لتنظيف النص للموديل
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return " ".join(text.split())


# 4. بناء الـ Pydantic Schema لبيانات العقار المدخلة
class ValuationRequest(BaseModel):
    property_type: str = Field(..., example="شقة")
    governorate: str = Field(..., example="الجيزة")
    city: str = Field(..., example="الدقي")
    detailed_address: str = Field(..., example="الدقي خلف نادي الصيد")
    bedrooms: Union[int, str] = Field(..., example=3)
    bathrooms: Union[int, str] = Field(..., example=2)
    size_sqm: float = Field(..., example=150.0)
    amenities: List[str] = Field(default=[], example=["مصعد", "غاز طبيعي", "أمن وحراسة"])
    asking_price: float = Field(..., example=5500000.0)
    description: str = Field(default="", example="شقة لقطة للبيع في الدقي...")


# 5. الـ Endpoints الرئيسية

@app.get("/health", tags=["System"])
def health_check():
    """لفحص حالة السيرفر والتأكد إن الموديل قايم"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "v22"
    }


@app.post("/api/v1/valuation/analyze", tags=["Valuation"])
def analyze_property(payload: ValuationRequest):
    """استقبال بيانات العقار وتحديد هل السعر لقطة، عادل، أو غالي"""
    try:
        # أ) معالجة وتنظيف الوصف المدخل فقط
        cleaned_desc = clean_arabic_description(payload.description)
        
        # ب) تحويل البيانات لـ DataFrame متوافق مع الموديل المدرب
        # (قم بتعديل الأسماء لتطابق الأعمدة الحقيقية للموديل عندك إذا لزم الأمر)
        input_data = pd.DataFrame([{
            "property_type": payload.property_type,
            "governorate": payload.governorate,
            "city": payload.city,
            "detailed_address": payload.detailed_address,
            "bedrooms": int(payload.bedrooms),
            "bathrooms": int(payload.bathrooms),
            "size_sqm": payload.size_sqm,
            "amenities_count": len(payload.amenities),
            "cleaned_description": cleaned_desc
        }])

        # ج) التنبؤ بالسعر العادل من خلال الموديل
        if model is not None:
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])
        else:
            # حساب افتراضي احتياطي في حال عدم وجود ملف الموديل أثناء التست
            predicted_price = payload.size_sqm * 35000 

        # د) حساب النطاق السعري المتوقع (+/- 10%) وتحويله للملايين (M)
        min_range = round((predicted_price * 0.9) / 1_000_000, 2)
        max_range = round((predicted_price * 1.1) / 1_000_000, 2)

        # هـ) تطبيق شروط الـ Thresholds لتحديد الـ Alert
        asking_price = payload.asking_price
        
        if asking_price < predicted_price * 0.85:
            alert_status = "GOOD DEAL"
        elif asking_price > predicted_price * 1.15:
            alert_status = "OVERPRICED"
        else:
            alert_status = "FAIR"

        # و) صياغة النص العربي الصافي مع الحفاظ التام على أرقام وحروف النطاق السعري
        response_reason = f"السعر ضمن النطاق المتوقع ({min_range}M-{max_range}M) للعقارات المشابهة."

        return {
            "alert": alert_status,
            "reason": response_reason
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# 6. معالج البورت الديناميكي الخاص بـ Railway لتشغيل السيرفر
if __name__ == "__main__":
    import uvicorn
    # لقط البورت المتغير اللي بتفرضه منصة Railway تلقائياً
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)