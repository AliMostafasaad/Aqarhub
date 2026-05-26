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

# تفعيل الـ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تحميل موديل الـ Machine Learning مع معالجة الـ Dictionary ذكياً
MODEL_PATH = "aqar_hub_v22_model.joblib"
model = None

try:
    if os.path.exists(MODEL_PATH):
        loaded_object = joblib.load(MODEL_PATH)
        
        # إذا كان الملف المرفوع عبارة عن Dictionary يحتوي على الموديل بداخله
        if isinstance(loaded_object, dict):
            print(f"ℹ️ Loaded object is a dictionary. Keys found: {list(loaded_object.keys())}")
            
            # محاولة استخراج الموديل بأشهر الأسماء الشائعة
            if 'model' in loaded_object:
                model = loaded_object['model']
            elif 'catboost' in loaded_object:
                model = loaded_object['catboost']
            else:
                # البحث التلقائي الذكي عن أي عنصر جوه الـ dict عنده دالة predict
                for key, value in loaded_object.items():
                    if hasattr(value, 'predict'):
                        model = value
                        print(f"🎯 Automatically extracted model from key: '{key}'")
                        break
        else:
            # لو الموديل متسيف لوحده كـ Object عادي من الأول
            model = loaded_object
            
        if model is not None:
            print(f"✅ Model loaded successfully: {MODEL_PATH}")
        else:
            print("❌ Error: Loaded file is a dictionary but no model object with '.predict' was found inside it.")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Using dummy predictor for fallback.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None


# 3. دالة تنظيف المعالجة اللغوية (تطبق على المدخلات فقط)
def clean_arabic_description(text: str) -> str:
    if not text:
        return ""
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "v22"
    }


@app.post("/api/v1/valuation/analyze", tags=["Valuation"])
def analyze_property(payload: ValuationRequest):
    try:
        cleaned_desc = clean_arabic_description(payload.description)
        
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

        # التنبؤ من خلال الموديل المستخرج
        if model is not None:
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])
        else:
            # حساب افتراضي احتياطي لحماية السيرفر من السقوط
            predicted_price = payload.size_sqm * 35000 

        min_range = round((predicted_price * 0.9) / 1_000_000, 2)
        max_range = round((predicted_price * 1.1) / 1_000_000, 2)

        asking_price = payload.asking_price
        
        if asking_price < predicted_price * 0.85:
            alert_status = "GOOD DEAL"
        elif asking_price > predicted_price * 1.15:
            alert_status = "OVERPRICED"
        else:
            alert_status = "FAIR"

        response_reason = f"السعر ضمن النطاق المتوقع ({min_range}M-{max_range}M) للعقارات المشابهة."

        return {
            "alert": alert_status,
            "reason": response_reason
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)