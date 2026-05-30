# Aqar Hub 🏠

An automated real estate valuation system for the Egyptian property market. It uses a machine learning model to evaluate property prices and instantly determine if an asking price is a "FAIR price", "GOOD DEAL", or "OVERPRICED".

## 🔗 Live Links
- **API Docs (Swagger):** https://aqarhub-production-29e5.up.railway.app/docs
- **Base URL:** https://aqarhub-production-29e5.up.railway.app

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **AI Engine:** CatBoost, Joblib, Pandas

## 📂 Project Structure
- `main.py`: Core FastAPI backend and deployment code.
- `aqar_hub_v22_model.joblib`: The trained CatBoost valuation model (v22).
- `requirements.txt`: Project dependencies.
