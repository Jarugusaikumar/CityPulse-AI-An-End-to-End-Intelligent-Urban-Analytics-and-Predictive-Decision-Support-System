"""
CityPulse AI - FastAPI service (Section 94).

Run with:
    uvicorn api.main:app --reload

Endpoints:
    POST /predict     - congestion classification
    POST /forecast     - traffic forecast (returns saved SARIMA forecast for a zone)
    POST /cluster       - zone cluster lookup
    POST /sentiment     - complaint category/sentiment/priority
    POST /anomaly       - check whether a given reading looks anomalous
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.nlp.complaint_nlp import classify_complaint  # noqa: E402

MODELS_DIR = ROOT / "models"

app = FastAPI(title="CityPulse AI API", version="1.0")

_congestion_bundle = None
_energy_bundle = None
_cluster_df = None
_anomaly_bundle = None


def _load_models():
    global _congestion_bundle, _energy_bundle, _cluster_df, _anomaly_bundle
    if _congestion_bundle is None and (MODELS_DIR / "congestion_model.pkl").exists():
        _congestion_bundle = joblib.load(MODELS_DIR / "congestion_model.pkl")
    if _energy_bundle is None and (MODELS_DIR / "energy_regression_model.pkl").exists():
        _energy_bundle = joblib.load(MODELS_DIR / "energy_regression_model.pkl")
    if _cluster_df is None and (MODELS_DIR / "zone_clusters.csv").exists():
        _cluster_df = pd.read_csv(MODELS_DIR / "zone_clusters.csv")
    if _anomaly_bundle is None and (MODELS_DIR / "anomaly_model.pkl").exists():
        _anomaly_bundle = joblib.load(MODELS_DIR / "anomaly_model.pkl")


class PredictRequest(BaseModel):
    vehicle_count: float
    average_speed: float
    rainfall: float
    temperature: float
    is_peak: int
    is_weekend: int
    day_of_week: int
    AQI: float
    energy_consumption: float
    road_type: str = "Highway"


class ForecastRequest(BaseModel):
    zone_id: str = "Z1"


class ClusterRequest(BaseModel):
    zone_id: str


class SentimentRequest(BaseModel):
    text: str


class AnomalyRequest(BaseModel):
    vehicle_count: float
    average_speed: float
    AQI: float
    energy_consumption: float


@app.get("/")
def root():
    return {"service": "CityPulse AI API", "status": "ok"}


@app.post("/predict")
def predict_congestion(req: PredictRequest):
    _load_models()
    if _congestion_bundle is None:
        return {"error": "Model not trained yet. Run src/models/classification.py first."}

    pipeline = _congestion_bundle["pipeline"]
    X = pd.DataFrame([req.dict()])
    X = X[_congestion_bundle["features_num"] + _congestion_bundle["features_cat"]]

    if _congestion_bundle["name"] == "XGBoost":
        code = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        classes = _congestion_bundle["classes"]
        prediction = classes[int(code)]
        probability = float(np.max(proba))
    else:
        prediction = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        probability = float(np.max(proba))

    return {"predicted_congestion": str(prediction), "probability": round(probability, 4)}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    path = MODELS_DIR / f"forecast_{req.zone_id}.csv"
    if not path.exists():
        return {"error": f"No forecast available for zone {req.zone_id}. "
                          f"Run src/forecasting/forecast.py for that zone first."}
    df = pd.read_csv(path)
    return {"zone_id": req.zone_id, "forecast": df.to_dict(orient="records")}


@app.post("/cluster")
def cluster(req: ClusterRequest):
    _load_models()
    if _cluster_df is None:
        return {"error": "Clustering not run yet. Run src/models/clustering.py first."}
    row = _cluster_df[_cluster_df["zone_id"] == req.zone_id]
    if row.empty:
        return {"error": f"Zone {req.zone_id} not found."}
    return row.iloc[0].to_dict()


@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    if not (MODELS_DIR / "complaint_classifier.pkl").exists():
        return {"error": "NLP model not trained yet. Run src/nlp/complaint_nlp.py first."}
    return classify_complaint(req.text)


@app.post("/anomaly")
def anomaly(req: AnomalyRequest):
    _load_models()
    if _anomaly_bundle is None:
        return {"error": "Anomaly model not trained yet. Run src/anomaly/detect_anomalies.py first."}
    model = _anomaly_bundle["model"]
    features = _anomaly_bundle["features"]
    X = pd.DataFrame([req.dict()])[features]
    flag = model.predict(X)[0]
    score = float(model.decision_function(X)[0])
    return {"is_anomaly": bool(flag == -1), "anomaly_score": round(score, 4)}
