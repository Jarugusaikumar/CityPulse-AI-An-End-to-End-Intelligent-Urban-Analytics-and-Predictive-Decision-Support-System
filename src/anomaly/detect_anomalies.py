"""Anomaly detection (Sections 72-73) using Isolation Forest on traffic volume."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = ["vehicle_count", "average_speed", "AQI", "energy_consumption"]


def run(contamination=0.02):
    df = pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv", parse_dates=["timestamp"])
    X = df[FEATURES]

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    df["anomaly_flag"] = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"] = (df["anomaly_flag"] == -1).astype(int)

    anomalies = df[df["is_anomaly"] == 1][
        ["timestamp", "zone_id", "vehicle_count", "AQI", "energy_consumption", "anomaly_score"]
    ].sort_values("anomaly_score")

    print(f"Detected {len(anomalies)} anomalies out of {len(df)} records ({len(anomalies)/len(df):.2%})")
    print(anomalies.head(10))

    anomalies.to_csv(MODELS_DIR / "anomalies.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES}, MODELS_DIR / "anomaly_model.pkl")
    print(f"\nSaved -> {MODELS_DIR / 'anomalies.csv'} and anomaly_model.pkl")
    return anomalies


if __name__ == "__main__":
    run()
