"""
Regression task (Sections 34-36): predict energy_consumption from
traffic/weather/context features. Compares Linear Regression, Random
Forest, and XGBoost regressors using MAE/RMSE.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = ["vehicle_count", "average_speed", "temperature", "humidity",
            "rainfall", "hour", "is_peak", "is_weekend", "AQI"]
TARGET = "energy_consumption"


def load_data():
    return pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv")


def run():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([("scale", StandardScaler(), FEATURES)])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=300, max_depth=6, learning_rate=0.05),
    }

    results, best_name, best_pipe, best_rmse = [], None, None, np.inf
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)})
        print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name, best_pipe = rmse, name, pipe

    print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f})")
    joblib.dump({"pipeline": best_pipe, "name": best_name, "features": FEATURES},
                MODELS_DIR / "energy_regression_model.pkl")
    pd.DataFrame(results).to_csv(MODELS_DIR / "regression_comparison.csv", index=False)
    print(f"Saved best model -> {MODELS_DIR / 'energy_regression_model.pkl'}")
    return results


if __name__ == "__main__":
    run()
