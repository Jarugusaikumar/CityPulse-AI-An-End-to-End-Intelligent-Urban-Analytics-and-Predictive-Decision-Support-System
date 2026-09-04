"""
Time-series forecasting (Sections 46-56).
Forecasts hourly traffic volume for a given zone using SARIMA, with a
naive seasonal baseline for comparison. Uses chronological train/test
split (never shuffle time series).
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_series(zone_id="Z1", freq_hours=1):
    df = pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv", parse_dates=["timestamp"])
    df = df[df["zone_id"] == zone_id].sort_values("timestamp")
    # Resample hourly to a clean, evenly-spaced series and use the last N days for speed
    series = df.set_index("timestamp")["vehicle_count"].resample("h").mean().interpolate()
    return series.tail(24 * 30)  # last 30 days


def naive_seasonal_forecast(train, horizon, season=24):
    """Baseline: repeat the value from exactly one day (season) earlier."""
    last_season = train[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season.values, reps)[:horizon]


def run(zone_id="Z1", horizon=24):
    series = load_series(zone_id)
    train, test = series[:-horizon], series[-horizon:]

    print(f"Forecasting traffic for zone {zone_id}: {len(train)} train points, {horizon} test points")

    # SARIMA(1,1,1)(1,1,1,24) - daily seasonality on hourly data
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    sarima_forecast = fitted.forecast(steps=horizon)

    naive_forecast = naive_seasonal_forecast(train, horizon)

    sarima_mae = mean_absolute_error(test, sarima_forecast)
    sarima_rmse = mean_squared_error(test, sarima_forecast) ** 0.5
    naive_mae = mean_absolute_error(test, naive_forecast)
    naive_rmse = mean_squared_error(test, naive_forecast) ** 0.5

    print(f"SARIMA   -> MAE={sarima_mae:.1f} RMSE={sarima_rmse:.1f}")
    print(f"Naive24  -> MAE={naive_mae:.1f} RMSE={naive_rmse:.1f}")

    result = pd.DataFrame({
        "timestamp": test.index, "actual": test.values,
        "sarima_forecast": sarima_forecast.values, "naive_forecast": naive_forecast,
    })
    out_path = MODELS_DIR / f"forecast_{zone_id}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved forecast comparison -> {out_path}")
    return result


if __name__ == "__main__":
    for zone_id in [f"Z{i + 1}" for i in range(6)]:
        run(zone_id=zone_id)
