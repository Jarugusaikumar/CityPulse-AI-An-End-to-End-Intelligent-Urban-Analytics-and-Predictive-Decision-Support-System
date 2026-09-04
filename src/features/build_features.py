"""
Feature Engineering (Sections 20-23 of the spec).

Joins traffic + weather + air quality + energy into a single master
table, adds datetime/peak/lag/rolling features, and saves it to
data/feature_engineered/master.csv for use by every downstream model.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.preprocessing.clean import fill_numeric_median, cap_outliers_iqr, drop_duplicates  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
FE_DIR = ROOT / "data" / "feature_engineered"
FE_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    traffic = pd.read_csv(RAW_DIR / "traffic.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(RAW_DIR / "weather.csv", parse_dates=["timestamp"])
    aqi = pd.read_csv(RAW_DIR / "air_quality.csv", parse_dates=["timestamp"])
    energy = pd.read_csv(RAW_DIR / "energy.csv", parse_dates=["timestamp"])
    return traffic, weather, aqi, energy


def add_datetime_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    return df


def add_time_series_features(df, group_col, target_col):
    df = df.sort_values([group_col, "timestamp"])
    df[f"{target_col}_lag_1"] = df.groupby(group_col)[target_col].shift(1)
    df[f"{target_col}_lag_24"] = df.groupby(group_col)[target_col].shift(24)
    df[f"{target_col}_rolling_mean_24"] = (
        df.groupby(group_col)[target_col].transform(lambda s: s.rolling(24, min_periods=1).mean())
    )
    df[f"{target_col}_rolling_std_24"] = (
        df.groupby(group_col)[target_col].transform(lambda s: s.rolling(24, min_periods=1).std())
    )
    return df


def build_master():
    traffic, weather, aqi, energy = load_raw()

    # Clean
    traffic = drop_duplicates(traffic)
    traffic = cap_outliers_iqr(traffic, ["vehicle_count", "average_speed"])
    weather = fill_numeric_median(weather, ["temperature", "humidity", "rainfall", "wind_speed"])
    aqi = cap_outliers_iqr(aqi, ["AQI", "PM2_5", "PM10"])
    energy = cap_outliers_iqr(energy, ["energy_consumption", "peak_demand"])

    # Merge (traffic is the base grain: timestamp x zone_id)
    df = traffic.merge(weather, on="timestamp", how="left")
    df = df.merge(aqi, on=["timestamp", "zone_id"], how="left")
    df = df.merge(energy, on=["timestamp", "zone_id"], how="left")

    # Feature engineering
    df = add_datetime_features(df)
    df = add_time_series_features(df, "zone_id", "vehicle_count")
    df = df.bfill().ffill()

    out_path = FE_DIR / "master.csv"
    df.to_csv(out_path, index=False)
    print(f"Master feature-engineered dataset saved: {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    build_master()
