"""
CityPulse AI - Synthetic Data Generator
Generates realistic synthetic urban datasets so the whole pipeline
can run end-to-end without needing external APIs or paid data sources.
Replace these with real data collectors when available.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

N_ZONES = 6
START = pd.Timestamp("2024-01-01")
HOURS = 24 * 180  # ~6 months of hourly data
timestamps = pd.date_range(START, periods=HOURS, freq="h")


def zone_ids():
    return [f"Z{i+1}" for i in range(N_ZONES)]


def generate_weather():
    df = pd.DataFrame({"timestamp": timestamps})
    hour = df["timestamp"].dt.hour
    day_of_year = df["timestamp"].dt.dayofyear
    df["temperature"] = (
        22 + 8 * np.sin(2 * np.pi * day_of_year / 365)
        + 4 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1.5, len(df))
    ).round(1)
    df["humidity"] = np.clip(60 + 15 * np.sin(2 * np.pi * hour / 24 + 1)
                              + np.random.normal(0, 5, len(df)), 20, 100).round(1)
    df["rainfall"] = np.clip(np.random.exponential(0.4, len(df)) - 0.3, 0, None).round(2)
    df["wind_speed"] = np.clip(np.random.normal(10, 4, len(df)), 0, None).round(1)
    df["pressure"] = np.random.normal(1013, 4, len(df)).round(1)
    df["visibility"] = np.clip(10 - df["rainfall"] * 2 + np.random.normal(0, 0.5, len(df)), 0.5, 10).round(1)
    return df


def generate_events():
    n_events = 40
    rows = []
    for i in range(n_events):
        start = START + pd.Timedelta(hours=int(np.random.uniform(0, HOURS - 10)))
        rows.append({
            "event_id": f"E{i+1:03d}",
            "event_name": np.random.choice(["Concert", "Festival", "Marathon", "Trade Fair", "Sports Match"]),
            "zone_id": np.random.choice(zone_ids()),
            "start_time": start,
            "end_time": start + pd.Timedelta(hours=int(np.random.uniform(2, 8))),
            "expected_crowd": int(np.random.uniform(500, 20000)),
        })
    return pd.DataFrame(rows)


def generate_traffic(weather, events):
    rows = []
    event_hours = {}
    for _, e in events.iterrows():
        rng = pd.date_range(e["start_time"], e["end_time"], freq="h")
        event_hours.setdefault(e["zone_id"], set()).update(rng)

    for zone in zone_ids():
        base = np.random.uniform(2000, 6000)
        for ts, temp, rain in zip(weather["timestamp"], weather["temperature"], weather["rainfall"]):
            hour = ts.hour
            dow = ts.dayofweek
            is_peak = hour in [7, 8, 9, 17, 18, 19, 20]
            is_weekend = dow >= 5
            is_event = ts in event_hours.get(zone, set())

            vehicle_count = base
            vehicle_count *= 1.8 if is_peak else 1.0
            vehicle_count *= 0.6 if is_weekend else 1.0
            vehicle_count *= 1.5 if is_event else 1.0
            vehicle_count *= max(0.5, 1 - rain * 0.15)
            vehicle_count += np.random.normal(0, base * 0.08)
            vehicle_count = max(200, vehicle_count)

            avg_speed = 60 - (vehicle_count / base) * 15 - rain * 3 + np.random.normal(0, 3)
            avg_speed = float(np.clip(avg_speed, 5, 70))

            congestion_ratio = vehicle_count / base
            if congestion_ratio > 1.6:
                congestion = "High"
            elif congestion_ratio > 1.1:
                congestion = "Medium"
            else:
                congestion = "Low"

            accident_count = np.random.poisson(0.15 + (0.3 if congestion == "High" else 0))

            rows.append({
                "timestamp": ts, "zone_id": zone,
                "vehicle_count": int(vehicle_count),
                "average_speed": round(avg_speed, 1),
                "congestion_level": congestion,
                "road_type": np.random.choice(["Highway", "Residential", "Commercial"]),
                "accident_count": int(accident_count),
                "is_event": int(is_event),
            })
    return pd.DataFrame(rows)


def generate_air_quality(traffic, weather):
    wmap = weather.set_index("timestamp")
    rows = []
    for zone, g in traffic.groupby("zone_id"):
        base_aqi = np.random.uniform(40, 90)
        for _, r in g.iterrows():
            rain = wmap.loc[r["timestamp"], "rainfall"]
            traffic_factor = r["vehicle_count"] / 4000
            aqi = base_aqi * (0.7 + 0.5 * traffic_factor) - rain * 8 + np.random.normal(0, 6)
            aqi = float(np.clip(aqi, 10, 400))
            rows.append({
                "timestamp": r["timestamp"], "zone_id": zone,
                "AQI": round(aqi, 1),
                "PM2_5": round(aqi * 0.6 + np.random.normal(0, 4), 1),
                "PM10": round(aqi * 0.9 + np.random.normal(0, 5), 1),
                "NO2": round(aqi * 0.3 + np.random.normal(0, 3), 1),
                "CO": round(0.5 + aqi * 0.01 + np.random.normal(0, 0.1), 2),
                "SO2": round(aqi * 0.15 + np.random.normal(0, 2), 1),
                "O3": round(30 + np.random.normal(0, 8), 1),
            })
    return pd.DataFrame(rows)


def generate_energy(weather, traffic):
    wmap = weather.set_index("timestamp")
    rows = []
    for zone, g in traffic.groupby("zone_id"):
        base = np.random.uniform(300, 800)
        for _, r in g.iterrows():
            temp = wmap.loc[r["timestamp"], "temperature"]
            hour = r["timestamp"].hour
            cooling_load = max(0, temp - 24) * 8
            heating_load = max(0, 16 - temp) * 6
            peak_factor = 1.3 if hour in [18, 19, 20, 21] else 1.0
            consumption = (base + cooling_load + heating_load) * peak_factor + np.random.normal(0, 15)
            rows.append({
                "timestamp": r["timestamp"], "zone_id": zone,
                "energy_consumption": round(max(50, consumption), 1),
                "peak_demand": round(max(50, consumption * np.random.uniform(1.05, 1.25)), 1),
            })
    return pd.DataFrame(rows)


def generate_transport():
    routes = [f"R{i+1}" for i in range(10)]
    sample_ts = timestamps[::3]  # every 3 hours
    rows = []
    for route in routes:
        base_pax = np.random.uniform(50, 400)
        for ts in sample_ts:
            hour = ts.hour
            is_peak = hour in [7, 8, 9, 17, 18, 19]
            pax = base_pax * (1.6 if is_peak else 1.0) + np.random.normal(0, base_pax * 0.1)
            delay = max(0, np.random.exponential(3) + (5 if is_peak else 0))
            rows.append({
                "timestamp": ts, "route_id": route,
                "passenger_count": int(max(0, pax)),
                "delay_minutes": round(delay, 1),
                "vehicle_count": np.random.randint(2, 15),
            })
    return pd.DataFrame(rows)


COMPLAINT_TEMPLATES = {
    "Traffic": ["Traffic is terrible near {loc}.", "Severe congestion reported at {loc} again.",
                "Road blocked due to heavy traffic near {loc}."],
    "Road": ["The road near {loc} is flooded and damaged.", "Potholes near {loc} need urgent repair."],
    "Water": ["No water supply in {loc} for two days.", "Water leakage reported near {loc}."],
    "Electricity": ["Frequent power cuts in {loc}.", "Street lights not working near {loc}."],
    "Pollution": ["Air quality near {loc} is very poor.", "Bad smell and smoke reported near {loc}."],
    "Transport": ["Bus delays are frequent on {loc} route.", "Public transport overcrowded near {loc}."],
    "Other": ["General maintenance issue reported near {loc}.", "Noise complaint near {loc}."],
}


def generate_complaints():
    n = 1500
    rows = []
    for i in range(n):
        category = np.random.choice(list(COMPLAINT_TEMPLATES.keys()))
        zone = np.random.choice(zone_ids())
        template = np.random.choice(COMPLAINT_TEMPLATES[category])
        text = template.format(loc=f"the {zone} station")
        priority = np.random.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2])
        ts = START + pd.Timedelta(hours=int(np.random.uniform(0, HOURS)))
        rows.append({
            "complaint_id": f"C{i+1:05d}", "timestamp": ts, "zone_id": zone,
            "complaint_text": text, "category": category,
            "status": np.random.choice(["Open", "Closed"], p=[0.3, 0.7]),
            "priority": priority,
        })
    return pd.DataFrame(rows)


def main():
    print("Generating weather...")
    weather = generate_weather()
    print("Generating events...")
    events = generate_events()
    print("Generating traffic...")
    traffic = generate_traffic(weather, events)
    print("Generating air quality...")
    air_quality = generate_air_quality(traffic, weather)
    print("Generating energy...")
    energy = generate_energy(weather, traffic)
    print("Generating transport...")
    transport = generate_transport()
    print("Generating complaints...")
    complaints = generate_complaints()

    weather.to_csv(RAW_DIR / "weather.csv", index=False)
    events.to_csv(RAW_DIR / "events.csv", index=False)
    traffic.to_csv(RAW_DIR / "traffic.csv", index=False)
    air_quality.to_csv(RAW_DIR / "air_quality.csv", index=False)
    energy.to_csv(RAW_DIR / "energy.csv", index=False)
    transport.to_csv(RAW_DIR / "transport.csv", index=False)
    complaints.to_csv(RAW_DIR / "complaints.csv", index=False)

    print(f"\nAll raw datasets saved to {RAW_DIR}")
    for f in RAW_DIR.glob("*.csv"):
        print(f" - {f.name}: {sum(1 for _ in open(f)) - 1} rows")


if __name__ == "__main__":
    main()
