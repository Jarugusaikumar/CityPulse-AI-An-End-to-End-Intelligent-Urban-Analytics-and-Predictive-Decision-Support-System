"""
CityPulse AI - Streamlit Application (Sections 83-91)

Run with:
    streamlit run app/app.py

Pages: Home, EDA, ML Prediction, Clustering, Forecasting, NLP, Anomaly Detection, AI Assistant
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.nlp.complaint_nlp import classify_complaint  # noqa: E402

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "feature_engineered"

st.set_page_config(page_title="CityPulse AI", layout="wide", page_icon="🏙️")


@st.cache_data
def load_master():
    path = DATA_DIR / "master.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_resource
def load_model(name):
    path = MODELS_DIR / name
    return joblib.load(path) if path.exists() else None


def page_home(df):
    st.title("🏙️ CityPulse AI — Urban Intelligence Platform — Jarugu Saikumar")
    st.caption("Traffic · Air Quality · Energy · Transport · Citizen Complaints · Forecasting · AI Insights")

    if df is None:
        st.warning("No data found. Run `python run_pipeline.py` first to generate data and train models.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Traffic (veh/hr)", f"{df['vehicle_count'].mean():,.0f}")
    col3.metric("Avg AQI", f"{df['AQI'].mean():.1f}")
    col4.metric("Avg Energy (kWh)", f"{df['energy_consumption'].mean():.1f}")

    st.subheader("Traffic Trend (hourly average across zones)")
    trend = df.groupby("timestamp")["vehicle_count"].mean()
    st.line_chart(trend)

    st.subheader("Zone Comparison")
    zone_summary = df.groupby("zone_id")[["vehicle_count", "AQI", "energy_consumption"]].mean()
    st.bar_chart(zone_summary)


def page_eda(df):
    st.header("📊 Exploratory Data Analysis")
    if df is None:
        st.warning("No data available. Run the pipeline first.")
        return

    st.write(df.describe())

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    st.subheader("Distribution")
    col = st.selectbox("Choose a column", numeric_df.columns, index=list(numeric_df.columns).index("vehicle_count"))
    st.bar_chart(df[col].value_counts(bins=20).sort_index())


def page_prediction():
    st.header("🚦 ML Prediction — Congestion Classifier")
    bundle = load_model("congestion_model.pkl")
    if bundle is None:
        st.warning("Model not trained. Run `python src/models/classification.py` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        vehicle_count = st.slider("Vehicle Count", 200, 15000, 5000)
        average_speed = st.slider("Average Speed (km/h)", 5, 70, 30)
        rainfall = st.slider("Rainfall (mm)", 0.0, 5.0, 0.0)
        temperature = st.slider("Temperature (°C)", 0, 45, 25)
    with col2:
        is_peak = st.checkbox("Peak Hour", value=True)
        is_weekend = st.checkbox("Weekend", value=False)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)
        aqi = st.slider("AQI", 10, 400, 90)
        energy = st.slider("Energy Consumption", 50, 1200, 500)
        road_type = st.selectbox("Road Type", ["Highway", "Residential", "Commercial"])

    if st.button("Predict Congestion"):
        pipeline = bundle["pipeline"]
        X = pd.DataFrame([{
            "vehicle_count": vehicle_count, "average_speed": average_speed, "rainfall": rainfall,
            "temperature": temperature, "is_peak": int(is_peak), "is_weekend": int(is_weekend),
            "day_of_week": day_of_week, "AQI": aqi, "energy_consumption": energy, "road_type": road_type,
        }])
        if bundle["name"] == "XGBoost":
            code = pipeline.predict(X)[0]
            proba = pipeline.predict_proba(X)[0]
            prediction = bundle["classes"][int(code)]
        else:
            prediction = pipeline.predict(X)[0]
            proba = pipeline.predict_proba(X)[0]

        st.success(f"Predicted Congestion: **{prediction}**  (confidence: {np.max(proba):.1%})")

        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            st.write("**Important factors (model feature importance):**")
            importances = model.feature_importances_
            ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
            names = bundle["features_num"] + list(ohe.get_feature_names_out(bundle["features_cat"]))
            top = sorted(zip(names, importances), key=lambda x: -x[1])[:5]
            for name, imp in top:
                st.write(f"- {name}: {imp:.3f}")


def page_clustering():
    st.header("🗺️ Zone Clustering")
    path = MODELS_DIR / "zone_clusters.csv"
    if not path.exists():
        st.warning("Run `python src/models/clustering.py` first.")
        return
    clusters = pd.read_csv(path)
    st.dataframe(clusters)
    st.subheader("Cluster Characteristics")
    st.bar_chart(clusters.set_index("zone_id")[["avg_traffic", "avg_aqi", "avg_energy"]])


def page_forecasting():
    st.header("📈 Forecasting")
    zone = st.selectbox("Select Zone", [f"Z{i+1}" for i in range(6)])
    path = MODELS_DIR / f"forecast_{zone}.csv"
    if not path.exists():
        st.warning(f"No forecast saved for {zone}. Run `python src/forecasting/forecast.py` first.")
        return
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    st.line_chart(df[["actual", "sarima_forecast", "naive_forecast"]])
    mae = (df["actual"] - df["sarima_forecast"]).abs().mean()
    st.metric("SARIMA MAE", f"{mae:.1f} vehicles/hr")


def page_nlp():
    st.header("💬 Citizen Complaint Analysis")
    if not (MODELS_DIR / "complaint_classifier.pkl").exists():
        st.warning("Run `python src/nlp/complaint_nlp.py` first.")
        return
    text = st.text_area("Enter a citizen complaint", "Traffic is terrible near the station and the road is flooded.")
    if st.button("Analyze"):
        result = classify_complaint(text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", result["category"])
        col2.metric("Sentiment", result["sentiment"])
        col3.metric("Priority", result["priority"])


def page_anomaly(df):
    st.header("🚨 Anomaly Detection")
    path = MODELS_DIR / "anomalies.csv"
    if not path.exists():
        st.warning("Run `python src/anomaly/detect_anomalies.py` first.")
        return
    anomalies = pd.read_csv(path)
    st.write(f"**{len(anomalies)} anomalies detected**")
    st.dataframe(anomalies.head(50))
    if df is not None:
        zone = st.selectbox("Zone", sorted(df["zone_id"].unique()))
        zone_anom = anomalies[anomalies["zone_id"] == zone]
        st.write(f"{len(zone_anom)} anomalies in {zone}")


def page_assistant():
    st.header("🤖 GenAI Assistant (Grounded)")
    st.caption("Answers are grounded in retrieved data/model results — not invented numbers "
               "(Section 92-93 of the spec). This demo uses simple retrieval + templated "
               "explanations; plug in an LLM call for richer natural-language responses.")

    clusters_path = MODELS_DIR / "zone_clusters.csv"
    question = st.text_input("Ask about a zone, e.g. 'Why is traffic high in Zone 3?'")
    if st.button("Ask") and question:
        zone_found = None
        for z in [f"Z{i+1}" for i in range(6)]:
            if z.lower() in question.lower() or z[1:] in question:
                zone_found = z
                break

        if zone_found and clusters_path.exists():
            row = pd.read_csv(clusters_path).set_index("zone_id").loc[zone_found]
            st.write(
                f"**Zone {zone_found} summary (retrieved from data):**\n\n"
                f"- Average traffic: {row['avg_traffic']:.0f} vehicles/hr\n"
                f"- Average AQI: {row['avg_aqi']:.1f}\n"
                f"- Average energy consumption: {row['avg_energy']:.1f}\n"
                f"- Complaint count: {int(row['complaint_count'])}\n\n"
                f"Based on this retrieved data, {zone_found} "
                f"{'shows above-average congestion' if row['avg_traffic'] > 4500 else 'shows moderate traffic levels'} "
                f"and {'elevated' if row['avg_aqi'] > 75 else 'moderate'} air quality readings."
            )
        else:
            st.info("Mention a specific zone (e.g. Z1-Z6) so I can retrieve grounded data for it.")


def ensure_project_assets():
    if (DATA_DIR / "master.csv").exists() and (MODELS_DIR / "congestion_model.pkl").exists():
        return True

    st.warning("No data found. Running the pipeline to generate datasets and train models...")
    with st.spinner("Generating synthetic data and training models. This may take a few minutes..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_pipeline.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        st.error("The pipeline failed while generating assets. Please check the console output.")
        st.code(result.stdout)
        st.code(result.stderr)
        return False

    load_master.clear()
    load_model.clear()
    st.success("Data and models are ready. Refreshing the app...")
    st.rerun()


def main():
    if not ensure_project_assets():
        return

    df = load_master()
    st.sidebar.title("CityPulse AI")
    page = st.sidebar.radio("Navigate", [
        "Home", "EDA", "ML Prediction", "Clustering", "Forecasting",
        "NLP", "Anomaly Detection", "AI Assistant",
    ])

    if page == "Home":
        page_home(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "ML Prediction":
        page_prediction()
    elif page == "Clustering":
        page_clustering()
    elif page == "Forecasting":
        page_forecasting()
    elif page == "NLP":
        page_nlp()
    elif page == "Anomaly Detection":
        page_anomaly(df)
    elif page == "AI Assistant":
        page_assistant()


if __name__ == "__main__":
    main()
