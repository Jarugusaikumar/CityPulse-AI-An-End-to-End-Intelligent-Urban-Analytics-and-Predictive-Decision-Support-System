# CityPulse-AI-An-End-to-End-Intelligent-Urban-Analytics-and-Predictive-Decision-Support-System — Jarugu Saikumar

An end-to-end Data Science platform designed to analyze heterogeneous urban data streams—including traffic, weather, air quality (AQI), energy consumption, public transportation, citizen complaints, and city events—to provide intelligent insights and predictive decision support.

This repository contains a runnable implementation of the CityPulse AI project flow: synthetic data → SQL storage → cleaning → EDA → feature engineering → supervised ML → clustering → time-series forecasting → NLP → anomaly detection → explainability → Streamlit app → FastAPI → Docker.

Everything here executes out of the box using synthetically generated
urban data, so you can run the whole pipeline immediately and later
swap in real data sources (see `data/DATA_DICTIONARY.md`).

## Quick Start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (data → models, ~2-3 minutes)
python run_pipeline.py

# 4. Launch the interactive dashboard
streamlit run app/app.py

# 5. (Optional, separate terminal) Launch the REST API
uvicorn api.main:app --reload
```

Then open the Streamlit URL it prints (usually http://localhost:8501)
and/or the API docs at http://localhost:8000/docs.

## What `run_pipeline.py` does

| Step | Script | Output |
|---|---|---|
| 1. Generate data | `src/data/generate_data.py` | `data/raw/*.csv` |
| 2. Load to SQL | `src/data/load_to_sql.py` | `citypulse.db` (SQLite) |
| 3. Feature engineering | `src/features/build_features.py` | `data/feature_engineered/master.csv` |
| 4. Classification | `src/models/classification.py` | `models/congestion_model.pkl` |
| 5. Regression | `src/models/regression.py` | `models/energy_regression_model.pkl` |
| 6. Clustering | `src/models/clustering.py` | `models/zone_clusters.csv` |
| 7. Forecasting | `src/forecasting/forecast.py` | `models/forecast_Z1.csv` |
| 8. NLP | `src/nlp/complaint_nlp.py` | `models/complaint_classifier.pkl` |
| 9. Anomaly detection | `src/anomaly/detect_anomalies.py` | `models/anomalies.csv` |
| 10. Explainability | `src/explain/explain_model.py` | printed feature importance / SHAP |

Every stage can also be run individually with `python <script path>`.

## Streamlit app pages

Home · EDA · ML Prediction (congestion) · Clustering · Forecasting ·
NLP (complaint category/sentiment/priority) · Anomaly Detection ·
AI Assistant (grounded retrieval demo — see Section 92 of the spec:
it reads real numbers from `models/zone_clusters.csv` rather than
inventing them; plug in an LLM call for richer natural-language
phrasing).

## API endpoints (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Congestion classification |
| `POST /forecast` | Traffic forecast for a zone |
| `POST /cluster` | Zone cluster lookup |
| `POST /sentiment` | Complaint category/sentiment/priority |
| `POST /anomaly` | Anomaly check for a reading |

## Docker

```bash
docker build -t citypulse-ai .
docker run -p 8501:8501 citypulse-ai
```

(This builds data + trains all models at image-build time, so the
container is immediately ready to serve on first run.)

## SQL

`sql/schema.sql` creates the full CityPulseDB schema (locations,
traffic, weather, air_quality, energy, transport, events, complaints,
predictions, model_results) plus an example view. It's written for
SQLite (used by `src/data/load_to_sql.py`) but is close to standard
SQL Server syntax — see the comment at the top of the file for how to
port it.

## Project structure

```
CityPulse-AI/
├── data/
│   ├── raw/                    # generated CSVs
│   ├── processed/
│   ├── feature_engineered/     # master.csv used by all models
│   └── DATA_DICTIONARY.md
├── sql/
│   └── schema.sql
├── src/
│   ├── data/                   # generation + SQL loading
│   ├── preprocessing/          # cleaning utilities
│   ├── features/               # feature engineering
│   ├── models/                 # classification, regression, clustering
│   ├── forecasting/            # SARIMA forecasting
│   ├── nlp/                    # complaint classification + sentiment
│   ├── anomaly/                # Isolation Forest anomaly detection
│   └── explain/                # feature importance + SHAP
├── models/                     # trained model artifacts (generated)
├── app/
│   └── app.py                  # Streamlit dashboard
├── api/
│   └── main.py                 # FastAPI service
├── run_pipeline.py             # orchestrates every stage
├── Dockerfile
├── requirements.txt
└── README.md
```

## Extending toward the full spec

This implementation covers the executable core of every stage in the
CityPulse AI spec (SQL → EDA → ML → clustering → forecasting → NLP →
anomaly detection → explainability → app → API → Docker). Two pieces
from the original spec are intentionally left as extension points
since they depend on licensed tools rather than code:

- **Power BI dashboards** — connect Power BI Desktop directly to
  `citypulse.db` (or export `data/feature_engineered/master.csv`) and
  build the 7 report pages described in the spec.
- **Cloud deployment** — the Dockerfile is ready to push to any
  container registry / cloud run service (AWS ECS, Azure Container
  Apps, GCP Cloud Run, etc.).

Everything else — SQL, statistics, ML, clustering, forecasting, deep
learning-adjacent modeling, NLP, anomaly detection, explainability,
the interactive app, and the API — runs locally with the commands
above.
>>>>>>> 3345881 (Fix project execution and update title)
