"""
CityPulse AI - Master Pipeline
Runs the full Collect -> Store -> Clean -> Engineer -> Model -> Explain
workflow end-to-end with a single command:

    python run_pipeline.py

Each stage is idempotent and can also be run individually (see README).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ("1. Generating synthetic urban datasets", "src/data/generate_data.py"),
    ("2. Loading data into SQL database", "src/data/load_to_sql.py"),
    ("3. Building feature-engineered master dataset", "src/features/build_features.py"),
    ("4. Training classification models (congestion)", "src/models/classification.py"),
    ("5. Training regression models (energy)", "src/models/regression.py"),
    ("6. Clustering city zones", "src/models/clustering.py"),
    ("7. Forecasting traffic (SARIMA)", "src/forecasting/forecast.py"),
    ("8. Training NLP complaint classifier", "src/nlp/complaint_nlp.py"),
    ("9. Detecting anomalies", "src/anomaly/detect_anomalies.py"),
    ("10. Explaining model predictions", "src/explain/explain_model.py"),
]


def main():
    for title, script in STAGES:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        result = subprocess.run([PY, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStage failed: {script} (exit code {result.returncode})")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("Pipeline complete. Next steps:")
    print("  streamlit run app/app.py     # interactive dashboard")
    print("  uvicorn api.main:app --reload   # REST API")
    print("=" * 70)


if __name__ == "__main__":
    main()
