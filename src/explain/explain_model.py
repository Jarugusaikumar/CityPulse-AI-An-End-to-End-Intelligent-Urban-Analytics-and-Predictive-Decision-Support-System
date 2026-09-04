"""
Explainable AI (Section 74): shows the important factors behind a
congestion prediction using both built-in feature importance and SHAP.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def get_feature_names(pipeline, features_num, features_cat):
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(features_cat))
    return features_num + cat_names


def explain_with_importance():
    bundle = joblib.load(MODELS_DIR / "congestion_model.pkl")
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline, bundle["features_num"], bundle["features_cat"])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print(f"Top contributing factors for model '{bundle['name']}':")
        for i in order[:8]:
            print(f"  {names[i]:<25} importance={importances[i]:.4f}")
    else:
        print("Model does not expose feature_importances_; falling back to coefficients.")
        coefs = np.abs(model.coef_).mean(axis=0)
        order = np.argsort(coefs)[::-1]
        for i in order[:8]:
            print(f"  {names[i]:<25} |coef|={coefs[i]:.4f}")


def explain_with_shap(n_samples=200):
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP explanation.")
        return

    bundle = joblib.load(MODELS_DIR / "congestion_model.pkl")
    pipeline = bundle["pipeline"]
    df = pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv")
    X = df[bundle["features_num"] + bundle["features_cat"]].sample(n_samples, random_state=42)
    X_trans = pipeline.named_steps["preprocess"].transform(X)
    names = get_feature_names(pipeline, bundle["features_num"], bundle["features_cat"])

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    print("\nSHAP mean |impact| per feature:")
    for i in order[:8]:
        print(f"  {names[i]:<25} mean|SHAP|={mean_abs[i]:.4f}")


if __name__ == "__main__":
    explain_with_importance()
    explain_with_shap()
