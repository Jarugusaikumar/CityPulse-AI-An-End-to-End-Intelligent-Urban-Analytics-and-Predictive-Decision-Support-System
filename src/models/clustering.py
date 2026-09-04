"""
Unsupervised clustering of city zones (Sections 41-45).
Uses aggregated zone-level features: average traffic, AQI, energy,
complaint count. Compares K-Means (with elbow/silhouette) and GMM (BIC).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_zone_features():
    df = pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv")
    complaints = pd.read_csv(ROOT / "data" / "raw" / "complaints.csv")

    agg = df.groupby("zone_id").agg(
        avg_traffic=("vehicle_count", "mean"),
        avg_aqi=("AQI", "mean"),
        avg_energy=("energy_consumption", "mean"),
    ).reset_index()

    complaint_counts = complaints.groupby("zone_id").size().rename("complaint_count").reset_index()
    agg = agg.merge(complaint_counts, on="zone_id", how="left").fillna(0)
    return agg


def run():
    agg = build_zone_features()
    feature_cols = ["avg_traffic", "avg_aqi", "avg_energy", "complaint_count"]
    X = agg[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow + silhouette for K-Means (bounded by n_zones)
    max_k = min(5, len(agg) - 1)
    inertias, sil_scores = [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
        print(f"K={k}: inertia={km.inertia_:.2f} silhouette={sil_scores[-1]:.3f}")

    best_k = list(range(2, max_k + 1))[int(np.argmax(sil_scores))]
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
    agg["kmeans_cluster"] = kmeans.labels_

    # GMM with BIC comparison
    bic_values = []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
        bic_values.append(gmm.bic(X_scaled))
    best_gmm_k = int(np.argmin(bic_values)) + 1
    gmm_final = GaussianMixture(n_components=best_gmm_k, random_state=42).fit(X_scaled)
    agg["gmm_cluster"] = gmm_final.predict(X_scaled)

    print(f"\nBest K-Means k (by silhouette): {best_k}")
    print(f"Best GMM components (by BIC): {best_gmm_k}")
    print("\nZone cluster summary:\n", agg)

    agg.to_csv(MODELS_DIR / "zone_clusters.csv", index=False)
    joblib.dump({"kmeans": kmeans, "gmm": gmm_final, "scaler": StandardScaler().fit(X),
                 "features": feature_cols}, MODELS_DIR / "clustering_model.pkl")
    print(f"\nSaved -> {MODELS_DIR / 'zone_clusters.csv'} and clustering_model.pkl")
    return agg


if __name__ == "__main__":
    run()
