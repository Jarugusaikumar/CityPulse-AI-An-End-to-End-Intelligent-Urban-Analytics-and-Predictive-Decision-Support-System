"""
Congestion classification (Sections 30-33).
Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost;
saves the best pipeline (preprocessing + model) to models/congestion_model.pkl
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

FEATURES_NUM = ["vehicle_count", "average_speed", "rainfall", "temperature",
                "is_peak", "is_weekend", "day_of_week", "AQI", "energy_consumption"]
FEATURES_CAT = ["road_type"]
TARGET = "congestion_level"

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(ROOT / "data" / "feature_engineered" / "master.csv")
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])


def run():
    df = load_data()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
    }

    results = []
    best_name, best_pipe, best_f1 = None, None, -1
    y_train_enc = y_train.astype("category")
    y_test_enc = y_test.astype("category").cat.set_categories(y_train_enc.cat.categories)

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        # XGBoost needs numeric labels
        if name == "XGBoost":
            pipe.fit(X_train, y_train_enc.cat.codes)
            preds_codes = pipe.predict(X_test)
            preds = pd.Categorical.from_codes(preds_codes, categories=y_train_enc.cat.categories)
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results.append({"model": name, "accuracy": round(acc, 4), "f1_weighted": round(f1, 4)})
        print(f"{name}: accuracy={acc:.4f} f1={f1:.4f}")

        if f1 > best_f1:
            best_f1, best_name, best_pipe = f1, name, pipe

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    print(classification_report(y_test, preds if best_name == "XGBoost" else best_pipe.predict(X_test)))

    joblib.dump({"pipeline": best_pipe, "name": best_name,
                 "classes": list(y_train_enc.cat.categories) if best_name == "XGBoost" else None,
                 "features_num": FEATURES_NUM, "features_cat": FEATURES_CAT},
                MODELS_DIR / "congestion_model.pkl")

    pd.DataFrame(results).to_csv(MODELS_DIR / "classification_comparison.csv", index=False)
    print(f"\nSaved best model -> {MODELS_DIR / 'congestion_model.pkl'}")
    return results


if __name__ == "__main__":
    run()
