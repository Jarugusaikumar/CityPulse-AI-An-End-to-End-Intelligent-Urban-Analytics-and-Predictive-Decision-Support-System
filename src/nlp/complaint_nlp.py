"""
NLP module (Sections 68-71): complaint category classification using
TF-IDF + Logistic Regression, plus a lightweight rule-based sentiment
scorer (keeps the project dependency-light; swap in a proper sentiment
model or transformer if available).
"""
import re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

NEGATIVE_WORDS = {"terrible", "bad", "poor", "flooded", "blocked", "damaged", "overcrowded",
                   "leakage", "smell", "smoke", "delay", "delays", "noise", "no"}
POSITIVE_WORDS = {"good", "great", "improved", "resolved", "excellent", "clean"}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_sentiment(text: str) -> str:
    words = set(clean_text(text).split())
    neg = len(words & NEGATIVE_WORDS)
    pos = len(words & POSITIVE_WORDS)
    if neg > pos:
        return "Negative"
    if pos > neg:
        return "Positive"
    return "Neutral"


def train_category_classifier():
    df = pd.read_csv(ROOT / "data" / "raw" / "complaints.csv")
    df["clean_text"] = df["complaint_text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print("Complaint category classification report:")
    print(classification_report(y_test, preds))

    joblib.dump({"vectorizer": vectorizer, "model": model}, MODELS_DIR / "complaint_classifier.pkl")
    print(f"Saved -> {MODELS_DIR / 'complaint_classifier.pkl'}")
    return vectorizer, model


def classify_complaint(text: str, vectorizer=None, model=None):
    if vectorizer is None or model is None:
        bundle = joblib.load(MODELS_DIR / "complaint_classifier.pkl")
        vectorizer, model = bundle["vectorizer"], bundle["model"]

    clean = clean_text(text)
    category = model.predict(vectorizer.transform([clean]))[0]
    sentiment = rule_based_sentiment(text)
    priority = "High" if sentiment == "Negative" and category in {"Traffic", "Road", "Water"} else \
        ("Medium" if sentiment == "Negative" else "Low")
    return {"category": category, "sentiment": sentiment, "priority": priority}


if __name__ == "__main__":
    train_category_classifier()
    example = "Traffic is terrible near the station and the road is flooded."
    print("\nExample:", example)
    print(classify_complaint(example))
