"""Train and compare TF-IDF classifiers, then save the best one.

Usage:
    python train_model.py                       # uses data/train_seed.csv (synthetic)
    python train_model.py --data reviews.csv    # your own dataset

The CSV needs a text column (text / review / reviewText ...) and either a `label`
column (positive / neutral / negative) or a numeric `rating` column (1-5, where
1-2 = negative, 3 = neutral, 4-5 = positive).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent
SEED_PATH = ROOT / "data" / "train_seed.csv"
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
REPORT_PATH = ROOT / "models" / "comparison.csv"

TEXT_COLUMNS = ("text", "review", "reviewtext", "review_text", "content", "comment")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lowered = {c.lower(): c for c in df.columns}
    text_col = next((lowered[c] for c in TEXT_COLUMNS if c in lowered), None)
    if text_col is None:
        raise SystemExit(f"No text column found. Columns: {list(df.columns)}")
    if "label" in lowered:
        labels = df[lowered["label"]].astype(str).str.lower().str.strip()
    elif "rating" in lowered or "score" in lowered:
        r = pd.to_numeric(df[lowered.get("rating", lowered.get("score"))], errors="coerce")
        labels = r.map(lambda x: "negative" if x <= 2 else "neutral" if x == 3 else "positive")
    else:
        raise SystemExit("Need a `label` or `rating` column.")
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": labels})
    out = out[out["label"].isin(["positive", "neutral", "negative"])].dropna()
    return out.drop_duplicates("text").reset_index(drop=True)


def candidates():
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=1000, C=3, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()),
                                ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))]),
        "Random Forest": Pipeline([("tfidf", tfidf()),
                                   ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    }


def train_and_save(data_path: Path | None = None, verbose: bool = True):
    df = load_dataset(data_path or SEED_PATH)
    df["clean"] = df["text"].map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    results, fitted = [], {}
    for name, pipe in candidates().items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        f1 = f1_score(y_test, pred, average="macro")
        results.append({"model": name, "macro_f1": round(f1, 4),
                        "accuracy": round((pred == y_test).mean(), 4)})
        fitted[name] = pipe
        if verbose:
            print(f"\n=== {name} ===")
            print(classification_report(y_test, pred, digits=3))
            print("Confusion matrix (rows=true, cols=pred; order negative/neutral/positive):")
            print(confusion_matrix(y_test, pred, labels=["negative", "neutral", "positive"]))

    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    best_name = table.iloc[0]["model"]
    # Refit the winner on ALL data before saving.
    best = candidates()[best_name].fit(df["clean"], df["label"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    table.to_csv(REPORT_PATH, index=False)
    if verbose:
        print("\n=== Model comparison (test split) ===")
        print(table.to_string(index=False))
        print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")
        if data_path is None:
            print("\nNOTE: trained on the small SYNTHETIC seed set. These scores are inflated and "
                  "only prove the pipeline works. Use a real dataset for meaningful numbers.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="CSV with text + label/rating columns")
    args = ap.parse_args()
    train_and_save(args.data)
