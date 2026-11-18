"""Sentiment model: TF-IDF + Logistic Regression blended with VADER.

* The scikit-learn pipeline is trained by `train_model.py` (best of LR / SVM / NB / RF).
* VADER adds a rule-based second opinion, which keeps predictions sensible even when the
  training data is small. Final probabilities = ML_WEIGHT * ML + (1 - ML_WEIGHT) * VADER.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
CLASSES = ["negative", "neutral", "positive"]
ML_WEIGHT = 0.5

_vader = SentimentIntensityAnalyzer()


def vader_proba(text: str) -> np.ndarray:
    """Map VADER's compound score to [neg, neu, pos] probabilities."""
    c = _vader.polarity_scores(text)["compound"]
    pos = max(c, 0.0)
    neg = max(-c, 0.0)
    neu = 1.0 - abs(c)
    # Sharpen the neutral band so mild text is not forced into pos/neg.
    probs = np.array([neg, neu * 0.9, pos]) + 1e-6
    return probs / probs.sum()


class SentimentModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes_ = list(pipeline.classes_)

    def _ml_proba(self, texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        raw = self.pipeline.predict_proba(cleaned)
        out = np.zeros((len(texts), 3))
        for j, cls in enumerate(self.classes_):
            out[:, CLASSES.index(cls)] = raw[:, j]
        return out

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        ml = self._ml_proba(texts)
        rule = np.vstack([vader_proba(t) for t in texts])
        return ML_WEIGHT * ml + (1 - ML_WEIGHT) * rule

    def predict(self, texts: List[str]) -> List[Dict]:
        probs = self.predict_proba(texts)
        results = []
        for p in probs:
            k = int(np.argmax(p))
            results.append({
                "sentiment": CLASSES[k],
                "confidence": float(p[k]),
                "p_negative": float(p[0]),
                "p_neutral": float(p[1]),
                "p_positive": float(p[2]),
            })
        return results


def load_model() -> SentimentModel:
    """Load the saved model, or train one from the seed data if it is missing/incompatible."""
    if MODEL_PATH.exists():
        try:
            return SentimentModel(joblib.load(MODEL_PATH))
        except Exception:
            pass  # e.g. scikit-learn version mismatch -> retrain below
    from train_model import train_and_save

    return SentimentModel(train_and_save(verbose=False))
