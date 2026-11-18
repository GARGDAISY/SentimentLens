"""Optional fallback: direct multilingual sentiment with XLM-RoBERTa.

Used only for reviews whose translation failed. Requires `transformers` and `torch`
(see requirements-optional.txt). If they are not installed, available() is False and
the app simply labels those reviews with the English model on the original text.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
_LABELS = {"positive": "positive", "neutral": "neutral", "negative": "negative"}


def available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)


def predict_proba(texts: List[str]) -> Optional[List[Dict[str, float]]]:
    """Return [{'positive': p, 'neutral': p, 'negative': p}, ...] or None on failure."""
    if not texts or not available():
        return None
    try:
        raw = _pipeline()(texts)
    except Exception:
        return None
    out = []
    for scores in raw:
        d = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for item in scores:
            label = _LABELS.get(item["label"].lower())
            if label:
                d[label] = float(item["score"])
        out.append(d)
    return out
