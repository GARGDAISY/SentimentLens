"""End-to-end review pipeline: language -> translation -> emoji/text sentiment -> insights."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional

import pandas as pd

from . import multilingual
from .emoji_sentiment import combine_emoji_sentiment
from .language import prepare_reviews
from .model import SentimentModel
from .similarity import find_duplicates


def verdict_from_counts(pos: int, neu: int, neg: int) -> Dict[str, str]:
    total = max(pos + neu + neg, 1)
    p, n = pos / total * 100, neg / total * 100
    if p >= 50 and p - n >= 25:
        return {"label": "Positive", "color": "#39FF88",
                "text": "Most reviewers are happy with this product. It looks like a safe pick."}
    if n >= 50 and n - p >= 25:
        return {"label": "Negative", "color": "#FF3D81",
                "text": "Most reviewers are unhappy with this product. Think twice before buying."}
    return {"label": "Mixed", "color": "#FFD23F",
            "text": "Opinions are split. Read the details and check what matters most to you."}


def analyze(
    reviews: List[str],
    model: SentimentModel,
    translate_fn: Optional[Callable] = None,
    cache: Optional[dict] = None,
    duplicate_threshold: float = 0.9,
) -> Dict:
    prepared = prepare_reviews(reviews, translate_fn=translate_fn, cache=cache)
    english = [p.english for p in prepared]

    # Batch predictions to avoid unnecessarily large intermediate arrays.
    preds = []
    for start in range(0, len(english), 512):
        preds.extend(model.predict(english[start:start + 512]))

    failed = [i for i, item in enumerate(prepared) if not item.translated]
    fallback_used = 0
    if failed:
        probs = multilingual.predict_proba([prepared[i].original for i in failed])
        if probs:
            for i, values in zip(failed, probs):
                label = max(values, key=values.get)
                preds[i] = {
                    "sentiment": label, "confidence": values[label],
                    "p_negative": values["negative"], "p_neutral": values["neutral"],
                    "p_positive": values["positive"],
                }
            fallback_used = len(failed)

    rows = []
    for item, prediction in zip(prepared, preds):
        adjusted = combine_emoji_sentiment(item.original, prediction)
        rows.append({
            "review": item.original,
            "english": item.english if item.translated else "(translation failed)",
            "language": item.language,
            "sentiment": adjusted["sentiment"],
        })
    # Keep the original review-detail fields; confidence and duplicate flags are intentionally hidden.
    df = pd.DataFrame(rows, columns=["review", "english", "language", "sentiment"])

    counts = Counter(df["sentiment"]) if len(df) else Counter()
    pos, neu, neg = counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)
    total = len(df)
    lang_counts = Counter(df["language"]) if total else Counter()
    languages = [
        {"language": lang, "count": count, "pct": round(count / total * 100, 1)}
        for lang, count in lang_counts.most_common()
    ]
    failed_languages = Counter(prepared[i].language for i in failed)
    return {
        "df": df,
        "total": total,
        "counts": {"positive": pos, "neutral": neu, "negative": neg},
        "pct": {key: round(value / total * 100, 1) if total else 0.0
                for key, value in {"positive": pos, "neutral": neu, "negative": neg}.items()},
        "verdict": verdict_from_counts(pos, neu, neg),
        "languages": languages,
        "duplicates": find_duplicates(english, threshold=duplicate_threshold),
        "translation_failed": len(failed),
        "translation_failed_by_language": dict(failed_languages),
        "fallback_used": fallback_used,
    }
