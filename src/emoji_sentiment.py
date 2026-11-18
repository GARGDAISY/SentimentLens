"""Transparent emoji sentiment signals, combined conservatively with text sentiment."""
from __future__ import annotations

import re
from typing import Dict, Optional

# Deliberately conservative: ambiguous/context-dependent emojis are not assigned a label.
EMOJI_POLARITY = {
    "😍": "positive", "🥰": "positive", "😊": "positive", "😃": "positive",
    "😄": "positive", "😁": "positive", "😀": "positive", "🙂": "positive",
    "❤️": "positive", "❤": "positive", "💕": "positive", "💖": "positive",
    "👍": "positive", "👏": "positive", "🎉": "positive", "🤩": "positive",
    "😘": "positive", "💯": "positive", "🙌": "positive", "✨": "positive",
    "😡": "negative", "🤬": "negative", "😠": "negative", "😤": "negative",
    "😞": "negative", "😔": "negative", "😢": "negative",
    "☹️": "negative", "🙁": "negative", "👎": "negative", "💔": "negative",
    "🤮": "negative", "🤢": "negative", "😫": "negative", "😩": "negative",
    "😐": "neutral", "😑": "neutral", "😶": "neutral", "🤔": "neutral",
    "😕": "neutral", "🙄": "neutral",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F\u0600-\u06FF\u4e00-\u9fff]+")


def emoji_signal(text: str) -> Optional[Dict[str, float]]:
    found = [EMOJI_POLARITY[e] for e in EMOJI_POLARITY if e in str(text)]
    if not found:
        return None
    counts = {label: found.count(label) for label in ("negative", "neutral", "positive")}
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def has_text_content(text: str) -> bool:
    return bool(_TOKEN_RE.search(str(text)))


def combine_emoji_sentiment(text: str, prediction: Dict[str, float]) -> Dict[str, float]:
    """Use emoji-only signal directly; use a modest 20% emoji weight for mixed reviews."""
    signal = emoji_signal(text)
    if not signal:
        return prediction
    keys = ("negative", "neutral", "positive")
    if not has_text_content(text):
        probs = signal
    else:
        weight = 0.20
        probs = {
            key: (1.0 - weight) * float(prediction.get(f"p_{key}", 0.0)) + weight * signal[key]
            for key in keys
        }
    total = sum(probs.values()) or 1.0
    probs = {key: value / total for key, value in probs.items()}
    label = max(probs, key=probs.get)
    return {
        **prediction,
        "sentiment": label,
        "confidence": probs[label],
        "p_negative": probs["negative"],
        "p_neutral": probs["neutral"],
        "p_positive": probs["positive"],
    }
