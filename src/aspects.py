"""Lightweight keyword-based aspect extraction for transparent review insights.

This is a heuristic, not a trained aspect-based sentiment model. It works on the
English text produced by the language pipeline and reports aspects only when found.
"""
from __future__ import annotations

from collections import defaultdict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ASPECTS = {
    "Battery": ("battery", "charging", "charge", "backup", "backup time"),
    "Camera": ("camera", "photo", "picture", "video quality"),
    "Price": ("price", "cost", "expensive", "cheap", "value for money", "overpriced"),
    "Delivery": ("delivery", "shipping", "arrived", "late", "courier"),
    "Product quality": ("quality", "build", "durable", "material", "broken", "broke"),
    "Customer service": ("customer service", "support", "refund", "replacement", "seller"),
    "Performance": ("performance", "speed", "slow", "lag", "fast", "working"),
    "Sound": ("sound", "audio", "bass", "volume", "speaker", "microphone"),
}
_ANALYZER = SentimentIntensityAnalyzer()


def extract_aspects(text: str) -> list[dict[str, str]]:
    lower = str(text).lower()
    tokens = list(re.finditer(r"\b[\w']+\b", lower))
    words = [match.group(0) for match in tokens]
    found = []
    for aspect, terms in _ASPECTS.items():
        positions = []
        for term in terms:
            term_words = term.split()
            width = len(term_words)
            for index in range(0, max(0, len(words) - width + 1)):
                if words[index:index + width] == term_words:
                    positions.append((index, index + width))
        if not positions:
            continue
        context_scores = []
        for start, end in positions:
            left, right = max(0, start - 3), min(len(words), end + 3)
            context = " ".join(words[left:right])
            context_scores.append(_ANALYZER.polarity_scores(context)["compound"])
        score = sum(context_scores) / len(context_scores)
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        found.append({"aspect": aspect, "sentiment": label})
    return found


def summarize_aspects(reviews: list[list[dict[str, str]]]) -> list[dict[str, int | str]]:
    counts = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for review_aspects in reviews:
        for item in review_aspects:
            counts[item["aspect"]][item["sentiment"]] += 1
    return [
        {"aspect": aspect, **values, "mentions": sum(values.values())}
        for aspect, values in sorted(counts.items(), key=lambda x: -sum(x[1].values()))
    ]
