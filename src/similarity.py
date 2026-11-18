"""Sparse TF-IDF similarity helpers for duplicate review detection."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.sparse import triu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text


def _tfidf(texts: List[str]):
    cleaned = [clean_text(text) for text in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        return vectorizer.fit_transform(cleaned), cleaned
    except ValueError:
        return None, cleaned


def find_duplicates(texts: List[str], threshold: float = 0.9) -> List[Dict]:
    """Return near-duplicate pairs using sparse matrix multiplication (no dense NxN matrix)."""
    if len(texts) < 2:
        return []
    matrix, cleaned = _tfidf(texts)
    if matrix is None:
        return []
    similarities = triu(matrix @ matrix.T, k=1, format="coo")
    pairs = []
    for i, j, score in zip(similarities.row, similarities.col, similarities.data):
        if score >= threshold and cleaned[i] and cleaned[j]:
            pairs.append({"review_a": int(i), "review_b": int(j), "similarity": round(float(score), 3)})
    return sorted(pairs, key=lambda item: -item["similarity"])


def duplicate_indices(pairs: List[Dict]) -> set:
    return {pair["review_b"] for pair in pairs}


def representative_reviews(texts: List[str], labels: List[str], sentiment: str, top_n: int = 3) -> List[str]:
    indices = [i for i, label in enumerate(labels) if label == sentiment]
    if not indices:
        return []
    matrix, _ = _tfidf(texts)
    if matrix is None:
        return [texts[i] for i in indices[:top_n]]
    subset = matrix[indices]
    centroid = np.asarray(subset.mean(axis=0))
    similarities = cosine_similarity(subset, centroid).ravel()
    order = np.argsort(-similarities)[:top_n]
    return [texts[indices[k]] for k in order]
