"""Text cleaning used before TF-IDF."""
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9'\s]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # sooooo -> soo

# Keep negations: they flip sentiment ("not good").
_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "it",
    "its", "this", "that", "these", "those", "of", "to", "in", "on", "at", "for",
    "with", "as", "by", "from", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "and", "or", "so", "if", "then", "there", "here",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = text.replace("n't", " not").replace("’", "'")
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)
