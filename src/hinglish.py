"""Rule-based Hinglish (romanized Hindi) detection and normalisation to English.

Google Translate often fails on romanized Hindi ("product mast hai"), so Hinglish
is converted with a small dictionary (data/hinglish_dictionary.csv) instead.
Negation ("nahi", "na", "mat") is handled by moving "not" in front of the word.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "hinglish_dictionary.csv"

# Filler / grammar words that carry no sentiment and are dropped.
FILLERS = {
    "hai", "hain", "h", "hu", "hun", "ho", "tha", "thi", "the", "hoga", "hogi",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "main", "par", "pe", "ne",
    "to", "toh", "bhi", "hi", "ye", "yeh", "yah", "wo", "woh", "iska", "isme",
    "is", "ek", "aur", "or", "lekin", "par", "magar", "kyunki", "kyuki",
    "mujhe", "mera", "meri", "mere", "humne", "maine", "hum", "main",
}
NEGATIONS = {"nahi", "nahin", "nai", "na", "mat", "nhi", "nahii"}

# Words that mostly occur in romanized Hindi; used for detection only.
_MARKERS = {
    "hai", "hain", "nahi", "nahin", "nhi", "bahut", "bohot", "boht", "accha", "acha",
    "achha", "achi", "achhi", "bekar", "bekaar", "mast", "kharab", "paisa", "paise",
    "vasool", "wasool", "badhiya", "badiya", "zabardast", "jabardast", "bakwas",
    "faltu", "ghatiya", "sasta", "mehnga", "mehenga", "kaafi", "bilkul", "ekdum",
    "tha", "thi", "kya", "kyun", "lekin", "magar", "mujhe", "maine", "mera", "meri",
    "ka", "ki", "ke", "ko", "mein", "tez", "dheema", "sahi", "galat", "pasand",
    "kamaal", "shandar", "lajawab", "toot", "tut", "dikkat", "jaldi",
}

_TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^\sA-Za-z\d]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=1)
def load_dictionary() -> dict[str, str]:
    df = pd.read_csv(DICT_PATH).dropna()
    return {str(h).strip().lower(): str(e).strip().lower() for h, e in zip(df["hinglish"], df["english"])}


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def is_hinglish(text: str) -> bool:
    """True if the text is written in Latin script and contains Hindi marker words."""
    if has_devanagari(text) or not _LATIN_RE.search(text):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _MARKERS)
    # Strong markers (not shared with English words like "to", "or", "me") count more.
    return hits >= 2 or (hits >= 1 and len(words) <= 4)


def normalize_hinglish(text: str) -> str:
    """Convert a Hinglish review into (rough) English for the sentiment pipeline."""
    dictionary = load_dictionary()
    words = re.findall(r"[A-Za-z']+|[^\sA-Za-z']+", text.lower())
    out: list[str] = []
    i = 0
    while i < len(words):
        # Longest phrase match first (3, 2, then 1 word), e.g. "paisa vasool".
        matched = False
        for n in (3, 2):
            phrase = " ".join(words[i:i + n])
            if len(words[i:i + n]) == n and phrase in dictionary:
                out.append(dictionary[phrase])
                i += n
                matched = True
                break
        if matched:
            continue
        w = words[i]
        if w in NEGATIONS:
            # "accha nahi" -> "not good": put "not" in front of the previous word.
            if out and out[-1] not in {"not", "never"}:
                out.insert(len(out) - 1, "not")
            else:
                out.append("not")
        elif w in dictionary:
            out.append(dictionary[w])
        elif w in FILLERS:
            pass
        else:
            out.append(w)
        i += 1
    return re.sub(r"\s+", " ", " ".join(out)).strip()
