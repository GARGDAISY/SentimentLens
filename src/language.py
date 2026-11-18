"""Offline language detection and translation to English with Argos Translate.

Supported language routes: English, Hindi, Hinglish, Mandarin Chinese, Spanish,
Arabic, French and Bengali. Argos language packages are installed explicitly via
scripts/install_argos_models.py; the application never downloads models at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import re

from langdetect import DetectorFactory, detect

from .hinglish import has_devanagari, is_hinglish, normalize_hinglish

DetectorFactory.seed = 0

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "hinglish": "Hinglish",
    "zh": "Mandarin Chinese", "zh-cn": "Mandarin Chinese", "zh-tw": "Mandarin Chinese",
    "es": "Spanish", "ar": "Arabic", "fr": "French", "bn": "Bengali",
    "unknown": "Unknown",
}
SUPPORTED_ARGOS_CODES = {"hi", "zh", "zh-cn", "zh-tw", "es", "ar", "fr", "bn"}


@dataclass
class Prepared:
    original: str
    english: str
    lang_code: str
    language: str
    translated: bool


def detect_language(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown"
    if is_hinglish(text):
        return "hinglish"
    # Explicit script checks improve short Hindi/Arabic/Bengali/Chinese samples.
    if has_devanagari(text):
        return "hi"
    if re.search(r"[\u0980-\u09ff]", text):
        return "bn"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    try:
        code = detect(text).lower()
        if code.startswith("zh"):
            return "zh"
        return code
    except Exception:
        return "unknown"


def _installed_argos_languages():
    try:
        import argostranslate.translate
        return argostranslate.translate.get_installed_languages()
    except Exception:
        return []


def installed_argos_pairs() -> set[tuple[str, str]]:
    pairs = set()
    for language in _installed_argos_languages():
        for translation in getattr(language, "translations", []):
            pairs.add((language.code, translation.to_lang.code))
    return pairs


def argos_model_status() -> dict:
    installed = installed_argos_pairs()
    required = {"zh", "hi", "es", "ar", "fr", "bn"}
    return {
        "installed": sorted(code for code in required if (code, "en") in installed),
        "missing": sorted(code for code in required if (code, "en") not in installed),
        "hindi_installed": ("hi", "en") in installed,
    }


def _argos_translate(text: str, lang_code: str) -> Optional[str]:
    """Translate with an already-installed Argos model; never downloads anything."""
    normalized = "zh" if lang_code.startswith("zh") else lang_code
    try:
        import argostranslate.translate
        languages = argostranslate.translate.get_installed_languages()
        source = next((x for x in languages if x.code == normalized), None)
        target = next((x for x in languages if x.code == "en"), None)
        if not source or not target:
            return None
        translation = source.get_translation(target)
        if translation is None:
            return None
        result = translation.translate(text)
        return result.strip() if result and result.strip() else None
    except Exception:
        return None


def translate_texts(
    texts: List[str],
    translate_fn: Optional[Callable[[List[str]], List[str]]] = None,
    retries: int = 0,
    delay: float = 0.0,
    max_consecutive_failures: int = 3,
) -> List[Optional[str]]:
    """Compatibility helper for injected batch translators used by tests/tools."""
    if translate_fn is None:
        return [None for _ in texts]
    out: List[Optional[str]] = []
    failures = 0
    for text in texts:
        if failures >= max_consecutive_failures:
            out.append(None)
            continue
        try:
            value = translate_fn([text])[0]
            value = value.strip() if value else None
            out.append(value or None)
            failures = 0 if value else failures + 1
        except Exception:
            out.append(None)
            failures += 1
    return out


def prepare_reviews(
    reviews: List[str],
    translate_fn: Optional[Callable[[List[str]], List[str]]] = None,
    cache: Optional[dict] = None,
) -> List[Prepared]:
    """Detect and translate reviews. Cache keys include source language and text."""
    cache = cache if cache is not None else {}
    prepared: List[Optional[Prepared]] = [None] * len(reviews)
    pending: dict[tuple[str, str], list[int]] = {}

    for i, raw in enumerate(reviews):
        text = str(raw or "").strip()
        code = detect_language(text)
        name = LANGUAGE_NAMES.get(code, code.upper() if code != "unknown" else "Unknown")
        if not text:
            prepared[i] = Prepared(text, text, code, name, False)
        elif code == "en":
            prepared[i] = Prepared(text, text, code, name, True)
        elif code == "hinglish":
            prepared[i] = Prepared(text, normalize_hinglish(text), code, name, True)
        elif code in SUPPORTED_ARGOS_CODES:
            key = (code, text)
            cached = cache.get(key)
            if cached:
                prepared[i] = Prepared(text, cached, code, name, True)
            else:
                pending.setdefault(key, []).append(i)
        else:
            prepared[i] = Prepared(text, text, code, name, False)

    for (code, text), indices in pending.items():
        translated = None
        if translate_fn is not None:
            translated_values = translate_texts([text], translate_fn)
            translated = translated_values[0] if translated_values else None
        else:
            translated = _argos_translate(text, code)
        if translated:
            cache[(code, text)] = translated
        name = LANGUAGE_NAMES.get(code, code.upper())
        for i in indices:
            prepared[i] = Prepared(text, translated or text, code, name, bool(translated))

    return [item for item in prepared if item is not None]
