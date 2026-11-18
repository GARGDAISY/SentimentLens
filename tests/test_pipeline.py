"""Run with:  python -m pytest -q   (or:  python tests/test_pipeline.py)"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis import analyze, verdict_from_counts
from src.extractors import extract_reviews, split_into_reviews
from src.hinglish import is_hinglish, normalize_hinglish
from src.model import load_model
from src.similarity import find_duplicates

MODEL = load_model()


def fake_translate(texts):
    return ["This product is very good" for _ in texts]


def test_split_lines_and_paragraphs():
    assert split_into_reviews("1. Great\n2. Bad one\n\n") == ["Great", "Bad one"]
    assert split_into_reviews("Line one\nline two\n\nSecond review") == ["Line one line two", "Second review"]


def test_hinglish_detection_and_negation():
    assert is_hinglish("Battery bahut accha hai")
    assert not is_hinglish("The battery is very good")
    assert normalize_hinglish("accha nahi hai") == "not good"
    assert "value for money" in normalize_hinglish("paisa vasool product")


def test_extractors_all_formats():
    reviews = ["Great product, loved it", "Terrible, broke in a day", "It is okay I guess"]
    txt = "\n".join(reviews).encode()
    assert extract_reviews("a.txt", txt) == reviews

    csv = pd.DataFrame({"id": [1, 2, 3], "Review": reviews}).to_csv(index=False).encode()
    assert extract_reviews("a.csv", csv) == reviews

    buf = io.BytesIO()
    pd.DataFrame({"id": [1, 2, 3], "comment": reviews}).to_excel(buf, index=False)
    assert extract_reviews("a.xlsx", buf.getvalue()) == reviews

    from docx import Document
    d = Document()
    for r in reviews:
        d.add_paragraph(r)
    b = io.BytesIO()
    d.save(b)
    assert extract_reviews("a.docx", b.getvalue()) == reviews

    from reportlab.pdfgen import canvas
    b = io.BytesIO()
    c = canvas.Canvas(b)
    for i, r in enumerate(reviews):
        c.drawString(72, 750 - 20 * i, r)
    c.save()
    assert extract_reviews("a.pdf", b.getvalue()) == reviews


def test_duplicates():
    pairs = find_duplicates(["Great sound quality", "Great sound quality", "Awful battery life"])
    assert len(pairs) == 1 and pairs[0]["similarity"] > 0.99


def test_analysis_end_to_end():
    reviews = ["Great sound and amazing battery", "Terrible quality, broke in a week",
               "It is okay, nothing special", "Battery bahut accha hai, sound mast hai",
               "La entrega fue rápida"]
    r = analyze(reviews, MODEL, translate_fn=fake_translate)
    assert r["total"] == 5
    assert sum(r["counts"].values()) == 5
    assert {x["language"] for x in r["languages"]} >= {"English", "Hinglish"}
    assert r["df"].loc[0, "sentiment"] == "positive"
    assert r["df"].loc[1, "sentiment"] == "negative"


def test_translation_failure_does_not_crash():
    def boom(_):
        raise RuntimeError("blocked")
    r = analyze(["La entrega fue rápida"], MODEL, translate_fn=boom)
    assert r["translation_failed"] == 1 and r["total"] == 1


def test_verdict_rules():
    assert verdict_from_counts(80, 10, 10)["label"] == "Positive"
    assert verdict_from_counts(10, 10, 80)["label"] == "Negative"
    assert verdict_from_counts(45, 10, 45)["label"] == "Mixed"


def test_reports():
    from src.report import to_csv_bytes, to_pdf_bytes
    r = analyze(["Great product", "Bad product"], MODEL)
    assert to_pdf_bytes(r, "Test").startswith(b"%PDF")
    assert b"sentiment" in to_csv_bytes(r)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_emoji_only_and_mixed_sentiment():
    from src.emoji_sentiment import combine_emoji_sentiment
    neutral = {"sentiment": "neutral", "confidence": 0.6,
               "p_negative": 0.1, "p_neutral": 0.8, "p_positive": 0.1}
    positive = combine_emoji_sentiment("😍❤️🥰", neutral)
    assert positive["sentiment"] == "positive"
    negative = combine_emoji_sentiment("😡🤬", neutral)
    assert negative["sentiment"] == "negative"
    mixed = combine_emoji_sentiment("Amazing product 😍", {
        "sentiment": "positive", "confidence": 0.9,
        "p_negative": 0.02, "p_neutral": 0.08, "p_positive": 0.9,
    })
    assert mixed["sentiment"] == "positive"


def test_result_table_keeps_original_fields_without_extra_columns():
    result = analyze(["Great battery life", "Bad delivery"], MODEL)
    assert list(result["df"].columns) == ["review", "english", "language", "sentiment"]
    assert "confidence" not in result
    assert "aspect_summary" not in result
    assert "possible_duplicate" not in result["df"].columns


def test_language_routes_detect_requested_languages():
    from src.language import detect_language
    assert detect_language("Este producto es excelente y funciona muy bien") == "es"
    assert detect_language("Ce produit est vraiment excellent") == "fr"
    assert detect_language("هذا المنتج رائع جدا") == "ar"
    assert detect_language("यह उत्पाद बहुत अच्छा है") == "hi"
    assert detect_language("这个产品非常好用") in {"zh", "zh-cn"}
    assert detect_language("এই পণ্যটি খুব ভালো") == "bn"


def test_all_argos_language_routes_and_translation_cache(monkeypatch):
    from src import language
    route_by_text = {
        "sample chinese": "zh",
        "sample hindi": "hi",
        "sample spanish": "es",
        "sample arabic": "ar",
        "sample french": "fr",
        "sample bengali": "bn",
    }
    monkeypatch.setattr(language, "detect_language", lambda text: route_by_text[text])
    calls = []
    monkeypatch.setattr(language, "_argos_translate",
                        lambda text, code: calls.append((text, code)) or f"English {code}")
    cache = {}
    reviews = list(route_by_text)
    first = language.prepare_reviews(reviews, cache=cache)
    second = language.prepare_reviews(reviews, cache=cache)
    assert [item.lang_code for item in first] == list(route_by_text.values())
    assert all(item.translated for item in first + second)
    assert len(calls) == 6  # second pass is served from cache


def test_context_dependent_emojis_are_not_forced_into_a_polarity():
    from src.emoji_sentiment import emoji_signal
    assert emoji_signal("😂🔥💀😭") is None
