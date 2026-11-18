"""Downloadable PDF and CSV reports without confidence/duplicate filler columns."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def to_csv_bytes(result: dict) -> bytes:
    return result["df"].to_csv(index=False).encode("utf-8-sig")


def _safe(text: str, limit: int = 140) -> str:
    text = "".join(ch if ord(ch) < 256 else "?" for ch in str(text))
    return escape(text if len(text) <= limit else text[: limit - 1] + "...")


def to_pdf_bytes(result: dict, product_name: str = "Product") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=6)
    body = styles["BodyText"]
    story = [
        Paragraph("Review Sentiment Report", title),
        Paragraph(f"{_safe(product_name)} | {datetime.now():%d %b %Y, %H:%M}", body),
        Spacer(1, 12),
    ]
    verdict = result["verdict"]
    story += [Paragraph(f"<b>Overall verdict: {verdict['label']}</b>", styles["Heading2"]),
              Paragraph(escape(verdict["text"]), body), Spacer(1, 10)]
    counts, percentages = result["counts"], result["pct"]
    summary = [["Sentiment", "Reviews", "Share"],
               ["Positive", counts["positive"], f"{percentages['positive']}%"],
               ["Neutral", counts["neutral"], f"{percentages['neutral']}%"],
               ["Negative", counts["negative"], f"{percentages['negative']}%"],
               ["Total", result["total"], "100%"]]
    table = Table(summary, hAlign="LEFT", colWidths=[5 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213D")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8BED6"))]))
    story += [table, Spacer(1, 14), Paragraph("Languages detected", styles["Heading3"])]
    lang = [["Language", "Reviews", "Share"]] + [
        [x["language"], x["count"], f"{x['pct']}%"] for x in result["languages"]]
    lang_table = Table(lang, hAlign="LEFT", colWidths=[5 * cm, 3 * cm, 3 * cm])
    lang_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213D")),
                                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8BED6"))]))
    story += [lang_table, Spacer(1, 12)]
    if result["duplicates"]:
        story.append(Paragraph(f"{len(result['duplicates'])} pair(s) of near-identical reviews detected. Similarity is a signal, not proof of fake reviews.", body))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Sample of analysed reviews (first 25)", styles["Heading3"]))
    rows = [["#", "Review (English)", "Sentiment"]]
    for index, row in enumerate(result["df"].head(25).itertuples(index=False), 1):
        rows.append([index, Paragraph(_safe(row.english), body), row.sentiment.title()])
    review_table = Table(rows, hAlign="LEFT", colWidths=[1 * cm, 12.5 * cm, 3 * cm], repeatRows=1)
    review_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213D")),
                                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                      ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8BED6")),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(review_table)
    doc.build(story)
    return buf.getvalue()
