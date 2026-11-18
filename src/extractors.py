"""Extract individual reviews from PDF, DOCX, XLSX/XLS, CSV, TXT files or raw text."""
from __future__ import annotations

import io
import re
from typing import List

import pandas as pd

SUPPORTED_EXTENSIONS = ("pdf", "docx", "xlsx", "xls", "csv", "txt")

# Column names that usually hold the review text in a spreadsheet.
_TEXT_COLUMN_HINTS = (
    "review", "reviews", "review_text", "reviewtext", "text", "comment",
    "comments", "feedback", "content", "body", "description", "message",
)

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u25CF]|\d{1,4}[.)])\s+")


def _clean_line(line: str) -> str:
    line = _BULLET_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def split_into_reviews(text: str, min_chars: int = 3) -> List[str]:
    """Split a block of text into reviews.

    - If reviews are separated by blank lines, each paragraph is one review.
    - Otherwise every non-empty line is one review.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        reviews = [_clean_line(" ".join(p.splitlines())) for p in paragraphs]
    else:
        reviews = [_clean_line(line) for line in text.split("\n")]
    return [r for r in reviews if len(r) >= min_chars]


def _pick_text_column(df: pd.DataFrame) -> str:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for hint in _TEXT_COLUMN_HINTS:
        if hint in lowered:
            return lowered[hint]
    # Fallback: the text column with the longest average length.
    best, best_len = None, -1.0
    for col in df.columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        avg = series.str.len().mean()
        if avg > best_len:
            best, best_len = col, avg
    if best is None:
        raise ValueError("The spreadsheet has no usable text column.")
    return best


def _from_dataframe(df: pd.DataFrame) -> List[str]:
    col = _pick_text_column(df)
    values = df[col].dropna().astype(str).tolist()
    return [_clean_line(v) for v in values if len(_clean_line(v)) >= 3]


def _read_txt(data: bytes) -> List[str]:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return split_into_reviews(data.decode(enc))
        except (UnicodeDecodeError, UnicodeError):
            continue
    return []


def _read_pdf(data: bytes) -> List[str]:
    import pdfplumber

    chunks = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return split_into_reviews("\n".join(chunks))


def _read_docx(data: bytes) -> List[str]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    # Reviews stored inside tables (one per cell) are picked up too.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    lines.append(cell.text)
    return split_into_reviews("\n".join(lines))


def extract_reviews(filename: str, data: bytes) -> List[str]:
    """Return a list of reviews from one uploaded file."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        return _read_txt(data)
    if ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        return _from_dataframe(df)
    if ext in ("xlsx", "xls"):
        return _from_dataframe(pd.read_excel(io.BytesIO(data)))
    if ext == "docx":
        return _read_docx(data)
    if ext == "pdf":
        return _read_pdf(data)
    raise ValueError(f"Unsupported file type: .{ext}")


def collect_reviews(files=None, pasted_text: str = "") -> tuple[List[str], List[str]]:
    """Combine reviews from uploaded files and pasted text.

    Returns (reviews, warnings).
    """
    reviews: List[str] = []
    warnings: List[str] = []
    for f in files or []:
        try:
            found = extract_reviews(f.name, f.getvalue())
            if not found:
                warnings.append(f"No reviews found in {f.name}.")
            reviews.extend(found)
        except Exception as exc:  # keep the app alive on a bad file
            warnings.append(f"Could not read {f.name}: {exc}")
    if pasted_text and pasted_text.strip():
        reviews.extend(split_into_reviews(pasted_text))
    return reviews, warnings
