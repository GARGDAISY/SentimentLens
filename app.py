"""Review Sentiment Analyzer - Streamlit app (neon dark theme)."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.analysis import analyze
from src.extractors import SUPPORTED_EXTENSIONS, collect_reviews, split_into_reviews
from src.model import load_model
from src.language import argos_model_status
from src.report import to_csv_bytes, to_pdf_bytes

ROOT = Path(__file__).parent
DEMO_PATH = ROOT / "data" / "sample_reviews" / "demo_reviews.txt"

GREEN, YELLOW, PINK, CYAN, PURPLE = "#39FF88", "#FFD23F", "#FF3D81", "#22D3EE", "#A78BFA"

st.set_page_config(page_title="Review Sentiment Analyzer", layout="wide")

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid #262A52; }}
.neon-title {{ font-size: 1.25rem; font-weight: 700; color: {CYAN};
  text-shadow: 0 0 14px rgba(34,211,238,.6); margin-bottom: .5rem; }}
.card {{ background: #12142A; border: 1px solid #2A2D55; border-radius: 14px; padding: 22px; }}
.cap {{ font-size: .75rem; font-weight: 600; color: #9AA0C8; letter-spacing: .08em; }}
.big {{ font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin-top: 6px; }}
.sub {{ color: #B9BEE0; margin-top: 6px; }}
.bar {{ height: 8px; border-radius: 4px; background: #1E2145; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 4px; }}
.lang-row {{ display: flex; justify-content: space-between; margin: 12px 0 6px; color: #E8EAFF; }}
.lang-row span:last-child {{ color: #9AA0C8; }}
div.stButton > button {{ border-radius: 10px; font-weight: 700; }}
div.stDownloadButton > button {{ border-radius: 10px; font-weight: 700;
  border: 1px solid {CYAN}; color: {CYAN}; background: transparent; }}
div.stDownloadButton > button:hover {{ box-shadow: 0 0 16px rgba(34,211,238,.45); }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_model():
    return load_model()


def metric_card(label: str, pct: float, count: int, color: str) -> str:
    return (
        f'<div class="card" style="border-color:{color}">'
        f'<div class="cap">{label}</div>'
        f'<div class="big" style="color:{color};text-shadow:0 0 14px {color}80">{pct:g}%</div>'
        f'<div class="sub">{count:,} reviews</div></div>'
    )


def run_analysis(reviews):
    cache = st.session_state.setdefault("translation_cache", {})
    result = analyze(reviews, get_model(), cache=cache)
    st.session_state["result"] = result
    st.session_state["pdf"] = to_pdf_bytes(result)
    st.session_state["csv"] = to_csv_bytes(result)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="neon-title">Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload reviews (drop files here)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help="PDF, DOCX, XLSX, CSV or TXT. One review per line or paragraph; "
             "for spreadsheets the review column is detected automatically.",
    )
    pasted = st.text_area("Paste reviews (one per line)", height=180,
                          placeholder="Battery bahut accha hai, sound mast hai\nCase quality is poor...")
    analyze_clicked = st.button("Analyze reviews", type="primary", width="stretch")
    demo_clicked = st.button("Try demo data", width="stretch")
    with st.expander("Language model setup"):
        status = argos_model_status()
        st.caption("Argos Translate runs locally. Models are never downloaded automatically.")
        st.write("Installed:", ", ".join(status["installed"]) or "None")
        st.write("Missing:", ", ".join(status["missing"]) or "None")
        st.caption("Install missing models with: python scripts/install_argos_models.py")

if demo_clicked:
    run_reviews = split_into_reviews(DEMO_PATH.read_text(encoding="utf-8"))
    with st.spinner(f"Analyzing {len(run_reviews)} demo reviews..."):
        run_analysis(run_reviews)
elif analyze_clicked:
    run_reviews, warns = collect_reviews(files, pasted)
    for w in warns:
        st.warning(w)
    if not run_reviews:
        st.error("Please upload a file or paste some reviews first.")
    else:
        with st.spinner(f"Analyzing {len(run_reviews)} reviews (translating where needed)..."):
            run_analysis(run_reviews)

# ---------------------------------------------------------------- main
st.title("Analysis results")
result = st.session_state.get("result")

if not result:
    st.markdown(
        '<div class="card"><div class="cap">GET STARTED</div>'
        '<div class="sub">Upload review files or paste reviews in the sidebar, then click '
        '<b>Analyze reviews</b>. Or press <b>Try demo data</b> to see an example with English, '
        'Hinglish, Hindi, Spanish, French and other sample reviews.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

v, counts, pct = result["verdict"], result["counts"], result["pct"]
st.caption(f"{result['total']:,} reviews analysed")

st.markdown(
    f'<div class="card" style="border-color:{v["color"]};box-shadow:0 0 28px {v["color"]}40>'
    f'<div class="cap">OVERALL VERDICT</div>'
    f'<div style="font-size:2.8rem;font-weight:700;color:{v["color"]};'
    f'text-shadow:0 0 18px {v["color"]}90;margin:8px 0">{v["label"]}</div>'
    f'<div class="sub" style="max-width:640px">{v["text"]}</div></div>',
    unsafe_allow_html=True,
)

st.write("")

c1, c2, c3 = st.columns(3)
c1.markdown(metric_card("POSITIVE", pct["positive"], counts["positive"], GREEN), unsafe_allow_html=True)
c2.markdown(metric_card("NEUTRAL", pct["neutral"], counts["neutral"], YELLOW), unsafe_allow_html=True)
c3.markdown(metric_card("NEGATIVE", pct["negative"], counts["negative"], PINK), unsafe_allow_html=True)
st.write("")

left, right = st.columns(2)
with left:
    st.markdown('<div class="card"><div class="cap">SENTIMENT SPLIT</div></div>', unsafe_allow_html=True)
    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[counts["positive"], counts["neutral"], counts["negative"]],
        hole=0.62, sort=False,
        marker=dict(colors=[GREEN, YELLOW, PINK], line=dict(color="#0A0B14", width=3)),
        textinfo="percent", textfont=dict(color="#0A0B14", size=14),
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EAFF"), legend=dict(orientation="h", y=-0.05),
        annotations=[dict(text=f"{result['total']:,}<br>reviews", showarrow=False,
                          font=dict(size=18, color="#E8EAFF"))],
    )
    st.plotly_chart(fig, width="stretch")

with right:
    palette = [CYAN, PURPLE]
    rows = "".join(
        f'<div class="lang-row"><span>{x["language"]}</span><span>{x["pct"]:g}% ({x["count"]})</span></div>'
        f'<div class="bar"><i style="width:{x["pct"]}%;background:{palette[i % 2]};'
        f'box-shadow:0 0 8px {palette[i % 2]}"></i></div>'
        for i, x in enumerate(result["languages"])
    )
    st.markdown(f'<div class="card"><div class="cap">LANGUAGES DETECTED</div>{rows}</div>',
                unsafe_allow_html=True)

if result["translation_failed"]:
    msg = f"{result['translation_failed']} review(s) could not be translated."
    if result["fallback_used"]:
        msg += " The optional multilingual model handled some of them."
    else:
        msg += " They were analysed in their original language by the English-focused model; treat these predictions with caution."
    st.warning(msg)
    if result.get("translation_failed_by_language"):
        st.caption("Affected languages: " + ", ".join(
            f"{language} ({count})" for language, count in result["translation_failed_by_language"].items()
        ))

st.write("")
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download PDF report", st.session_state["pdf"],
                   file_name="review_sentiment_report.pdf", mime="application/pdf")
d2.download_button("Download CSV", st.session_state["csv"],
                   file_name="review_sentiment_results.csv", mime="text/csv")

with st.expander("Review-level details"):
    if result["duplicates"]:
        st.info(f"{len(result['duplicates'])} pair(s) of near-identical reviews found (TF-IDF cosine similarity >= 0.9). They are flagged by the duplicate-detection feature.")
    st.dataframe(result["df"], width="stretch", hide_index=True)
