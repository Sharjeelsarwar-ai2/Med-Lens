# ──────────────────────────────────────────────────────────────
# MedLens — AI Medical Report Translator  (with OCR support)
# Python + Streamlit + Groq  |  Streamlit Cloud ready
# ──────────────────────────────────────────────────────────────

import hashlib
import html
import io
import json
import os
import re
import textwrap
import types as _types
import typing as _typing
from typing import Literal

import fitz  # PyMuPDF
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from groq import (
    Groq,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field

# ─── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="MedLens · Medical Report Translator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Constants ────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_UPLOAD_MB = 10
MAX_TEXT_CHARS = 12_000
MAX_QUESTION_CHARS = 500

STATUS_COLORS = {
    "Normal": "#2dd4bf",
    "Borderline": "#f59e0b",
    "Abnormal": "#fb7185",
    "Informational": "#a78bfa",
}

STATUS_ICONS = {
    "Normal": "✅",
    "Borderline": "⚠️",
    "Abnormal": "🔴",
    "Informational": "ℹ️",
}

# ─── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Sora:wght@400;500;600;700;800&display=swap');

/* ── Palette ──
   Primary:  Teal   #2dd4bf / #14b8a6
   Accent:   Violet #a78bfa / #8b5cf6
   Warn:     Amber  #f59e0b
   Danger:   Rose   #fb7185
   Base:     Near-black graphite, not navy — deliberately distinct from
   the previous indigo/emerald build.
*/

/* ── Remove Streamlit chrome that masks the hero ── */
[data-testid="stHeader"]{
    background:transparent !important;
    height:0 !important;
    min-height:0 !important;
}
[data-testid="stToolbar"]{display:none !important}
[data-testid="stDecoration"]{display:none !important}
footer{visibility:hidden}
#MainMenu{visibility:hidden}

/* ── Global ── */
html, body, [class*="css"]{
    font-family:'Space Grotesk',-apple-system,BlinkMacSystemFont,sans-serif;
}
.stApp{
    background:
        radial-gradient(circle at 12% 10%, rgba(45,212,191,.11), transparent 38%),
        radial-gradient(circle at 90% 15%, rgba(167,139,250,.10), transparent 42%),
        radial-gradient(circle at 50% 105%, rgba(251,113,133,.05), transparent 48%),
        #05070a;
}
.block-container{max-width:1480px;padding-top:0.6rem;padding-bottom:3rem}

::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:#05070a}
::-webkit-scrollbar-thumb{
    background:linear-gradient(180deg,#2dd4bf,#a78bfa);
    border-radius:10px;
}

@keyframes ml-gradient-shift{
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
@keyframes ml-float{
    0%,100%{transform:translateY(0) translateX(0)}
    50%{transform:translateY(-8px) translateX(6px)}
}
@keyframes ml-fade-in{
    from{opacity:0;transform:translateY(8px)}
    to{opacity:1;transform:translateY(0)}
}

/* ── Top bar (replaces sidebar) ── */
.ml-topbar{
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 22px;margin-bottom:18px;
    border-radius:18px;
    background:linear-gradient(120deg,rgba(20,20,28,.75),rgba(10,12,18,.6));
    border:1px solid rgba(148,163,184,.12);
    backdrop-filter:blur(12px);
}
.ml-brand{
    font-family:'Sora',sans-serif;font-weight:700;font-size:18px;
    color:#f1f5f9;letter-spacing:-.3px;
    display:flex;align-items:center;gap:10px;
}
.ml-brand .ml-dot{
    width:9px;height:9px;border-radius:50%;
    background:linear-gradient(135deg,#2dd4bf,#a78bfa);
    box-shadow:0 0 12px rgba(45,212,191,.8);
}
.ml-status-pill{
    display:inline-flex;align-items:center;gap:6px;
    border-radius:999px;padding:6px 14px;
    font-size:11.5px;font-weight:700;letter-spacing:.4px;
}
.ml-status-on{background:rgba(45,212,191,.12);color:#5eead4;border:1px solid rgba(45,212,191,.3)}
.ml-status-off{background:rgba(251,113,133,.12);color:#fda4af;border:1px solid rgba(251,113,133,.3)}

/* ── Hero banner (asymmetric split) ── */
.ml-hero{
    display:grid;grid-template-columns:1.5fr 1fr;gap:36px;align-items:center;
    background:
        radial-gradient(ellipse at 95% 10%,rgba(167,139,250,.30),transparent 55%),
        radial-gradient(ellipse at 5% 100%,rgba(45,212,191,.24),transparent 50%),
        linear-gradient(155deg,#0a0e14 0%,#12161f 55%,#0c1015 100%);
    background-size:200% 200%;
    animation:ml-gradient-shift 16s ease infinite;
    border:1px solid rgba(148,163,184,.16);
    border-radius:26px;
    padding:44px 44px;
    margin-bottom:22px;
    color:#f8fafc;
    position:relative;
    overflow:hidden;
    box-shadow:0 30px 80px -24px rgba(45,212,191,.22), inset 0 1px 0 rgba(255,255,255,.05);
}
.ml-hero::after{
    content:"";position:absolute;top:-35%;right:-8%;width:380px;height:380px;
    background:radial-gradient(circle,rgba(167,139,250,.32),transparent 70%);
    filter:blur(8px);animation:ml-float 9s ease-in-out infinite;pointer-events:none;
}
.ml-hero-main{position:relative;z-index:1}
.ml-eyebrow{
    display:inline-flex;align-items:center;gap:8px;
    color:#5eead4;font-size:11px;font-weight:700;
    letter-spacing:2.8px;margin-bottom:16px;
    text-transform:uppercase;
}
.ml-eyebrow::before{
    content:"";width:7px;height:7px;border-radius:50%;
    background:linear-gradient(135deg,#2dd4bf,#a78bfa);
    box-shadow:0 0 10px rgba(45,212,191,.9);
}
.ml-hero h1{
    font-family:'Sora',sans-serif;
    background:linear-gradient(100deg,#ffffff 15%,#99f6e4 50%,#c4b5fd 85%);
    -webkit-background-clip:text;background-clip:text;color:transparent;
    font-size:clamp(30px,4vw,46px);font-weight:800;
    letter-spacing:-1.6px;margin:0 0 14px;line-height:1.12;
}
.ml-hero p{
    color:#cbd5e1;font-size:15px;max-width:640px;
    line-height:1.75;margin:0 0 8px;
}
.ml-pill{
    display:inline-block;border-radius:999px;
    padding:6px 14px;margin-top:14px;margin-right:8px;
    background:linear-gradient(135deg,rgba(45,212,191,.14),rgba(167,139,250,.12));
    border:1px solid rgba(94,234,212,.25);
    color:#d9f2ee;font-size:10.5px;font-weight:700;
    letter-spacing:.7px;
    box-shadow:0 4px 14px rgba(45,212,191,.12);
    transition:transform .2s, box-shadow .2s;
}
.ml-pill:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(45,212,191,.25)}

/* ── Hero side panel ── */
.ml-hero-side{
    position:relative;z-index:1;
    background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.01));
    border:1px solid rgba(148,163,184,.16);
    border-radius:20px;padding:22px 24px;
    backdrop-filter:blur(6px);
}
.ml-hero-side .ml-side-item{
    display:flex;align-items:flex-start;gap:12px;
    padding:10px 0;border-bottom:1px solid rgba(148,163,184,.1);
}
.ml-hero-side .ml-side-item:last-child{border-bottom:none}
.ml-hero-side .ml-side-ico{font-size:18px;line-height:1}
.ml-hero-side .ml-side-title{font-size:13px;font-weight:700;color:#e2e8f0}
.ml-hero-side .ml-side-sub{font-size:11.5px;color:#94a3b8;margin-top:1px}

/* ── Cards ── */
.ml-card{
    border:1px solid rgba(148,163,184,.14);
    border-radius:18px;padding:22px 26px;
    background:linear-gradient(135deg,rgba(24,26,32,.6),rgba(10,12,16,.7));
    backdrop-filter:blur(14px);
    margin-bottom:14px;
    box-shadow:0 8px 24px -12px rgba(0,0,0,.55);
    transition:border-color .25s, transform .25s, box-shadow .25s;
    animation:ml-fade-in .4s ease both;
}
.ml-card:hover{
    border-color:rgba(45,212,191,.45);
    transform:translateY(-3px);
    box-shadow:0 16px 34px -14px rgba(45,212,191,.3);
}

.ml-card-normal{border-left:4px solid #2dd4bf}
.ml-card-borderline{border-left:4px solid #f59e0b}
.ml-card-abnormal{border-left:4px solid #fb7185}
.ml-card-informational{border-left:4px solid #a78bfa}

.ml-card h4{margin:0 0 6px;font-size:16.5px;color:#f8fafc;font-family:'Sora',sans-serif;font-weight:600}
.ml-card p{margin:0 0 4px;font-size:14px;color:#cbd5e1;line-height:1.65}
.ml-card .ml-label{
    font-size:11px;font-weight:700;letter-spacing:1.4px;
    text-transform:uppercase;margin-bottom:8px;
}

/* ── Metric chip strip ── */
.ml-stat-strip{
    display:flex;gap:12px;flex-wrap:wrap;
    padding:16px 18px;margin-bottom:22px;
    border-radius:18px;
    background:linear-gradient(120deg,rgba(24,26,32,.6),rgba(10,12,16,.5));
    border:1px solid rgba(148,163,184,.12);
}
.ml-metric{
    flex:1;min-width:130px;text-align:center;padding:16px 10px;
    border-radius:14px;
    background:rgba(255,255,255,.02);
    border:1px solid rgba(148,163,184,.1);
    transition:transform .25s, box-shadow .25s, border-color .25s, background .25s;
}
.ml-metric:hover{
    transform:translateY(-3px);
    border-color:rgba(45,212,191,.35);
    background:rgba(45,212,191,.04);
    box-shadow:0 14px 28px -18px rgba(45,212,191,.35);
}
.ml-metric .ico{font-size:16px;margin-bottom:4px;opacity:.85}
.ml-metric .num{font-family:'Sora',sans-serif;font-size:30px;font-weight:800;letter-spacing:-1px;line-height:1.1}
.ml-metric .lbl{font-size:10.5px;color:#94a3b8;margin-top:4px;letter-spacing:.7px;text-transform:uppercase;font-weight:600}

/* ── Summary box ── */
.ml-summary{
    border-radius:20px;padding:30px 34px;
    background:linear-gradient(135deg,rgba(45,212,191,.09),rgba(167,139,250,.07));
    border:1px solid rgba(45,212,191,.2);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.05), 0 12px 30px -18px rgba(45,212,191,.25);
    line-height:1.8;color:#e2e8f0;font-size:15px;
}

/* ── Glossary ── */
.ml-glossary{
    border-radius:16px;padding:16px 22px;
    background:linear-gradient(135deg,rgba(24,26,32,.6),rgba(10,12,16,.5));
    border:1px solid rgba(148,163,184,.13);
    border-left:3px solid #a78bfa;
    margin-bottom:10px;
    transition:border-color .2s, transform .2s;
}
.ml-glossary:hover{border-color:#c4b5fd;transform:translateX(2px)}
.ml-glossary strong{color:#c4b5fd;font-size:14px;font-family:'Sora',sans-serif}
.ml-glossary span{color:#cbd5e1;font-size:13.5px;line-height:1.65}

/* ── Questions card ── */
.ml-question{
    border-radius:14px;padding:16px 22px;
    background:linear-gradient(135deg,rgba(245,158,11,.09),rgba(245,158,11,.04));
    border:1px solid rgba(245,158,11,.22);
    margin-bottom:10px;color:#fbbf24;
    font-size:14px;line-height:1.6;
    box-shadow:0 8px 20px -14px rgba(245,158,11,.3);
}

/* ── Disclaimer ── */
.ml-disclaimer{
    border-radius:16px;padding:18px 24px;
    background:linear-gradient(135deg,rgba(251,113,133,.09),rgba(251,113,133,.03));
    border:1px solid rgba(251,113,133,.22);
    color:#fda4af;font-size:13px;line-height:1.7;
    margin-top:10px;
}

/* ── Evidence quote ── */
.ml-evidence{
    border-left:3px solid #2dd4bf;
    padding:12px 18px;margin:8px 0;
    background:linear-gradient(90deg,rgba(45,212,191,.07),transparent);
    border-radius:0 14px 14px 0;
    font-size:13.5px;color:#cbd5e1;
    line-height:1.65;white-space:pre-wrap;
    overflow-wrap:anywhere;
}

/* ── OCR preview ── */
.ml-ocr-badge{
    display:inline-block;border-radius:999px;
    padding:5px 13px;margin:4px 4px 4px 0;
    font-size:11px;font-weight:700;letter-spacing:.8px;
}
.ml-ocr-good{background:rgba(45,212,191,.14);color:#5eead4;border:1px solid rgba(45,212,191,.3)}
.ml-ocr-fair{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.3)}
.ml-ocr-poor{background:rgba(251,113,133,.14);color:#fda4af;border:1px solid rgba(251,113,133,.3)}

/* ── Buttons ── */
.stButton>button, .stDownloadButton>button{
    border-radius:12px !important;
    font-weight:700 !important;
    letter-spacing:.3px;
    transition:transform .18s, box-shadow .18s !important;
    border:1px solid rgba(148,163,184,.18) !important;
}
button[kind="primary"], .stButton>button[kind="primary"]{
    background:linear-gradient(120deg,#14b8a6,#2dd4bf 45%,#a78bfa) !important;
    background-size:180% 180% !important;
    border:none !important;
    color:#05070a !important;
    box-shadow:0 10px 28px -10px rgba(45,212,191,.5) !important;
}
button[kind="primary"]:hover{
    transform:translateY(-2px);
    box-shadow:0 16px 36px -10px rgba(45,212,191,.65) !important;
    background-position:100% 0 !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{transform:translateY(-2px)}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
    gap:6px;background:rgba(10,12,16,.5);
    padding:6px;border-radius:14px;
    border:1px solid rgba(148,163,184,.12);
}
.stTabs [data-baseweb="tab"]{
    border-radius:10px;color:#94a3b8;font-weight:600;
    padding:10px 18px;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(120deg,rgba(45,212,191,.22),rgba(167,139,250,.18)) !important;
    color:#f1f5f9 !important;
}

/* ── Inputs / expander / uploader / popover ── */
.stTextInput input, .stTextArea textarea{
    background:rgba(10,12,16,.6) !important;
    border-radius:12px !important;
    border:1px solid rgba(148,163,184,.18) !important;
}
[data-testid="stFileUploaderDropzone"]{
    background:linear-gradient(135deg,rgba(45,212,191,.06),rgba(167,139,250,.04)) !important;
    border:1.5px dashed rgba(94,234,212,.3) !important;
    border-radius:16px !important;
}
.streamlit-expanderHeader{
    background:rgba(20,22,28,.5) !important;
    border-radius:12px !important;
}
div[data-testid="stExpander"]{
    border:1px solid rgba(148,163,184,.13) !important;
    border-radius:16px !important;
    overflow:hidden;
}
div[data-testid="stPopoverBody"]{
    background:linear-gradient(160deg,#12141a,#0a0c10) !important;
    border:1px solid rgba(148,163,184,.16) !important;
    border-radius:16px !important;
}

/* ── Code block ── */
.stCodeBlock, pre{
    border-radius:14px !important;
    border:1px solid rgba(148,163,184,.13) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"]{
    border-radius:14px !important;
    backdrop-filter:blur(10px);
}
</style>
""", unsafe_allow_html=True)


# ─── Pydantic schemas ────────────────────────────────────────

class LabValue(BaseModel):
    name: str = Field(max_length=120)
    reported_value: str = Field(max_length=80)
    reference_range: str = Field(max_length=100)
    unit: str = Field(max_length=40)
    status: Literal["Normal", "Borderline", "Abnormal", "Informational"]
    plain_explanation: str = Field(max_length=500)
    possible_causes: list[str] = Field(max_length=4)
    source_quote: str = Field(max_length=300)


class GlossaryTerm(BaseModel):
    term: str = Field(max_length=80)
    definition: str = Field(max_length=300)
    source_quote: str = Field(max_length=300)


class DoctorQuestion(BaseModel):
    question: str = Field(max_length=200)
    reason: str = Field(max_length=300)


class ReportTranslation(BaseModel):
    # NOTE: limits below are generous — the sanitizer trims to fit if the
    # model gets verbose. Earlier caps (100/600/1200) tripped ValidationError
    # whenever the model listed multiple panels or was unusually detailed.
    report_type: str = Field(max_length=300)
    patient_summary: str = Field(max_length=800)
    overall_impression: str = Field(max_length=2000)
    lab_values: list[LabValue] = Field(default_factory=list)
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    doctor_questions: list[DoctorQuestion] = Field(
        default_factory=list, max_length=8
    )
    important_notes: list[str] = Field(default_factory=list, max_length=5)
    confidence_note: str = Field(max_length=400)


class FollowUpAnswer(BaseModel):
    answer: str = Field(max_length=1500)
    source_quotes: list[str] = Field(max_length=5)
    cannot_answer: bool = False
    reason_cannot_answer: str = Field(default="", max_length=300)


# ─── Helpers ──────────────────────────────────────────────────

def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default))


_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Strip any HTML/markup the model emitted, then escape for safe display.

    LLM output is untrusted text, not markup — the translation prompt tells
    gpt-oss to write plain language, but models don't always comply (this is
    exactly what produced literal '</p>' tags showing up in the UI). Stripping
    tags here means a stray '<p style=...>' in a field can never leak through
    to the page again, regardless of what the model decides to emit.
    """
    if not text:
        return ""
    stripped = _TAG_RE.sub(" ", str(text))
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return html.escape(stripped)


def friendly_error(error: Exception) -> str:
    if isinstance(error, AuthenticationError):
        return "🔑 Authentication failed — check GROQ_API_KEY in your Streamlit secrets."
    if isinstance(error, RateLimitError):
        return "⏳ Rate limit reached — wait a moment and retry."
    if isinstance(error, APIConnectionError):
        return "🌐 Cannot reach Groq — check your connection and retry."
    if isinstance(error, APIStatusError):
        return "❌ Groq rejected the request — verify model availability and account limits."
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return f"⚠️ {error}"
    return f"⚠️ Unexpected error: {type(error).__name__}. Try again or choose a different model."


def strip_tags(text: str) -> str:
    """Like clean_text, but for values passed straight to st.markdown/
    st.info/st.warning (which render markdown, not raw HTML) — strip stray
    tags without HTML-escaping, since escaping here would show '&amp;' etc.
    literally instead of being interpreted as markdown.
    """
    if not text:
        return ""
    stripped = _TAG_RE.sub(" ", str(text))
    return re.sub(r"\s+", " ", stripped).strip()


def sanitize_for_schema(data: dict, model_cls: type[BaseModel]) -> dict:
    """Truncate strings/lists in raw LLM JSON to fit the pydantic schema.

    The model doesn't reliably respect max_length in the JSON schema, so a
    single over-long field (e.g. report_type listing six panels) would raise
    a ValidationError and kill the whole translation. This walks the raw
    dict against the model's field metadata and clips anything too long
    before validation, turning a hard failure into a harmless trim.
    """
    if not isinstance(data, dict):
        return data

    out: dict = {}
    for name, field in model_cls.model_fields.items():
        if name not in data:
            continue
        value = data[name]
        ann = field.annotation

        # Unwrap Optional[X]
        origin = _typing.get_origin(ann)
        if origin in (_typing.Union, _types.UnionType):
            non_none = [a for a in _typing.get_args(ann) if a is not type(None)]
            if non_none:
                ann = non_none[0]
                origin = _typing.get_origin(ann)

        # Field-level max_length — applies to str length OR list length
        max_len = None
        for meta in (field.metadata or ()):
            ml = getattr(meta, "max_length", None)
            if ml is not None:
                max_len = ml if max_len is None else min(max_len, ml)

        args = _typing.get_args(ann)

        if isinstance(value, str):
            out[name] = value[:max_len] if max_len else value

        elif isinstance(value, list):
            if max_len:
                value = value[:max_len]
            inner = args[0] if args else None
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                value = [
                    sanitize_for_schema(v, inner) if isinstance(v, dict) else v
                    for v in value
                ]
            out[name] = value

        elif isinstance(ann, type) and issubclass(ann, BaseModel) and isinstance(value, dict):
            out[name] = sanitize_for_schema(value, ann)

        else:
            out[name] = value

    # Preserve any extra keys the model invented (pydantic will ignore them)
    for k, v in data.items():
        out.setdefault(k, v)
    return out


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF using PyMuPDF."""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_pdf_ocr(file_bytes: bytes) -> tuple[str, list[Image.Image]]:
    """Render each PDF page as an image and OCR it (for scanned PDFs)."""
    images = []
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
            processed = preprocess_image_for_ocr(img)
            text_parts.append(pytesseract.image_to_string(processed))
    return "\n".join(text_parts).strip(), images


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Enhance an image for better OCR accuracy."""
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if too small (below 1500px width)
    w, h = img.size
    if w < 1500:
        scale = 1500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert to grayscale
    img = img.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)

    # Apply slight blur to reduce noise then sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)

    # Binarize (adaptive-like thresholding via point)
    threshold = 140
    img = img.point(lambda p: 255 if p > threshold else 0, mode="1")

    # Convert back to grayscale for Tesseract
    img = img.convert("L")

    return img


def estimate_ocr_quality(text: str) -> tuple[str, str, str]:
    """Heuristic OCR quality estimation."""
    if not text.strip():
        return "poor", "ml-ocr-poor", "No text could be extracted"

    total_chars = len(text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    digit_chars = sum(1 for c in text if c.isdigit())
    space_chars = sum(1 for c in text if c.isspace())
    garbage_chars = total_chars - alpha_chars - digit_chars - space_chars

    if total_chars == 0:
        return "poor", "ml-ocr-poor", "No text extracted"

    garbage_ratio = garbage_chars / total_chars
    alpha_ratio = alpha_chars / total_chars

    # Count recognizable medical/lab terms
    medical_terms = [
        "blood", "test", "result", "range", "reference", "normal",
        "high", "low", "mg", "dl", "ml", "patient", "date", "lab",
        "cholesterol", "glucose", "hemoglobin", "wbc", "rbc",
        "platelet", "sodium", "potassium", "creatinine", "urine",
        "report", "specimen", "doctor", "physician", "panel",
    ]
    text_lower = text.lower()
    term_matches = sum(1 for t in medical_terms if t in text_lower)

    if garbage_ratio < 0.08 and alpha_ratio > 0.5 and term_matches >= 3:
        return "good", "ml-ocr-good", f"Good quality · {term_matches} medical terms detected"
    elif garbage_ratio < 0.18 and alpha_ratio > 0.35 and term_matches >= 1:
        return "fair", "ml-ocr-fair", f"Fair quality · some characters may be misread"
    else:
        return "poor", "ml-ocr-poor", "Poor quality · results may be unreliable"


def extract_image_ocr(file_bytes: bytes) -> tuple[str, Image.Image]:
    """OCR a single image file."""
    img = Image.open(io.BytesIO(file_bytes))
    processed = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(processed)
    return text.strip(), img


def fp(text: str, extra: str = "") -> str:
    return hashlib.sha256((text + extra).encode()).hexdigest()


def request_json(
    api_key: str, model: str,
    system_prompt: str, user_content: str,
    max_tokens: int = 6000,
    reasoning_effort: str = "low",
) -> dict:
    kwargs = dict(
        model=model,
        temperature=0.05,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    # gpt-oss models are reasoning models: hidden "thinking" tokens count
    # against max_tokens too. Default reasoning effort is "medium", which
    # can eat most of the budget before any JSON is written and truncate
    # the response. Pin it to "low" for this structured-extraction task —
    # only send it for gpt-oss models, since other Groq models reject the
    # param outright.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = reasoning_effort

    with Groq(api_key=api_key, timeout=90.0, max_retries=1) as client:
        result = client.chat.completions.create(**kwargs)

    if result.choices[0].finish_reason == "length":
        raise ValueError(
            f"Response was cut off after {max_tokens:,} tokens before it "
            "finished — the report likely has more values than fit in "
            "this budget. Try a shorter report, or raise max_tokens in "
            "request_json()."
        )
    return json.loads(result.choices[0].message.content or "{}")


def render_disclaimer():
    st.markdown("""
    <div class="ml-disclaimer">
        <strong>⚕️ Medical Disclaimer</strong><br>
        This tool translates medical terminology into plain language for
        <strong>educational purposes only</strong>. It is <strong>NOT</strong>
        medical advice, diagnosis, or treatment. All explanations are derived
        strictly from the text you uploaded — nothing is invented or sourced
        externally. <strong>Always consult your physician</strong> before making
        any health decisions. If you are experiencing a medical emergency,
        call your local emergency number immediately.
    </div>
    """, unsafe_allow_html=True)


def render_evidence(quote: str):
    if quote.strip():
        escaped = clean_text(quote.strip())
        st.markdown(
            f'<div class="ml-evidence">📄 <em>From your report:</em> "{escaped}"</div>',
            unsafe_allow_html=True,
        )


# ─── Sample report ────────────────────────────────────────────

SAMPLE_REPORT = textwrap.dedent("""\
COMPLETE BLOOD COUNT (CBC) WITH DIFFERENTIAL
Patient: Jane Doe | DOB: 1985-03-12 | Date: 2025-04-28
Ordering Physician: Dr. Sarah Mitchell | Lab: CityHealth Reference Lab

Test                    Result      Reference Range     Units       Flag
─────────────────────────────────────────────────────────────────────────
WBC (White Blood Cells) 11.8        4.5 – 11.0          x10^3/uL    HIGH
RBC (Red Blood Cells)   4.52        3.80 – 5.10         x10^6/uL
Hemoglobin (Hgb)        12.1        12.0 – 16.0         g/dL
Hematocrit (Hct)        36.8        36.0 – 46.0         %
MCV                     81.4        80.0 – 100.0        fL
MCH                     26.8        27.0 – 33.0         pg          LOW
MCHC                    32.9        32.0 – 36.0         g/dL
RDW                     14.8        11.5 – 14.5         %           HIGH
Platelets               142         150 – 400           x10^3/uL    LOW
MPV                     11.2        7.5 – 11.5          fL

DIFFERENTIAL:
Neutrophils             72.1        40.0 – 70.0         %           HIGH
Lymphocytes             18.3        20.0 – 40.0         %           LOW
Monocytes               7.2         2.0 – 8.0           %
Eosinophils             1.8         1.0 – 4.0           %
Basophils               0.6         0.0 – 1.0           %

BASIC METABOLIC PANEL (BMP)
Test                    Result      Reference Range     Units       Flag
─────────────────────────────────────────────────────────────────────────
Glucose (Fasting)       118         70 – 99             mg/dL       HIGH
BUN                     22          7 – 20              mg/dL       HIGH
Creatinine              1.3         0.6 – 1.2           mg/dL       HIGH
eGFR                    68          >60                  mL/min
Sodium                  141         136 – 145           mEq/L
Potassium               4.1         3.5 – 5.0           mEq/L
Chloride                102         98 – 106            mEq/L
CO2 (Bicarbonate)       24          23 – 29             mEq/L
Calcium                 9.4         8.5 – 10.5          mg/dL

LIPID PANEL
Test                    Result      Reference Range     Units       Flag
─────────────────────────────────────────────────────────────────────────
Total Cholesterol       238         <200                mg/dL       HIGH
LDL Cholesterol         162         <100                mg/dL       HIGH
HDL Cholesterol         42          >40                 mg/dL
Triglycerides           186         <150                mg/dL       HIGH
VLDL                    37          5 – 40              mg/dL

LIVER FUNCTION TESTS (LFT)
Test                    Result      Reference Range     Units       Flag
─────────────────────────────────────────────────────────────────────────
ALT (SGPT)              52          7 – 56              U/L
AST (SGOT)              48          10 – 40             U/L         HIGH
ALP                     88          44 – 147            U/L
Total Bilirubin         1.4         0.1 – 1.2           mg/dL       HIGH
Direct Bilirubin        0.4         0.0 – 0.3           mg/dL       HIGH
Albumin                 3.9         3.5 – 5.5           g/dL
Total Protein           7.2         6.0 – 8.3           g/dL

THYROID PANEL
Test                    Result      Reference Range     Units       Flag
─────────────────────────────────────────────────────────────────────────
TSH                     6.8         0.4 – 4.0           mIU/L       HIGH
Free T4                 0.7         0.8 – 1.8           ng/dL       LOW
Free T3                 2.1         2.3 – 4.2           pg/mL       LOW

HbA1c                   6.2         <5.7                %           HIGH

URINALYSIS
Test                    Result      Reference
─────────────────────────────────────────────────────────────────────────
Color                   Dark Yellow Pale to Dark Yellow
Appearance              Slightly Hazy   Clear
pH                      5.5         5.0 – 8.0
Specific Gravity        1.028       1.005 – 1.030
Protein                 Trace       Negative
Glucose                 Negative    Negative
Ketones                 Negative    Negative
Blood                   Negative    Negative
WBC (Urine)             5           0 – 5 /HPF

COMMENTS:
- Thyroid results suggest subclinical hypothyroidism. Recommend clinical
  correlation and consider repeat testing in 6-8 weeks.
- Fasting glucose and HbA1c indicate prediabetes. Lifestyle modification
  and monitoring recommended.
- Mildly elevated bilirubin may warrant further evaluation if symptoms
  of jaundice are present.
- Lipid panel indicates dyslipidemia. Consider dietary counseling and
  cardiovascular risk assessment.
- Platelet count is mildly low; recommend repeat CBC in 4 weeks to
  confirm trend.
""")


# ─── AI Prompts ───────────────────────────────────────────────

TRANSLATION_PROMPT = f"""
You are MedLens, a medical report translator that converts clinical
reports into plain language a patient can understand.

ABSOLUTE RULES — VIOLATING ANY RULE IS FORBIDDEN:
1. Use ONLY information explicitly present in the uploaded report.
2. NEVER invent, assume, or hallucinate any data, diagnosis, or advice.
3. NEVER add information from external medical knowledge beyond explaining
   what the terms in the report mean.
4. If a value, term, or context is unclear or missing from the report,
   say "This information is not provided in your report."
5. NEVER provide a diagnosis. You EXPLAIN what the report says.
6. NEVER recommend treatments, medications, or lifestyle changes on
   your own — only relay recommendations already stated in the report.
7. For every explanation, include a direct quote from the report as
   evidence in the source_quote field.
8. If the report contains physician comments or recommendations, relay
   them exactly and note they came from the report.
9. patient_summary must only contain facts from the report header.
10. confidence_note must honestly state any limitations you noticed
    (e.g., partially illegible text, missing reference ranges, possible
    OCR errors in scanned documents).
11. If the text appears to contain OCR artifacts or garbled characters,
    note this in confidence_note and do your best with readable parts.
    Do NOT guess values that are illegible.
12. Every text field must be PLAIN TEXT ONLY — no HTML tags (e.g. <p>,
    <br>, <div>), no Markdown formatting, no code blocks. Write normal
    sentences and paragraphs as plain strings.
13. Keep report_type short: list only the panel names (e.g. "CBC + BMP
    + Lipid Panel"), max 300 characters.

STATUS CLASSIFICATION (use ONLY report data):
- "Normal": value is within the stated reference range.
- "Borderline": value is at or very near the edge of the reference range.
- "Abnormal": value is clearly outside the stated reference range.
- "Informational": qualitative results or values without a clear range.

Return ONLY valid JSON matching this schema:
{json.dumps(ReportTranslation.model_json_schema(), indent=2)}
"""

FOLLOWUP_PROMPT = f"""
You are MedLens, answering a patient's follow-up question about their
medical report.

ABSOLUTE RULES:
1. Answer ONLY from the report text provided. Quote relevant parts.
2. If the answer is not in the report, set cannot_answer=true and
   explain what information is missing.
3. NEVER invent data, diagnoses, or medical advice.
4. NEVER recommend treatments not already mentioned in the report.
5. Keep language simple and compassionate.
6. Include source_quotes from the report to support every claim.
7. If parts of the report appear garbled from OCR, acknowledge this
   limitation honestly.
8. Every text field must be PLAIN TEXT ONLY — no HTML tags, no Markdown
   formatting, no code blocks.

Return ONLY valid JSON matching this schema:
{json.dumps(FollowUpAnswer.model_json_schema(), indent=2)}
"""


# ─── Top bar (replaces sidebar — settings live in a popover) ──

api_key = get_setting("GROQ_API_KEY")

topbar_l, topbar_r = st.columns([3, 1.3])

with topbar_l:
    status_class = "ml-status-on" if api_key else "ml-status-off"
    status_text = "🟢 Groq connected" if api_key else "🔑 API key missing"
    st.markdown(f"""
    <div class="ml-topbar">
        <div class="ml-brand"><span class="ml-dot"></span> MedLens</div>
        <span class="ml-status-pill {status_class}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

with topbar_r:
    with st.popover("⚙️ Settings", use_container_width=True):
        st.markdown("##### AI Model")
        model = st.text_input(
            "AI Model",
            value=get_setting("GROQ_MODEL", DEFAULT_MODEL),
            help="Any Groq model supporting chat + JSON output.",
            label_visibility="collapsed",
        ).strip()

        st.markdown("##### How it works")
        st.markdown("""
        1. 📄 Upload your lab report (PDF, image, or text)
        2. 🔍 OCR reads scanned/photo reports automatically
        3. 🤖 AI reads **only your report** — nothing else
        4. 📊 See every value explained in plain language
        5. ❓ Ask follow-up questions about your results
        """)

        st.markdown("##### Supported formats")
        st.markdown("""
        | Format | Method |
        |---|---|
        | 📋 Paste text | Direct input |
        | 📎 PDF (digital) | Text extraction |
        | 📎 PDF (scanned) | OCR fallback |
        | 📷 Image (JPG/PNG/WEBP) | OCR |
        """)

        render_disclaimer()

        if st.button("🗑️ Clear workspace", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if not api_key:
    st.warning("Add GROQ_API_KEY to Streamlit secrets — open ⚙️ Settings above.", icon="🔑")


# ─── Hero ─────────────────────────────────────────────────────

st.markdown("""
<div class="ml-hero">
    <div class="ml-hero-main">
        <div class="ml-eyebrow">AI · MEDICAL REPORT INTELLIGENCE</div>
        <h1>Understand your lab results<br>in plain English.</h1>
        <p>
            Upload any medical report — typed, scanned, or photographed —
            and get a clear, jargon-free translation.
            Every explanation comes directly from <strong>your report</strong> —
            nothing is invented, nothing is sourced externally.
        </p>
        <span class="ml-pill">🔒 YOUR DATA ONLY</span>
        <span class="ml-pill">🎯 GROUNDED IN YOUR TEXT</span>
        <span class="ml-pill">📷 OCR SUPPORTED</span>
        <span class="ml-pill">📄 EVIDENCE CITED</span>
        <span class="ml-pill">⚕️ NOT MEDICAL ADVICE</span>
    </div>
    <div class="ml-hero-side">
        <div class="ml-side-item">
            <div class="ml-side-ico">🔬</div>
            <div>
                <div class="ml-side-title">Value-by-value breakdown</div>
                <div class="ml-side-sub">Status, range &amp; plain-language meaning</div>
            </div>
        </div>
        <div class="ml-side-item">
            <div class="ml-side-ico">📖</div>
            <div>
                <div class="ml-side-title">Jargon decoded</div>
                <div class="ml-side-sub">Every term explained with a source quote</div>
            </div>
        </div>
        <div class="ml-side-item">
            <div class="ml-side-ico">❓</div>
            <div>
                <div class="ml-side-title">Doctor-ready questions</div>
                <div class="ml-side-sub">Generated from your specific results</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Upload section ──────────────────────────────────────────

with st.expander(
    "📄 Upload or paste your medical report",
    expanded="translation" not in st.session_state,
):

    source = st.radio(
        "Input method",
        [
            "📋 Paste text",
            "📎 Upload PDF",
            "📷 Upload image (JPG / PNG / WEBP)",
            "🧪 Use sample report",
        ],
        horizontal=True,
    )

    report_text = ""
    ocr_used = False
    ocr_quality = None
    preview_image = None

    # ── Paste text ──
    if source == "📋 Paste text":
        report_text = st.text_area(
            "Paste your medical report here",
            height=300,
            placeholder="Paste your lab report, imaging report, or discharge summary…",
            max_chars=MAX_TEXT_CHARS,
        ).strip()

    # ── Upload PDF ──
    elif source == "📎 Upload PDF":
        uploaded = st.file_uploader(
            "Upload a medical report PDF",
            type=["pdf"],
            help=f"Max {MAX_UPLOAD_MB} MB. Works with both digital and scanned PDFs.",
        )
        if uploaded:
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
            else:
                file_bytes = uploaded.getvalue()

                # Try text extraction first
                with st.spinner("📄 Extracting text from PDF…"):
                    try:
                        report_text = extract_pdf_text(file_bytes)
                    except Exception:
                        report_text = ""

                # If text extraction yields very little, fall back to OCR
                if len(report_text.strip()) < 50:
                    st.info("📷 Limited text found — switching to OCR mode for scanned pages…")
                    with st.spinner("🔍 Running OCR on PDF pages (this may take 30–60 seconds)…"):
                        try:
                            report_text, page_images = extract_pdf_ocr(file_bytes)
                            ocr_used = True
                            if page_images:
                                preview_image = page_images[0]
                        except Exception as e:
                            st.error(f"OCR failed: {e}")
                            report_text = ""
                else:
                    st.success(f"✅ Extracted {len(report_text):,} characters from {uploaded.name}")

    # ── Upload image ──
    elif source == "📷 Upload image (JPG / PNG / WEBP)":
        uploaded_img = st.file_uploader(
            "Upload a photo or scan of your medical report",
            type=["jpg", "jpeg", "png", "webp"],
            help=f"Max {MAX_UPLOAD_MB} MB. Clear, well-lit photos work best.",
        )
        if uploaded_img:
            if uploaded_img.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
            else:
                with st.spinner("🔍 Reading your report image with OCR…"):
                    try:
                        report_text, preview_image = extract_image_ocr(
                            uploaded_img.getvalue()
                        )
                        ocr_used = True
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

    # ── Sample report ──
    elif source == "🧪 Use sample report":
        report_text = SAMPLE_REPORT
        st.info("Using a sample CBC + metabolic panel for demonstration.")

    # ── OCR quality feedback ──
    if ocr_used and report_text:
        quality_level, quality_class, quality_msg = estimate_ocr_quality(report_text)
        ocr_quality = quality_level

        st.markdown(f"""
        <div style="margin:12px 0">
            <span class="ml-ocr-badge {quality_class}">
                📷 OCR {quality_level.upper()}
            </span>
            <span style="color:#94a3b8;font-size:13px;margin-left:8px">
                {quality_msg} · {len(report_text):,} characters extracted
            </span>
        </div>
        """, unsafe_allow_html=True)

        if quality_level == "poor":
            st.warning(
                "⚠️ OCR quality is poor. Results may be unreliable. "
                "Try uploading a clearer image with good lighting, "
                "or use a digital PDF if available."
            )
        elif quality_level == "fair":
            st.info(
                "💡 OCR quality is fair. Some values might be misread. "
                "Please verify critical numbers against your original report."
            )

    # ── Image preview ──
    if preview_image is not None:
        with st.expander("🖼️ Preview uploaded image"):
            st.image(preview_image, caption="Your uploaded report", use_container_width=True)

    # ── Report text preview ──
    if report_text:
        char_count = len(report_text)
        truncated = False
        if char_count > MAX_TEXT_CHARS:
            report_text = report_text[:MAX_TEXT_CHARS]
            truncated = True
            st.warning(
                f"Report truncated to {MAX_TEXT_CHARS:,} characters. "
                "Very long reports may lose some data at the end."
            )

        with st.container(border=True):
            header_parts = [f"📏 {char_count:,} characters"]
            if truncated:
                header_parts.append("(truncated)")
            if ocr_used:
                header_parts.append("· 📷 OCR extracted")
            st.caption(" ".join(header_parts))

            display_text = report_text[:3000]
            if len(report_text) > 3000:
                display_text += "\n\n… [showing first 3000 chars]"
            st.code(display_text, language=None)

        if ocr_used:
            st.markdown("""
            <div style="border-radius:14px;padding:14px 20px;
                background:linear-gradient(135deg,rgba(45,212,191,.08),rgba(45,212,191,.03));
                border:1px solid rgba(45,212,191,.18);
                color:#5eead4;font-size:13px;line-height:1.6;margin-bottom:12px">
                <strong>📷 OCR Notice:</strong> This text was extracted using
                optical character recognition. Some characters, numbers, or
                formatting may be incorrect. The AI will note any suspected
                OCR errors in its analysis. Always verify critical values
                against your original document.
            </div>
            """, unsafe_allow_html=True)

    # ── Fingerprint and translation trigger ──
    current_fp = fp(report_text, model) if report_text else ""

    if st.session_state.get("report_fp") != current_fp:
        for k in ["translation", "followup", "report_fp"]:
            st.session_state.pop(k, None)
        if current_fp:
            st.session_state["report_fp"] = current_fp

    can_translate = bool(report_text) and bool(api_key) and bool(model)

    ocr_context = ""
    if ocr_used:
        ocr_context = (
            "\n\n[SYSTEM NOTE: This text was extracted via OCR from a "
            "scanned/photographed document. Some characters may be "
            "misread. Flag any suspected OCR errors in confidence_note. "
            "Do NOT guess illegible values.]"
        )

    if st.button(
        "🩺 Translate My Report",
        type="primary",
        disabled=not can_translate,
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "🔬 Reading your report carefully — this takes 15–30 seconds…"
            ):
                raw = request_json(
                    api_key, model,
                    TRANSLATION_PROMPT,
                    report_text + ocr_context,
                    max_tokens=6000,
                )
                translation = ReportTranslation.model_validate(
                    sanitize_for_schema(raw, ReportTranslation)
                )
                st.session_state["translation"] = translation.model_dump()
                st.session_state["report_text"] = report_text
                st.session_state["ocr_used"] = ocr_used
                st.session_state.pop("followup", None)
        except Exception as e:
            st.error(friendly_error(e))


# ─── Stop if no translation yet ──────────────────────────────

if "translation" not in st.session_state:
    st.info(
        "Upload or paste a medical report above and click "
        "**Translate My Report** to begin."
    )
    st.stop()

tr = ReportTranslation.model_validate(st.session_state["translation"])
report_text = st.session_state.get("report_text", "")
ocr_used = st.session_state.get("ocr_used", False)

render_disclaimer()
st.markdown("")

# ── OCR reminder banner ──
if ocr_used:
    st.markdown("""
    <div style="border-radius:16px;padding:16px 22px;
        background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(245,158,11,.03));
        border:1px solid rgba(245,158,11,.2);
        color:#fcd34d;font-size:13px;line-height:1.7;margin-bottom:16px">
        <strong>📷 OCR Mode Active</strong> — This report was read from an
        image/scan. Some values may have been misread by OCR.
        Please verify all critical numbers against your original document.
    </div>
    """, unsafe_allow_html=True)

# ─── Metrics row ──────────────────────────────────────────────

total = len(tr.lab_values)
normal_n = sum(1 for v in tr.lab_values if v.status == "Normal")
borderline_n = sum(1 for v in tr.lab_values if v.status == "Borderline")
abnormal_n = sum(1 for v in tr.lab_values if v.status == "Abnormal")
info_n = sum(1 for v in tr.lab_values if v.status == "Informational")

chips = "".join(
    f'<div class="ml-metric">'
    f'<div class="ico">{icon}</div>'
    f'<div class="num" style="color:{color}">{num}</div>'
    f'<div class="lbl">{label}</div></div>'
    for num, label, color, icon in [
        (total, "Values Found", "#c4b5fd", "📊"),
        (normal_n, "Normal", STATUS_COLORS["Normal"], "✅"),
        (borderline_n, "Borderline", STATUS_COLORS["Borderline"], "⚠️"),
        (abnormal_n, "Abnormal", STATUS_COLORS["Abnormal"], "🔴"),
        (info_n, "Informational", STATUS_COLORS["Informational"], "ℹ️"),
    ]
)
st.markdown(f'<div class="ml-stat-strip">{chips}</div>', unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────

tab_summary, tab_values, tab_glossary, tab_questions, tab_ask = st.tabs([
    "📋 Summary",
    "🔬 Lab Values",
    "📖 Glossary",
    "❓ Doctor Questions",
    "💬 Ask About Your Report",
])


# ── Summary tab ───────────────────────────────────────────────

with tab_summary:
    st.markdown("### Your Report at a Glance")
    st.markdown("")

    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            st.markdown("##### 📄 Report Type")
            st.markdown(f"**{strip_tags(tr.report_type)}**")

        with st.container(border=True):
            st.markdown("##### 👤 Patient Information")
            st.markdown(strip_tags(tr.patient_summary))

        st.markdown(
            f'<div class="ml-summary">'
            f'<strong>🔍 Overall Impression</strong><br><br>'
            f'{clean_text(tr.overall_impression)}</div>',
            unsafe_allow_html=True,
        )

        if tr.confidence_note:
            st.markdown("")
            st.info(f"🔎 **AI Confidence Note:** {strip_tags(tr.confidence_note)}")

    with right:
        if tr.lab_values:
            status_counts = {
                "Normal": normal_n,
                "Borderline": borderline_n,
                "Abnormal": abnormal_n,
                "Informational": info_n,
            }
            status_counts = {k: v for k, v in status_counts.items() if v > 0}

            fig = go.Figure(go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                hole=0.72,
                marker=dict(
                    colors=[STATUS_COLORS[s] for s in status_counts.keys()]
                ),
                textinfo="label+value",
                textposition="outside",
                textfont=dict(size=13),
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope, sans-serif", color="#e2e8f0"),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{total}</b><br>values",
                    x=0.5, y=0.5, font_size=20,
                    font_color="#e2e8f0",
                    showarrow=False,
                )],
            )
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": False},
            )

        if tr.important_notes:
            st.markdown("##### ⚠️ Important Notes from Your Report")
            for note in tr.important_notes:
                st.warning(strip_tags(note), icon="📌")


# ── Lab Values tab ────────────────────────────────────────────

with tab_values:
    st.markdown("### Your Results Explained")
    st.caption(
        "Every explanation below comes directly from your uploaded report. "
        "Source quotes are shown as evidence."
    )
    if ocr_used:
        st.caption(
            "📷 Values were extracted via OCR — verify numbers against "
            "your original document."
        )
    st.markdown("")

    if not tr.lab_values:
        st.info("No quantitative lab values were found in this report.")
    else:
        filter_status = st.multiselect(
            "Filter by status",
            ["Normal", "Borderline", "Abnormal", "Informational"],
            default=[],
            placeholder="Show all",
        )

        search_val = st.text_input(
            "🔍 Search values",
            placeholder="e.g. cholesterol, TSH, glucose…",
        )

        display_values = tr.lab_values
        if filter_status:
            display_values = [
                v for v in display_values if v.status in filter_status
            ]
        if search_val:
            display_values = [
                v for v in display_values
                if search_val.lower() in v.name.lower()
            ]

        if not display_values:
            st.warning("No values match your filters.")
        else:
            order = {
                "Abnormal": 0, "Borderline": 1,
                "Informational": 2, "Normal": 3,
            }
            display_values = sorted(
                display_values, key=lambda v: order.get(v.status, 4)
            )

            for row_start in range(0, len(display_values), 2):
                row_values = display_values[row_start:row_start + 2]
                grid_cols = st.columns(2)

                for col, val in zip(grid_cols, row_values):
                    css_class = f"ml-card ml-card-{val.status.lower()}"
                    icon = STATUS_ICONS.get(val.status, "")
                    causes_html = ""
                    if val.possible_causes:
                        causes_items = "".join(
                            f"<li>{clean_text(c)}</li>"
                            for c in val.possible_causes
                        )
                        causes_html = (
                            f'<p style="margin-top:8px">'
                            f'<strong>Possible causes mentioned/implied in '
                            f'report:</strong></p>'
                            f'<ul style="color:#cbd5e1;font-size:13.5px">'
                            f'{causes_items}</ul>'
                        )

                    with col:
                        st.markdown(f"""
                        <div class="{css_class}">
                            <div class="ml-label" style="color:{STATUS_COLORS[val.status]}">
                                {icon} {val.status.upper()}
                            </div>
                            <h4>{clean_text(val.name)}</h4>
                            <p>
                                <strong>Your result:</strong>
                                {clean_text(val.reported_value)}
                                {clean_text(val.unit)}&nbsp;&nbsp;|&nbsp;&nbsp;
                                <strong>Normal range:</strong>
                                {clean_text(val.reference_range)}
                                {clean_text(val.unit)}
                            </p>
                            <p style="margin-top:8px">
                                {clean_text(val.plain_explanation)}
                            </p>
                            {causes_html}
                        </div>
                        """, unsafe_allow_html=True)

                        render_evidence(val.source_quote)


# ── Glossary tab ──────────────────────────────────────────────

with tab_glossary:
    st.markdown("### Medical Terms Decoded")
    st.caption(
        "Every term below appeared in your report. "
        "Definitions explain what these terms mean in context."
    )
    st.markdown("")

    if not tr.glossary:
        st.info("No medical terminology required explanation in this report.")
    else:
        search_term = st.text_input(
            "🔍 Search glossary",
            placeholder="e.g. hemoglobin, eGFR…",
            key="gloss_search",
        )

        display_glossary = tr.glossary
        if search_term:
            display_glossary = [
                g for g in display_glossary
                if search_term.lower() in g.term.lower()
            ]

        for term in display_glossary:
            st.markdown(f"""
            <div class="ml-glossary">
                <strong>🔬 {clean_text(term.term)}</strong><br>
                <span>{clean_text(term.definition)}</span>
            </div>
            """, unsafe_allow_html=True)
            render_evidence(term.source_quote)


# ── Doctor Questions tab ──────────────────────────────────────

with tab_questions:
    st.markdown("### Questions to Ask Your Doctor")
    st.caption(
        "Based on your report results, these are specific questions "
        "worth discussing at your next appointment."
    )
    st.markdown("")

    if not tr.doctor_questions:
        st.info(
            "No specific questions were generated — "
            "your results may be straightforward."
        )
    else:
        for i, q in enumerate(tr.doctor_questions, 1):
            st.markdown(f"""
            <div class="ml-question">
                <strong>❓ Question {i}:</strong>
                {clean_text(q.question)}<br>
                <span style="color:#fde68a;font-size:13px">
                    💡 <em>Why ask this:</em> {clean_text(q.reason)}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        questions_text = "\n\n".join(
            f"Q{i}: {strip_tags(q.question)}\nWhy: {strip_tags(q.reason)}"
            for i, q in enumerate(tr.doctor_questions, 1)
        )
        st.download_button(
            "📥 Download questions card",
            questions_text,
            "my_doctor_questions.txt",
            "text/plain",
            use_container_width=True,
        )


# ── Ask tab (follow-up questions) ─────────────────────────────

with tab_ask:
    st.markdown("### Ask About Your Report")
    st.caption(
        "Ask any question about your results. The AI will answer "
        "**only** from your report — if the answer isn't there, "
        "it will tell you."
    )
    st.markdown("")

    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. What does my elevated TSH mean? "
            "Is my cholesterol dangerous?"
        ),
        max_chars=MAX_QUESTION_CHARS,
        key="followup_q",
    )

    ocr_followup_note = ""
    if ocr_used:
        ocr_followup_note = (
            "\n\n[SYSTEM NOTE: The report text was extracted via OCR. "
            "Some values may be misread. Acknowledge this if relevant.]"
        )

    if st.button(
        "💬 Ask MedLens",
        type="primary",
        disabled=not question or not api_key or not model,
        use_container_width=True,
    ):
        payload = json.dumps({
            "report_text": report_text[:MAX_TEXT_CHARS] + ocr_followup_note,
            "question": question[:MAX_QUESTION_CHARS],
        }, ensure_ascii=False)

        try:
            with st.spinner("🔎 Searching your report for an answer…"):
                raw = request_json(
                    api_key, model,
                    FOLLOWUP_PROMPT,
                    payload,
                    max_tokens=2500,
                )
                answer = FollowUpAnswer.model_validate(
                    sanitize_for_schema(raw, FollowUpAnswer)
                )
                st.session_state["followup"] = {
                    "question": question,
                    "answer": answer.model_dump(),
                }
        except Exception as e:
            st.error(friendly_error(e))

    saved = st.session_state.get("followup")
    if saved:
        ans = FollowUpAnswer.model_validate(saved["answer"])

        with st.container(border=True):
            st.markdown(f"**You asked:** {clean_text(saved['question'])}")
            st.markdown("")

            if ans.cannot_answer:
                st.warning(
                    f"🚫 **Cannot answer from your report.** "
                    f"{strip_tags(ans.reason_cannot_answer)}"
                )
            else:
                st.markdown(
                    f'<div class="ml-summary">'
                    f'{clean_text(ans.answer)}</div>',
                    unsafe_allow_html=True,
                )

            if ans.source_quotes:
                st.markdown("")
                st.markdown("##### 📄 Evidence from your report")
                for sq in ans.source_quotes:
                    render_evidence(sq)

    render_disclaimer()


# ─── Footer ──────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="text-align:center;padding:24px 0 10px">
    <span style="color:#64748b;font-size:13px">
        🩺 <strong style="color:#94a3b8">MedLens</strong> · Built with Streamlit + Groq (openai/gpt-oss-120b) ·
        Your data stays in this session · AI explains, never diagnoses ·
        📷 OCR powered by Tesseract ·
        <strong>Always consult your physician</strong>
    </span>
</div>
""", unsafe_allow_html=True)
