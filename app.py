# ──────────────────────────────────────────────────────────────
# MedLens — AI Medical Report Translator  (with OCR support)
# Python + Streamlit + Groq  |  Streamlit Cloud ready
# ──────────────────────────────────────────────────────────────

import hashlib
import html
import io
import json
import os
import textwrap
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
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_UPLOAD_MB = 10
MAX_TEXT_CHARS = 12_000
MAX_QUESTION_CHARS = 500

STATUS_COLORS = {
    "Normal": "#10b981",
    "Borderline": "#f59e0b",
    "Abnormal": "#f43f5e",
    "Informational": "#6366f1",
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
/* ── Global ── */
.block-container{max-width:1480px;padding-top:1.5rem;padding-bottom:3rem}

/* ── Hero banner ── */
.ml-hero{
    background:
        radial-gradient(ellipse at 90% 0%,rgba(99,102,241,.32),transparent 55%),
        radial-gradient(ellipse at 10% 100%,rgba(16,185,129,.18),transparent 50%),
        linear-gradient(145deg,#0f172a 0%,#1e293b 100%);
    border:1px solid rgba(148,163,184,.18);
    border-radius:24px;
    padding:42px 44px;
    margin-bottom:28px;
    color:#f8fafc;
    position:relative;
    overflow:hidden;
}
.ml-hero::before{
    content:"";position:absolute;inset:0;
    background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events:none;
}
.ml-eyebrow{
    color:#a5b4fc;font-size:11px;font-weight:700;
    letter-spacing:2.8px;margin-bottom:14px;
}
.ml-hero h1{
    color:#f8fafc;font-size:clamp(30px,4.5vw,48px);
    letter-spacing:-1.8px;margin:0 0 14px;line-height:1.12;
}
.ml-hero p{
    color:#cbd5e1;font-size:15.5px;max-width:740px;
    line-height:1.75;margin:0 0 8px;
}
.ml-pill{
    display:inline-block;border-radius:999px;
    padding:5px 13px;margin-top:16px;margin-right:8px;
    background:rgba(165,180,252,.10);
    border:1px solid rgba(165,180,252,.22);
    color:#c7d2fe;font-size:11px;font-weight:600;
    letter-spacing:1px;
}

/* ── Cards ── */
.ml-card{
    border:1px solid rgba(148,163,184,.15);
    border-radius:16px;padding:22px 26px;
    background:linear-gradient(135deg,rgba(30,41,59,.45),rgba(15,23,42,.55));
    margin-bottom:14px;
    transition:border-color .2s;
}
.ml-card:hover{border-color:rgba(99,102,241,.4)}

.ml-card-normal{border-left:4px solid #10b981}
.ml-card-borderline{border-left:4px solid #f59e0b}
.ml-card-abnormal{border-left:4px solid #f43f5e}
.ml-card-informational{border-left:4px solid #6366f1}

.ml-card h4{margin:0 0 6px;font-size:16px;color:#f1f5f9}
.ml-card p{margin:0 0 4px;font-size:14px;color:#cbd5e1;line-height:1.65}
.ml-card .ml-label{
    font-size:11px;font-weight:600;letter-spacing:1.2px;
    text-transform:uppercase;margin-bottom:8px;
}

/* ── Metric tiles ── */
.ml-metric{
    text-align:center;padding:20px 14px;
    border-radius:16px;
    background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(15,23,42,.6));
    border:1px solid rgba(148,163,184,.12);
}
.ml-metric .num{font-size:36px;font-weight:800;letter-spacing:-1px;line-height:1.1}
.ml-metric .lbl{font-size:12px;color:#94a3b8;margin-top:4px;letter-spacing:.5px}

/* ── Summary box ── */
.ml-summary{
    border-radius:18px;padding:28px 32px;
    background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(16,185,129,.06));
    border:1px solid rgba(99,102,241,.18);
    line-height:1.8;color:#e2e8f0;font-size:15px;
}

/* ── Glossary ── */
.ml-glossary{
    border-radius:14px;padding:16px 22px;
    background:rgba(30,41,59,.45);
    border:1px solid rgba(148,163,184,.12);
    margin-bottom:10px;
}
.ml-glossary strong{color:#a5b4fc;font-size:14px}
.ml-glossary span{color:#cbd5e1;font-size:13.5px;line-height:1.65}

/* ── Questions card ── */
.ml-question{
    border-radius:12px;padding:14px 20px;
    background:rgba(245,158,11,.06);
    border:1px solid rgba(245,158,11,.18);
    margin-bottom:8px;color:#fbbf24;
    font-size:14px;line-height:1.6;
}

/* ── Disclaimer ── */
.ml-disclaimer{
    border-radius:14px;padding:18px 24px;
    background:rgba(239,68,68,.06);
    border:1px solid rgba(239,68,68,.18);
    color:#fca5a5;font-size:13px;line-height:1.7;
    margin-top:10px;
}

/* ── Evidence quote ── */
.ml-evidence{
    border-left:3px solid #6366f1;
    padding:12px 18px;margin:8px 0;
    background:rgba(99,102,241,.05);
    border-radius:0 12px 12px 0;
    font-size:13.5px;color:#cbd5e1;
    line-height:1.65;white-space:pre-wrap;
    overflow-wrap:anywhere;
}

/* ── OCR preview ── */
.ml-ocr-badge{
    display:inline-block;border-radius:999px;
    padding:4px 12px;margin:4px 4px 4px 0;
    font-size:11px;font-weight:600;letter-spacing:.8px;
}
.ml-ocr-good{background:rgba(16,185,129,.12);color:#6ee7b7;border:1px solid rgba(16,185,129,.25)}
.ml-ocr-fair{background:rgba(245,158,11,.12);color:#fcd34d;border:1px solid rgba(245,158,11,.25)}
.ml-ocr-poor{background:rgba(239,68,68,.12);color:#fca5a5;border:1px solid rgba(239,68,68,.25)}

/* ── Hide Streamlit branding ── */
footer{visibility:hidden}
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
    report_type: str = Field(max_length=100)
    patient_summary: str = Field(max_length=600)
    overall_impression: str = Field(max_length=1200)
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


def friendly_error(error: Exception) -> str:
    if isinstance(error, AuthenticationError):
        return "🔑 Authentication failed — check GROQ_API_KEY in your Streamlit secrets."
    if isinstance(error, RateLimitError):
        return "⏳ Rate limit reached — wait a moment and retry."
    if isinstance(error, APIConnectionError):
        return "🌐 Cannot reach Groq — check your connection and retry."
    if isinstance(error, APIStatusError):
        return "❌ Groq rejected the request — verify model availability and account limits."
    return f"⚠️ Unexpected error: {type(error).__name__}. Try again or choose a different model."


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
) -> dict:
    with Groq(api_key=api_key, timeout=90.0, max_retries=1) as client:
        result = client.chat.completions.create(
            model=model,
            temperature=0.05,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    if result.choices[0].finish_reason == "length":
        raise ValueError("Response was truncated — try a shorter report.")
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
        escaped = html.escape(quote.strip())
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

Return ONLY valid JSON matching this schema:
{json.dumps(FollowUpAnswer.model_json_schema(), indent=2)}
"""


# ─── Sidebar ──────────────────────────────────────────────────

api_key = get_setting("GROQ_API_KEY")

with st.sidebar:
    st.markdown("## 🩺 MedLens")
    st.caption("MEDICAL REPORT TRANSLATOR")
    st.divider()

    if api_key:
        st.success("Groq API connected", icon="✅")
    else:
        st.warning("Add GROQ_API_KEY to Streamlit secrets.", icon="🔑")

    model = st.text_input(
        "AI Model",
        value=get_setting("GROQ_MODEL", DEFAULT_MODEL),
        help="Any Groq model supporting chat + JSON output.",
    ).strip()

    st.divider()
    st.markdown("##### How it works")
    st.markdown("""
    1. 📄 Upload your lab report (PDF, image, or text)
    2. 🔍 OCR reads scanned/photo reports automatically
    3. 🤖 AI reads **only your report** — nothing else
    4. 📊 See every value explained in plain language
    5. ❓ Ask follow-up questions about your results
    """)

    st.divider()
    st.markdown("##### Supported formats")
    st.markdown("""
    | Format | Method |
    |---|---|
    | 📋 Paste text | Direct input |
    | 📎 PDF (digital) | Text extraction |
    | 📎 PDF (scanned) | OCR fallback |
    | 📷 Image (JPG/PNG/WEBP) | OCR |
    """)

    st.divider()
    render_disclaimer()

    st.divider()
    if st.button("🗑️ Clear workspace", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ─── Hero ─────────────────────────────────────────────────────

st.markdown("""
<div class="ml-hero">
    <div class="ml-eyebrow">AI · MEDICAL REPORT INTELLIGENCE</div>
    <h1>Understand your lab results<br>in plain English.</h1>
    <p>
        Upload any medical report — typed, scanned, or photographed —
        and get a clear, jargon-free translation.
        Every explanation comes directly from <strong>your report</strong> —
        nothing is invented, nothing is sourced externally.
    </p>
    <span class="ml-pill">🔒 YOUR DATA ONLY</span>
    <span class="ml-pill">🚫 ZERO HALLUCINATION</span>
    <span class="ml-pill">📷 OCR SUPPORTED</span>
    <span class="ml-pill">📄 EVIDENCE CITED</span>
    <span class="ml-pill">⚕️ NOT MEDICAL ADVICE</span>
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
            <div style="border-radius:12px;padding:14px 20px;
                background:rgba(99,102,241,.06);
                border:1px solid rgba(99,102,241,.15);
                color:#a5b4fc;font-size:13px;line-height:1.6;margin-bottom:12px">
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
                translation = ReportTranslation.model_validate(raw)
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
    <div style="border-radius:14px;padding:16px 22px;
        background:rgba(245,158,11,.06);
        border:1px solid rgba(245,158,11,.18);
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

c1, c2, c3, c4, c5 = st.columns(5)

for col, num, label, color in [
    (c1, total, "Values Found", "#a5b4fc"),
    (c2, normal_n, "Normal", STATUS_COLORS["Normal"]),
    (c3, borderline_n, "Borderline", STATUS_COLORS["Borderline"]),
    (c4, abnormal_n, "Abnormal", STATUS_COLORS["Abnormal"]),
    (c5, info_n, "Informational", STATUS_COLORS["Informational"]),
]:
    col.markdown(
        f'<div class="ml-metric">'
        f'<div class="num" style="color:{color}">{num}</div>'
        f'<div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")


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
            st.markdown(f"**{tr.report_type}**")

        with st.container(border=True):
            st.markdown("##### 👤 Patient Information")
            st.markdown(f"{tr.patient_summary}")

        st.markdown(
            f'<div class="ml-summary">'
            f'<strong>🔍 Overall Impression</strong><br><br>'
            f'{html.escape(tr.overall_impression)}</div>',
            unsafe_allow_html=True,
        )

        if tr.confidence_note:
            st.markdown("")
            st.info(f"🔎 **AI Confidence Note:** {tr.confidence_note}")

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
                font=dict(family="Inter, sans-serif", color="#e2e8f0"),
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
                st.warning(note, icon="📌")


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

            for val in display_values:
                css_class = f"ml-card ml-card-{val.status.lower()}"
                icon = STATUS_ICONS.get(val.status, "")
                causes_html = ""
                if val.possible_causes:
                    causes_items = "".join(
                        f"<li>{html.escape(c)}</li>"
                        for c in val.possible_causes
                    )
                    causes_html = (
                        f'<p style="margin-top:8px">'
                        f'<strong>Possible causes mentioned/implied in '
                        f'report:</strong></p>'
                        f'<ul style="color:#cbd5e1;font-size:13.5px">'
                        f'{causes_items}</ul>'
                    )

                st.markdown(f"""
                <div class="{css_class}">
                    <div class="ml-label" style="color:{STATUS_COLORS[val.status]}">
                        {icon} {val.status.upper()}
                    </div>
                    <h4>{html.escape(val.name)}</h4>
                    <p>
                        <strong>Your result:</strong>
                        {html.escape(val.reported_value)}
                        {html.escape(val.unit)}&nbsp;&nbsp;|&nbsp;&nbsp;
                        <strong>Normal range:</strong>
                        {html.escape(val.reference_range)}
                        {html.escape(val.unit)}
                    </p>
                    <p style="margin-top:8px">
                        {html.escape(val.plain_explanation)}
                    </p>
                    {causes_html}
                </div>
                """, unsafe_allow_html=True)

                render_evidence(val.source_quote)
                st.markdown("")


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
                <strong>🔬 {html.escape(term.term)}</strong><br>
                <span>{html.escape(term.definition)}</span>
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
                {html.escape(q.question)}<br>
                <span style="color:#fde68a;font-size:13px">
                    💡 <em>Why ask this:</em> {html.escape(q.reason)}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        questions_text = "\n\n".join(
            f"Q{i}: {q.question}\nWhy: {q.reason}"
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
                answer = FollowUpAnswer.model_validate(raw)
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
            st.markdown(f"**You asked:** {html.escape(saved['question'])}")
            st.markdown("")

            if ans.cannot_answer:
                st.warning(
                    f"🚫 **Cannot answer from your report.** "
                    f"{ans.reason_cannot_answer}"
                )
            else:
                st.markdown(
                    f'<div class="ml-summary">'
                    f'{html.escape(ans.answer)}</div>',
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
<div style="text-align:center;padding:20px 0 10px">
    <span style="color:#64748b;font-size:13px">
        🩺 MedLens · Built with Streamlit + Groq ·
        Your data stays in this session · AI explains, never diagnoses ·
        📷 OCR powered by Tesseract ·
        <strong>Always consult your physician</strong>
    </span>
</div>
""", unsafe_allow_html=True)
