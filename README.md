# 🩺 MedLens — AI Medical Report Translator

**Understand your lab results in plain English — grounded strictly in your own report.**

MedLens is a Streamlit app that translates medical reports (blood tests, metabolic panels, lipid panels, imaging summaries, etc.) into clear, jargon-free language a patient can actually understand. Every explanation is derived **only** from the text you upload — no external medical knowledge, no invented values, no hallucinated diagnoses.

---

## ✨ Features

- **📄 Multiple input formats** — paste text, upload digital PDFs, scanned PDFs, or photos (JPG / PNG / WEBP)
- **📷 Built-in OCR** — Tesseract reads scanned reports and photos, with automatic quality assessment (Good / Fair / Poor) and preprocessing tuned for lab tables
- **🔬 Value-by-value breakdown** — every test result explained with status (✅ Normal · ⚠️ Borderline · 🔴 Abnormal · ℹ️ Informational), reference range, and plain-language meaning
- **📖 Jargon decoder** — medical terms explained in context, with a source quote from your report
- **❓ Doctor-ready questions** — questions generated from *your specific* results, downloadable as a text card to bring to your appointment
- **💬 Follow-up Q&A** — ask questions about your report; the AI answers **only** from the report and says so honestly when something isn't there
- **🛡️ Anti-hallucination verification** — every value and quote is re-checked against the source text before it reaches the screen
- **📄 Evidence citations** — every explanation shows the exact text it was derived from
- **🎨 Custom UI** — teal/violet on graphite, animated hero, custom cards, no default Streamlit chrome

---

## 🛡️ How MedLens Fights Hallucination

Medical AI output demands more than "trust the model." MedLens layers five defenses:

| Layer | What it does |
|---|---|
| **`temperature=0.0`** | Removes randomness from model output |
| **Strict prompt** | Forbids adding external knowledge; requires verbatim `source_quote` for every claim |
| **Quote verification** | Each quoted passage is fuzzy-matched back to the report — invented quotes get dropped |
| **Value verification** | Each numeric value is digit-level checked against the report — invented numbers get dropped |
| **Graceful schema handling** | Over-long model output is trimmed to fit rather than crashing the whole translation |

Values that fail **both** checks are removed entirely. Values that fail only one are kept but **flagged** to the user with a reason (usually an OCR read error). The UI shows a verification banner summarising what passed, what was flagged, and what was dropped.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI  (custom CSS, no default chrome)          │
├─────────────────────────────────────────────────────────┤
│  Input layer:  Paste · PDF · Image · Sample             │
│      ↓                                                   │
│  OCR layer:    PyMuPDF (digital) → Tesseract (scanned)  │
│      ↓                                                   │
│  Prompt layer: Grounded translation + follow-up prompts │
│      ↓                                                   │
│  LLM:          Groq · openai/gpt-oss-120b               │
│      ↓                                                   │
│  Verify:       Quote check + value check + flag/drop    │
│      ↓                                                   │
│  Render:       Cards · pie chart · glossary · Q&A       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/medlens.git
cd medlens
pip install streamlit groq pydantic pymupdf pillow pytesseract plotly
```

You'll also need the **Tesseract OCR binary** installed on your system:

- **macOS**: `brew install tesseract`
- **Ubuntu / Debian**: `sudo apt install tesseract-ocr`
- **Windows**: [installer here](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) and grab a free API key.

### 3. Configure secrets

Create `.streamlit/secrets.toml` in the project root:

```toml
GROQ_API_KEY = "gsk_..."
# Optional — override the default model
# GROQ_MODEL = "openai/gpt-oss-120b"
```

For Streamlit Cloud, add the same keys under **Settings → Secrets**.

### 4. Run

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
medlens/
├── app.py                    # Entire application (single-file)
├── README.md
└── .streamlit/
    └── secrets.toml          # GROQ_API_KEY, optional GROQ_MODEL
```

The app is intentionally single-file so it can be dropped into Streamlit Cloud as-is.

---

## 🧠 Models

- **Default:** `openai/gpt-oss-120b` via Groq — fast, cheap, strong at structured JSON extraction
- **Any Groq-hosted chat model** that supports `response_format={"type": "json_object"}` works — change it in ⚙️ Settings inside the app
- gpt-oss models are reasoning models; the app pins `reasoning_effort` appropriately for the free tier's 8,000 TPM ceiling

---

## ⚠️ Known Limitations

- **OCR is the weakest link.** Phone photos of lab reports with uneven lighting can lose digit columns. The app warns you when quality drops and lets you verify against your original document.
- **Free-tier Groq** has an 8,000 TPM cap, so very long reports (100+ values) may need a paid tier or a shorter upload.
- **`plain_explanation` is not verified** — only source quotes and numeric values are. The model is heavily constrained by the prompt, but no LLM is 100% immune to narrative drift.
- **Not a diagnostic tool.** This is educational only.

---

## ⚕️ Medical Disclaimer

MedLens translates medical terminology into plain language for **educational purposes only**. It is **not** medical advice, diagnosis, or treatment. All explanations are derived strictly from the text you upload — nothing is invented or sourced externally. **Always consult your physician** before making any health decisions. If you are experiencing a medical emergency, call your local emergency number immediately.

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[Groq](https://groq.com)** — LLM inference (openai/gpt-oss-120b)
- **[PyMuPDF](https://pymupdf.readthedocs.io)** — PDF text extraction and page rendering
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** (via `pytesseract`) — OCR for scanned documents and photos
- **[Pillow](https://python-pillow.org)** — Image preprocessing
- **[Pydantic](https://docs.pydantic.dev)** — Schema validation and defensive clipping
- **[Plotly](https://plotly.com/python/)** — Status-distribution donut chart

---

## 📜 License

MIT — free to use, modify, and ship.

---

## 🙏 Acknowledgements

Built as a demonstration of **grounded AI** — where every claim is traceable back to the source, and verification is a first-class feature rather than an afterthought.

If you build on this, please keep the medical disclaimer intact and don't pretend it's a diagnostic tool. The goal is comprehension, not diagnosis.
