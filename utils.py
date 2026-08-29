"""
utils.py
Shared utilities for the Indonesian News Classification Dashboard.

Contains:
- Cached data/model/metrics loaders
- Text preprocessing pipeline (mirrors notebook 01_dataCleaning.ipynb)
- Shared design tokens / CSS injection
"""

import json
import re
import html
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "df_final.csv"
METRICS_PATH = BASE_DIR / "data" / "metrics.json"
MODEL_PATH = BASE_DIR / "model" / "final_pipeline.pkl"

LABEL_COL = "article_topic_grouped"

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PRIMARY = "#4F46E5"       # indigo
PRIMARY_DARK = "#3730A3"
ACCENT = "#06B6D4"        # cyan
BG = "#0F172A"
CARD_BG = "#1E293B"
TEXT_MUTED = "#94A3B8"

CATEGORY_COLORS = {
    "Hiburan & Lifestyle": "#F472B6",
    "Ekonomi & Bisnis": "#34D399",
    "Olahraga": "#FBBF24",
    "Haji": "#A78BFA",
    "Internasional & Sejarah": "#60A5FA",
    "Teknologi & Sains": "#22D3EE",
    "Kesehatan": "#F87171",
    "Regional": "#FB923C",
    "Politik & Hukum": "#818CF8",
    "Otomotif": "#4ADE80",
    "Pendidikan": "#E879F9",
}


def inject_global_css():
    st.markdown(
        f"""
        <style>
        .main {{
            padding-top: 1rem;
        }}
        .stApp {{
            background: radial-gradient(circle at 10% 0%, #111827 0%, #0B1120 45%, #0A0F1C 100%);
        }}
        .metric-card {{
            background: linear-gradient(160deg, {CARD_BG} 0%, #16213A 100%);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        }}
        .metric-label {{
            color: {TEXT_MUTED};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: #F8FAFC;
            margin-top: 0.15rem;
        }}
        .metric-sub {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }}
        .section-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 0.15rem;
        }}
        .section-sub {{
            color: {TEXT_MUTED};
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }}
        .insight-box {{
            background: rgba(79,70,229,0.10);
            border-left: 4px solid {PRIMARY};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin: 0.6rem 0;
            color: #E2E8F0;
            font-size: 0.93rem;
        }}
        .warn-box {{
            background: rgba(251,191,36,0.10);
            border-left: 4px solid #FBBF24;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin: 0.6rem 0;
            color: #E2E8F0;
            font-size: 0.93rem;
        }}
        .pred-badge {{
            display: inline-block;
            padding: 0.55rem 1.3rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 1.15rem;
            color: white;
            background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
            box-shadow: 0 6px 20px rgba(79,70,229,0.35);
        }}
        .pill {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: rgba(148,163,184,0.15);
            color: #CBD5E1;
            font-size: 0.75rem;
            margin-right: 0.3rem;
        }}
        section[data-testid="stSidebar"] {{
            background: #0B1120;
            border-right: 1px solid rgba(148,163,184,0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["content_length"] = df["article_content"].astype(str).str.split().str.len()
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = joblib.load(MODEL_PATH)
    return pipe


# ---------------------------------------------------------------------------
# Text preprocessing — mirrors notebook 01_dataCleaning.ipynb exactly
# so that raw user-submitted news text is normalized the same way the
# training corpus was, before being fed into the TF-IDF + LinearSVC pipeline.
# ---------------------------------------------------------------------------
_DEEP_CLEAN_PATTERNS = [
    (r"<.*?>", " "),
    (r"(?i)\b(baca juga|baca selengkapnya|artikel terkait)\s*:?\s*.*?(?=\n|$)", " "),
    (r"(?i)\b(penulis|editor|reporter|pewarta|sumber)\s*:\s*[a-zA-Z\s\.\,]+(?=\n|$)", " "),
    (r"^\s*[A-Z][a-zA-Z\s]+(?:\s*\([A-Za-z\s]+\))?(?:,\s*[A-Za-z0-9\.\s]+)?\s*[-–]\s+", ""),
    (r"\([^\)]*?(?i:reporter|editor|pewarta)[^\)]*?\)", " "),
    (r"http\S+|www\.\S+", " "),
]

_CUSTOM_STOPWORDS = {
    "kompas", "detik", "tribun", "antara", "cnn", "cnbc",
    "com", "co", "id", "jakarta", "redaksi", "pewarta",
    "reporter", "editor", "baca", "juga", "artikel", "terkait",
}


@st.cache_resource(show_spinner=False)
def get_stopword_set():
    import nltk
    try:
        from nltk.corpus import stopwords
        base = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        base = set(stopwords.words("indonesian"))
    return base.union(_CUSTOM_STOPWORDS)


@st.cache_resource(show_spinner=False)
def get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


def _apply_deep_cleaning(text: str) -> str:
    cleaned = text
    for pattern, repl in _DEEP_CLEAN_PATTERNS:
        cleaned = re.sub(pattern, repl, cleaned)
    return cleaned


def _split_camel_case(text: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def _normalize_text(text: str) -> str:
    text = _split_camel_case(text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(raw_text: str) -> str:
    """
    Full preprocessing pipeline for a single raw news article string,
    replicating: HTML unescape -> deep cleaning -> normalization ->
    stopword removal -> Sastrawi stemming.
    Returns the cleaned string ready to be passed into the TF-IDF pipeline.
    """
    if not raw_text or not str(raw_text).strip():
        return ""

    stopword_set = get_stopword_set()
    stemmer = get_stemmer()

    text = html.unescape(str(raw_text))
    text = _apply_deep_cleaning(text)
    text = _normalize_text(text)

    tokens = [w for w in text.split() if w.isalpha() and w not in stopword_set]
    stemmed = [stemmer.stem(w) for w in tokens]
    stemmed = [w for w in stemmed if w and w not in stopword_set]
    return " ".join(stemmed)
