import streamlit as st
import pickle
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(
    page_title="InsureNLP — French Insurance Review Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 12px;
    border-radius: 8px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Page header banner ── */
.page-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    color: white;
}
.page-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.9rem;
    font-weight: 700;
    color: white !important;
}
.page-header p {
    margin: 0;
    font-size: 0.95rem;
    opacity: 0.88;
    color: white !important;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    text-align: center;
}
.kpi-card .kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-card .kpi-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Section headings ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
    border-left: 4px solid #0ea5e9;
    padding-left: 12px;
    margin: 28px 0 14px 0;
}

/* ── Review cards ── */
.review-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.review-card .review-meta {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 6px;
}
.review-card .review-text {
    font-size: 0.92rem;
    color: #1e293b;
    line-height: 1.6;
}

/* ── Score bar ── */
.score-bar-wrap { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.score-bar-bg {
    flex: 1; background: #f1f5f9; border-radius: 999px; height: 8px; overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}
.score-label { font-size: 0.78rem; color: #475569; min-width: 48px; text-align: right; }

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.pill-positive { background: #dcfce7; color: #166534; }
.pill-negative { background: #fee2e2; color: #991b1b; }
.pill-neutral  { background: #fef9c3; color: #854d0e; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
}

/* ── Info / Tip box ── */
.tip-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #1e40af;
    margin-bottom: 16px;
}

/* ── Matplotlib figure border ── */
.stPlotlyChart, [data-testid="stImage"] { border-radius: 12px; overflow: hidden; }

/* ── Metric overrides ── */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em;
    padding: 4px 0 16px 0;
    display: block;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: #94a3b8 !important;
    margin-top: -12px;
    margin-bottom: 20px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_DIR = os.path.join(BASE_DIR, "Classical_Models")

_candidate_paths = [
    os.path.join(BASE_DIR, "avis_traduit_final.xlsx"),
    os.path.join(os.path.dirname(BASE_DIR), "avis_traduit_final.xlsx"),
]
DATA_PATH = next((p for p in _candidate_paths if os.path.exists(p)), None)

# ── Optional: set a public URL so the app auto-downloads the dataset ────────────
# Paste a direct-download link (Google Drive, HuggingFace, etc.) here.
# Leave as empty string "" to disable auto-download.
DATA_URL = "https://huggingface.co/datasets/kleber099100/avis_traduit_final.xlsx/resolve/main/avis_traduit_final.xlsx"

# ── Loaders ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_classical_models():
    tfidf = joblib.load(os.path.join(CLASSICAL_DIR, "tfidf_vectorizer.joblib"))
    lr_sent = joblib.load(os.path.join(CLASSICAL_DIR, "lr_sentiment_model.joblib"))
    svc_stars = joblib.load(os.path.join(CLASSICAL_DIR, "svc_stars_model.joblib"))
    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
        sent_enc = pickle.load(f)
    with open(os.path.join(BASE_DIR, "categories.json")) as f:
        categories = json.load(f)
    return tfidf, lr_sent, svc_stars, sent_enc, categories


@st.cache_resource
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_flan_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "google/flan-t5-base"
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


@st.cache_data
def load_data():
    target = os.path.join(BASE_DIR, "avis_traduit_final.xlsx")

    # 1. Use local file if present
    path = next((p for p in _candidate_paths if os.path.exists(p)), None)

    # 2. Try auto-download if a URL is configured
    if path is None and DATA_URL:
        with st.spinner("Downloading dataset…"):
            try:
                import requests
                response = requests.get(DATA_URL, stream=True, timeout=120)
                response.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Verify it looks like a real Excel file (PK zip magic bytes)
                with open(target, "rb") as f:
                    magic = f.read(4)
                if magic != b"PK\x03\x04":
                    os.remove(target)
                    st.error(
                        "Download URL did not return a valid Excel file — "
                        "got an HTML page instead. Check that DATA_URL is a "
                        "**direct download** link, not a preview/viewer URL."
                    )
                    return None
                path = target
            except Exception as e:
                st.error(f"Auto-download failed: {e}")
                return None

    if path is None:
        return None  # caller will show the missing-data message

    df = pd.read_excel(path, engine="openpyxl")
    df = df.dropna(subset=["note"]).copy()
    df["note"] = df["note"].astype(int)
    df["avis_cor_en"] = df["avis_cor_en"].fillna(df["avis_en"]).fillna(df["avis"]).astype(str)
    return df


def _require_data(df) -> bool:
    """Show a friendly message and return False when the dataset is unavailable."""
    if df is not None:
        return True
    st.markdown("""
    <div style="background:#fef9c3;border:1px solid #fde047;border-radius:12px;padding:24px 28px;margin-top:12px;">
        <div style="font-size:1.4rem;font-weight:700;color:#854d0e;margin-bottom:8px;">📂 Dataset not found</div>
        <div style="color:#713f12;line-height:1.7;">
            This page needs <code>avis_traduit_final.xlsx</code> to work.<br><br>
            <strong>Option A — local file:</strong> place <code>avis_traduit_final.xlsx</code>
            in the project root and restart the app.<br>
            <strong>Option B — auto-download:</strong> set <code>DATA_URL</code> at the top of
            <code>app.py</code> to a direct-download link (Google Drive, HuggingFace, etc.).
        </div>
    </div>
    """, unsafe_allow_html=True)
    return False


@st.cache_data
def predict_sentiments(texts):
    tfidf, lr_sent, _, sent_enc, _ = load_classical_models()
    vec = tfidf.transform(texts)
    return sent_enc.inverse_transform(lr_sent.predict(vec))


@st.cache_resource
def compute_all_embeddings():
    cache_path = os.path.join(BASE_DIR, "corpus_embeddings.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    df = load_data()
    model = load_sentence_model()
    embs = model.encode(list(df["avis_cor_en"]), batch_size=64, show_progress_bar=False)
    np.save(cache_path, embs)
    return embs


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# Matplotlib style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#475569",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "text.color": "#0f172a",
    "grid.color": "#e2e8f0",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
})

COLORS_SENT = {"negative": "#f43f5e", "neutral": "#f59e0b", "positive": "#10b981"}
ACCENT = "#0ea5e9"

SENT_PILL = {
    "positive": '<span class="pill pill-positive">POSITIVE</span>',
    "negative": '<span class="pill pill-negative">NEGATIVE</span>',
    "neutral":  '<span class="pill pill-neutral">NEUTRAL</span>',
}

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<span class="sidebar-brand">NLP Project #2 </span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-tagline">By Andrea ZANIN and Antoine URSEL</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Insurer Deep-dive",
            "Prediction & Explanation",
            "Review Search",
            "RAG Q&A",
        ],
        label_visibility="collapsed",
    )
    st.divider()

# ── Page: Overview ─────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <h1>Overview Dashboard</h1>
        <p>Statistics on the whole dataset, sentiment distribution, and insurer comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if not _require_data(df):
        st.stop()
    df["pred_sent"] = predict_sentiments(list(df["avis_cor_en"]))

    pos_pct = (df["pred_sent"] == "positive").mean()
    neg_pct = (df["pred_sent"] == "negative").mean()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-label">Total Reviews</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{df["assureur"].nunique()}</div>
            <div class="kpi-label">Insurers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{df["note"].mean():.2f}<span style="font-size:1rem;color:#64748b;"> / 5</span></div>
            <div class="kpi-label">Avg Star Rating</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#10b981;">{pos_pct:.0%}</div>
            <div class="kpi-label">Positive Sentiment</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f43f5e;">{neg_pct:.0%}</div>
            <div class="kpi-label">Negative Sentiment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-heading">Star Rating Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = df["note"].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color=ACCENT, edgecolor="white", width=0.65, zorder=3)
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, color="#475569")
        ax.set_xlabel("Stars")
        ax.set_ylabel("Reviews")
        ax.set_xticks([1, 2, 3, 4, 5])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-heading">Predicted Sentiment Split</div>', unsafe_allow_html=True)
        sent_counts = df["pred_sent"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        wedge_colors = [COLORS_SENT.get(l, "#94a3b8") for l in sent_counts.index]
        wedges, texts, autotexts = ax2.pie(
            sent_counts.values, labels=sent_counts.index,
            autopct="%1.1f%%", colors=wedge_colors,
            startangle=90, pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")
            at.set_fontweight("bold")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown('<div class="section-heading">Top Insurers by Average Rating (min. 20 reviews)</div>', unsafe_allow_html=True)
    avg = df.groupby("assureur")["note"].agg(["mean", "count"]).reset_index()
    avg.columns = ["Insurer", "Avg Stars", "Reviews"]
    avg = avg[avg["Reviews"] >= 20].sort_values("Avg Stars", ascending=False).head(20)

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    bar_colors = [ACCENT if v >= 3.5 else "#f43f5e" for v in avg["Avg Stars"]]
    bars3 = ax3.bar(avg["Insurer"], avg["Avg Stars"], color=bar_colors, edgecolor="white", zorder=3)
    ax3.bar_label(bars3, fmt="%.2f", padding=4, fontsize=8, color="#475569")
    ax3.set_ylim(0, 5.5)
    ax3.set_xticklabels(avg["Insurer"], rotation=35, ha="right", fontsize=9)
    ax3.axhline(df["note"].mean(), color="#6366f1", linewidth=1.2, linestyle="--", label=f"Global avg ({df['note'].mean():.2f})")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    with st.expander("View full insurer table"):
        st.dataframe(avg.reset_index(drop=True), use_container_width=True)

# ── Page: Insurer Deep-dive ────────────────────────────────────────────────────

elif page == "Insurer Deep-dive":
    st.markdown("""
    <div class="page-header">
        <h1>Insurer Deep-dive</h1>
        <p>Per-insurer metrics, sentiment breakdown, and sample reviews</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if not _require_data(df):
        st.stop()
    df["pred_sent"] = predict_sentiments(list(df["avis_cor_en"]))

    insurer = st.selectbox("Select an Insurer", sorted(df["assureur"].unique()))
    sub = df[df["assureur"] == insurer]

    pos_pct_i = (sub["pred_sent"] == "positive").mean()
    neg_pct_i = (sub["pred_sent"] == "negative").mean()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{len(sub):,}</div>
            <div class="kpi-label">Total Reviews</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{sub["note"].mean():.2f}<span style="font-size:1rem;color:#64748b;"> / 5</span></div>
            <div class="kpi-label">Avg Stars</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#10b981;">{pos_pct_i:.0%}</div>
            <div class="kpi-label">Positive</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f43f5e;">{neg_pct_i:.0%}</div>
            <div class="kpi-label">Negative</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-heading">Star Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        counts_i = sub["note"].value_counts().sort_index()
        ax.bar(counts_i.index, counts_i.values, color=ACCENT, edgecolor="white", width=0.65, zorder=3)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Stars")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-heading">Sentiment Breakdown</div>', unsafe_allow_html=True)
        sent_c = sub["pred_sent"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        ax2.pie(
            sent_c.values, labels=sent_c.index,
            autopct="%1.1f%%",
            colors=[COLORS_SENT.get(l, "#94a3b8") for l in sent_c.index],
            startangle=90, pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown('<div class="section-heading">Sample Reviews by Stars</div>', unsafe_allow_html=True)
    star_buckets = [
        ("positive", "Positive (4–5 stars)", sub[sub["note"] >= 4]),
        ("neutral",  "Neutral (3 stars)",    sub[sub["note"] == 3]),
        ("negative", "Negative (1–2 stars)", sub[sub["note"] <= 2]),
    ]
    for pill_key, bucket_label, pool in star_buckets:
        if len(pool) == 0:
            continue
        sample = pool.sample(min(2, len(pool)), random_state=42)
        st.markdown(f'<div style="font-size:0.82rem;font-weight:600;color:#475569;margin:14px 0 6px 0;">{bucket_label}</div>', unsafe_allow_html=True)
        for _, row in sample.iterrows():
            pill = SENT_PILL.get(pill_key, "")
            st.markdown(f"""
            <div class="review-card">
                <div class="review-meta">{pill} &nbsp; ⭐ {row['note']} / 5 &nbsp;·&nbsp; {row['assureur']}</div>
                <div class="review-text">{str(row['avis_cor_en'])[:350]}…</div>
            </div>
            """, unsafe_allow_html=True)

# ── Page: Prediction & Explanation ────────────────────────────────────────────

elif page == "Prediction & Explanation":
    st.markdown("""
    <div class="page-header">
        <h1>Prediction &amp; Explanation</h1>
        <p>Enter a review to get its predicted star rating, sentiment, topic, and word-level explanations</p>
    </div>
    """, unsafe_allow_html=True)

    tfidf, lr_sent, svc_stars, sent_enc, categories = load_classical_models()

    POSITIVE_EXAMPLE = "Very excellent insurer and good service."
    NEGATIVE_EXAMPLE = "Bad insurer, the pricing is too high."

    ex_col1, ex_col2, _ = st.columns([1, 1, 4])
    if ex_col1.button("✅ Positive example"):
        st.session_state["review_input"] = POSITIVE_EXAMPLE
        st.session_state["auto_run"] = True
    if ex_col2.button("❌ Negative example"):
        st.session_state["review_input"] = NEGATIVE_EXAMPLE
        st.session_state["auto_run"] = True

    review_text = st.text_area(
        "Enter your review (in English):",
        height=140,
        placeholder="e.g. The claim process was very smooth and the agents were helpful...",
        value=st.session_state.get("review_input", ""),
        key="review_input",
    )

    run = st.button("Analyse Review", type="primary") or st.session_state.pop("auto_run", False)

    if run and review_text.strip():
        with st.spinner("Analysing…"):
            vec = tfidf.transform([review_text])

            sent_proba = lr_sent.predict_proba(vec)[0]
            sent_idx = int(np.argmax(sent_proba))
            sent_label = sent_enc.inverse_transform([sent_idx])[0]
            sent_conf = sent_proba[sent_idx]

            dec_scores = svc_stars.decision_function(vec)[0]
            star_proba = softmax(dec_scores)
            star_pred = int(svc_stars.predict(vec)[0]) + 1
            star_conf = star_proba[star_pred - 1]

            text_lower = review_text.lower()
            keyword_map = {
                "Pricing": ["price", "cost", "expensive", "cheap", "premium", "rate"],
                "Claims Processing": ["claim", "claims", "reimburs", "payout"],
                "Customer Service": ["service", "agent", "staff", "support", "help", "respond"],
                "Coverage": ["cover", "coverage", "policy", "benefit"],
                "Enrollment": ["enrol", "sign up", "register", "join"],
                "Cancellation": ["cancel", "terminat", "end contract"],
            }
            kw_scores = {c: sum(kw in text_lower for kw in kws) for c, kws in keyword_map.items()}
            top_cat = max(kw_scores, key=kw_scores.get) if max(kw_scores.values()) > 0 else "Overall Experience"

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="section-heading">Predictions</div>', unsafe_allow_html=True)
            stars_str = "⭐" * star_pred + "☆" * (5 - star_pred)
            pill = SENT_PILL.get(sent_label, "")
            st.markdown(f"""
            <div class="review-card">
                <div class="review-meta">Star Rating</div>
                <div class="kpi-value">{stars_str} &nbsp; {star_pred} / 5</div>
                <div class="review-meta" style="margin-top:6px;">Confidence: {star_conf:.0%}</div>
            </div>
            <div class="review-card">
                <div class="review-meta">Sentiment</div>
                <div style="margin:4px 0;">{pill}</div>
                <div class="review-meta">Confidence: {sent_conf:.0%}</div>
            </div>
            <div class="review-card">
                <div class="review-meta">Topic Category</div>
                <div style="font-weight:600;color:#0f172a;">{top_cat}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-heading">Confidence Breakdown</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))

            sent_colors = [COLORS_SENT.get(c, "#94a3b8") for c in sent_enc.classes_]
            axes[0].barh(sent_enc.classes_, sent_proba, color=sent_colors, edgecolor="white", zorder=3)
            axes[0].set_xlim(0, 1)
            axes[0].set_title("Sentiment Probabilities", fontsize=10)
            for i, v in enumerate(sent_proba):
                axes[0].text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=8, color="#475569")

            axes[1].bar(range(1, 6), star_proba, color=ACCENT, edgecolor="white", zorder=3)
            axes[1].set_xticks(range(1, 6))
            axes[1].set_ylim(0, 1)
            axes[1].set_title("Star Rating Probabilities", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        feat_names = tfidf.get_feature_names_out()
        feat_vec = vec.toarray()[0]
        top_idx_w = feat_vec.argsort()[-15:][::-1]
        top_words = [(feat_names[i], float(feat_vec[i])) for i in top_idx_w if feat_vec[i] > 0]

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-heading">Word Importance (TF-IDF)</div>', unsafe_allow_html=True)
            if top_words:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.barh([x[0] for x in top_words[::-1]], [x[1] for x in top_words[::-1]],
                         color="#6366f1", edgecolor="white", zorder=3)
                ax2.set_xlabel("TF-IDF Weight")
                ax2.set_title("Most influential terms", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        with col4:
            st.markdown('<div class="section-heading">Sentiment Driver (LR Coefficients)</div>', unsafe_allow_html=True)
            coef_sent = lr_sent.coef_[sent_idx]
            influence = [
                (feat_names[i], float(coef_sent[i] * feat_vec[i]))
                for i in range(len(feat_names)) if feat_vec[i] > 0
            ]
            influence.sort(key=lambda x: abs(x[1]), reverse=True)
            df_inf = pd.DataFrame(influence[:15], columns=["Word", "Contribution"])
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            colors3 = ["#10b981" if c > 0 else "#f43f5e" for c in df_inf["Contribution"]]
            ax3.barh(df_inf["Word"][::-1], df_inf["Contribution"][::-1], color=colors3[::-1], edgecolor="white", zorder=3)
            ax3.axvline(0, color="#94a3b8", linewidth=0.8)
            ax3.set_title(f"Contributions toward: {sent_label}", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    elif not review_text.strip():
        st.markdown('<div class="tip-box">💡 Enter a review above (or click an example) and press <strong>Analyse Review</strong>.</div>', unsafe_allow_html=True)

# ── Page: Review Search ────────────────────────────────────────────────────────

elif page == "Review Search":
    st.markdown("""
    <div class="page-header">
        <h1>Review Search</h1>
        <p>Find relevant reviews using Semantic embeddings, BM25 keyword ranking, or a Hybrid of both</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if not _require_data(df):
        st.stop()
    df["pred_sent"] = predict_sentiments(list(df["avis_cor_en"]))

    search_mode = st.radio(
        "Search method:",
        ["Semantic (MiniLM)", "BM25 (keyword)", "Hybrid (BM25 + Semantic)"],
        horizontal=True,
    )
    mode_desc = {
        "Semantic (MiniLM)": "Finds reviews by meaning using all-MiniLM-L6-v2 sentence embeddings + cosine similarity.",
        "BM25 (keyword)": "Classic keyword-based ranking — fast, no model needed. Best for exact terms.",
        "Hybrid (BM25 + Semantic)": "Combines semantic + BM25 scores (equal weight). Best overall relevance.",
    }
    st.markdown(f'<div class="tip-box">{mode_desc[search_mode]}</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. slow claims reimbursement, friendly customer support…",
    )

    with st.expander("Filters"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            star_filter = st.multiselect("Stars", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        with fc2:
            sent_filter = st.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                         default=["positive", "neutral", "negative"])
        with fc3:
            insurer_filter = st.multiselect("Insurer", sorted(df["assureur"].unique()), default=[])
        top_n = st.slider("Number of results", 5, 50, 10)

    pool = df[df["note"].isin(star_filter) & df["pred_sent"].isin(sent_filter)]
    if insurer_filter:
        pool = pool[pool["assureur"].isin(insurer_filter)]
    pool = pool.dropna(subset=["avis_cor_en"])

    if query.strip():
        use_semantic = search_mode in ("Semantic (MiniLM)", "Hybrid (BM25 + Semantic)")
        use_bm25 = search_mode in ("BM25 (keyword)", "Hybrid (BM25 + Semantic)")
        corpus = list(pool["avis_cor_en"])

        if use_bm25:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            bm25_raw = np.array(bm25.get_scores(query.lower().split()), dtype=float)
            mx = bm25_raw.max()
            bm25_scores = bm25_raw / mx if mx > 0 else bm25_raw

        if use_semantic:
            with st.spinner("Computing semantic similarity…"):
                all_embs = compute_all_embeddings()
                pool_embs = all_embs[pool.index]
                sem_scores = cosine_similarity(
                    load_sentence_model().encode([query]), pool_embs
                )[0]

        if use_semantic and use_bm25:
            final_scores = (sem_scores + bm25_scores) / 2
        elif use_semantic:
            final_scores = sem_scores
        else:
            final_scores = bm25_scores

        top_idx = final_scores.argsort()[-top_n:][::-1]
        result = pool.iloc[top_idx].copy()
        result["score"] = final_scores[top_idx]

        st.markdown(f'<div class="section-heading">Top {top_n} results · {len(pool):,} reviews searched</div>', unsafe_allow_html=True)

        for _, row in result.iterrows():
            score_pct = row["score"] * 100
            fill_w = max(0, min(100, int(score_pct)))
            pill = SENT_PILL.get(row["pred_sent"], "")
            stars_str = "⭐" * int(row["note"])
            st.markdown(f"""
            <div class="review-card">
                <div class="review-meta">
                    {pill} &nbsp; {stars_str} ({row['note']}/5) &nbsp;·&nbsp; <strong>{row['assureur']}</strong>
                </div>
                <div class="review-text" style="margin:8px 0;">{str(row['avis_cor_en'])[:420]}</div>
                <div class="score-bar-wrap">
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{fill_w}%;"></div></div>
                    <span class="score-label">{score_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="tip-box">💡 Enter a search query above. Use Semantic for meaning-based search, BM25 for exact keywords, or Hybrid for best results.</div>', unsafe_allow_html=True)

# ── Page: RAG Q&A ──────────────────────────────────────────────────────────────

elif page == "RAG Q&A":
    st.markdown("""
    <div class="page-header">
        <h1>RAG Q&amp;A</h1>
        <p>Retrieval-Augmented Generation — ask questions answered from real reviews using flan-t5-base</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if not _require_data(df):
        st.stop()
    df["pred_sent"] = predict_sentiments(list(df["avis_cor_en"]))

    st.markdown('<div class="tip-box">The system retrieves the most relevant reviews via TF-IDF, then generates an answer with <strong>flan-t5-base</strong> (≈1 GB download on first use, loaded on demand).</div>', unsafe_allow_html=True)

    question = st.text_input("Your question:", placeholder="What do customers say about claims processing?")
    qa_col1, qa_col2 = st.columns([2, 1])
    with qa_col1:
        insurer_qa = st.selectbox("Filter by insurer (optional)", ["All"] + sorted(df["assureur"].unique()))
    with qa_col2:
        n_retrieve = st.slider("Reviews to retrieve", 3, 20, 5)

    if st.button("Ask", type="primary") and question:
        sub_qa = df if insurer_qa == "All" else df[df["assureur"] == insurer_qa]
        sub_qa = sub_qa.dropna(subset=["avis_cor_en"]).reset_index(drop=True)

        with st.spinner("Retrieving relevant reviews…"):
            tfidf_qa = TfidfVectorizer(max_features=10000, stop_words="english")
            doc_vecs = tfidf_qa.fit_transform(sub_qa["avis_cor_en"])
            q_vec = tfidf_qa.transform([question])
            scores = cosine_similarity(q_vec, doc_vecs)[0]
            top_idx = scores.argsort()[-n_retrieve:][::-1]
            retrieved = sub_qa.iloc[top_idx]

        st.markdown(f'<div class="section-heading">Top {n_retrieve} Retrieved Reviews</div>', unsafe_allow_html=True)
        for _, row in retrieved.iterrows():
            pill = SENT_PILL.get(row["pred_sent"], "")
            st.markdown(f"""
            <div class="review-card">
                <div class="review-meta">{pill} &nbsp; ⭐ {row['note']} / 5 &nbsp;·&nbsp; <strong>{row['assureur']}</strong></div>
                <div class="review-text">{str(row['avis_cor_en'])[:320]}…</div>
            </div>
            """, unsafe_allow_html=True)

        context = "\n\n".join([
            f"Review (star {r['note']}): {str(r['avis_cor_en'])[:200]}"
            for _, r in retrieved.iterrows()
        ])
        st.session_state["rag_context"] = context
        st.session_state["rag_question"] = question

    if "rag_context" in st.session_state:
        st.divider()
        if st.button("Generate Answer with flan-t5", type="secondary"):
            with st.spinner("Generating answer (loading flan-t5-base on first use)…"):
                try:
                    import torch
                    tokenizer_t5, model_t5 = load_flan_t5()
                    prompt = (
                        f"Based on these insurance reviews:\n{st.session_state['rag_context']}"
                        f"\n\nQuestion: {st.session_state['rag_question']}\nAnswer:"
                    )
                    inputs = tokenizer_t5(prompt, return_tensors="pt", max_length=512, truncation=True)
                    with torch.no_grad():
                        outputs = model_t5.generate(**inputs, max_new_tokens=200)
                    answer = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
                    st.markdown('<div class="section-heading">Generated Answer</div>', unsafe_allow_html=True)
                    st.success(answer)
                except Exception as e:
                    st.error(f"LLM generation failed: {e}")
