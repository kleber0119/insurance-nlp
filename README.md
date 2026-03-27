# NLP Project 2 - By Andrea ZANIN and Antoine URSEL

**Streamlit App link :** https://insurance-nlp-andreazanin.streamlit.app/

> End-to-end NLP pipeline for sentiment analysis, star-rating prediction, semantic search, and RAG Q&A on French insurance reviews.

---

## Overview

This project builds a complete supervised-learning NLP pipeline on a dataset of 34 thousand French insurance reviews. Reviews are translated to English and processed through classical ML models, deep learning architectures, and modern transformer-based retrieval, with an interactive Streamlit analytics dashboard.

**Key results**

| Task | Model | Accuracy |
|---|---|---|
| Sentiment classification (3 classes) | TF-IDF + Logistic Regression | ~85 % |
| Star rating prediction (1–5) | TF-IDF + LinearSVC | ~65 % |
| Sentiment (deep learning) | BiLSTM + custom embeddings | ~83 % |
| Star rating (deep learning) | BiLSTM | ~60 % |

---

## Pipeline

```
Raw French reviews
        │
        ▼
  [1] Data collection & preprocessing
      • Language detection & translation (FR → EN)
      • Text cleaning, de-duplication
        │
        ▼
  [2] Exploratory Data Analysis
      • Star distribution, insurer benchmarks
      • Word clouds, n-gram frequencies
        │
        ▼
  [3] Classical ML (scikit-learn)
      • TF-IDF vectorisation (n-grams 1–2, 50 k features)
      • Logistic Regression  → sentiment (neg / neu / pos)
      • LinearSVC            → star rating (1–5)
      • Evaluation: accuracy, F1-macro, confusion matrices
        │
        ▼
  [4] Deep Learning (Keras / TensorFlow)
      • Tokenisation + padding
      • BiLSTM with custom trained embeddings → sentiment
      • BiLSTM with custom trained embeddings → stars
      • BiLSTM with pre-trained French embeddings → sentiment
        │
        ▼
  [5] Explainability
      • SHAP LinearExplainer on LR model
      • Global feature importance (per sentiment class)
      • Individual review explanations
        │
        ▼
  [6] Information Retrieval
      • BM25 (rank-bm25) keyword ranking
      • Semantic search — sentence-transformers all-MiniLM-L6-v2
      • Hybrid search (BM25 + semantic, normalised average)
        │
        ▼
  [7] RAG Q&A
      • TF-IDF retrieval of top-k relevant reviews
      • Answer generation with google/flan-t5-base
        │
        ▼
  [8] Streamlit dashboard (app.py)
      • Overview, Insurer Deep-dive, Prediction & Explanation,
        Review Search, RAG Q&A
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/kleber0119/insurance-nlp.git
cd NLP_French_Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

The dataset (`avis_traduit_final.xlsx`) is not included in the repository. There are two ways to provide it:

**Option A — Auto-download (recommended)**

The app can download the dataset automatically on first launch from HuggingFace.
Open `app.py` and set the `DATA_URL` constant near the top of the file:

```python
DATA_URL = "https://huggingface.co/datasets/kleber099100/avis_traduit_final.xlsx/resolve/main/avis_traduit_final.xlsx"
```

The app will download and save the file locally on first run. Subsequent runs use the local copy.


**Option B — Manual**

Place `avis_traduit_final.xlsx` directly in the project root folder. The app detects it automatically.

---

## Running the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On the first launch it will:

1. Download the dataset if `DATA_URL` is set and no local file is found
2. Load all classical ML models from `Classical_Models/`
3. Compute and cache sentence embeddings to `corpus_embeddings.npy` (~35 MB, one-time, ~1 min)

---

## Dashboard pages

| Page | Description |
|---|---|
| **Overview** | Corpus-wide KPIs, star distribution, sentiment pie chart, insurer benchmark bar chart |
| **Insurer Deep-dive** | Per-insurer metrics, sentiment breakdown, sample reviews grouped by star rating |
| **Prediction & Explanation** | Live sentiment + star prediction on custom text, with TF-IDF word importance and LR coefficient attribution charts |
| **Review Search** | BM25 / Semantic (MiniLM) / Hybrid search with filters on stars, sentiment, and insurer |
| **RAG Q&A** | TF-IDF retrieval + flan-t5-base answer generation (model loaded on demand) |

---

## Models & techniques

| Component | Library / Model |
|---|---|
| Text vectorisation | scikit-learn `TfidfVectorizer` (unigrams + bigrams) |
| Classical classifiers | `LogisticRegression`, `LinearSVC` |
| Deep learning | Keras BiLSTM with custom + pre-trained French embeddings |
| Explainability | SHAP `LinearExplainer` with `Independent` masker |
| Semantic search | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Keyword search | `rank-bm25` · `BM25Okapi` |
| LLM generation | HuggingFace `transformers` · `google/flan-t5-base` |
| Dashboard | Streamlit |

---
