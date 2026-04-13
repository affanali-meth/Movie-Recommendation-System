"""
🎬 CineMatch — Content-Based Movie Recommendation System
========================================================
A production-ready Streamlit app that delivers personalized movie
recommendations using TF-IDF vectorization and cosine similarity.

Author  : [Mohammed Affan Ali]
GitHub  : https://github.com/affanali-meth
LinkedIn: https://www.linkedin.com/in/mohdaffanali/
"""

import os
import warnings
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")  # suppress sklearn version warnings

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# INJECT GOOGLE FONTS via <link> tag
# NOTE: Using @import inside <style> triggers Streamlit's CSP and
#       causes a blank/black screen. <link> tags work reliably.
# ─────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', Georgia, sans-serif !important;
    }

    /* ── App background ── */
    .stApp {
        background-color: #0d0d12 !important;
        color: #e8e6e1 !important;
    }

    /* ── Container ── */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden !important; }
    [data-testid="stToolbar"]  { visibility: hidden !important; }

    /* ══════════════════════════════════
       HERO
    ══════════════════════════════════ */
    .hero-wrapper {
        text-align: center;
        padding: 3.5rem 1rem 2.5rem;
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #c8a96e;
        border: 1px solid rgba(200,169,110,.35);
        border-radius: 100px;
        padding: 6px 18px;
        margin-bottom: 1.4rem;
        background: rgba(200,169,110,.06);
    }
    .hero-title {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: clamp(3rem, 7vw, 5.5rem);
        line-height: 1.05;
        color: #f0ece4;
        margin: 0 0 0.6rem;
        letter-spacing: -0.02em;
    }
    .hero-title em {
        color: #c8a96e;
        font-style: italic;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #6a6762;
        font-weight: 300;
        max-width: 500px;
        margin: 0 auto 0.5rem;
        line-height: 1.7;
    }
    .hero-divider {
        width: 48px;
        height: 2px;
        background: linear-gradient(90deg, #c8a96e, transparent);
        margin: 1.8rem auto 0;
        border-radius: 2px;
    }

    /* ══════════════════════════════════
       SEARCH SECTION LABEL
    ══════════════════════════════════ */
    .search-label-html {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: #c8a96e;
        margin-bottom: .5rem;
        padding: 0 2px;
    }

    /* ══════════════════════════════════
       SELECTBOX
    ══════════════════════════════════ */
    div[data-baseweb="select"] > div {
        background-color: #17161f !important;
        border: 1.5px solid #2a2838 !important;
        border-radius: 12px !important;
        color: #e8e6e1 !important;
        font-size: .95rem !important;
        transition: border-color .2s !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #c8a96e !important;
        box-shadow: 0 0 0 3px rgba(200,169,110,.12) !important;
    }
    div[data-baseweb="select"] input {
        color: #e8e6e1 !important;
        caret-color: #c8a96e !important;
    }
    div[data-baseweb="select"] svg { fill: #c8a96e !important; }
    [data-baseweb="menu"] { background: #17161f !important; border: 1px solid #2a2838 !important; border-radius: 12px !important; }
    ul[data-baseweb="menu"] li { background: #17161f !important; color: #c8c5be !important; font-size: .9rem !important; }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] [aria-selected="true"] { background: #22212d !important; color: #f0ece4 !important; }

    /* ══════════════════════════════════
       SLIDER  — full gold theme
    ══════════════════════════════════ */
    /* Track background */
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
        background: #22212d !important;
        height: 4px !important;
        border-radius: 4px !important;
    }
    /* Filled portion of track */
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, #c8a96e, #a0773a) !important;
        height: 4px !important;
    }
    /* Thumb */
    [data-testid="stSlider"] div[role="slider"] {
        background: #c8a96e !important;
        border: 3px solid #0d0d12 !important;
        box-shadow: 0 0 0 2px #c8a96e, 0 4px 12px rgba(200,169,110,.4) !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
    }
    /* Tick marks */
    [data-testid="stSlider"] [role="slider"] + div { color: #c8a96e !important; font-weight: 700 !important; }
    /* Min/max labels */
    [data-testid="stSlider"] > div > div > div:last-child > div {
        color: #44423e !important;
        font-size: .72rem !important;
    }
    /* "Results" label above slider */
    [data-testid="stSlider"] label {
        color: #7a7672 !important;
        font-size: .78rem !important;
        font-weight: 500 !important;
        letter-spacing: .04em !important;
    }
    /* Current value tooltip */
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #c8a96e !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
        background: #17161f !important;
        border: 1px solid #2a2838 !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }

    /* ══════════════════════════════════
       PRIMARY BUTTON
    ══════════════════════════════════ */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #c8a96e 0%, #a0773a 100%) !important;
        color: #0d0d12 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: .9rem !important;
        letter-spacing: .08em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 12px !important;
        padding: .8rem 1.5rem !important;
        margin-top: .6rem !important;
        cursor: pointer !important;
        transition: all .2s ease !important;
        box-shadow: 0 4px 20px rgba(200,169,110,.25) !important;
    }
    div.stButton > button:hover {
        opacity: .88 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(200,169,110,.35) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(200,169,110,.2) !important;
    }

    /* ══════════════════════════════════
       SECTION HEADER
    ══════════════════════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: 2.5rem 0 1.5rem;
        padding-bottom: .8rem;
        border-bottom: 1px solid #1a1928;
    }
    .section-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #c8a96e;
        box-shadow: 0 0 8px rgba(200,169,110,.5);
        flex-shrink: 0;
    }
    .section-header h2 {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.5rem;
        color: #f0ece4;
        margin: 0;
    }
    .section-count {
        margin-left: auto;
        font-size: .72rem;
        font-weight: 600;
        color: #44423e;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    /* ══════════════════════════════════
       MOVIE CARD
    ══════════════════════════════════ */
    .movie-card {
        background: #111119;
        border: 1px solid #1c1b26;
        border-radius: 18px;
        padding: 1.5rem 1.4rem 1.3rem;
        margin-bottom: .6rem;
        position: relative;
        overflow: hidden;
        transition: border-color .3s, transform .3s, box-shadow .3s;
        height: 100%;
    }
    .movie-card::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 18px;
        background: radial-gradient(ellipse at top left, rgba(200,169,110,.04) 0%, transparent 60%);
        pointer-events: none;
    }
    .movie-card:hover {
        border-color: rgba(200,169,110,.35);
        transform: translateY(-4px);
        box-shadow: 0 20px 48px rgba(0,0,0,.6), 0 0 0 1px rgba(200,169,110,.1);
    }
    .card-accent {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #c8a96e 0%, rgba(200,169,110,0) 100%);
    }
    .card-number {
        position: absolute;
        top: 1rem; right: 1.2rem;
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 3rem;
        color: rgba(200,169,110,.06);
        line-height: 1;
        font-weight: 700;
        user-select: none;
    }
    .card-icon {
        font-size: 2rem;
        margin-bottom: .7rem;
        display: block;
    }
    .card-rank {
        font-size: .6rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: #c8a96e;
        margin-bottom: .5rem;
    }
    .card-title {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.12rem;
        color: #f0ece4;
        margin-bottom: .5rem;
        line-height: 1.28;
        padding-right: 1.5rem;
    }
    .card-meta {
        display: flex;
        gap: 1rem;
        font-size: .76rem;
        color: #4a4752;
        margin-bottom: .65rem;
        flex-wrap: wrap;
    }
    .genre-pill {
        display: inline-block;
        font-size: .63rem;
        font-weight: 600;
        background: rgba(200,169,110,.08);
        color: #8a7e68;
        border: 1px solid rgba(200,169,110,.15);
        border-radius: 100px;
        padding: 2px 10px;
        margin: 0 4px 5px 0;
        letter-spacing: .03em;
    }
    .card-overview {
        font-size: .8rem;
        color: #5a5754;
        line-height: 1.62;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: .9rem;
    }
    .match-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: .68rem;
        color: #38363a;
        margin-bottom: 5px;
    }
    .match-pct { color: #c8a96e !important; font-weight: 700 !important; }
    .match-bar-bg {
        background: #1c1b26;
        border-radius: 100px;
        height: 3px;
        overflow: hidden;
    }
    .match-bar-fill {
        height: 3px;
        background: linear-gradient(90deg, #c8a96e 0%, #a0773a 100%);
        border-radius: 100px;
    }

    /* ══════════════════════════════════
       SELECTED MOVIE BANNER
    ══════════════════════════════════ */
    .selected-banner {
        background: linear-gradient(135deg, #111119 0%, #14121e 100%);
        border: 1px solid rgba(200,169,110,.25);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,.4), inset 0 1px 0 rgba(200,169,110,.1);
    }
    .selected-banner::before {
        content: '🎬';
        position: absolute;
        right: 1.5rem; top: 1.2rem;
        font-size: 4rem;
        opacity: .07;
    }
    .selected-label {
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .2em;
        text-transform: uppercase;
        color: #c8a96e;
        margin-bottom: .5rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }
    .selected-label::before {
        content: '';
        display: inline-block;
        width: 16px; height: 1px;
        background: #c8a96e;
    }
    .selected-title {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2rem;
        color: #f0ece4;
        margin-bottom: .3rem;
        line-height: 1.15;
    }
    .selected-tagline {
        font-size: .88rem;
        color: #55524e;
        font-style: italic;
        margin-bottom: .7rem;
    }
    .selected-meta {
        display: flex;
        gap: 1.2rem;
        font-size: .8rem;
        color: #4a4752;
        margin-bottom: .8rem;
    }
    .selected-overview {
        font-size: .85rem;
        color: #6a6762;
        line-height: 1.7;
        max-width: 680px;
    }

    /* ══════════════════════════════════
       HOW IT WORKS
    ══════════════════════════════════ */
    .steps-grid {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: .8rem;
    }
    .step-box {
        flex: 1;
        min-width: 180px;
        background: #0d0d12;
        border: 1px solid #1a1928;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        position: relative;
        overflow: hidden;
    }
    .step-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: linear-gradient(180deg, #c8a96e, transparent);
    }
    .step-num   { font-family:'DM Serif Display',Georgia,serif; font-size:1.5rem; color:#c8a96e; margin-bottom:.3rem; line-height:1; }
    .step-title { font-size:.8rem; font-weight:600; color:#c8c5be; margin-bottom:.3rem; }
    .step-desc  { font-size:.75rem; color:#44423e; line-height:1.58; }

    /* ══════════════════════════════════
       STATS ROW
    ══════════════════════════════════ */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1.5rem 0 2rem;
    }
    .stat-item { text-align: center; }
    .stat-num {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.8rem;
        color: #c8a96e;
        line-height: 1;
        margin-bottom: .2rem;
    }
    .stat-label { font-size: .68rem; font-weight: 600; color: #38363a; letter-spacing: .1em; text-transform: uppercase; }

    /* ══════════════════════════════════
       FOOTER
    ══════════════════════════════════ */
    .footer {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        color: #2c2b2e;
        font-size: .78rem;
        border-top: 1px solid #111119;
        margin-top: 4rem;
    }
    .footer-links { display: flex; justify-content: center; gap: 1.5rem; margin: .6rem 0; }
    .footer a { color: #55524e; text-decoration: none; transition: color .2s; }
    .footer a:hover { color: #c8a96e; }

    /* ══════════════════════════════════
       MISC
    ══════════════════════════════════ */
    div[data-testid="stAlert"] {
        background: #111119 !important;
        border-radius: 12px !important;
        border-color: rgba(200,169,110,.2) !important;
    }
    details > summary { color: #55524e !important; font-size: .85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# DATA LOADING  (cached so it runs only once)
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_pkl(filename: str):
    """Load a pickle file from the same directory as app.py."""
    with open(os.path.join(BASE_DIR, filename), "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all ML artifacts and enrich movies_df with metadata columns."""
    movies_df    = _load_pkl("movies_df.pkl")
    indices      = _load_pkl("indices.pkl")
    tfidf_matrix = _load_pkl("tfidf_matrix.pkl")

    # Merge richer info from movies_metadata.csv
    meta = pd.read_csv(os.path.join(BASE_DIR, "movies_metadata.csv"), low_memory=False)
    keep = ["title", "poster_path", "release_date", "vote_count", "imdb_id"]
    meta_slim = meta[keep].drop_duplicates(subset="title")
    merged = movies_df.merge(meta_slim, on="title", how="left")

    return merged, indices, tfidf_matrix


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def get_recommendations(title: str, movies_df, indices, tfidf_matrix, top_n: int = 10):
    """
    Return the top-N most content-similar movies using cosine similarity.

    Key robustness fixes:
    - Duplicate titles: indices may return a Series; we take the first value.
    - Index alignment: similarity is computed only against the first
      len(movies_df) rows of tfidf_matrix to prevent out-of-bounds errors.
    """
    if title not in indices:
        return pd.DataFrame()

    # Safely resolve to a scalar row index
    idx_raw = indices[title]
    idx = int(idx_raw.iloc[0]) if hasattr(idx_raw, "iloc") else int(idx_raw)

    n = len(movies_df)
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix[:n]).flatten()

    sim_series = pd.Series(sims)
    sim_series.iloc[idx] = -1          # exclude the query movie itself

    top_idx = list(sim_series.nlargest(top_n).index)
    recs = movies_df.iloc[top_idx].copy()
    recs["similarity"] = sim_series.iloc[top_idx].values
    return recs.reset_index(drop=True)


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

CARD_ICONS = ["🎬", "🎥", "🍿", "🎞️", "⭐", "🎭", "🌟", "🎦", "🏆", "🎪"]


def _year(val) -> str:
    try:
        return str(val)[:4] if pd.notna(val) else "—"
    except Exception:
        return "—"


def _rating(val) -> str:
    try:
        v = float(val)
        return f"⭐ {v:.1f}" if v > 0 else "—"
    except Exception:
        return "—"


def _genre_pills(genres_str) -> str:
    """Return up to 3 inline genre pill HTML elements."""
    if not isinstance(genres_str, str) or not genres_str.strip():
        return ""
    return "".join(
        f'<span class="genre-pill">{g.strip()}</span>'
        for g in genres_str.split()[:3]
    )


def render_movie_card(row, rank: int):
    """Render a single recommendation card."""
    icon     = CARD_ICONS[rank % len(CARD_ICONS)]
    title    = row.get("title",        "Unknown")
    genres   = _genre_pills(row.get("genres", ""))
    rating   = _rating(row.get("vote_average", 0))
    year     = _year(row.get("release_date", ""))
    overview = str(row.get("overview", "")).strip() or "No overview available."
    pct      = int(row.get("similarity", 0) * 100)

    st.markdown(
        f"""
        <div class="movie-card">
          <div class="card-accent"></div>
          <div class="card-number">{rank + 1:02d}</div>
          <span class="card-icon">{icon}</span>
          <div class="card-rank">Recommendation #{rank + 1}</div>
          <div class="card-title">{title}</div>
          <div style="margin-bottom:.55rem;">{genres}</div>
          <div class="card-meta">
            <span>🗓 {year}</span>
            <span>{rating}</span>
          </div>
          <div class="card-overview">{overview}</div>
          <div class="match-row">
            <span>Match score</span>
            <span class="match-pct">{pct}%</span>
          </div>
          <div class="match-bar-bg">
            <div class="match-bar-fill" style="width:{pct}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_banner(row):
    """Render the highlighted banner for the chosen movie."""
    title    = row.get("title",        "Unknown")
    genres   = _genre_pills(row.get("genres", ""))
    rating   = _rating(row.get("vote_average", 0))
    year     = _year(row.get("release_date", ""))
    tagline  = str(row.get("tagline", "")).strip()
    overview = str(row.get("overview", "")).strip() or "No overview available."

    tagline_html = (
        f'<div class="selected-tagline">&ldquo;{tagline}&rdquo;</div>'
        if tagline else ""
    )
    st.markdown(
        f"""
        <div class="selected-banner">
          <div class="selected-label">Now Recommending Based On</div>
          <div class="selected-title">{title}</div>
          {tagline_html}
          <div style="margin-bottom:.65rem;">{genres}</div>
          <div class="selected-meta">
            <span>🗓 {year}</span>
            <span>{rating}</span>
          </div>
          <div class="selected-overview">{overview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    # ── Load data ───────────────────────────────
    with st.spinner("🎬 Loading model & data…"):
        try:
            movies_df, indices, tfidf_matrix = load_artifacts()
        except Exception as e:
            st.error(f"❌ Failed to load model files. Make sure all .pkl files are in the same folder.\n\nError: {e}")
            st.stop()

    # Sort titles: letters first (A-Z), then numbers, then symbols
    def _sort_key(t):
        t = str(t).strip()
        if t and t[0].isalpha():
            return (0, t.lower())   # letters first
        elif t and t[0].isdigit():
            return (1, t.lower())   # digits second
        else:
            return (2, t.lower())   # symbols last

    movie_titles = sorted(movies_df["title"].dropna().unique().tolist(), key=_sort_key)

    # ── Hero ────────────────────────────────────
    n_movies = len(movies_df)
    st.markdown(
        f"""
        <div class="hero-wrapper">
          <div class="hero-tag">✦ AI-Powered Recommendations</div>
          <h1 class="hero-title">Cine<em>Match</em></h1>
          <p class="hero-sub">
            Discover films you'll love — powered by TF-IDF embeddings
            &amp; cosine similarity. No bubble, no bias.
          </p>
          <div class="hero-divider"></div>
        </div>
        <div class="stats-row">
          <div class="stat-item">
            <div class="stat-num">{n_movies:,}</div>
            <div class="stat-label">Movies Indexed</div>
          </div>
          <div class="stat-item">
            <div class="stat-num">50K</div>
            <div class="stat-label">TF-IDF Features</div>
          </div>
          <div class="stat-item">
            <div class="stat-num">Cosine</div>
            <div class="stat-label">Similarity Engine</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Search card ─────────────────────────────
    # Streamlit does NOT render native widgets inside raw HTML divs.
    # We use centered columns + a styled header div above the widgets.
    _, card_col, _ = st.columns([1, 4, 1])
    with card_col:
        st.markdown(
            """
            <div style="
                background:#13131a;
                border:1px solid #22212d;
                border-radius:20px 20px 0 0;
                padding:1.5rem 1.8rem 0.8rem;
                box-shadow:0 -4px 30px rgba(0,0,0,.3);
            ">
              <p style="
                font-size:.72rem;font-weight:600;letter-spacing:.14em;
                text-transform:uppercase;color:#c8a96e;margin:0;
              ">🎬 Select a movie you enjoyed</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Wrap widgets in a styled bottom-half of card
        st.markdown(
            """<div style="
                background:#13131a;
                border:1px solid #22212d;
                border-top:none;
                border-radius:0 0 20px 20px;
                padding:0.6rem 1.8rem 1.6rem;
                box-shadow:0 20px 50px rgba(0,0,0,.45);
                margin-bottom:1.5rem;
            ">""",
            unsafe_allow_html=True,
        )
        selected = st.selectbox(
            label="Movie",
            options=[""] + movie_titles,
            index=0,
            label_visibility="collapsed",
            placeholder="Type to search a movie…",
        )
        col_btn, col_num = st.columns([3, 1])
        with col_num:
            top_n = st.slider("Results", min_value=5, max_value=20, value=10, step=1)
        with col_btn:
            go = st.button("✦  Get Recommendations", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── How it works ────────────────────────────
    with st.expander("⚙️  How does CineMatch work?"):
        st.markdown(
            """
            <div class="steps-grid">
              <div class="step-box">
                <div class="step-num">01</div>
                <div class="step-title">Tag Extraction</div>
                <div class="step-desc">
                  Each movie's overview, genres, tagline, and keywords
                  are combined into a rich "tags" string.
                </div>
              </div>
              <div class="step-box">
                <div class="step-num">02</div>
                <div class="step-title">TF-IDF Embedding</div>
                <div class="step-desc">
                  A TF-IDF vectorizer converts tags into high-dimensional
                  sparse vectors, capturing thematic importance.
                </div>
              </div>
              <div class="step-box">
                <div class="step-num">03</div>
                <div class="step-title">Cosine Similarity</div>
                <div class="step-desc">
                  We measure the angle between vectors — smallest angle
                  means most similar content.
                </div>
              </div>
            </div>
            <p style="font-size:.78rem;color:#44423e;margin-top:.9rem;margin-bottom:0;">
              Purely content-driven — no user data, no collaborative filtering.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # ── Recommendations ─────────────────────────
    if go:
        if not selected:
            st.info("⚡ Please select a movie to get started.")
        else:
            with st.spinner(f"Finding films similar to **{selected}**…"):
                recs = get_recommendations(
                    selected, movies_df, indices, tfidf_matrix, top_n=top_n
                )

            if recs.empty:
                st.warning("😕 Couldn't find recommendations for that title. Try another!")
            else:
                # Show the selected movie info
                query_rows = movies_df[movies_df["title"] == selected]
                if not query_rows.empty:
                    render_selected_banner(query_rows.iloc[0])

                # Section title
                st.markdown(
                    f"""
                    <div class="section-header">
                      <div class="section-dot"></div>
                      <h2>Top {len(recs)} Recommendations</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 2-column grid
                for i in range(0, len(recs), 2):
                    c1, c2 = st.columns(2, gap="medium")
                    with c1:
                        render_movie_card(recs.iloc[i], rank=i)
                    if i + 1 < len(recs):
                        with c2:
                            render_movie_card(recs.iloc[i + 1], rank=i + 1)
                    st.write("")

    # ── Footer ──────────────────────────────────
    st.markdown(
        """
        <div class="footer">
          Built with ❤️ using Streamlit · scikit-learn · Python
          &nbsp;·&nbsp;
          <a href="https://github.com/yourusername" target="_blank">⭐ GitHub</a>
          &nbsp;·&nbsp;
          <a href="https://linkedin.com/in/yourprofile" target="_blank">💼 LinkedIn</a>
          <br><br>
          © 2025 · CineMatch · Content-Based Movie Recommendation System
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()