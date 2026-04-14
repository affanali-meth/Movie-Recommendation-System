"""
🎬 CineMatch — Content-Based Movie Recommendation System

A production-ready Streamlit app that delivers personalized movie
recommendations using TF-IDF vectorization and cosine similarity.

Author  : Mohammed Affan Ali
GitHub  : https://github.com/affanali-meth
LinkedIn: https://www.linkedin.com/in/mohdaffanali/
"""

import os
import warnings
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── FONTS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e8e0d0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(180,130,60,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(100,60,180,0.08) 0%, transparent 60%),
        #080c14 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

[data-testid="block-container"] {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #c8922a;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700;
    line-height: 1.05;
    color: #f0e8d8;
    letter-spacing: -0.02em;
}
.hero-title em {
    font-style: italic;
    color: #c8922a;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: rgba(220,210,195,0.60);
    margin-top: 1rem;
    letter-spacing: 0.02em;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c8922a, transparent);
    margin: 1.8rem auto 0;
}

/* ── Controls Bar ── */
.controls-wrapper {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(200,146,42,0.18);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 2rem 0;
    backdrop-filter: blur(10px);
}
.controls-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(200,146,42,0.7);
    margin-bottom: 0.5rem;
}

/* Streamlit select box */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(200,146,42,0.30) !important;
    border-radius: 10px !important;
    color: #f0e8d8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #c8922a !important;
    box-shadow: 0 0 0 2px rgba(200,146,42,0.15) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: #c8922a !important;
}
[data-testid="stSlider"] .rc-slider-track { background: #c8922a !important; }
[data-testid="stSlider"] .rc-slider-handle {
    border-color: #c8922a !important;
    background: #c8922a !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c8922a 0%, #e0a83a 100%) !important;
    color: #080c14 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.72rem 2.2rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(200,146,42,0.3) !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(200,146,42,0.45) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Selected Movie Banner ── */
.selected-banner {
    background: linear-gradient(135deg,
        rgba(200,146,42,0.12) 0%,
        rgba(200,146,42,0.04) 100%);
    border: 1px solid rgba(200,146,42,0.35);
    border-left: 4px solid #c8922a;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin: 2rem 0 2.5rem;
    position: relative;
    overflow: hidden;
}
.selected-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(200,146,42,0.10), transparent 70%);
    pointer-events: none;
}
.selected-tag {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c8922a;
    margin-bottom: 0.6rem;
}
.selected-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0e8d8;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}
.selected-meta {
    font-size: 0.82rem;
    color: rgba(200,146,42,0.85);
    margin: 0.5rem 0 0.9rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.selected-overview {
    font-size: 0.9rem;
    line-height: 1.7;
    color: rgba(220,210,195,0.70);
    max-width: 800px;
}

/* ── Genre Pills ── */
.genre-pill {
    display: inline-block;
    background: rgba(200,146,42,0.15);
    border: 1px solid rgba(200,146,42,0.30);
    color: #c8922a;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    margin: 0.1rem 0.2rem 0.1rem 0;
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0 0 1.5rem;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(200,146,42,0.3), transparent);
}
.section-header-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0e8d8;
    white-space: nowrap;
}
.section-header-count {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: rgba(200,146,42,0.6);
    text-transform: uppercase;
}

/* ── Movie Card ── */
.movie-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(200,146,42,0.12);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: default;
}
.movie-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(200,146,42,0.04) 0%,
        transparent 60%);
    pointer-events: none;
    transition: opacity 0.3s;
    opacity: 0;
}
.movie-card:hover {
    border-color: rgba(200,146,42,0.35);
    background: rgba(255,255,255,0.045);
    transform: translateY(-2px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(200,146,42,0.15);
}
.movie-card:hover::before { opacity: 1; }

.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.8rem;
}
.card-rank-badge {
    background: rgba(200,146,42,0.15);
    border: 1px solid rgba(200,146,42,0.25);
    color: #c8922a;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.card-match {
    font-size: 0.78rem;
    font-weight: 700;
    color: #c8922a;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0e8d8;
    line-height: 1.3;
    margin-bottom: 0.5rem;
}
.card-meta {
    font-size: 0.78rem;
    color: rgba(200,146,42,0.75);
    margin: 0.55rem 0 0.8rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.card-overview {
    font-size: 0.83rem;
    line-height: 1.65;
    color: rgba(200,200,185,0.60);
    margin-bottom: 1rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.match-bar-bg {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.match-bar-fill {
    height: 3px;
    background: linear-gradient(90deg, #c8922a, #e0c060);
    border-radius: 2px;
    transition: width 0.6s ease;
}
.match-label-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.match-label-text {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(200,146,42,0.55);
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 1.5rem;
    margin: 0 0 2.5rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(200,146,42,0.15);
    border-radius: 10px;
    padding: 0.7rem 1.3rem;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #c8922a;
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(200,200,180,0.45);
}

/* ── Spinner override ── */
[data-testid="stSpinner"] > div {
    border-top-color: #c8922a !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(200,146,42,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200,146,42,0.5); }

/* ── Stagger animation ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.animate-in {
    animation: fadeSlideUp 0.45s ease both;
}
.delay-1 { animation-delay: 0.05s; }
.delay-2 { animation-delay: 0.10s; }
.delay-3 { animation-delay: 0.15s; }
.delay-4 { animation-delay: 0.20s; }
.delay-5 { animation-delay: 0.25s; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: rgba(200,190,175,0.35);
}
.empty-state-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
}
.empty-state-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-style: italic;
}

/* ── Footer ── */
.cinefooter {
    text-align: center;
    padding: 3rem 0 1rem;
    font-size: 0.75rem;
    color: rgba(200,190,175,0.25);
    letter-spacing: 0.05em;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 4rem;
}
.cinefooter a {
    color: rgba(200,146,42,0.5);
    text-decoration: none;
}
.cinefooter a:hover { color: #c8922a; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_pkl(filename: str):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    movies_df   = _load_pkl("movies_df.pkl")
    indices     = _load_pkl("indices.pkl")
    tfidf_matrix = _load_pkl("tfidf_matrix.pkl")

    meta = pd.read_csv(
        os.path.join(BASE_DIR, "movies_metadata.csv"), low_memory=False
    )
    keep = ["title", "poster_path", "release_date", "vote_count", "vote_average", "imdb_id"]
    # Coerce vote_average in metadata to numeric
    meta["vote_average"] = pd.to_numeric(meta["vote_average"], errors="coerce")
    meta_slim = meta[keep].drop_duplicates(subset="title")

    # Merge — keep movies_df vote_average if available, fallback to meta
    merged = movies_df.merge(meta_slim, on="title", how="left", suffixes=("", "_meta"))

    # Resolve vote_average: prefer movies_df column if valid, else meta column
    if "vote_average_meta" in merged.columns:
        merged["vote_average"] = merged["vote_average"].fillna(merged["vote_average_meta"])
        merged.drop(columns=["vote_average_meta"], inplace=True)

    return merged, indices, tfidf_matrix


def get_recommendations(
    title: str,
    movies_df: pd.DataFrame,
    indices: pd.Series,
    tfidf_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    if title not in indices.index:
        return pd.DataFrame()

    idx_raw = indices[title]
    # indices may map to a scalar or a Series (duplicate titles)
    idx = int(idx_raw.iloc[0]) if hasattr(idx_raw, "iloc") else int(idx_raw)

    n = len(movies_df)
    if idx >= n:
        return pd.DataFrame()

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix[:n]).flatten()
    sim_series = pd.Series(sims, dtype=float)
    sim_series.iloc[idx] = -1.0          # exclude the query movie itself

    top_idx = list(sim_series.nlargest(top_n).index)
    recs = movies_df.iloc[top_idx].copy()
    recs["similarity"] = sim_series.iloc[top_idx].values
    return recs.reset_index(drop=True)


# ── UI HELPERS ─────────────────────────────────────────────────────────────────

def _year(val) -> str:
    try:
        return str(val)[:4] if pd.notna(val) and str(val).strip() else "—"
    except Exception:
        return "—"


def _rating(val) -> str:
    try:
        v = float(val)
        return f"⭐ {v:.1f}" if v > 0 else "—"
    except Exception:
        return "—"


def _genre_pills(genres_str) -> str:
    if not isinstance(genres_str, str) or not genres_str.strip():
        return ""
    genres = [g.strip() for g in genres_str.split() if g.strip()][:4]
    return "".join(f'<span class="genre-pill">{g}</span>' for g in genres)


def render_selected_banner(row: pd.Series) -> None:
    title    = row.get("title", "Unknown")
    genres   = _genre_pills(row.get("genres", ""))
    rating   = _rating(row.get("vote_average", 0))
    year     = _year(row.get("release_date", ""))
    overview = str(row.get("overview", "")).strip() or "No overview available."

    st.markdown(f"""
    <div class="selected-banner animate-in">
        <div class="selected-tag">✦ Now Exploring</div>
        <div class="selected-title">{title}</div>
        <div style="margin-bottom:.4rem;">{genres}</div>
        <div class="selected-meta">
            <span>🗓 {year}</span>
            <span>{rating}</span>
        </div>
        <div class="selected-overview">{overview}</div>
    </div>
    """, unsafe_allow_html=True)


def render_movie_card(row: pd.Series, rank: int) -> None:
    title    = row.get("title", "Unknown")
    genres   = _genre_pills(row.get("genres", ""))
    rating   = _rating(row.get("vote_average", 0))
    year     = _year(row.get("release_date", ""))
    overview = str(row.get("overview", "")).strip() or "No overview available."
    pct      = int(round(row.get("similarity", 0) * 100))
    delay    = (rank % 5) + 1

    st.markdown(f"""
    <div class="movie-card animate-in delay-{delay}">
        <div class="card-top">
            <span class="card-rank-badge">#{rank + 1}</span>
            <span class="card-match">{pct}% match</span>
        </div>
        <div class="card-title">{title}</div>
        <div style="margin-bottom:.2rem;">{genres}</div>
        <div class="card-meta">
            <span>🗓 {year}</span>
            <span>{rating}</span>
        </div>
        <div class="card-overview">{overview}</div>
        <div class="match-label-row">
            <span class="match-label-text">Similarity</span>
            <span class="match-label-text">{pct}%</span>
        </div>
        <div class="match-bar-bg">
            <div class="match-bar-fill" style="width:{pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Header
    st.markdown("""
    <div class="hero animate-in">
        <div class="hero-eyebrow">✦ AI-Powered Discovery ✦</div>
        <div class="hero-title">Cine<em>Match</em></div>
        <div class="hero-sub">Find films that speak your language — powered by content similarity</div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Curating the collection…"):
        movies_df, indices, tfidf_matrix = load_artifacts()

    total_movies = movies_df["title"].dropna().nunique()

    # Stats row
    st.markdown(f"""
    <div class="stats-row animate-in delay-2">
        <div class="stat-pill">
            <span class="stat-value">{total_movies:,}</span>
            <span class="stat-label">Films Indexed</span>
        </div>
        <div class="stat-pill">
            <span class="stat-value">TF-IDF</span>
            <span class="stat-label">Algorithm</span>
        </div>
        <div class="stat-pill">
            <span class="stat-value">Cosine</span>
            <span class="stat-label">Similarity</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    st.markdown('<div class="controls-wrapper animate-in delay-3">', unsafe_allow_html=True)

    col_sel, col_n, col_btn = st.columns([5, 2, 1.5])

    movie_titles = sorted(movies_df["title"].dropna().unique().tolist())

    with col_sel:
        st.markdown('<div class="controls-label">Choose a Movie</div>', unsafe_allow_html=True)
        selected = st.selectbox(
            label="movie_selector",
            options=[""] + movie_titles,
            index=0,
            label_visibility="collapsed",
            placeholder="Search for a movie…",
        )

    with col_n:
        st.markdown('<div class="controls-label">Results to Show</div>', unsafe_allow_html=True)
        top_n = st.slider(
            label="top_n_slider",
            min_value=5,
            max_value=20,
            value=10,
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown('<div class="controls-label">&nbsp;</div>', unsafe_allow_html=True)
        go = st.button("Discover →", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Results
    if go and selected:
        with st.spinner("Finding your perfect matches…"):
            recs = get_recommendations(selected, movies_df, indices, tfidf_matrix, top_n)

        if recs.empty:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">🎭</span>
                <div class="empty-state-text">No recommendations found for this title.</div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Selected banner
        source_row = movies_df[movies_df["title"] == selected]
        if not source_row.empty:
            render_selected_banner(source_row.iloc[0])

        # Section header
        st.markdown(f"""
        <div class="section-header animate-in">
            <span class="section-header-text">Recommended for You</span>
            <span class="section-header-count">{len(recs)} films</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        # Cards in 2-column grid
        for i in range(0, len(recs), 2):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                render_movie_card(recs.iloc[i], i)
            if i + 1 < len(recs):
                with c2:
                    render_movie_card(recs.iloc[i + 1], i + 1)

    elif not selected and go:
        st.warning("Please select a movie first.")

    else:
        # Idle state
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">🎬</span>
            <div class="empty-state-text">Select a film above and click <strong>Discover →</strong> to begin</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="cinefooter">
        CineMatch · Built with Streamlit &amp; Scikit-learn ·
        <a href="https://github.com/affanali-meth" target="_blank">GitHub</a> ·
        <a href="https://www.linkedin.com/in/mohdaffanali/" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
