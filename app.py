import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from functools import lru_cache

DATA_DIR               = Path("data")
MOVIE_FILE             = DATA_DIR / "movies_enhanced.pkl"
SIMILARITY_FILE        = DATA_DIR / "similarity_matrix.npy"
USER_MATRIX_FILE       = DATA_DIR / "user_movie_matrix.pkl"
PREDICTED_RATINGS_FILE = DATA_DIR / "predicted_ratings.pkl"
RATINGS_FILE           = "ratings.csv"
TMDB_API_KEY           = "b4af919b50b76eee5974d1c06e5dbbde"
TMDB_IMG_BASE          = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_IMG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='342' height='513' viewBox='0 0 342 513'%3E"
    "%3Crect width='342' height='513' fill='%230F1420'/%3E"
    "%3Ctext x='171' y='260' font-family='sans-serif' font-size='18' "
    "fill='%23E8A838' text-anchor='middle'%3ENo Poster%3C/text%3E"
    "%3C/svg%3E"
)

CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,600&family=Raleway:wght@300;400;500;600&display=swap');

:root {
    --bg:     #080B14;
    --card:   #0F1420;
    --gold:   #E8A838;
    --gold2:  #A87820;
    --text:   #E8E4DC;
    --muted:  #6B6760;
    --border: rgba(232,168,56,0.18);
    --radius: 10px;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: Raleway, sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0A0D18 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden !important; }

.stButton > button {
    background: linear-gradient(135deg, #E8A838, #C07820) !important;
    color: #080B14 !important; border: none !important;
    font-family: Raleway, sans-serif !important; font-weight: 700 !important;
    font-size: 0.82rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important; border-radius: 4px !important;
    transition: all 0.2s ease !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #F0B848, #D08830) !important;
    box-shadow: 0 6px 20px rgba(232,168,56,0.3) !important;
    transform: translateY(-1px) !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput input {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: var(--radius) !important;
}
hr { border-color: var(--border) !important; }

/* ── Movie card CSS classes (avoids inline single-quote font-family bug) ── */
.cin-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.cin-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
}
.cin-poster-wrap {
    position: relative; overflow: hidden;
    background: #0F1420;   /* dark fill while poster loads / if it fails */
    aspect-ratio: 2 / 3;   /* reserve space on the wrapper, not just the img */
}
.cin-poster {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
}
.cin-year {
    position: absolute; top: 8px; right: 8px;
    background: rgba(8,11,20,0.82); color: var(--gold);
    font-size: 0.66rem; font-weight: 600;
    padding: 3px 7px; border-radius: 4px;
    font-family: Raleway, sans-serif;
}
.cin-body { padding: 12px 13px 14px; }
.cin-title {
    font-family: Playfair Display, Georgia, serif;
    color: var(--text); font-size: 0.95rem;
    margin: 0 0 7px; line-height: 1.3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.cin-bar-wrap { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
.cin-bar-bg { flex: 1; height: 4px; background: #1E2436; border-radius: 2px; overflow: hidden; }
.cin-bar-fill { height: 100%; border-radius: 2px; }
.cin-bar-val { font-size: 0.76rem; font-weight: 600; min-width: 28px; }
.cin-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.cin-pill {
    background: #141A2A; color: var(--gold);
    border: 1px solid rgba(232,168,56,0.28);
    padding: 2px 8px; border-radius: 20px;
    font-size: 0.68rem; font-family: Raleway, sans-serif; font-weight: 500;
}
.cin-overview {
    color: var(--muted); font-size: 0.73rem;
    line-height: 1.55; margin: 0;
    font-family: Raleway, sans-serif;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}
.cin-gap { height: 16px; }
</style>
<script>
window._cinePlaceholder = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='342' height='513' viewBox='0 0 342 513'%3E%3Crect width='342' height='513' fill='%230F1420'/%3E%3Ctext x='171' y='260' font-family='sans-serif' font-size='18' fill='%23E8A838' text-anchor='middle'%3ENo Poster%3C/text%3E%3C/svg%3E";
</script>
"""

def build_card(movie, poster_url):
    title    = str(movie.get("title", "Untitled")).replace('"', "&quot;").replace("'", "&#39;")
    overview = str(movie.get("overview", ""))[:200].strip()
    rating   = float(movie.get("vote_average") or 0)
    year     = str(movie.get("year") or "")
    genres   = movie.get("genres", []) if isinstance(movie.get("genres"), list) else []

    pct   = min(int(rating / 10 * 100), 100)
    color = "#E8A838" if rating >= 7 else ("#C07820" if rating >= 5 else "#C0392B")
    pills = "".join(f'<span class="cin-pill">{g}</span>' for g in genres[:4])

    return (
        '<div class="cin-card">'
        '<div class="cin-poster-wrap">'
        f'<img class="cin-poster" src="{poster_url}" alt="{title}" '
        'onerror="this.onerror=null;this.src=window._cinePlaceholder||this.src;">'
        f'<span class="cin-year">{year}</span>'
        '</div>'
        '<div class="cin-body">'
        f'<div class="cin-title" title="{title}">{title}</div>'
        '<div class="cin-bar-wrap">'
        '<div class="cin-bar-bg">'
        f'<div class="cin-bar-fill" style="width:{pct}%;background:{color};"></div>'
        '</div>'
        f'<span class="cin-bar-val" style="color:{color};">{rating:.1f}</span>'
        '</div>'
        f'<div class="cin-pills">{pills}</div>'
        f'<p class="cin-overview">{overview}</p>'
        '</div>'
        '</div>'
    )

def hero_header():
    st.markdown(
        '<div style="padding:2.5rem 0 1.5rem;text-align:center;'
        'border-bottom:1px solid rgba(232,168,56,0.2);">'
        '<div style="font-size:0.7rem;letter-spacing:0.35em;color:#E8A838;'
        'text-transform:uppercase;margin-bottom:0.6rem;font-family:Raleway,sans-serif;font-weight:600;">'
        '&#10022; &nbsp; Discover &middot; Explore &middot; Watch &nbsp; &#10022;'
        '</div>'
        '<h1 style="font-family:Playfair Display,Georgia,serif;font-size:3rem;font-weight:800;'
        'color:#E8E4DC;margin:0;line-height:1.1;">'
        'Cin<span style="color:#E8A838;font-style:italic;">e</span>Match'
        '</h1>'
        '<p style="color:#4A4740;font-size:0.84rem;margin-top:0.7rem;'
        'font-family:Raleway,sans-serif;letter-spacing:0.06em;">'
        'Intelligent film recommendations &nbsp;&middot;&nbsp; TF-IDF &amp; Matrix Factorisation'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def section_heading(text, sub=""):
    sub_html = (
        f'<p style="color:#6B6760;font-size:0.8rem;margin:0.3rem 0 0;font-family:Raleway,sans-serif;">{sub}</p>'
        if sub else ""
    )
    st.markdown(
        f'<div style="margin:2rem 0 1.2rem;">'
        f'<h2 style="font-family:Playfair Display,Georgia,serif;color:#E8E4DC;font-size:1.7rem;margin:0;">{text}</h2>'
        f'{sub_html}'
        f'<div style="width:44px;height:2px;background:#E8A838;margin-top:0.5rem;border-radius:2px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        with open(MOVIE_FILE, "rb") as f:
            movies = pickle.load(f)
        missing = {"id", "title", "overview", "genres"} - set(movies.columns)
        if missing:
            st.error(f"Missing columns in processed data: {missing}")
            return None, None, None, None
        similarity = np.load(SIMILARITY_FILE)
        user_matrix = pred_ratings = None
        if USER_MATRIX_FILE.exists():
            with open(USER_MATRIX_FILE, "rb") as f:
                user_matrix = pickle.load(f)
        if PREDICTED_RATINGS_FILE.exists():
            with open(PREDICTED_RATINGS_FILE, "rb") as f:
                pred_ratings = pickle.load(f)
        return movies, similarity, user_matrix, pred_ratings
    except FileNotFoundError:
        st.error("Data files not found — please run Preprocessing.py first.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Load error: {e}")
        return None, None, None, None

@lru_cache(maxsize=2000)
def fetch_poster(tmdb_id):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY}, timeout=4,
        )
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return f"{TMDB_IMG_BASE}{path}"
    except Exception:
        pass
    return PLACEHOLDER_IMG


def poster_from_row(movie):
    local = movie.get("poster_path")
    if local and str(local).startswith("/"):
        return f"https://image.tmdb.org/t/p/w342{local}"
    try:
        return fetch_poster(int(movie.get("id", 0)))
    except Exception:
        return PLACEHOLDER_IMG

_TITLE_STOP = {
    "the","a","an","of","in","on","at","to","and","or","for",
    "with","de","la","le","das","die","der","el","los","les",
}

def _title_overlap(query_title, candidate_titles):
    """
    Returns a float array [0..1] — fraction of query title words found in each
    candidate title.  Short common words (≤2 chars) and stopwords are ignored.
    Example: query="One Piece Film Red", candidate="One Piece: Stampede" → 0.5
    (both "one" and "piece" match out of 4 query words)
    """
    q_words = [
        w for w in query_title.lower().split()
        if len(w) >= 3 and w not in _TITLE_STOP
    ]
    if not q_words:
        return np.zeros(len(candidate_titles))

    scores = []
    for t in candidate_titles:
        t_words = set(
            w for w in str(t).lower().split()
            if len(w) >= 3 and w not in _TITLE_STOP
        )
        match = sum(1 for w in q_words if w in t_words)
        scores.append(match / len(q_words))
    return np.array(scores)


def _title_boost(query_title, sim_scores, candidate_titles, quality=None):
    """
    Final ranking score:
      60% cosine similarity  — content relevance
      35% title overlap      — franchise / sequel grouping
       5% quality score      — tiebreak on well-rated films

    The title overlap weight is intentionally high: a franchise film with
    slightly lower content similarity should still outrank an unrelated film.
    """
    overlap = _title_overlap(query_title, candidate_titles)
    q = quality if quality is not None else np.zeros(len(sim_scores))
    return 0.60 * np.array(sim_scores) + 0.35 * overlap + 0.05 * np.array(q)


def recommend_content(title, movies, similarity, top_n=12, genre_filter=None):
    mask = movies["title"].str.lower() == title.lower()
    idx  = movies[mask].index
    if len(idx) == 0:
        mask = movies["title"].str.lower().str.contains(title.lower(), na=False)
        idx  = movies[mask].index
    if len(idx) == 0:
        return pd.DataFrame()

    sim_scores = sorted(enumerate(similarity[idx[0]]), key=lambda x: x[1], reverse=True)[1:top_n * 6 + 1]
    recs = movies.iloc[[i for i, _ in sim_scores]].copy()
    recs["_sim"] = [s for _, s in sim_scores]

    if genre_filter:
        recs = recs[recs["genres"].apply(
            lambda g: any(genre_filter.lower() in gg.lower() for gg in g)
            if isinstance(g, list) else False
        )]

    recs["_score"] = _title_boost(title, recs["_sim"].values, recs["title"].tolist())
    return recs.nlargest(top_n, "_score")


def recommend_hybrid(title, movies, similarity, top_n=12, genre_filter=None):
    mask = movies["title"].str.lower() == title.lower()
    idx  = movies[mask].index
    if len(idx) == 0:
        mask = movies["title"].str.lower().str.contains(title.lower(), na=False)
        idx  = movies[mask].index
    if len(idx) == 0:
        return pd.DataFrame()

    sim_scores = sorted(enumerate(similarity[idx[0]]), key=lambda x: x[1], reverse=True)[1:top_n * 6 + 1]
    pool = movies.iloc[[i for i, _ in sim_scores]].copy()
    pool["_sim"] = [s for _, s in sim_scores]

    if genre_filter:
        pool = pool[pool["genres"].apply(
            lambda g: any(genre_filter.lower() in gg.lower() for gg in g)
            if isinstance(g, list) else False
        )]

    q_max = max(float(movies["quality_score"].max()), 1e-6) if "quality_score" in movies.columns else 1
    quality_norm = pool.get("quality_score", pd.Series(0, index=pool.index)).values / q_max

    pool["_score"] = _title_boost(title, pool["_sim"].values, pool["title"].tolist(), quality=quality_norm)
    return pool.nlargest(top_n, "_score")


def recommend_collaborative(user_id, user_matrix, pred_ratings, movies, top_n=12, genre_filter=None):
    if user_matrix is None or user_id not in user_matrix.index:
        return pd.DataFrame()
    user_idx = user_matrix.index.get_loc(user_id)
    scores   = pred_ratings[user_idx]
    unrated  = user_matrix.columns[user_matrix.loc[user_id] == 0]
    top      = sorted(
        zip(unrated, scores[user_matrix.columns.get_indexer(unrated)]),
        key=lambda x: x[1], reverse=True
    )[:top_n * 3]
    recs = movies[movies["id"].isin([r[0] for r in top])].copy()
    recs["_pred"] = recs["id"].map(dict(top)).fillna(0)
    if genre_filter:
        recs = recs[recs["genres"].apply(
            lambda g: any(genre_filter.lower() in gg.lower() for gg in g)
            if isinstance(g, list) else False
        )]
    return recs.nlargest(top_n, "_pred")


def render_grid(recs, n_cols=3):
    items = list(recs.iterrows())
    for row_start in range(0, len(items), n_cols):
        row_items = items[row_start:row_start + n_cols]
        cols = st.columns(n_cols, gap="medium")
        for col, (_, movie) in zip(cols, row_items):
            with col:
                st.markdown(build_card(movie, poster_from_row(movie)), unsafe_allow_html=True)
        for empty_col in cols[len(row_items):]:
            with empty_col:
                st.empty()
        st.markdown('<div class="cin-gap"></div>', unsafe_allow_html=True)

def build_sidebar(movies):
    with st.sidebar:
        st.markdown(
            '<div style="padding:1rem 0 0.8rem;border-bottom:1px solid rgba(232,168,56,0.2);margin-bottom:1rem;">'
            '<p style="font-family:Playfair Display,Georgia,serif;font-size:1.25rem;color:#E8E4DC;margin:0;">'
            '🎬 CineMatch</p>'
            '<p style="font-size:0.7rem;color:#4A4740;margin:0.2rem 0 0;'
            'font-family:Raleway,sans-serif;letter-spacing:0.07em;">SMART FILM FINDER</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p style="color:#E8A838;font-size:0.7rem;letter-spacing:0.12em;font-weight:600;text-transform:uppercase;margin-bottom:2px;">Mode</p>', unsafe_allow_html=True)
        mode = st.radio("", ["Content-Based", "Hybrid (Recommended)", "Collaborative"],
                        index=1, label_visibility="collapsed")

        st.markdown("---")

        all_genres = sorted({
            g for genres in movies["genres"].dropna()
            for g in (genres if isinstance(genres, list) else [])
        })
        st.markdown('<p style="color:#E8A838;font-size:0.7rem;letter-spacing:0.12em;font-weight:600;text-transform:uppercase;margin-bottom:2px;">Genre Filter</p>', unsafe_allow_html=True)
        genre_raw = st.selectbox("", ["All Genres"] + all_genres, label_visibility="collapsed")
        genre_filter = None if genre_raw == "All Genres" else genre_raw

        st.markdown("---")

        st.markdown('<p style="color:#E8A838;font-size:0.7rem;letter-spacing:0.12em;font-weight:600;text-transform:uppercase;margin-bottom:2px;">Number of Results</p>', unsafe_allow_html=True)
        top_n = st.slider("", 3, 18, 9, 3, label_visibility="collapsed")

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.68rem;color:#3A3730;font-family:Raleway,sans-serif;line-height:1.65;">'
            '<b style="color:#4A4740;">Content</b> &mdash; TF-IDF cosine similarity<br>'
            '<b style="color:#4A4740;">Hybrid</b> &mdash; similarity + quality score<br>'
            '<b style="color:#4A4740;">Collaborative</b> &mdash; SVD matrix factorisation'
            '</p>',
            unsafe_allow_html=True,
        )

    return mode, genre_filter, top_n

def main():
    st.set_page_config(
        page_title="CineMatch", page_icon="🎬",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(CINEMA_CSS, unsafe_allow_html=True)

    with st.spinner("Loading the vault…"):
        movies, similarity, user_matrix, pred_ratings = load_data()
    if movies is None:
        return

    mode, genre_filter, top_n = build_sidebar(movies)
    hero_header()

    if mode in ("Content-Based", "Hybrid (Recommended)"):
        section_heading("Find Similar Films",
                        "Pick a movie you love — we will find what to watch next.")

        movie_list     = sorted(movies["title"].dropna().unique())
        selected_movie = st.selectbox("Choose a film you enjoy", movie_list,
                                      help="Start typing to search")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            go = st.button("Get Recommendations")

        if go:
            with st.spinner("Analysing your taste…"):
                fn   = recommend_hybrid if "Hybrid" in mode else recommend_content
                recs = fn(selected_movie, movies, similarity, top_n, genre_filter)
            if recs.empty:
                st.warning("No results — try a different title or remove the genre filter.")
            else:
                st.session_state.recs       = recs
                st.session_state.recs_label = selected_movie
                st.session_state.recs_mode  = mode

        if st.session_state.get("recs") is not None and not st.session_state.recs.empty:
            recs = st.session_state.recs
            section_heading(
                f"Because you liked {st.session_state.recs_label}",
                f"{len(recs)} films \u00b7 {st.session_state.recs_mode}"
                + (f" \u00b7 {genre_filter}" if genre_filter else ""),
            )
            render_grid(recs, n_cols=3)

    else:
        section_heading("Personalised For You",
                        "We use your rating history to surface films you will love.")

        if user_matrix is None:
            st.info("No collaborative data found — run Preprocessing.py with ratings.csv present.")
        else:
            user_ids      = sorted(user_matrix.index.unique())
            selected_user = st.selectbox("Select your User ID", user_ids)

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                go = st.button("Get My Recommendations")

            if go:
                with st.spinner("Crunching your preferences…"):
                    recs = recommend_collaborative(
                        selected_user, user_matrix, pred_ratings, movies, top_n, genre_filter
                    )
                if recs.empty:
                    st.warning("Not enough data for this user. Rate some movies first!")
                else:
                    st.session_state.collab_recs = recs
                    st.session_state.collab_user = selected_user

            if (st.session_state.get("collab_recs") is not None
                    and st.session_state.get("collab_user") == selected_user
                    and not st.session_state.collab_recs.empty):
                recs = st.session_state.collab_recs
                section_heading(f"Recommended for User {selected_user}", f"{len(recs)} films")
                render_grid(recs, n_cols=3)

    _r1 = st.session_state.get("recs")
    _r2 = st.session_state.get("collab_recs")
    all_recs = _r1 if (_r1 is not None and not _r1.empty) else _r2
    if all_recs is not None and not all_recs.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Rate These Films — helps future recommendations"):
            user_id = st.number_input("Your User ID", min_value=1, value=1, step=1)
            new_ratings = []
            for _, movie in all_recs.iterrows():
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown(
                        f'<p style="color:#E8E4DC;font-family:Raleway,sans-serif;'
                        f'font-size:0.88rem;margin:0.6rem 0;">{movie["title"]}</p>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    rating = st.slider("", 0.0, 10.0, 0.0, 0.5,
                                       key=f"r_{user_id}_{movie['id']}",
                                       label_visibility="collapsed")
                if rating > 0:
                    new_ratings.append({
                        "userId": user_id, "movieId": movie["id"], "rating": rating
                    })
            if new_ratings and st.button("Submit Ratings"):
                df = pd.DataFrame(new_ratings)
                if Path(RATINGS_FILE).exists():
                    old = pd.read_csv(RATINGS_FILE)
                    df  = pd.concat([old, df]).drop_duplicates(
                        subset=["userId", "movieId"], keep="last"
                    )
                df.to_csv(RATINGS_FILE, index=False)
                st.success(f"{len(new_ratings)} rating(s) saved — re-run Preprocessing.py to update the model.")

    st.markdown(
        '<div style="margin-top:4rem;padding:1.2rem 0;'
        'border-top:1px solid rgba(232,168,56,0.12);text-align:center;">'
        '<p style="color:#2A2720;font-size:0.72rem;font-family:Raleway,sans-serif;margin:0;">'
        'CineMatch &nbsp;&middot;&nbsp; Built by <span style="color:#E8A838;">Abhinav Aras and Shripad Joshi</span>'
        ' &nbsp;&middot;&nbsp; Powered by TMDB &amp; Streamlit'
        '</p></div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()