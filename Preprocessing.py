import pandas as pd
import ast
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

MAX_MOVIES       = 10_000
MAX_TFIDF_FEATS  = 15_000  
SVD_COMPONENTS   = 50     
DATA_DIR         = Path("data")

def parse_list_field(x, key="name"):
    """Handle JSON-list strings, plain comma strings, or actual lists."""
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [
                    item[key] if isinstance(item, dict) and key in item else str(item)
                    for item in parsed
                ]
        except Exception:
            pass
        return [t.strip() for t in x.replace("|", ",").split(",") if t.strip()]
    return []


def parse_plain_keywords(x):
    """Keywords column is plain comma-separated text (no JSON)."""
    if pd.isna(x) or x == "":
        return []
    return [t.strip().replace(" ", "_") for t in str(x).split(",") if t.strip()]


def safe_year(date_str):
    try:
        return int(str(date_str)[:4])
    except Exception:
        return None

def process_movies():
    print("🎬 Starting enhanced preprocessing…\n")
    try:
        movies_df = pd.read_csv("MovieDataset.csv")
        print(f"✅ Dataset loaded — {len(movies_df):,} rows")
        print(f"   Columns: {movies_df.columns.tolist()}\n")
    except Exception as e:
        print(f"❌ Cannot load MovieDataset.csv: {e}")
        return

    wanted = [
        "id", "title", "overview", "genres", "vote_average",
        "vote_count", "keywords", "tagline", "release_date",
        "popularity", "poster_path", "runtime",
    ]
    processed = pd.DataFrame()
    for col in wanted:
        processed[col] = movies_df[col] if col in movies_df.columns else None

    processed["genres"]   = processed["genres"].apply(parse_list_field)
    processed["keywords"] = processed["keywords"].apply(parse_plain_keywords)

    processed["overview"] = processed["overview"].fillna("").astype(str).str.strip()
    processed["tagline"]  = processed["tagline"].fillna("").astype(str).str.strip()
    processed["title"]    = processed["title"].fillna("Untitled").astype(str).str.strip()

    processed["vote_average"] = pd.to_numeric(processed["vote_average"], errors="coerce").fillna(0)
    processed["vote_count"]   = pd.to_numeric(processed["vote_count"],   errors="coerce").fillna(0)
    processed["popularity"]   = pd.to_numeric(processed["popularity"],   errors="coerce").fillna(0)
    processed["runtime"]      = pd.to_numeric(processed["runtime"],      errors="coerce").fillna(0)
    processed["year"]         = processed["release_date"].apply(safe_year)

    m = processed["vote_count"].quantile(0.80)
    C = processed["vote_average"].mean()
    processed["quality_score"] = (
        (processed["vote_count"] / (processed["vote_count"] + m)) * processed["vote_average"]
        + (m / (processed["vote_count"] + m)) * C
    ).round(3)

    if len(processed) > MAX_MOVIES:
        processed = processed.nlargest(MAX_MOVIES, "quality_score").reset_index(drop=True)
        print(f"⚠️  Dataset trimmed to top {MAX_MOVIES:,} by quality score")

    def build_tags(row):
        title_str    = " ".join(row["title"].split()) * 6
        genre_str    = " ".join(row["genres"]) * 3      
        keyword_str  = " ".join(row["keywords"][:30]) * 2 
        overview_str = row["overview"]
        tagline_str  = row["tagline"]
        return f"{title_str} {overview_str} {tagline_str} {genre_str} {keyword_str}".lower()

    print("⚙️  Building weighted tag corpus…")
    processed["tags"] = processed.apply(build_tags, axis=1)

    print("⚙️  Fitting TF-IDF vectoriser (this may take ~30 s)…")
    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATS,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,   
        min_df=2,            
    )
    vectors = tfidf.fit_transform(processed["tags"])
    print(f"   Vocabulary size: {len(tfidf.vocabulary_):,}")

    print("⚙️  Computing cosine similarity…")
    similarity = cosine_similarity(vectors)   

    pop_norm = (processed["quality_score"] / processed["quality_score"].max()).values
    similarity = 0.95 * similarity + 0.05 * pop_norm[np.newaxis, :]
    np.fill_diagonal(similarity, 0) 

    print("⚙️  Building collaborative filtering model…")
    try:
        ratings_df = pd.read_csv("ratings.csv")
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])
        ratings_df["userId"]  = ratings_df["userId"].astype(int)
        ratings_df["movieId"] = ratings_df["movieId"].astype(int)

        user_movie_matrix = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        user_means = user_movie_matrix.replace(0, np.nan).mean(axis=1).fillna(0)
        centred    = user_movie_matrix.sub(user_means, axis=0).fillna(0)

        n_comp = min(SVD_COMPONENTS, min(centred.shape) - 1)
        svd    = TruncatedSVD(n_components=n_comp, random_state=42)
        latent = svd.fit_transform(centred)

        pred_ratings = np.dot(latent, svd.components_) + user_means.values[:, np.newaxis]

        with open(DATA_DIR / "user_movie_matrix.pkl", "wb") as f:
            pickle.dump(user_movie_matrix, f)
        with open(DATA_DIR / "predicted_ratings.pkl", "wb") as f:
            pickle.dump(pred_ratings, f)

        print(f"   ✅ Collaborative model built — {user_movie_matrix.shape[0]} users, "
              f"{user_movie_matrix.shape[1]} movies")
        print(f"   SVD explained variance: {svd.explained_variance_ratio_.sum():.1%}")

    except Exception as e:
        print(f"   ⚠️  Collaborative filtering skipped: {e}")

    DATA_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "movies_enhanced.pkl", "wb") as f:
        pickle.dump(processed, f)

    np.save(DATA_DIR / "similarity_matrix.npy", similarity)

    with open(DATA_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print(f"""
✅  Preprocessing complete!
   Movies saved     : {len(processed):,}
   Similarity shape : {similarity.shape}
   Data dir         : {DATA_DIR.resolve()}

First 5 movies:
{processed[['id', 'title', 'genres', 'quality_score']].head().to_string(index=False)}
""")


if __name__ == "__main__":
    process_movies()
