"""
app.py
======
Flask web application for the Indian Movie Recommendation System.

Routes:
  GET  /           → Home page with movie dropdown
  POST /recommend  → Returns recommendations as JSON
  GET  /movie/<title> → Movie detail page
"""

import os
import pickle
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ─────────────────────────────────────────────────────────────────────────────
# Load pre-trained model artifacts (loaded ONCE at startup for efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_artifacts():
    """Load all pickled model files into memory at app startup."""
    try:
        with open(os.path.join(MODEL_DIR, "movies_df.pkl"), "rb") as f:
            movies_df = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "cosine_sim.pkl"), "rb") as f:
            cosine_sim = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "movie_indices.pkl"), "rb") as f:
            movie_indices = pickle.load(f)

        print("✓ Model artifacts loaded successfully!")
        return movies_df, cosine_sim, movie_indices

    except FileNotFoundError:
        print("⚠️  Model files not found. Run model_training.py first!")
        return None, None, None


# Load once at startup
movies_df, cosine_sim, movie_indices = load_model_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Core Recommendation Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(movie_title, num_recommendations=8):
    """
    Returns a list of similar movies based on cosine similarity.

    Algorithm:
      1. Find the index of the input movie in the dataframe
      2. Look up its similarity scores against ALL other movies
      3. Sort those scores in descending order
      4. Return the top-N movies (excluding the input movie itself)

    Args:
        movie_title (str): Title of the movie to find recommendations for
        num_recommendations (int): Number of movies to return (default: 8)

    Returns:
        list[dict]: Each dict contains movie metadata for display
    """
    if movies_df is None:
        return []

    # Step 1: Get the integer index of the movie
    if movie_title not in movie_indices:
        return []

    idx = movie_indices[movie_title]

    # Step 2: Get pairwise similarity scores for this movie vs all others
    # cosine_sim[idx] is a 1D array of scores, one per movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Step 3: Sort movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Step 4: Remove the movie itself (score = 1.0 with itself), take top N
    sim_scores = sim_scores[1: num_recommendations + 1]

    # Step 5: Build output list with full movie metadata
    recommendations = []
    for i, score in sim_scores:
        movie = movies_df.iloc[i]
        recommendations.append({
            "title":        movie["title"],
            "genre":        movie["genre"],
            "language":     movie["language"],
            "cast":         movie["cast"],
            "director":     movie["director"],
            "release_year": int(movie["release_year"]),
            "rating":       float(movie["rating"]),
            "description":  movie["description"],
            "similarity":   round(float(score) * 100, 1)   # as percentage
        })

    return recommendations


def get_movie_details(title):
    """Fetch full details for a single movie by title."""
    if movies_df is None:
        return None

    row = movies_df[movies_df["title"] == title]
    if row.empty:
        return None

    movie = row.iloc[0]
    return {
        "title":        movie["title"],
        "genre":        movie["genre"],
        "language":     movie["language"],
        "cast":         movie["cast"],
        "director":     movie["director"],
        "release_year": int(movie["release_year"]),
        "rating":       float(movie["rating"]),
        "description":  movie["description"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home page: renders the main UI with the movie dropdown."""
    if movies_df is None:
        return "⚠️ Model not trained yet. Please run model_training.py first.", 500

    # Sort movies alphabetically for the dropdown
    movie_list = sorted(movies_df["title"].tolist())
    return render_template("index.html", movies=movie_list)


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    API endpoint: accepts a movie title, returns JSON recommendations.

    Request body (JSON): { "movie": "Dangal", "count": 8 }
    Response (JSON):     { "query": "Dangal", "recommendations": [...] }
    """
    data = request.get_json()

    if not data or "movie" not in data:
        return jsonify({"error": "No movie title provided"}), 400

    movie_title = data["movie"].strip()
    count = int(data.get("count", 8))

    # Validate the movie exists in our dataset
    if movie_title not in movie_indices.index:
        return jsonify({"error": f"Movie '{movie_title}' not found in dataset"}), 404

    # Get the query movie details + recommendations
    query_movie  = get_movie_details(movie_title)
    recs         = get_recommendations(movie_title, count)

    return jsonify({
        "query":           movie_title,
        "query_details":   query_movie,
        "recommendations": recs,
        "count":           len(recs)
    })


@app.route("/movies")
def get_all_movies():
    """Returns the full movie list as JSON (for autocomplete or search)."""
    if movies_df is None:
        return jsonify([])

    movies = movies_df[["title", "genre", "language", "rating"]].to_dict(orient="records")
    return jsonify(movies)


@app.route("/search")
def search():
    """Simple search endpoint: filters movies by query string."""
    query = request.args.get("q", "").lower().strip()
    if not query or movies_df is None:
        return jsonify([])

    # Filter movies where title contains the query
    mask    = movies_df["title"].str.lower().str.contains(query, na=False)
    results = movies_df[mask][["title", "genre", "language", "rating"]].head(10)
    return jsonify(results.to_dict(orient="records"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎬 Starting Indian Movie Recommendation System...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
