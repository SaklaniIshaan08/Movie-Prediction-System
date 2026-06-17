"""
model_training.py
=================
This script handles:
  1. Loading and cleaning the Indian movies dataset
  2. Feature engineering — combining movie attributes into one text vector
  3. Vectorization using CountVectorizer (Bag of Words)
  4. Building the cosine similarity matrix
  5. Saving the trained model artifacts using pickle

Run this ONCE before starting the Flask server.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────────────────────────────────────

print("🎬 Indian Movie Recommendation System — Model Training")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "indian_movies.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    ✓ Loaded {len(df)} movies from dataset")
print(f"    ✓ Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Data Cleaning & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Cleaning and preprocessing data...")

# Fill any missing values with empty strings to avoid NaN errors
df.fillna("", inplace=True)

# Clean text: lowercase, strip spaces, remove special characters
def clean_text(text):
    """
    Converts text to lowercase and removes special chars.
    This ensures 'Shah Rukh Khan' and 'shah rukh khan' are treated the same.
    """
    text = str(text).lower()
    text = text.replace(",", " ").replace("-", " ").replace(".", "")
    return text.strip()

df["title_clean"]    = df["title"].apply(clean_text)
df["genre_clean"]    = df["genre"].apply(clean_text)
df["language_clean"] = df["language"].apply(clean_text)
df["cast_clean"]     = df["cast"].apply(clean_text)
df["director_clean"] = df["director"].apply(clean_text)

print(f"    ✓ Cleaned all text fields")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Engineering features (combining movie attributes)...")

def create_feature_soup(row):
    """
    Combines multiple movie features into a single text string (called 'soup').
    
    Why a 'soup'? The CountVectorizer treats each word as a feature.
    By repeating important features (like director & cast), we give them
    more weight in the similarity calculation.
    
    Feature weights via repetition:
      - Director (x3): Most important for style/genre similarity
      - Genre    (x3): Core similarity signal
      - Cast     (x2): Secondary but still important
      - Language (x2): Clusters Indian regional films correctly
    """
    genre    = " ".join(row["genre_clean"].split())    # repeated 3x
    director = " ".join(row["director_clean"].split()) # repeated 3x
    cast     = " ".join(row["cast_clean"].split())     # repeated 2x
    language = row["language_clean"]                   # repeated 2x

    # Build the weighted soup string
    soup = (
        f"{genre} {genre} {genre} "
        f"{director} {director} {director} "
        f"{cast} {cast} "
        f"{language} {language}"
    )
    return soup

df["soup"] = df.apply(create_feature_soup, axis=1)

print(f"    ✓ Feature soup created for all movies")
print(f"    Sample soup: {df['soup'].iloc[0][:80]}...")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Vectorization using CountVectorizer
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Vectorizing features with CountVectorizer...")

"""
CountVectorizer converts text into a sparse matrix of token counts.
Each movie becomes a vector in a high-dimensional space where each
dimension represents a unique word (feature).

Why CountVectorizer over TF-IDF here?
→ TF-IDF penalizes common words. But 'Rajkumar Hirani' appearing many
  times SHOULD matter — a fan of his films wants MORE Rajkumar Hirani.
  CountVectorizer respects that raw frequency.
"""

cv = CountVectorizer(
    stop_words="english",   # Remove common English words like 'the', 'a', 'is'
    max_features=5000,      # Limit to top 5000 features to save memory
    ngram_range=(1, 1)      # Unigrams only (single words)
)

# Fit and transform: creates a (num_movies × num_features) sparse matrix
count_matrix = cv.fit_transform(df["soup"])

print(f"    ✓ Count matrix shape: {count_matrix.shape}")
print(f"    ✓ Vocabulary size: {len(cv.vocabulary_)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Cosine Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Computing cosine similarity matrix...")

"""
Cosine Similarity measures the angle between two vectors.
Score = 1.0 → identical movies
Score = 0.0 → completely unrelated movies

Formula: cos(θ) = (A · B) / (||A|| × ||B||)

We get an (N × N) matrix where entry [i][j] = similarity between movie i and j.
"""

cosine_sim = cosine_similarity(count_matrix, count_matrix)

print(f"    ✓ Similarity matrix shape: {cosine_sim.shape}")
print(f"    ✓ Sample similarity (Dangal vs Lagaan): "
      f"{cosine_sim[0][9]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Save the Model Artifacts using Pickle
# ─────────────────────────────────────────────────────────────────────────────

print("\n💾 Saving model artifacts...")

# Save the cleaned dataframe (movie metadata for display)
df_save_path = os.path.join(MODEL_DIR, "movies_df.pkl")
with open(df_save_path, "wb") as f:
    pickle.dump(df, f)
print(f"    ✓ Saved movies dataframe → {df_save_path}")

# Save the cosine similarity matrix
sim_save_path = os.path.join(MODEL_DIR, "cosine_sim.pkl")
with open(sim_save_path, "wb") as f:
    pickle.dump(cosine_sim, f)
print(f"    ✓ Saved cosine similarity matrix → {sim_save_path}")

# Save the CountVectorizer (for reference / future reuse)
cv_save_path = os.path.join(MODEL_DIR, "count_vectorizer.pkl")
with open(cv_save_path, "wb") as f:
    pickle.dump(cv, f)
print(f"    ✓ Saved CountVectorizer → {cv_save_path}")

# Save index mapping: movie title → dataframe index
movie_indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
idx_save_path = os.path.join(MODEL_DIR, "movie_indices.pkl")
with open(idx_save_path, "wb") as f:
    pickle.dump(movie_indices, f)
print(f"    ✓ Saved movie index mapping → {idx_save_path}")

print("\n" + "=" * 60)
print("✅ Model training complete! All artifacts saved to /model/")
print("   You can now run: python app.py")
print("=" * 60)
