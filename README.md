# 🎬 Indian Movie Recommendation System

> A content-based ML recommendation engine for Indian cinema (Bollywood · Tollywood · Kollywood · Mollywood), built with Python, Flask, and scikit-learn.

---

## 📁 Project Structure

```
indian-movie-recommender/
│
├── app.py                  ← Flask web server + API routes
├── model_training.py       ← ML pipeline: data cleaning → vectorization → similarity
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── data/
│   └── indian_movies.csv   ← Dataset: 100 Indian films across 5 industries
│
├── model/                  ← Saved model artifacts (auto-created after training)
│   ├── movies_df.pkl       ← Cleaned dataframe
│   ├── cosine_sim.pkl      ← Precomputed similarity matrix
│   ├── movie_indices.pkl   ← Title → index mapping
│   └── count_vectorizer.pkl
│
├── static/
│   ├── css/style.css       ← UI styling
│   └── js/main.js          ← Frontend logic
│
└── templates/
    └── index.html          ← Main HTML page
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
python model_training.py
```

### 3. Start the Flask server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🧠 How It Works — Technical Overview

### Step 1: Data Loading & Cleaning
- Load `indian_movies.csv` with Pandas
- Fill missing values with empty strings
- Normalize text: lowercase, remove punctuation

### Step 2: Feature Engineering
Multiple movie attributes are combined into a **"feature soup"** — a single text string that represents each movie's identity:

```
genre (×3) + director (×3) + cast (×2) + language (×2)
```

Repetition acts as a **weight** — a movie's director and genre influence recommendations more strongly than its release year.

### Step 3: Vectorization (CountVectorizer)
scikit-learn's `CountVectorizer` converts each movie's feature soup into a **numeric vector** (Bag of Words):

```
"action drama rajamouli rajamouli prabhas..." → [0, 0, 3, 2, 1, ...]
```

Each dimension of the vector = one unique word in the entire corpus.

### Step 4: Cosine Similarity Matrix
For every pair of movies, cosine similarity is computed:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Result: an **N × N matrix** where `matrix[i][j]` is how similar movie `i` is to movie `j`.

- Score = 1.0 → identical feature profiles
- Score = 0.0 → nothing in common

### Step 5: Recommendation
When a user selects a movie:
1. Find its row in the similarity matrix
2. Sort all other movies by similarity score (descending)
3. Return top-N results

---

## 📊 Dataset

The dataset covers **100 curated Indian films** across:

| Industry | Language | Examples |
|---|---|---|
| Bollywood | Hindi | Dangal, 3 Idiots, Gangs of Wasseypur |
| Tollywood | Telugu | RRR, Bahubali, Arjun Reddy |
| Kollywood | Tamil | Vikram, 96, Super Deluxe |
| Mollywood | Malayalam | Drishyam, Kumbalangi Nights |
| Sandalwood | Kannada | KGF Chapter 1 & 2 |

Fields: `title`, `genre`, `language`, `cast`, `director`, `release_year`, `rating`, `description`

---

## 🎯 Key ML Concepts

### Content-Based Filtering
Recommends items **similar to what a user already likes**, based on item attributes — not user behavior. No need for user history or ratings data.

**Contrast with Collaborative Filtering** which uses other users' behavior patterns.

### Cosine Similarity
Measures the **angle** between two vectors in high-dimensional space. Independent of vector magnitude — only direction matters. Perfect for text-based feature comparison.

### Feature Engineering
The process of creating new inputs for your ML model from raw data. Here: combining genre + director + cast + language into a weighted text feature. Quality of features directly determines recommendation quality.

### Vectorization
Converting text into numbers (vectors). CountVectorizer uses raw word counts. TF-IDF additionally penalizes common words. Choice between them depends on your use case.

---

## 🗂 Model Artifacts (Pickle Files)

| File | Contents | Why saved? |
|---|---|---|
| `movies_df.pkl` | Full cleaned dataframe | Display movie metadata without re-loading CSV |
| `cosine_sim.pkl` | N×N similarity matrix | Avoid recomputing on every request |
| `movie_indices.pkl` | Title → row index map | O(1) movie lookup |
| `count_vectorizer.pkl` | Trained vectorizer | For adding new movies later |

---

## 🎤 Interview Questions & Answers

### Q1. What is content-based filtering?
**A:** Content-based filtering recommends items similar to those a user has shown interest in, based on item features (like genre, director, cast). It doesn't need data about other users. For example, if you liked *Dangal*, the system looks at its features (biography, sport, drama, Aamir Khan, Nitesh Tiwari) and finds movies with similar attributes.

**Contrast:** Collaborative filtering uses patterns from many users — "users who liked X also liked Y." Content-based works even for a single user with no history.

---

### Q2. Explain cosine similarity in simple terms.
**A:** Imagine each movie as an arrow pointing in some direction in a high-dimensional space. Cosine similarity measures the angle between two arrows — not their length, just their direction.

- Angle = 0° → similarity = 1.0 (identical direction)
- Angle = 90° → similarity = 0.0 (nothing in common)

Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`

We use cosine (not Euclidean distance) because it handles sparse vectors better — movies with few overlapping features shouldn't be penalized just for having short feature lists.

---

### Q3. Why CountVectorizer and not TF-IDF?
**A:** TF-IDF downweights terms that appear frequently across documents. But in our case, if the director "Rajamouli" appears three times, that *should* carry weight — a Rajamouli fan wants more Rajamouli films. CountVectorizer respects raw frequency, which aligns with our feature-weighting strategy via repetition.

TF-IDF would be better for a text search engine where common words like "the" should be ignored. For movie feature matching, raw counts work better.

---

### Q4. What is feature engineering and why did you repeat some features?
**A:** Feature engineering is creating useful input representations for ML models from raw data. In this project, I combined multiple fields into a single text string ("soup"). I repeated important features like director and genre to give them more weight — since CountVectorizer counts word frequency, repeating a word makes it more influential in the cosine similarity calculation.

Director ×3 means director similarity contributes 3× more than language ×1 in the final similarity score.

---

### Q5. What is a similarity matrix and how does it scale?
**A:** It's an N×N matrix where entry [i][j] = cosine similarity between movie i and movie j. For 100 movies, it's a 100×100 matrix = 10,000 values.

**Scaling issue:** For 10,000 movies, it becomes a 10,000×10,000 matrix = 100 million values. At that scale, you'd use approximate nearest neighbor algorithms (like FAISS or Annoy) rather than brute-force cosine similarity.

---

### Q6. What is pickle and why use it?
**A:** Pickle is Python's built-in serialization library that converts Python objects (like NumPy arrays, Pandas DataFrames, sklearn models) into byte streams that can be saved to disk and reloaded.

We save the similarity matrix with pickle so the Flask server doesn't need to retrain the model on every startup — it just loads the precomputed results in milliseconds.

---
---
---

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | CountVectorizer + cosine_similarity |
| Flask | Web framework and REST API |
| Pickle | Model serialization |
| HTML/CSS/JS | Frontend interface |

---

## 👨‍💻 Author

B.Tech CSE Final Year Student  
Indian Movie Recommendation System — Resume Project

---

*Built to demonstrate: content-based filtering, feature engineering, text vectorization, cosine similarity, Flask REST APIs, and model persistence.*
