import os
import pandas as pd
from pathlib import Path
import zipfile
import subprocess

SCRATCH = Path("/network/scratch/a/adls/")
DATA_DIR = SCRATCH / "ml-32m"
OUT_DIR = SCRATCH / "ml32m_tears_10k"
OUT_DIR.mkdir(exist_ok=True)

ZIP_PATH = SCRATCH / "ml-32m.zip"

# -----------------------------
# 1) Download dataset if needed
# -----------------------------
if not ZIP_PATH.exists():
    print("Downloading MovieLens 32M dataset...")
    subprocess.run([
        "wget",
        "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
        "-O",
        str(ZIP_PATH)
    ])

# -----------------------------
# 2) Unzip dataset
# -----------------------------
if not DATA_DIR.exists():
    print("Extracting ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(SCRATCH)

print("Dataset extracted at:", DATA_DIR)

# -----------------------------
# 3) Load raw CSVs
# -----------------------------
print("Loading ratings...")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies = pd.read_csv(DATA_DIR / "movies.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")

# -----------------------------
# Filtering parameters
# -----------------------------
NUM_ITEMS = 4000
NUM_USERS = 10000
MIN_USER_RATINGS = 30
MIN_ITEM_RATINGS = 200

# -----------------------------
# 4) Filter items
# -----------------------------
item_counts = ratings["movieId"].value_counts()
popular_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index
ratings = ratings[ratings["movieId"].isin(popular_items)]

# Top items
top_items = ratings["movieId"].value_counts().head(NUM_ITEMS).index
ratings = ratings[ratings["movieId"].isin(top_items)]

# -----------------------------
# 5) Filter users
# -----------------------------
user_counts = ratings["userId"].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index[:NUM_USERS]
ratings = ratings[ratings["userId"].isin(active_users)]

print("Filtered users:", ratings["userId"].nunique())
print("Filtered items:", ratings["movieId"].nunique())
print("Total ratings:", len(ratings))

# -----------------------------
# 6) Remap IDs
# -----------------------------
user_ids = sorted(ratings["userId"].unique())
movie_ids = sorted(ratings["movieId"].unique())

user_map = {old: i for i, old in enumerate(user_ids)}
movie_map = {old: i for i, old in enumerate(movie_ids)}

ratings["userId"] = ratings["userId"].map(user_map)
ratings["movieId"] = ratings["movieId"].map(movie_map)

# -----------------------------
# 7) Save output
# -----------------------------
ratings.to_csv(OUT_DIR / "ratings.csv", index=False)

movies = movies[movies["movieId"].isin(movie_map.keys())].copy()
movies["movieId"] = movies["movieId"].map(movie_map)
movies.to_csv(OUT_DIR / "movies.csv", index=False)

tags = tags[
    tags["userId"].isin(user_map.keys()) &
    tags["movieId"].isin(movie_map.keys())
].copy()
tags["userId"] = tags["userId"].map(user_map)
tags["movieId"] = tags["movieId"].map(movie_map)
tags.to_csv(OUT_DIR / "tags.csv", index=False)

print("\n✔ DONE — subset created at:", OUT_DIR)

