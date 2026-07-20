import os
import pandas as pd
import scipy.sparse as sp

# مسیر دیتاست MovieLens
BASE_DIR = os.path.dirname(__file__)
MOVIELENS_PATH = os.path.join(BASE_DIR, "TEARS_Project", "Code4Neda", "ml-1m")

# --- Map movie id -> title ---
def map_id_to_title(data_name="ml-1m"):
    movies_file = os.path.join(MOVIELENS_PATH, "movies.dat")
    mapping = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                mid, title, genres = parts
                mapping[int(mid)] = title
    return mapping

# --- Map movie id -> genres ---
def map_id_to_genre(data_name="ml-1m"):
    movies_file = os.path.join(MOVIELENS_PATH, "movies.dat")
    mapping = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                mid, title, genres = parts
                mapping[int(mid)] = genres.split("|")
    return mapping

# --- Load movies with full info ---
def load_movies(data_name="ml-1m"):
    movies_file = os.path.join(MOVIELENS_PATH, "movies.dat")
    movies = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                mid, title, genres = parts
                movies[int(mid)] = (title, genres.split("|"))
    return movies

# --- Load ratings ---
def load_ratings(data_name="ml-1m"):
    ratings_file = os.path.join(MOVIELENS_PATH, "ratings.dat")
    data = pd.read_csv(ratings_file, sep="::", engine="python", header=None,
                       names=["user", "movie", "rating", "timestamp"])
    return data

# --- Build sparse interaction matrix ---
def build_interaction_matrix(data_name="ml-1m"):
    ratings = load_ratings(data_name)
    num_users = ratings["user"].nunique()
    num_movies = ratings["movie"].nunique()
    R = sp.lil_matrix((num_users + 1, num_movies + 1))
    for row in ratings.itertuples():
        R[row.user, row.movie] = row.rating
    return R
