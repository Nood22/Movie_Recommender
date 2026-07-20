import json
from collections import defaultdict

MOVIES_FILE = "movies.dat"
RATINGS_FILE = "ratings.dat"

# -------- Load movies --------
movies = {}
movie_genres = {}

with open(MOVIES_FILE, encoding="latin-1") as f:
    for line in f:
        movieId, title, genres = line.strip().split("::")
        movies[int(movieId)] = title
        movie_genres[int(movieId)] = genres.split("|")

# -------- Count ratings --------
rating_count = defaultdict(int)

with open(RATINGS_FILE, encoding="latin-1") as f:
    for line in f:
        userId, movieId, rating, ts = line.strip().split("::")
        rating_count[int(movieId)] += 1

# -------- Sort by popularity --------
sorted_movies = sorted(
    rating_count.items(),
    key=lambda x: x[1],
    reverse=True
)

# -------- Select 50 with genre diversity --------
selected = []
genre_counter = defaultdict(int)

for movieId, _ in sorted_movies:
    if movieId not in movies:
        continue

    genres = movie_genres[movieId]

    # prevent genre domination
    if any(genre_counter[g] >= 10 for g in genres):
        continue

    selected.append({
        "movieId": movieId,
        "title": movies[movieId],
        "genres": genres
    })

    for g in genres:
        genre_counter[g] += 1

    if len(selected) == 50:
        break

# -------- Save output --------
output_path = (
    "/home/mila/a/adls/tears_project_final/"
    "ui/tearsApp/data/fixed_50_movies_ml1m.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print("✅ Saved fixed 50-movie list to:")
print(output_path)
