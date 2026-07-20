import requests
import csv
import json
import time

TMDB_KEY = "YOUR_TMDB_API_KEY"

# ---- 1) IMDb 150 Top Rated ----
def fetch_imdb_top150():
    url = "https://imdb-api.projects.thetuhin.com/top250"
    r = requests.get(url).json()

    movies = []
    for item in r["data"][:150]:
        movies.append({
            "id": item["id"],
            "title": item["title"],
            "year": int(item["year"]),
            "rank": int(item["rank"])
        })
    return movies


# ---- 2) TMDB 100 Popular Mainstream ----
def fetch_tmdb_popular(limit=100):
    movies = []
    page = 1

    while len(movies) < limit:
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page={page}"
        data = requests.get(url).json()

        for m in data["results"]:
            movies.append({
                "id": m["id"],
                "title": m["title"],
                "year": int(m["release_date"][:4]) if m.get("release_date") else None,
                "rank": None  # TMDB محبوبیت رتبه ندارد
            })
            if len(movies) >= limit:
                break
        page += 1
        time.sleep(0.25)

    return movies


# ---- 3) TMDB Enrichment (poster, overview, genres, vote_average) ----
def enrich_tmdb(movie):
    title = movie["title"]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={title}"
    data = requests.get(url).json()

    if not data["results"]:
        return None

    m = data["results"][0]
    return {
        "title": title,
        "year": movie["year"],
        "rank": movie["rank"],
        "tmdb_id": m["id"],
        "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
        "overview": m.get("overview", ""),
        "genres": m.get("genre_ids", []),
        "vote_average": m.get("vote_average", 0),
        "popularity": m.get("popularity", 0)
    }


# ---- MAIN ----
print("Fetching IMDb Top 150…")
imdb_movies = fetch_imdb_top150()

print("Fetching TMDB Popular 100…")
mainstream_movies = fetch_tmdb_popular(100)

print("Merging datasets…")
combined = imdb_movies + mainstream_movies

# remove duplicates
unique_titles = {}
for m in combined:
    unique_titles[m["title"].lower()] = m
merged = list(unique_titles.values())

print(f"Merged count: {len(merged)}")

# ---- Enrich all movies ----
final_dataset = []
for movie in merged:
    enriched = enrich_tmdb(movie)
    if enriched:
        final_dataset.append(enriched)
    time.sleep(0.20)

print(f"Enriched dataset size: {len(final_dataset)}")

# ---- Sort by rating (descending) ----
final_dataset.sort(key=lambda x: x["vote_average"], reverse=True)

# ---- Save JSON ----
with open("movies250.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2)

# ---- Save CSV ----
with open("movies250.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["title", "year", "rank", "tmdb_id", "poster", "genres", "vote_average", "overview"])
    for m in final_dataset:
        writer.writerow([
            m["title"], m["year"], m["rank"], m["tmdb_id"],
            m["poster"], "|".join(map(str, m["genres"])), m["vote_average"], m["overview"]
        ])

print("✔ Dataset saved: movies250.json + movies250.csv")
