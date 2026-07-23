"""FastAPI service for the deterministic quality-improved recommender."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
import numpy as np
import scipy.sparse as sp

from tears_recommender import EnhancedRecommender, load_catalog, load_ratings


class RecommendationRequest(BaseModel):
    description: str = ""
    summary: str = ""
    context: str = ""
    liked: list[str] = Field(default_factory=list)
    disliked_genres: list[str] = Field(default_factory=list)
    disliked: list[str] = Field(default_factory=list)
    exclude_titles: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)


class SummaryRequest(BaseModel):
    liked: list[str] = Field(default_factory=list)
    disliked: list[str] = Field(default_factory=list)
    context: str = ""


class GenreRequest(BaseModel):
    genres: list[str] = Field(default_factory=list)


class GERSRequest(GenreRequest):
    context: str = ""
    top_k: int = Field(default=12, ge=1, le=100)


class ML1MSummaryRequest(BaseModel):
    movie_ids: list[int] = Field(default_factory=list)


TASTE_SUMMARY_PROMPT = """
Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy.
Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task.
Keep the summary short at about 200 words. The summary should have the following format:
Summary: {{Specific details about genres the user enjoys}}. {{Specific details of plot points the user seems to enjoy}}. {{Specific details about genres the user does not enjoy}}. {{Specific details of plot points the user does not enjoy but other users may}}.

User information:
{user_input}
""".strip()


def build_recommender() -> EnhancedRecommender:
    catalog = load_catalog()
    ratings = load_ratings()
    positives = ratings[
        (ratings.rating >= 4.0) & ratings.movie_id.isin(catalog.movie_id_to_index)
    ]
    user_ids = {user_id: row for row, user_id in enumerate(positives.user_id.unique())}
    rows = positives.user_id.map(user_ids).to_numpy()
    columns = positives.movie_id.map(catalog.movie_id_to_index).to_numpy()
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(user_ids), catalog.size),
    )
    interactions.data[:] = 1.0
    return EnhancedRecommender(catalog).fit(interactions)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = build_recommender()
    yield


app = FastAPI(title="TEARS Quality Recommender", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "model": "EASE + explicit genre control",
        "catalog_items": app.state.recommender.catalog.size,
    }


def preference_text(parts: list[str]) -> str:
    """Combine legacy and current text fields without changing ranking semantics."""
    return "\nContext: ".join(part.strip() for part in parts if part.strip())


def genre_summary(genres: list[str], context: str = "") -> str:
    unique_genres = list(dict.fromkeys(genre.strip() for genre in genres if genre.strip()))
    base = f"The user prefers movies in these genres: {', '.join(unique_genres)}."
    return preference_text([base if unique_genres else "", context])


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        description = preference_text(
            [request.description, request.summary, request.context]
        )
        return {
            "items": app.state.recommender.recommend(
                description=description,
                liked_titles=request.liked,
                disliked_genres=list(
                    dict.fromkeys(request.disliked_genres + request.disliked)
                ),
                exclude_titles=request.exclude_titles,
                top_k=request.top_k,
            )
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/summarize")
def summarize(request: SummaryRequest):
    if not request.liked:
        return {"summary": ""}
    if not os.getenv("OPENAI_API_KEY"):
        liked = ", ".join(request.liked)
        disliked = ", ".join(request.disliked)
        parts = [f"The user likes movies such as {liked}."]
        if disliked:
            parts.append(f"The user does not enjoy {disliked}.")
        if request.context.strip():
            parts.append(f"Current viewing context: {request.context.strip()}")
        return {"summary": " ".join(parts)}

    user_input = (
        f"Liked movies:\n{', '.join(request.liked)}\n\n"
        f"Disliked movies or genres:\n{', '.join(request.disliked)}\n\n"
        f"Additional context:\n{request.context}"
    )
    try:
        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": TASTE_SUMMARY_PROMPT.format(user_input=user_input),
                }
            ],
            temperature=0.4,
            max_tokens=260,
        )
        return {"summary": response.choices[0].message.content.strip()}
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Summary generation failed: {error}") from error


@app.post("/summarize_genres")
def summarize_genres(request: GenreRequest):
    """Compatibility endpoint used by the original GERS interface."""
    return {"summary": genre_summary(request.genres)}


@app.post("/gers")
def gers(request: GERSRequest):
    """Run the original genre-driven screen through the improved EASE ranker."""
    summary = genre_summary(request.genres, request.context)
    try:
        items = app.state.recommender.recommend(
            description=summary,
            liked_titles=[],
            disliked_genres=[],
            top_k=request.top_k,
        )
        return {"summary": summary, "items": items}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/summary_from_ml1m")
def summary_from_ml1m(request: ML1MSummaryRequest):
    """Build a valid preference summary from original MovieLens movie IDs."""
    catalog = app.state.recommender.catalog
    titles: list[str] = []
    genres: list[str] = []
    for movie_id in request.movie_ids:
        index = catalog.movie_id_to_index.get(movie_id)
        if index is None:
            continue
        titles.append(catalog.titles[index])
        genres.extend(catalog.genres[index])

    if not titles:
        return {"summary": ""}

    unique_genres = list(dict.fromkeys(genres))
    return {
        "summary": (
            f"The user likes movies such as {', '.join(titles)}. "
            f"Their selected movies suggest interest in {', '.join(unique_genres)}."
        )
    }
