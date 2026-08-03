# ============================================================
# api.py — TEARS + GERS with GPT Genre Embedding
# Nova Final Stable Version — December 2025
# ============================================================

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import torch
import pickle
import os
import json
from openai import OpenAI


TASTE_SUMMARY_PROMPT = """
Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy
Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task
Keep the summary short at about 200 words. The summary should have the following format:
Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}.
""".strip()

# ---------------- OpenAI Setup ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

# ---------------- Internal Modules ----------------
from hybrid_recommender import HybridRecommender
from summary_encoder import SummaryEncoder
from tears_inference_adapter import TEARSInferenceAdapter
from tears_candidate_filter import filter_tears_items, normalize_movie_id
from tmdb_collections import TMDBCollectionResolver, split_movie_title_year

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tmdb_collection_resolver = TMDBCollectionResolver(
    api_key=os.getenv("TMDB_API_KEY"),
    bearer_token=os.getenv("TMDB_BEARER_TOKEN"),
)

MOVIE_EMB_PATH = "/home/mila/a/adls/tears_project_final/model/saved_models/movie_embeddings.pt"
TITLES_PATH = "/home/mila/a/adls/tears_project_final/model/saved_models/movie_titles_fixed.pkl"
GENRES_PATH = "/home/mila/a/adls/tears_project_final/model/saved_models/movie_genres_fixed.pkl"

SUMMARY_JSON_PATH = (
    "/home/mila/a/adls/tears_project_final/"
    "TEARS_Project/Code4Neda/saved_user_summary/ml-1m/"
    "user_summary_gpt-4-1106-preview_.json"
)

# ============================================================
# FastAPI + CORS
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tears_adapter = TEARSInferenceAdapter()
    try:
        yield
    finally:
        app.state.tears_adapter = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load Static Model Data
# ============================================================
print("🎬 Loading movie data...")

with open(TITLES_PATH, "rb") as f:
    movie_titles = pickle.load(f)

with open(GENRES_PATH, "rb") as f:
    movie_genres = pickle.load(f)

movie_embeddings = torch.load(MOVIE_EMB_PATH, map_location=DEVICE)
movie_embeddings = movie_embeddings.float().to(DEVICE)

encoder = SummaryEncoder(device=DEVICE)
recommender = HybridRecommender(
    summary_encoder=encoder,
    movie_embeddings=movie_embeddings,
    movie_titles=movie_titles,
    movie_genres=movie_genres,
    device=DEVICE,
)

print("🚀 API ready on", DEVICE)

# ============================================================
# Load ML-1M Precomputed Summaries
# ============================================================
print("📖 Loading ML-1M precomputed summaries...")

with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
    ML1M_SUMMARIES = json.load(f)

print(f"✅ Loaded {len(ML1M_SUMMARIES)} summaries")

# ============================================================
# Request Models
# ============================================================
class SummaryMovieEvidence(BaseModel):
    title: str = Field(min_length=1)
    rating: float | None = None
    genres: list[str] = Field(default_factory=list)


class SummaryRequest(BaseModel):
    movies: list[SummaryMovieEvidence] = Field(min_length=1)
    disliked: list[str] = Field(default_factory=list)
    context: str | None = ""


class ML1MSummaryRequest(BaseModel):
    movie_ids: list[int]


class TEARSRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=12, ge=1, le=100)


class TEARSRecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    rank: int
    rank_label: str


class TEARSRecommendationResponse(BaseModel):
    items: list[TEARSRecommendationItem]


class GERSRequest(BaseModel):
    genres: list[str]
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    context: str | None = ""
    top_k: int = 12


# ============================================================
# 1) ML-1M PRECOMPUTED SUMMARY
# ============================================================
@app.post("/summary_from_ml1m")
def get_ml1m_summary(req: ML1MSummaryRequest):
    summaries = []

    for mid in req.movie_ids:
        key_variants = [
            str(mid),
            f"{mid}.0",
            str(float(mid)),
        ]

        for k in key_variants:
            if k in ML1M_SUMMARIES:
                summaries.append(ML1M_SUMMARIES[k])
                break

    return {"summary": " ".join(summaries[:3])[:800]}


# ============================================================
# 2) GPT SHORT SUMMARIZER
# ============================================================
@app.post("/summarize")
def summarize(req: SummaryRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Summary generation is unavailable: OPENAI_API_KEY is not configured.",
        )

    evidence = []
    for movie in req.movies:
        evidence.append(movie.title)
        if movie.rating is not None:
            evidence.append(f"Rating: {movie.rating}")
        evidence.append(f"\\Genres: {', '.join(movie.genres)}")

    if req.disliked:
        evidence.extend(
            [
                "",
                f"Movies or genres the user does not enjoy: {', '.join(req.disliked)}",
            ]
        )
    if req.context and req.context.strip():
        evidence.extend(["", f"Additional context: {req.context.strip()}"])
    user_evidence = "\n".join(evidence)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TASTE_SUMMARY_PROMPT},
                {"role": "user", "content": user_evidence},
            ],
            temperature=0,
            seed=0,
            max_tokens=220,
        )

        summary_text = response.choices[0].message.content.strip()
        return {"summary": summary_text}

    except Exception as e:
        print("🔥 SUMMARY ERROR:", e)
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {e}",
        ) from e

# ============================================================
# 3) TEARS — Summary-Based Recommender
# ============================================================
@app.post("/recommend", response_model=TEARSRecommendationResponse)
async def tears(
    req: TEARSRecommendationRequest,
    request: Request,
) -> TEARSRecommendationResponse:
    adapter: TEARSInferenceAdapter = request.app.state.tears_adapter
    try:
        selected_movie_ids = {
            normalize_movie_id(movie_id) for movie_id in req.liked_movie_ids
        }
        catalog_movie_ids = {
            normalize_movie_id(movie_id) for movie_id in req.excluded_movie_ids
        }
        candidate_top_k = min(
            100,
            req.top_k + len(selected_movie_ids | catalog_movie_ids),
        )
        candidates = adapter.recommend(
            summary=req.summary,
            liked_movie_ids=req.liked_movie_ids,
            alpha=req.alpha,
            top_k=candidate_top_k,
        )

        selected_metadata = [
            (
                movie_id,
                adapter.movie_metadata[movie_id]["title"],
                split_movie_title_year(
                    adapter.movie_metadata[movie_id]["title"]
                )[1],
            )
            for movie_id in dict.fromkeys(req.liked_movie_ids)
        ]
        selected_resolutions = await asyncio.gather(
            *[
                asyncio.to_thread(
                    tmdb_collection_resolver.resolve,
                    movie_id,
                    title,
                    year,
                )
                for movie_id, title, year in selected_metadata
            ]
        )
        selected_collection_ids = {
            resolution.collection_id
            for resolution in selected_resolutions
            if resolution is not None and resolution.collection_id is not None
        }

        candidate_resolutions = await asyncio.gather(
            *[
                asyncio.to_thread(
                    tmdb_collection_resolver.resolve,
                    candidate["movie_id"],
                    candidate["title"],
                    split_movie_title_year(candidate["title"])[1],
                )
                for candidate in candidates
            ]
        )
        candidate_collections = {
            normalize_movie_id(candidate["movie_id"]): resolution
            for candidate, resolution in zip(candidates, candidate_resolutions)
        }
        items, removed_items = filter_tears_items(
            candidates,
            selected_movie_ids=selected_movie_ids,
            catalog_movie_ids=catalog_movie_ids,
            selected_collection_ids=selected_collection_ids,
            candidate_collections=candidate_collections,
            top_k=req.top_k,
        )
        removed_catalog = sum(
            item["reason"] == "catalog" for item in removed_items
        )
        removed_collection = sum(
            item["reason"] == "selected_collection" for item in removed_items
        )
        compact = lambda values: json.dumps(values, separators=(",", ":"))
        print(
            "TEARS /recommend: "
            f"selected={compact(req.liked_movie_ids)} "
            f"excluded_catalog={len(catalog_movie_ids)} "
            f"candidates={len(candidates)} "
            f"removed_catalog={removed_catalog} "
            f"removed_collection={removed_collection} "
            f"returned={compact([item['movie_id'] for item in items])}",
            flush=True,
        )
        return TEARSRecommendationResponse(items=items)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        print("🔥 TEARS INFERENCE ERROR:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="TEARS recommendation inference failed",
        ) from error


# ============================================================
# 4) GERS — Genre-Based Recommender
# ============================================================
@app.post("/gers")
def gers(req: GERSRequest, request: Request):
    adapter: TEARSInferenceAdapter = request.app.state.tears_adapter
    try:
        items = adapter.recommend_genres(
            req.genres,
            excluded_movie_ids=req.excluded_movie_ids,
            top_k=req.top_k,
        )
        print(
            "GERS /gers: "
            f"genres={json.dumps(req.genres, separators=(',', ':'))} "
            f"returned={json.dumps([item['movie_id'] for item in items], separators=(',', ':'))}",
            flush=True,
        )
        return {"items": items}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        print("🔥 GERS INFERENCE ERROR:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="GERS genre inference failed",
        ) from error
# ============================================================
# Root
# ============================================================
@app.get("/")
def root():
    return {"status": "running", "device": DEVICE}
