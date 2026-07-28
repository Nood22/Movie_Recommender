# ============================================================
# api.py — TEARS + GERS with GPT Genre Embedding
# Nova Final Stable Version — December 2025
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import torch
import pickle
import os
import json
from openai import OpenAI


TASTE_SUMMARY_PROMPT = """
You are an expert movie preference analyst.

Task:
Generate a detailed natural-language summary of a user's movie taste based on the provided information.

Input:
<<<
{USER_INPUT}
>>>

Instructions:
- Start with the exact word: "Summary:"
- In the first paragraph, describe what the user enjoys:
  - Preferred genres
  - Emotional tone
  - Narrative complexity
  - Themes and storytelling style
- Focus on patterns and inferred preferences, not listing items.
- Use fluent, analytical English.

- Then write a second paragraph starting with the exact word: "Conversely,"
  - Describe what the user tends not to enjoy
  - Clearly contrast with the positive preferences

- Do NOT use bullet points.
- Do NOT mention that you are an AI.
- Keep the total length between 120 and 180 words.
"""

# ---------------- OpenAI Setup ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    raise ValueError("❌ OPENAI_API_KEY not found in environment!")

# ---------------- Internal Modules ----------------
from hybrid_recommender import HybridRecommender
from summary_encoder import SummaryEncoder
from tears_inference_adapter import TEARSInferenceAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
class SummaryRequest(BaseModel):
    liked: list[str] = []
    disliked: list[str] = []
    context: str | None = ""


class ML1MSummaryRequest(BaseModel):
    movie_ids: list[int]


class TEARSRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
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
    user_input = f"""
Liked movies:
{", ".join(req.liked)}

Disliked movies or genres:
{", ".join(req.disliked)}

Additional context:
{req.context}
"""

    prompt = TASTE_SUMMARY_PROMPT.format(USER_INPUT=user_input)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=220,
        )

        summary_text = response.choices[0].message.content.strip()
        return {"summary": summary_text}

    except Exception as e:
        print("🔥 SUMMARY ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

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
        items = adapter.recommend(
            summary=req.summary,
            liked_movie_ids=req.liked_movie_ids,
            alpha=req.alpha,
            top_k=req.top_k,
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
def gers(req: GERSRequest):
    try:
        user_input = f"""
Selected genres:
{", ".join(req.genres)}

Additional context:
{req.context}
"""

        prompt = TASTE_SUMMARY_PROMPT.format(
            USER_INPUT=user_input
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.4,
        )

        genre_summary = response.choices[0].message.content.strip()

        items = recommender.recommend(
            summary_text=genre_summary,
            context_text="",
            liked_titles=[],
            disliked_genres=[],
            top_k=req.top_k,
            alpha=1.0,
        )

        return {
            "summary": genre_summary,
            "items": [
                {
                    "title": it["title"],
                    "score": float(it["score"]),
                    "rank": int(it["rank"]),
                    "rank_label": it["rank_label"],
                }
                for it in items
            ],
        }

    except Exception as e:
        print("🔥 GERS ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================
# Root
# ============================================================
@app.get("/")
def root():
    return {"status": "running", "device": DEVICE}
