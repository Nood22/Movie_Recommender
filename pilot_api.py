"""API and static site for full-corpus TEARS and the retained pilot GERS."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from pilot_recommender import (
    EXPECTED_MATRIX_FINGERPRINT,
    EXPECTED_PILOT_MATRIX_FINGERPRINT,
    PilotHybridRecommender,
)
from tears_training.summaries import (
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    render_history_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_BUILD = PROJECT_ROOT / "movie-recommender-pilot" / "build"
ONBOARDING_PATH = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "pilot_support20_onboarding.json"
)
ONBOARDING_MANIFEST_PATH = ONBOARDING_PATH.with_suffix(".manifest.json")
SUMMARY_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_MIN_RECOMMENDATION_YEAR = 2020

ONBOARDING_CATALOG = json.loads(ONBOARDING_PATH.read_text(encoding="utf-8"))
ONBOARDING_MANIFEST = json.loads(
    ONBOARDING_MANIFEST_PATH.read_text(encoding="utf-8")
)
EXPECTED_ONBOARDING_FINGERPRINT = str(ONBOARDING_MANIFEST["fingerprint"])


class SPAStaticFiles(StaticFiles):
    """Serve the React entry point for client-side routes and refreshes."""

    async def get_response(self, path: str, scope: dict):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code != 404 or Path(path).suffix:
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404 and not Path(path).suffix:
            return await super().get_response("index.html", scope)
        return response


class SummaryMovieEvidence(BaseModel):
    title: str = Field(min_length=1)
    rating: float | None = Field(default=None, ge=0, le=5)
    genres: list[str] = Field(default_factory=list)


class SummaryRequest(BaseModel):
    movies: list[SummaryMovieEvidence] = Field(min_length=1)
    disliked: list[str] = Field(default_factory=list)
    context: str | None = ""


class TEARSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    catalog_fingerprint: str = Field(min_length=64, max_length=64)
    onboarding_fingerprint: str = Field(min_length=64, max_length=64)
    alpha: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=12, ge=1, le=100)
    min_release_year: int = Field(
        default=DEFAULT_MIN_RECOMMENDATION_YEAR, ge=1874, le=2100
    )


class GERSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genres: list[str] = Field(min_length=1, max_length=50)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    catalog_fingerprint: str = Field(min_length=64, max_length=64)
    onboarding_fingerprint: str = Field(min_length=64, max_length=64)
    context: str | None = ""
    alpha: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=12, ge=1, le=100)
    min_release_year: int = Field(
        default=DEFAULT_MIN_RECOMMENDATION_YEAR, ge=1874, le=2100
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = PilotHybridRecommender()
    onboarding_ids = [int(movie["movieId"]) for movie in ONBOARDING_CATALOG]
    app.state.recommender.validate_movie_ids(onboarding_ids, "onboarding catalog")
    if (
        ONBOARDING_MANIFEST.get("matrix_fingerprint")
        != EXPECTED_PILOT_MATRIX_FINGERPRINT
    ):
        raise RuntimeError("Onboarding artifact references a different pilot matrix")
    yield
    app.state.recommender = None


app = FastAPI(title="TEARS Full-Corpus / GERS Pilot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health(request: Request) -> dict:
    status = request.app.state.recommender.status()
    status["onboarding"] = {
        "path": str(ONBOARDING_PATH),
        "manifest_path": str(ONBOARDING_MANIFEST_PATH),
        "fingerprint": EXPECTED_ONBOARDING_FINGERPRINT,
        "items": len(ONBOARDING_CATALOG),
    }
    return status


@app.get("/api/catalog")
def catalog(request: Request) -> dict:
    recommender = request.app.state.recommender
    catalog_by_id = recommender.catalog.set_index("movieId")
    for movie in ONBOARDING_CATALOG:
        row = catalog_by_id.loc[int(movie["movieId"])]
        if (
            int(row.modelItemId) != int(movie["modelItemId"])
            or str(row.title) != str(movie["title"])
            or str(row.genres).split("|") != list(movie["genres"])
        ):
            raise HTTPException(
                status_code=500,
                detail="Onboarding artifact does not match the serving catalog",
            )
    return {
        "catalog_fingerprint": EXPECTED_MATRIX_FINGERPRINT,
        "onboarding_fingerprint": EXPECTED_ONBOARDING_FINGERPRINT,
        "items": ONBOARDING_CATALOG,
    }


def verify_request_provenance(
    catalog_fingerprint: str,
    onboarding_fingerprint: str,
) -> None:
    # The support-20 item catalog and model-item mapping are byte-identical in
    # the pilot and full matrices. Accept the old fingerprint for browser tabs
    # opened before the promotion; the independently verified onboarding
    # fingerprint is still mandatory. New bundles send the full fingerprint.
    compatible_catalog_fingerprints = {
        EXPECTED_MATRIX_FINGERPRINT,
        EXPECTED_PILOT_MATRIX_FINGERPRINT,
    }
    if catalog_fingerprint not in compatible_catalog_fingerprints:
        raise HTTPException(
            status_code=409,
            detail="Frontend catalog fingerprint does not match the serving model",
        )
    if onboarding_fingerprint != EXPECTED_ONBOARDING_FINGERPRINT:
        raise HTTPException(
            status_code=409,
            detail="Frontend onboarding fingerprint does not match the serving artifact",
        )


@app.post("/api/summarize")
def summarize(payload: SummaryRequest) -> dict[str, str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Summary generation is unavailable: OPENAI_API_KEY is not configured.",
        )
    history_lines = []
    for movie in payload.movies:
        rating = "unrated" if movie.rating is None else f"{movie.rating:g}/5"
        history_lines.append(
            f"- title={movie.title!r}; private_rating={rating}; "
            f"genres={'|'.join(movie.genres) or 'Unknown'}"
        )
    if payload.disliked:
        history_lines.append(
            "- explicitly_disliked=" + "|".join(payload.disliked)
        )
    if payload.context and payload.context.strip():
        history_lines.append("- viewing_context=" + payload.context.strip())

    try:
        from openai import OpenAI

        response = OpenAI().responses.create(
            model=SUMMARY_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": render_history_prompt(
                        "\n".join(history_lines), min_words=120, max_words=260
                    ),
                },
            ],
            reasoning={"effort": "minimal"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "viewer_profile",
                    "strict": True,
                    "schema": SUMMARY_SCHEMA,
                },
                "verbosity": "low",
            },
            max_output_tokens=450,
            store=False,
        )
        parsed = json.loads(response.output_text)
        summary = str(parsed["summary"]).strip()
        if not summary:
            raise ValueError("empty summary")
        return {"summary": summary}
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {error}",
        ) from error


@app.post("/api/recommend")
def recommend(payload: TEARSRequest, request: Request) -> dict[str, list[dict]]:
    verify_request_provenance(
        payload.catalog_fingerprint, payload.onboarding_fingerprint
    )
    try:
        items = request.app.state.recommender.recommend_tears(
            summary=payload.summary,
            liked_movie_ids=payload.liked_movie_ids,
            excluded_movie_ids=payload.excluded_movie_ids,
            alpha=payload.alpha,
            top_k=payload.top_k,
            min_release_year=payload.min_release_year,
        )
        return {"items": items}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail="Full-corpus TEARS inference failed"
        ) from error


@app.post("/api/gers")
def gers(payload: GERSRequest, request: Request) -> dict[str, list[dict]]:
    verify_request_provenance(
        payload.catalog_fingerprint, payload.onboarding_fingerprint
    )
    try:
        items = request.app.state.recommender.recommend_gers(
            genres=payload.genres,
            liked_movie_ids=payload.liked_movie_ids,
            excluded_movie_ids=payload.excluded_movie_ids,
            alpha=payload.alpha,
            top_k=payload.top_k,
            min_release_year=payload.min_release_year,
        )
        return {"items": items}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail="Pilot GERS inference failed"
        ) from error


if FRONTEND_BUILD.is_dir():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")
