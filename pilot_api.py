"""API and static site for the promoted full-corpus TEARS and GERS models."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from math import isclose
import os
from pathlib import Path
import unicodedata
from time import perf_counter
from typing import Any, Literal

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
    validate_text,
)
from study_runtime import (
    CANONICAL_CONTEXT,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    StudyConflict,
    StudyNotFound,
    StudyStore,
    YEAR_FILTER_POLICY,
    effective_gers_input,
    effective_tears_input,
    file_fingerprint,
    immutable_trial_snapshot,
    recommendation_input_snapshot,
    representation_snapshot,
    signature,
    summary_input_snapshot,
    target_observation,
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
SUMMARY_MAX_ATTEMPTS = 3
SUMMARY_TARGET_MIN_WORDS = 140
SUMMARY_TARGET_MAX_WORDS = 180
DEFAULT_MIN_RECOMMENDATION_YEAR = 2020
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "CANONICAL_STUDY_TASK_CONTRACT.md"
DEPLOYMENT_PATH = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "serving_deployment.json"
)
INSTRUMENT_PATH = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "study_instrument.json"
)

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


class RecommendationPreferenceMovie(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int
    rating: float | None = Field(default=None, ge=0, le=5)


class StudyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: Literal[PROTOCOL_ID]
    protocol_version: Literal[PROTOCOL_VERSION]
    participant_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    system: Literal["TEARS", "GERS"]
    task_id: Literal["1a", "1b", "2", "3", "4"]
    trial_id: str | None = Field(default=None, min_length=1, max_length=128)
    attempt: int | None = Field(default=None, ge=1, le=5)
    representation_revision: int = Field(ge=0)
    request_id: str = Field(min_length=1, max_length=128)
    input_signature: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_movie_id: int | None = None


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movies: list[SummaryMovieEvidence] = Field(min_length=1)
    disliked: list[str] = Field(default_factory=list)
    context: str | None = ""
    study: StudyMetadata


class TEARSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    preference_evidence: list[RecommendationPreferenceMovie] = Field(
        default_factory=list, max_length=500
    )
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    catalog_fingerprint: str = Field(min_length=64, max_length=64)
    onboarding_fingerprint: str = Field(min_length=64, max_length=64)
    context: str | None = ""
    alpha: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=12, ge=1, le=100)
    min_release_year: int = Field(
        default=DEFAULT_MIN_RECOMMENDATION_YEAR, ge=1874, le=2100
    )
    study: StudyMetadata


class GERSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genres: list[str] = Field(min_length=1, max_length=500)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    preference_evidence: list[RecommendationPreferenceMovie] = Field(
        default_factory=list, max_length=500
    )
    excluded_movie_ids: list[int] = Field(default_factory=list, max_length=500)
    catalog_fingerprint: str = Field(min_length=64, max_length=64)
    onboarding_fingerprint: str = Field(min_length=64, max_length=64)
    context: str | None = ""
    alpha: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=12, ge=1, le=100)
    min_release_year: int = Field(
        default=DEFAULT_MIN_RECOMMENDATION_YEAR, ge=1874, le=2100
    )
    study: StudyMetadata


class TrialCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_request_id: str = Field(min_length=1, max_length=128)
    target_movie_id: int
    study: StudyMetadata


class TrialCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study: StudyMetadata


class RenderedRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int
    rank: int = Field(ge=1)
    score: float


class RenderEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendations: list[RenderedRecommendation]
    study: StudyMetadata


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_id: str = Field(min_length=1, max_length=128)
    instrument_version: str = Field(min_length=1, max_length=64)
    responses: dict[str, Any]
    representation: str | list[str]
    preference_evidence: list[RecommendationPreferenceMovie] = Field(
        default_factory=list, max_length=500
    )
    context: str | None = ""
    source_request_id: str | None = Field(default=None, min_length=1, max_length=128)
    study: StudyMetadata


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
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    if (
        deployment.get("protocol_id") != PROTOCOL_ID
        or deployment.get("protocol_version") != PROTOCOL_VERSION
        or deployment.get("matrix_fingerprint") != EXPECTED_MATRIX_FINGERPRINT
        or deployment.get("candidate_filter") != YEAR_FILTER_POLICY
        or deployment.get("tears_alpha")
        != {"status": "not_applicable", "request_compatibility_value": 0.5}
        or deployment.get("gers_model") != "gers_recvae"
        or deployment.get("gers_seed") != 2022
        or deployment.get("gers_checkpoint_sha256")
        != "f4755f87b133b95ee6e9919c498011efd2096ee24da3e13895f48e3d30669dea"
    ):
        raise RuntimeError("Deployment manifest differs from the frozen study policy")
    instrument_path = Path(os.getenv("STUDY_INSTRUMENT_PATH", INSTRUMENT_PATH))
    instrument = json.loads(instrument_path.read_text(encoding="utf-8"))
    if (
        instrument.get("protocol_id") != PROTOCOL_ID
        or instrument.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise RuntimeError("Study instrument references a different task contract")
    status = app.state.recommender.status()
    provenance = {
        "protocol": {
            "id": PROTOCOL_ID,
            "version": PROTOCOL_VERSION,
            "sha256": file_fingerprint(PROTOCOL_PATH),
        },
        "deployment": {
            "id": deployment["deployment"],
            "sha256": file_fingerprint(DEPLOYMENT_PATH),
        },
        "instrument": {
            "id": instrument.get("instrument_id"),
            "version": instrument.get("instrument_version"),
            "status": instrument.get("status"),
            "sha256": file_fingerprint(instrument_path),
        },
        "models": status["models"],
        "data": {
            "matrix_fingerprint": EXPECTED_MATRIX_FINGERPRINT,
            "onboarding_fingerprint": EXPECTED_ONBOARDING_FINGERPRINT,
            "catalog_sha256": status["catalog"]["sha256"],
        },
        "candidate_filter": YEAR_FILTER_POLICY,
    }
    app.state.study_instrument = instrument
    app.state.study_store = StudyStore(
        Path(os.getenv("STUDY_DB_PATH", PROJECT_ROOT / "logs" / "pilot_study.sqlite3")),
        provenance,
    )
    yield
    app.state.recommender = None
    app.state.study_store = None


app = FastAPI(title="TEARS / GERS Full-Corpus Recommenders", lifespan=lifespan)
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
    status["study"] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "instrument_configured": bool(
            request.app.state.study_instrument.get("configured")
        ),
        "durable_log_counts": request.app.state.study_store.counts(),
        "candidate_filter": YEAR_FILTER_POLICY,
    }
    return status


@app.get("/api/study/instrument")
def study_instrument(request: Request) -> dict[str, Any]:
    return request.app.state.study_instrument


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


def _verify_study_route(meta: StudyMetadata, system: str) -> None:
    if meta.system != system:
        raise HTTPException(
            status_code=422,
            detail=f"Study system must be {system} for this endpoint",
        )
    if meta.task_id in {"2", "3"}:
        if not meta.trial_id or meta.attempt is None or meta.target_movie_id is None:
            raise HTTPException(
                status_code=422,
                detail="Task 2/3 requests require trial, attempt, and target metadata",
            )
    elif meta.attempt is not None:
        raise HTTPException(
            status_code=422,
            detail="Attempt numbers are only valid for Task 2/3 profile edits",
        )


def _verify_input_signature(meta: StudyMetadata, snapshot: dict[str, Any]) -> None:
    expected = signature(snapshot)
    if meta.input_signature != expected:
        raise HTTPException(
            status_code=409,
            detail="Input signature does not match the submitted immutable request snapshot",
        )


def _study_receipt(meta: StudyMetadata) -> dict[str, Any]:
    return {
        "protocol_id": meta.protocol_id,
        "protocol_version": meta.protocol_version,
        "request_id": meta.request_id,
        "input_signature": meta.input_signature,
        "task_id": meta.task_id,
        "trial_id": meta.trial_id,
        "attempt": meta.attempt,
    }


def _summary_validation_errors(
    summary: str, recommender: PilotHybridRecommender, titles: list[str]
) -> list[str]:
    return validate_text(summary, recommender.config, titles)


def _manual_edit_validation_errors(summary: str) -> list[str]:
    """Apply only transport/inference-safety checks to participant-authored text."""

    errors = []
    if not summary.strip():
        errors.append("empty")
    if any(
        unicodedata.category(character) == "Cc"
        and character not in {"\n", "\r", "\t"}
        for character in summary
    ):
        errors.append("unsupported_control_character")
    return errors


def _summary_generation_prompt(
    history_lines: list[str], previous_errors: list[str] | None = None
) -> str:
    prompt = render_history_prompt(
        "\n".join(history_lines),
        min_words=SUMMARY_TARGET_MIN_WORDS,
        max_words=SUMMARY_TARGET_MAX_WORDS,
    )
    prompt += (
        "\n\nBefore returning, perform this contract preflight without adding unsupported "
        "claims: write roughly 35-45 words in each sentence on the first attempt, "
        "preferably 36-40 words per sentence, and aim near 152 words without going "
        f"outside {SUMMARY_TARGET_MIN_WORDS}-{SUMMARY_TARGET_MAX_WORDS} words total. Sentence "
        "one must contain only supported liked genres, or state that no strong liked "
        "genre is supported. Sentence two must contain only supported liked themes or "
        "content, or state that none is supported. Sentence three must contain only "
        "supported disliked genres or styles, or state that none is supported. Sentence "
        "four must contain only supported disliked plot or content preferences and what "
        "other viewers may enjoy, or state that none is supported. Do not move dislike "
        "claims into sentences one or two, and do not move the viewer's like claims into "
        "sentences three or four. Count the words in the complete four-sentence summary "
        "before returning it. For private evidence direction, values at or below 2 "
        "support dislikes, values at or above 4 support likes, and middle values are "
        "neutral or inconclusive; apply one item's direction consistently rather than "
        "splitting its genres into opposing preferences. Private-rating semantic "
        "conversion is mandatory: use each private value only to infer supported "
        "preference direction and strength, then "
        "express the underlying preference semantically only as a like, dislike, relative "
        "preference, or abstention. Do not explain a conclusion by referring to how the "
        "preference was measured. The summary must not contain the words rating, ratings, rated, star, "
        "stars, score, scores, scoring, scale, or measurement; star counts; numeric "
        "rating values; phrases such as '1-star', '5-star', 'rated X', or 'X/5'; or any "
        "equivalent rating metadata. For sparse or negative-only evidence, reach the "
        "length target with grounded uncertainty and abstention language in unsupported "
        "slots, plus detail about supported negative genres or content where available; "
        "never invent likes or unsupported preferences merely to add words. A dislike "
        "does not imply liking its opposite, so never infer an unstated complementary "
        "preference. When a sentence has no supported preference, expand only on the "
        "limits of the available evidence and the resulting uncertainty."
    )
    if not previous_errors:
        return prompt
    rating_leakage_repair = ""
    if "rating_leakage" in previous_errors:
        rating_leakage_repair = (
            " The rating_leakage error means the previous draft exposed how a "
            "preference was measured. Preserve the evidence by expressing the underlying "
            "preference semantically as a like, dislike, relative preference, or "
            "abstention. Do not merely delete the preference claim. The replacement must "
            "not mention the evidence measurement or use any form of rating, rated, star, "
            "score, scoring, scale, numeric value, or equivalent rating metadata."
        )
    word_count_repair = ""
    if any(
        error.startswith(("word_count:", "generation_word_target:"))
        for error in previous_errors
    ):
        word_count_repair = (
            " The word_count error requires a complete rewrite, not a short correction "
            "or an extension of the rejected draft. Write exactly four complete "
            f"sentences totaling {SUMMARY_TARGET_MIN_WORDS}-{SUMMARY_TARGET_MAX_WORDS} "
            "words, with roughly 35-45 words in every sentence and preferably 36-40, "
            "aim near 152 words, and silently count the words before returning. If "
            "evidence is sparse or negative-only, expand "
            "unsupported slots with grounded uncertainty or abstention language and "
            "expand supported negative preferences only with details grounded in the "
            "supplied genres, explicit dislikes, or context. Do not invent positive or "
            "negative preferences to reach the length target, and do not infer that "
            "disliking one element means liking its opposite."
        )
    return (
        prompt
        + "\n\nThe previous draft was rejected by the unchanged TEARS validator "
        + "for: "
        + ", ".join(previous_errors)
        + ". Generate a fresh replacement from the original private evidence above. "
        + "Preserve preference direction and do not add genres, themes, plot elements, "
        + "or viewing needs that the evidence does not support. Where a required part "
        + "has no support, explicitly say that no strong preference is supported rather "
        + "than filling it with invented detail. Re-run the complete contract preflight "
        + "and return all four sentences."
        + rating_leakage_repair
        + word_count_repair
    )


def _generate_valid_tears_summary(
    client: Any,
    history_lines: list[str],
    recommender: PilotHybridRecommender,
    titles: list[str],
    attempts: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    previous_errors: list[str] | None = None
    final_retry_errors: list[str] = []
    final_validator_errors: list[str] = []
    for attempt_number in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        response = client.responses.create(
            model=SUMMARY_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _summary_generation_prompt(
                        history_lines, previous_errors
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
                "verbosity": "high",
            },
            max_output_tokens=600,
            store=False,
        )
        parsed = json.loads(response.output_text)
        summary = str(parsed["summary"]).strip()
        final_validator_errors = _summary_validation_errors(
            summary, recommender, titles
        )
        word_count = len(summary.split())
        generation_errors = []
        if (
            not final_validator_errors
            and not SUMMARY_TARGET_MIN_WORDS
            <= word_count
            <= SUMMARY_TARGET_MAX_WORDS
        ):
            generation_errors.append(f"generation_word_target:{word_count}")
        final_retry_errors = list(final_validator_errors) + generation_errors
        attempts.append(
            {
                "attempt": attempt_number,
                "kind": "initial" if attempt_number == 1 else "validator_repair",
                "word_count": word_count,
                "validator_errors": list(final_validator_errors),
                "generation_errors": generation_errors,
                "retry_errors": list(final_retry_errors),
            }
        )
        if not final_retry_errors:
            final_validator_errors = _summary_validation_errors(
                summary, recommender, titles
            )
            if final_validator_errors:
                final_retry_errors = list(final_validator_errors)
                previous_errors = final_retry_errors
                continue
            return summary, attempts
        previous_errors = final_retry_errors
    failure_gate = (
        "frozen online validator"
        if final_validator_errors
        else "online generation word target"
    )
    raise HTTPException(
        status_code=422,
        detail=(
            f"Generated TEARS summary failed the {failure_gate} after "
            f"{SUMMARY_MAX_ATTEMPTS} attempts: " + ", ".join(final_retry_errors)
        ),
    )


def _verify_frozen_serving_settings(
    payload: TEARSRequest | GERSRequest, system: str
) -> None:
    if payload.min_release_year != DEFAULT_MIN_RECOMMENDATION_YEAR:
        raise HTTPException(
            status_code=409,
            detail="The frozen candidate policy requires release_year >= 2020",
        )
    if system == "TEARS" and payload.alpha != 0.5:
        raise HTTPException(
            status_code=409,
            detail="TEARS alpha is a fixed non-applicable compatibility value of 0.5",
        )

    evidence_ids = [movie.movie_id for movie in payload.preference_evidence]
    if evidence_ids != list(payload.liked_movie_ids):
        raise HTTPException(
            status_code=422,
            detail="Preference evidence must exactly match liked_movie_ids",
        )
    if system == "TEARS" and any(
        movie.rating is None for movie in payload.preference_evidence
    ):
        raise HTTPException(
            status_code=422,
            detail="TEARS preference evidence requires a rating for every selected movie",
        )


def _evaluation_input_snapshot(payload: EvaluationRequest) -> dict[str, Any]:
    return {
        "instrument_id": payload.instrument_id,
        "instrument_version": payload.instrument_version,
        "responses": payload.responses,
        "representation": payload.representation,
        "preference_evidence": [
            movie.model_dump(mode="json") for movie in payload.preference_evidence
        ],
        "context": (payload.context or "").strip(),
        "source_request_id": payload.source_request_id,
        "system": payload.study.system,
        "task_id": payload.study.task_id,
    }


@app.post("/api/summarize")
def summarize(payload: SummaryRequest, request: Request) -> dict[str, Any]:
    _verify_study_route(payload.study, "TEARS")
    if payload.study.task_id not in {"1a", "4"}:
        raise HTTPException(
            status_code=422, detail="Summary generation is only part of Task 1a or Task 4"
        )
    start = perf_counter()
    prompt_snapshot = summary_input_snapshot(payload)
    generation_attempts: list[dict[str, Any]] = []
    try:
        _verify_input_signature(payload.study, prompt_snapshot)
    except HTTPException as error:
        request.app.state.study_store.log_event(
            "summary_error",
            payload.study,
            "failed",
            {
                "effective_generation_input": prompt_snapshot,
                "generation_attempts": generation_attempts,
            },
            latency_ms=(perf_counter() - start) * 1000,
            error={"status_code": error.status_code, "detail": error.detail},
        )
        raise
    if not os.getenv("OPENAI_API_KEY"):
        error = {
            "type": "configuration_error",
            "message": "OPENAI_API_KEY is not configured",
        }
        request.app.state.study_store.log_event(
            "summary_error",
            payload.study,
            "failed",
            {"effective_generation_input": prompt_snapshot},
            latency_ms=(perf_counter() - start) * 1000,
            error=error,
        )
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

        summary, generation_attempts = _generate_valid_tears_summary(
            OpenAI(),
            history_lines,
            request.app.state.recommender,
            [movie.title for movie in payload.movies],
            generation_attempts,
        )
        latency_ms = (perf_counter() - start) * 1000
        request.app.state.study_store.log_event(
            "summary_result",
            payload.study,
            "completed",
            {
                "preference_evidence": [
                    movie.model_dump(mode="json") for movie in payload.movies
                ],
                "context": (payload.context or "").strip(),
                "effective_generation_input": {
                    **prompt_snapshot,
                    "history_lines": history_lines,
                    "summary_model": SUMMARY_MODEL,
                    "max_attempts": SUMMARY_MAX_ATTEMPTS,
                    "initial_verbosity": "high",
                    "generation_word_target": [
                        SUMMARY_TARGET_MIN_WORDS,
                        SUMMARY_TARGET_MAX_WORDS,
                    ],
                },
                "representation": summary,
                "validator": {"valid": True, "errors": []},
                "generation_attempts": generation_attempts,
            },
            latency_ms=latency_ms,
        )
        return {
            "summary": summary,
            "validator": {"valid": True, "errors": []},
            "generation": {
                "attempt_count": len(generation_attempts),
                "retry_count": len(generation_attempts) - 1,
                "word_count": len(summary.split()),
            },
            "study": _study_receipt(payload.study),
        }
    except HTTPException as error:
        request.app.state.study_store.log_event(
            "summary_error",
            payload.study,
            "failed",
            {
                "effective_generation_input": prompt_snapshot,
                "generation_attempts": generation_attempts,
            },
            latency_ms=(perf_counter() - start) * 1000,
            error={"status_code": error.status_code, "detail": error.detail},
        )
        raise
    except Exception as error:
        request.app.state.study_store.log_event(
            "summary_error",
            payload.study,
            "failed",
            {
                "effective_generation_input": prompt_snapshot,
                "generation_attempts": generation_attempts,
            },
            latency_ms=(perf_counter() - start) * 1000,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {error}",
        ) from error


def _recommendation_error(
    request: Request,
    payload: TEARSRequest | GERSRequest,
    system: str,
    start: float,
    error: Exception,
    reserved_attempt: bool,
) -> None:
    latency_ms = (perf_counter() - start) * 1000
    if isinstance(error, HTTPException):
        error_payload = {"status_code": error.status_code, "detail": error.detail}
    else:
        error_payload = {"type": type(error).__name__, "message": str(error)}
    if reserved_attempt and payload.study.trial_id and payload.study.attempt:
        request.app.state.study_store.fail_attempt(
            payload.study.trial_id,
            payload.study.attempt,
            error_payload,
            latency_ms,
        )
    request.app.state.study_store.log_event(
        "recommendation_error",
        payload.study,
        "failed",
        {
            "request": recommendation_input_snapshot(system, payload),
            "immutable_trial_input": immutable_trial_snapshot(system, payload),
            "representation": representation_snapshot(system, payload),
            "candidate_filter": YEAR_FILTER_POLICY,
        },
        latency_ms=latency_ms,
        error=error_payload,
    )


def _recommend(
    payload: TEARSRequest | GERSRequest,
    request: Request,
    system: Literal["TEARS", "GERS"],
) -> dict[str, Any]:
    start = perf_counter()
    reserved_attempt = False
    try:
        _verify_study_route(payload.study, system)
        if payload.study.task_id not in {"1b", "2", "3", "4"}:
            raise HTTPException(
                status_code=422,
                detail="Recommendations are only valid for Tasks 1b, 2, 3, or 4",
            )
        if system == "GERS" and payload.study.task_id == "4":
            raise HTTPException(
                status_code=409,
                detail=(
                    "GERS Task 4 is blocked: context is not an effective frozen-model input"
                ),
            )
        if system == "GERS" and (payload.context or "").strip():
            raise HTTPException(
                status_code=409,
                detail=(
                    "GERS context is unavailable: the frozen model has no effective "
                    "context input"
                ),
            )
        if (
            system == "TEARS"
            and payload.study.task_id == "4"
            and (payload.context or "").strip() != CANONICAL_CONTEXT
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Task 4 requires the exact context: {CANONICAL_CONTEXT}",
            )
        verify_request_provenance(
            payload.catalog_fingerprint, payload.onboarding_fingerprint
        )
        _verify_frozen_serving_settings(payload, system)
        input_snapshot = recommendation_input_snapshot(system, payload)
        _verify_input_signature(payload.study, input_snapshot)
        immutable = immutable_trial_snapshot(system, payload)
        representation = representation_snapshot(system, payload)
        if payload.study.task_id in {"2", "3"}:
            request.app.state.study_store.reserve_attempt(
                payload.study, immutable, representation
            )
            reserved_attempt = True

        recommender = request.app.state.recommender
        if system == "TEARS":
            assert isinstance(payload, TEARSRequest)
            # Task 1a generation is already gated by the unchanged frozen
            # validator. Recommendation requests may contain participant edits,
            # so they receive only transport/inference-safety validation.
            validation_errors = _manual_edit_validation_errors(payload.summary)
            if validation_errors:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "TEARS summary failed the participant edit "
                        "inference-safety validator: "
                        + ", ".join(validation_errors)
                    ),
                )
            effective_input = effective_tears_input(recommender, payload.summary)
            items = recommender.recommend_tears(
                summary=payload.summary,
                liked_movie_ids=payload.liked_movie_ids,
                excluded_movie_ids=payload.excluded_movie_ids,
                alpha=payload.alpha,
                top_k=payload.top_k,
                min_release_year=payload.min_release_year,
            )
            alpha_semantics = "not_applicable_compatibility_value"
        else:
            assert isinstance(payload, GERSRequest)
            effective_input = effective_gers_input(
                recommender, payload.genres, payload.liked_movie_ids
            )
            items = recommender.recommend_gers(
                genres=payload.genres,
                liked_movie_ids=payload.liked_movie_ids,
                excluded_movie_ids=payload.excluded_movie_ids,
                alpha=payload.alpha,
                top_k=payload.top_k,
                min_release_year=payload.min_release_year,
            )
            alpha_semantics = "frozen_hybrid_blend_input"

        latency_ms = (perf_counter() - start) * 1000
        event_payload = {
            "request": input_snapshot,
            "preference_evidence": {
                "movies": [
                    movie.model_dump(mode="json")
                    for movie in payload.preference_evidence
                ],
                "liked_movie_ids": list(payload.liked_movie_ids),
                "selected_movie_rating_semantics": (
                    "participant_ratings_in_summary_evidence"
                    if system == "TEARS"
                    else "fixed_5.0_nonzero_unresolved_protocol_issue"
                ),
            },
            "representation": representation,
            "effective_model_input": effective_input,
            "immutable_trial_input": immutable,
            "target": (
                {
                    "movie_id": payload.study.target_movie_id,
                    **target_observation(items, payload.study.target_movie_id),
                }
                if payload.study.target_movie_id is not None
                else None
            ),
            "recommendations": items,
            "context": (payload.context or "").strip(),
            "candidate_filter": YEAR_FILTER_POLICY,
            "alpha": {"value": payload.alpha, "semantics": alpha_semantics},
        }
        request.app.state.study_store.log_event(
            "recommendation_result",
            payload.study,
            "completed",
            event_payload,
            latency_ms=latency_ms,
        )
        trial = None
        if reserved_attempt:
            assert payload.study.trial_id is not None
            assert payload.study.attempt is not None
            trial = request.app.state.study_store.complete_attempt(
                payload.study.trial_id,
                payload.study.attempt,
                items,
                latency_ms,
            )
        return {
            "items": items,
            "study": _study_receipt(payload.study),
            "trial": trial,
        }
    except (StudyConflict, StudyNotFound) as error:
        _recommendation_error(
            request, payload, system, start, error, reserved_attempt
        )
        raise HTTPException(status_code=409, detail=str(error)) from error
    except HTTPException as error:
        _recommendation_error(
            request, payload, system, start, error, reserved_attempt
        )
        raise
    except ValueError as error:
        _recommendation_error(
            request, payload, system, start, error, reserved_attempt
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _recommendation_error(
            request, payload, system, start, error, reserved_attempt
        )
        detail = (
            "Full-corpus TEARS inference failed"
            if system == "TEARS"
            else "Full-corpus GERS inference failed"
        )
        raise HTTPException(status_code=500, detail=detail) from error


@app.post("/api/recommend")
def recommend(payload: TEARSRequest, request: Request) -> dict[str, Any]:
    return _recommend(payload, request, "TEARS")


@app.post("/api/gers")
def gers(payload: GERSRequest, request: Request) -> dict[str, Any]:
    return _recommend(payload, request, "GERS")


@app.post("/api/study/trials")
def create_trial(payload: TrialCreateRequest, request: Request) -> dict[str, Any]:
    meta = payload.study
    if meta.task_id not in {"2", "3"} or not meta.trial_id:
        raise HTTPException(
            status_code=422,
            detail="A trial requires Task 2/3 metadata and a stable trial ID",
        )
    if meta.attempt is not None:
        raise HTTPException(status_code=422, detail="A new trial cannot have an attempt")
    if meta.target_movie_id != payload.target_movie_id:
        raise HTTPException(
            status_code=422, detail="Target metadata does not match the selected target"
        )
    trial_snapshot = {
        "system": meta.system,
        "task_id": meta.task_id,
        "trial_id": meta.trial_id,
        "baseline_request_id": payload.baseline_request_id,
        "target_movie_id": payload.target_movie_id,
    }
    _verify_input_signature(meta, trial_snapshot)
    try:
        trial = request.app.state.study_store.create_trial(
            meta, payload.baseline_request_id, payload.target_movie_id
        )
    except (StudyConflict, StudyNotFound) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    request.app.state.study_store.log_event(
        "trial_created",
        meta,
        "active",
        {"trial": trial, "candidate_filter": YEAR_FILTER_POLICY},
    )
    return {"trial": trial, "study": _study_receipt(meta)}


@app.get("/api/study/trials/{trial_id}")
def get_trial(trial_id: str, request: Request) -> dict[str, Any]:
    try:
        return {"trial": request.app.state.study_store.get_trial(trial_id)}
    except StudyNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/study/trials/{trial_id}/complete")
def complete_trial(
    trial_id: str, payload: TrialCloseRequest, request: Request
) -> dict[str, Any]:
    meta = payload.study
    if (
        meta.task_id not in {"2", "3"}
        or meta.trial_id != trial_id
        or meta.attempt is not None
        or meta.target_movie_id is None
    ):
        raise HTTPException(status_code=422, detail="Invalid trial completion metadata")
    close_snapshot = {
        "system": meta.system,
        "task_id": meta.task_id,
        "trial_id": trial_id,
        "target_movie_id": meta.target_movie_id,
        "action": "complete",
    }
    _verify_input_signature(meta, close_snapshot)
    try:
        trial = request.app.state.study_store.close_trial(meta)
    except (StudyConflict, StudyNotFound) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    request.app.state.study_store.log_event(
        "trial_completed", meta, "completed", {"trial": trial}
    )
    return {"trial": trial, "study": _study_receipt(meta)}


@app.post("/api/study/render")
def record_render(payload: RenderEventRequest, request: Request) -> dict[str, Any]:
    source = request.app.state.study_store.event_by_request(
        payload.study.request_id, "recommendation_result"
    )
    if source is None:
        raise HTTPException(status_code=409, detail="No matching model response exists")
    if any(
        (
            source["participant_id"] != payload.study.participant_id,
            source["session_id"] != payload.study.session_id,
            source["system"] != payload.study.system,
            source["task_id"] != payload.study.task_id,
            source["trial_id"] != payload.study.trial_id,
            source["attempt_number"] != payload.study.attempt,
            source["target_movie_id"] != payload.study.target_movie_id,
            source["input_signature"] != payload.study.input_signature,
        )
    ):
        raise HTTPException(
            status_code=409, detail="Rendered response metadata is stale or mismatched"
        )
    rendered = [item.model_dump(mode="json") for item in payload.recommendations]
    model_returned = [
        {
            "movie_id": int(item["movie_id"]),
            "rank": int(item["rank"]),
            "score": float(item["score"]),
        }
        for item in source["payload"]["recommendations"]
    ]
    if len(rendered) != len(model_returned) or any(
        rendered_item["movie_id"] != model_item["movie_id"]
        or rendered_item["rank"] != model_item["rank"]
        or not isclose(
            rendered_item["score"],
            model_item["score"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for rendered_item, model_item in zip(rendered, model_returned)
    ):
        raise HTTPException(
            status_code=409,
            detail="Rendered ordering does not exactly match the current model response",
        )
    request.app.state.study_store.log_event(
        "ui_render",
        payload.study,
        "completed",
        {
            "rendered_recommendations": rendered,
            "source_recommendation_event": source["event_id"],
            "candidate_filter": YEAR_FILTER_POLICY,
        },
    )
    return {"recorded": True, "study": _study_receipt(payload.study)}


@app.post("/api/study/evaluations")
def persist_evaluation(
    payload: EvaluationRequest, request: Request
) -> dict[str, Any]:
    _verify_study_route(payload.study, payload.study.system)
    _verify_input_signature(payload.study, _evaluation_input_snapshot(payload))
    instrument = request.app.state.study_instrument
    if not instrument.get("configured"):
        raise HTTPException(status_code=409, detail=instrument.get("blocker"))
    if (
        payload.instrument_id != instrument.get("instrument_id")
        or payload.instrument_version != instrument.get("instrument_version")
    ):
        raise HTTPException(status_code=409, detail="Study instrument version mismatch")
    task = instrument.get("tasks", {}).get(payload.study.task_id)
    if task is None:
        raise HTTPException(status_code=422, detail="No instrument exists for this task")
    expected_ids = {str(item["id"]) for item in task.get("items", [])}
    response_ids = set(payload.responses)
    if response_ids != expected_ids:
        raise HTTPException(
            status_code=422,
            detail="Questionnaire responses do not match the configured task items",
        )
    for item in task.get("items", []):
        options = item.get("options")
        if options is None:
            continue
        allowed = {option["value"] for option in options}
        if payload.responses[str(item["id"])] not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Response for {item['id']} is outside the configured options",
            )

    source = None
    if payload.study.task_id == "1b":
        if not payload.source_request_id:
            raise HTTPException(
                status_code=422,
                detail="Task 1b evaluation requires its rendered recommendation request",
            )
        source = request.app.state.study_store.event_by_request(
            payload.source_request_id, "recommendation_result"
        )
        rendered_source = request.app.state.study_store.event_by_request(
            payload.source_request_id, "ui_render"
        )
        if source is None or rendered_source is None or any(
            (
                source["participant_id"] != payload.study.participant_id,
                source["session_id"] != payload.study.session_id,
                source["system"] != payload.study.system,
                source["task_id"] != "1b",
                source["payload"]["representation"] != payload.representation,
                source["payload"]["preference_evidence"]["movies"]
                != [
                    movie.model_dump(mode="json")
                    for movie in payload.preference_evidence
                ],
                source["payload"]["context"] != (payload.context or "").strip(),
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="Task 1b evaluation state does not match a rendered baseline",
            )
    evaluation_id = request.app.state.study_store.persist_evaluation(
        payload.study,
        payload.instrument_id,
        payload.instrument_version,
        payload.responses,
    )
    request.app.state.study_store.log_event(
        "questionnaire_response",
        payload.study,
        "completed",
        {
            "evaluation_id": evaluation_id,
            "instrument_id": payload.instrument_id,
            "instrument_version": payload.instrument_version,
            "responses": payload.responses,
            "representation": payload.representation,
            "preference_evidence": [
                movie.model_dump(mode="json") for movie in payload.preference_evidence
            ],
            "context": (payload.context or "").strip(),
            "source_request_id": payload.source_request_id,
            "source_recommendation_event": source["event_id"] if source else None,
        },
    )
    return {
        "evaluation_id": evaluation_id,
        "recorded": True,
        "study": _study_receipt(payload.study),
    }


if FRONTEND_BUILD.is_dir():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")
