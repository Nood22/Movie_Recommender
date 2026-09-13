"""API and static site for the promoted full-corpus TEARS and GERS models."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
from math import isclose
import os
from pathlib import Path
import re
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
    validate_text,
)
from tears_training.evidence_gated_summary_harness_v11 import (
    build_separated_evidence as build_online_separated_evidence,
    validate_summary_contract as validate_online_summary_contract,
)
from study_runtime import (
    CANONICAL_CONTEXT,
    CANDIDATE_POLICY_ID,
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
from tears_summary_views import sparse_positive_fallback, synchronize_edit
from tears_preference_ranking import POLICY_ID as TEARS_RANKING_POLICY, explicit_genre_preferences, mentioned_genres as mentioned_online_genres


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_VIEWS_SHA256 = hashlib.sha256((PROJECT_ROOT / "tears_summary_views.py").read_bytes()).hexdigest()
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
SUMMARY_POLICY = "shared-editable-profile-v2"
ONLINE_TASK1A_SYSTEM_PROMPT = """Task: You will now help me generate a detailed summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not comment on ratings or the rating process; use participant-facing qualitative language such as the user likes, the user may prefer, or the user does not enjoy.
Write a natural, editable preference profile in third person so another expert given the same evidence could craft a similar summary. Refer to the participant as "the user," "the viewer," or "they" in the summary.
Use polished, grammatical prose and check subject-verb agreement, especially after the pronoun "they."

Use these four semantic categories as guidance when the evidence supports them, not as mandatory sentences or mandatory claims:
1. Genres the user enjoys.
2. Plot points, characteristics, themes, or content the user enjoys.
3. Genres or styles the user does not enjoy.
4. Plot points, characteristics, themes, or content the user does not enjoy.

Apply this strict grounding contract:
- Evidence at or above 4 supports positive preferences only. Evidence at or below 2 supports negative preferences only. Middle or missing values are neutral or inconclusive.
- Infer positive preferences only from positive evidence, and negative preferences only from negative evidence or an explicit dislike.
- Explicit participant dislikes are authoritative and must be retained. NONE and WEAK describe movie-derived negative evidence only; they never cancel an explicit participant dislike.
- State explicit genre dislikes in their own simple sentence, such as "They dislike horror." Do not describe them as expressed, reported or supported, and do not invent reasons or content aversions from the genre alone.
- A categorical positive genre claim may use only a genre labeled SUPPORTED POSITIVE. A categorical negative genre claim may use only a genre labeled SUPPORTED NEGATIVE. Isolated or insufficient evidence may be omitted or described only as narrow and tentative.
- If SUPPORTED POSITIVE GENRES is empty, do not call any genre a clear preference or say that the user likes or enjoys it. A narrow observation from isolated positive material may use explicitly tentative language such as "may" or "seems"; otherwise omit positive content silently. When supported genres already express that direction, omit every INSUFFICIENT GENRE rather than discussing it neutrally.
- Never infer a dislike from missing positive evidence, and never infer that disliking one element means liking its opposite.
- Apply each item's evidence direction consistently; do not split its genres into opposing preferences.
- Do not turn a genre labeled MIXED into a categorical like or dislike. If mentioned, describe it only as mixed, selective, context-dependent, or varying by execution; do not combine a categorical claim with a later qualification.
- Do not introduce a genre, theme, plot element, style, or viewing need that the supplied evidence does not support.
- Do not infer themes, plot elements, content, or styles merely from a genre label or from generic knowledge about that genre. Include them only when the positive or negative title evidence directly and coherently supports them; otherwise omit that category.
- Follow NEGATIVE EVIDENCE STATUS exactly: NONE means silently omit negative content rather than fabricating a dislike. WEAK means omit the negative inference or describe it once with cautious wording in every clause, such as "the viewer may be less interested in [genre] films, particularly [grounded content]." Under WEAK, do not use categorical forms such as dislikes, avoids, has an aversion to, or does not enjoy. STRONG means describe every REQUIRED NEGATIVE GENRE and only coherent additional negative patterns.
- The four categories may be combined or omitted. Do not fill unsupported positive or negative slots.
- Treat abstention as an internal decision. Silently omit every unsupported direction, genre, theme, plot, content, or style slot. Never describe what is missing, unknown, unclear, insufficient, unavailable, or unsupported when any preference signal can be expressed.
- Only when the entire input has no positive, negative, mixed, or tentative preference signal may the summary contain one short neutral statement that more preference information is needed.
- Do not use audit or meta language such as evidence, data, support, abstention, signal, status, viewing history, input, record, observed, reported, private item, missing, insufficient, indeterminate, unavailable, isolated, tentative, or categorical. Express caution naturally through words such as may, might, or seems.
- Do not optimize for minimum length. Cover all salient supported genres and all distinct grounded narrative, theme, content, and positive-versus-negative patterns. Let semantic coverage determine length; omit unsupported material without compressing supported detail.
- When the supplied preference evidence is sufficiently rich, aim for three to four natural sentences and roughly 90 to 150 words. These are stylistic targets, never permission to pad, repeat a claim, invent an unsupported slot, or fail an otherwise grounded summary.
- Sparse profiles may be shorter than 90 words, but develop the supported preference in natural context rather than returning an extremely short clause dominated by one genre term. When the evidence permits, use two natural sentences to explain the grounded tendency and its meaningful scope or execution. Do not add an opposite preference to create balance.

This exact profile is both shown to the participant and encoded by TEARS to rank
movies. Put the most distinctive grounded preferences first. Use concise natural
language, retaining concrete distinctions in tone, setting, narrative and themes
when the actual titles or explicit participant text justify them. With one liked
title, describe its recognizable content as a possible interest using may or
might, rather than asserting an established general taste or omitting all content.
When a title is unfamiliar or ambiguous, use only supplied metadata; never guess
its plot. Avoid boilerplate, repeated qualifications, and lists of generic genre
associations. For rich histories, preserve distinct preferences instead of reducing
everything to their broadest common genre. Aim to stay below 180 words so the
complete profile fits the encoder; brevity must not erase supported dislikes.

Let the length reflect the evidence: sparse evidence should produce a focused but sufficiently contextualized profile, while rich evidence should produce a fuller profile. Do not pad to reach a word count and do not treat the sentence or word targets as validation requirements.
Begin with the literal prefix `Summary:`. Return only the structured summary field requested by the response schema.
Never include movie titles, actors, years, numeric values, rating metadata, internal evidence labels, or the evidence process in the summary."""
ONLINE_TASK1A_PROMPT_SHA256 = hashlib.sha256(
    ONLINE_TASK1A_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
PILOT_API_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_MIN_RECOMMENDATION_YEAR = YEAR_FILTER_POLICY["value"]
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
    summary_source_request_id: str | None = Field(default=None, min_length=1, max_length=128)
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
    min_release_year: int | None = Field(
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
    min_release_year: int | None = Field(
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
        or deployment.get("candidate_policy_id") != CANDIDATE_POLICY_ID
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
        "candidate_policy_id": CANDIDATE_POLICY_ID,
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
        "candidate_policy_id": CANDIDATE_POLICY_ID,
        "candidate_filter": YEAR_FILTER_POLICY,
    }
    status["online_task1a"] = {
        "summary_policy": SUMMARY_POLICY,
        "backend_prompt_sha256": ONLINE_TASK1A_PROMPT_SHA256,
        "summary_views_sha256": SUMMARY_VIEWS_SHA256,
        "backend_reasoning_effort": "low",
        "edit_reasoning_effort": None,
        "legacy_edit_reasoning_effort": "medium",
        "backend_word_range": None,
        "backend_sentences": None,
        "negative_rating_maximum": 2,
        "source_sha256_at_process_start": PILOT_API_SOURCE_SHA256,
        "system_prompt_sha256": ONLINE_TASK1A_PROMPT_SHA256,
        "model": SUMMARY_MODEL,
        "max_attempts": SUMMARY_MAX_ATTEMPTS,
        "reasoning_effort": "low",
        "text_verbosity": "high",
        "max_output_tokens": 2000,
        "structured_output": "viewer_profile strict JSON schema",
        "store": False,
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
    summary: str,
    recommender: PilotHybridRecommender,
    titles: list[str],
    grounding_evidence: dict[str, Any] | None = None,
) -> list[str]:
    # The shared validator remains frozen for offline summary artifacts. Online
    # Task 1a deliberately drops only its four-sentence and word-count gates;
    # prefix, third-person, privacy, and leakage checks remain unchanged.
    errors = [
        error
        for error in validate_text(summary, recommender.config, titles)
        if not error.startswith(("format_parts:", "word_count:"))
    ]
    if re.search(
        r"\bthey(?:\s+[a-z]+){0,2}\s+(?:is|has|prefers|likes|enjoys|dislikes|avoids|shows|responds|"
        r"tends|appreciates|favors|favours|gravitates)\b",
        summary,
        re.IGNORECASE,
    ):
        errors.append("grammar_subject_verb_agreement")
    if re.search(
        r"\b(?:evidence|data|supports?|supported|abstention|signals?|status|"
        r"viewing history|input|records?|observed|reported|private item|missing|"
        r"insufficient|indeterminate|unavailable|expressed|examples?|profile|note|"
        r"isolated|tentative|categorical)\b",
        summary,
        re.IGNORECASE,
    ):
        errors.append("audit_process_language")
    if grounding_evidence is None:
        return errors
    payload = grounding_evidence["inference_payload"]
    if len(payload["positive_evidence"]) == 1 and not payload["negative_evidence"]:
        genres = set(payload["positive_evidence"][0]["movielens_genres"])
        genres -= set(grounding_evidence.get("explicit_negative_genres", []))
        missing = genres - mentioned_online_genres(summary)
        if missing:
            errors.append("grounding_missing_positive:" + "|".join(sorted(missing)))
    has_preference_signal = bool(
        payload["positive_evidence"]
        or payload["negative_evidence"]
        or payload["mixed_genres"]
        or grounding_evidence.get("explicit_negative_evidence")
    )
    if has_preference_signal and re.search(
        r"\b(?:not enough[^.!?;]{0,80}(?:information|detail|preferences?|likes?|dislikes?)|"
        r"no (?:clear|known|definitive) "
        r"(?:preferences?|likes?|dislikes?)|does not reveal|do not reveal|"
        r"remain(?:s)? unclear|not yet clear|cannot (?:identify|infer)|"
        r"unable to (?:identify|infer)|unknown|remain(?:s)? unspecified|"
        r"(?:has|have) not expressed|additional (?:examples|information)|"
        r"more examples|would (?:help|clarify|refine)|"
        r"more tailored recommendations?|no (?:definite|clear|known)?\s*"
        r"(?:negative |positive )?(?:preferences?|likes?|dislikes?)|"
        r"(?:has|have) not (?:expressed|indicated|shown)|"
        r"not clearly (?:indicated|established)|other [^.!?;]{0,50}preferences?|"
        r"more [^.!?;]{0,40}information|would (?:allow|enable)|"
        r"(?:fuller|balanced) (?:description|profile)|recommendations?)\b",
        summary,
        re.IGNORECASE,
    ):
        errors.append("participant_facing_abstention")
    supported_genres = (set(payload["supported_positive_genres"]) | set(
        payload["supported_negative_genres"]
    )) - set(grounding_evidence.get("explicit_negative_genres", []))
    mentioned_insufficient = set(payload["insufficient_genres"]) & mentioned_online_genres(
        summary
    )
    if supported_genres and mentioned_insufficient:
        errors.append(
            "unsupported_slot_mentioned:" + "|".join(sorted(mentioned_insufficient))
        )
    grounding = validate_online_summary_contract(
        re.sub(r"\badventures\b", "adventure", summary, flags=re.I), grounding_evidence)
    if grounding.missing_core_positive_genres:
        errors.append(
            "grounding_missing_positive:"
            + "|".join(grounding.missing_core_positive_genres)
        )
    if grounding.missing_required_negative_genres:
        errors.append(
            "grounding_missing_negative:"
            + "|".join(grounding.missing_required_negative_genres)
        )
    positive_unsupported = sorted(
        set(grounding.categorical_positive_on_negative_genres)
        | set(grounding.categorical_positive_on_mixed_genres)
        | set(grounding.categorical_positive_on_insufficient_genres)
    )
    if positive_unsupported:
        errors.append("grounding_positive_on_unsupported:" + "|".join(positive_unsupported))
    negative_unsupported = sorted(
        set(grounding.categorical_negative_on_positive_genres)
        | set(grounding.categorical_negative_on_mixed_genres)
        | set(grounding.categorical_negative_on_insufficient_genres)
    )
    if negative_unsupported:
        errors.append("grounding_negative_on_unsupported:" + "|".join(negative_unsupported))
    if grounding.none_fabricated_dislike:
        errors.append("grounding_none_fabricated_dislike")
    if grounding.weak_broad_unsupported_genres:
        errors.append(
            "grounding_weak_broad_dislike:"
            + "|".join(grounding.weak_broad_unsupported_genres)
        )
    if grounding.internal_contract_label_leak:
        errors.append("grounding_internal_label_leak")
    return errors


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


def _online_generation_evidence(
    payload: SummaryRequest,
) -> tuple[list[str], dict[str, Any]]:
    """Separate directions and classify genres before online verbalization."""

    movies = list(payload.movies)
    # V11's audited evidence builder treats 2.5 as negative. The existing online
    # Task 1a contract uses <=2, so normalize every middle value to neutral here.
    normalized_ratings = [
        3.0
        if movie.rating is None or 2 < movie.rating < 4
        else float(movie.rating)
        for movie in movies
    ]
    separated = build_online_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "online-task1a-request",
            "history_items": len(movies),
            "movie_ids": list(range(1, len(movies) + 1)),
            "titles": [movie.title for movie in movies],
            "ratings": normalized_ratings,
            "genres": ["|".join(movie.genres) or "(no genres listed)" for movie in movies],
        }
    )
    separated["explicit_negative_evidence"] = list(payload.disliked)
    # The frozen offline classifier only sees ratings. Add explicit online
    # genre dislikes as their own evidence, without manufacturing rated movies.
    explicit_genres = set().union(*(mentioned_online_genres(value) for value in payload.disliked))
    separated["explicit_negative_genres"] = sorted(explicit_genres)
    inference = separated["inference_payload"]
    # Preserve the release year privately so remakes and ambiguous titles can
    # be identified. The frozen training evidence builder strips it for audits.
    for direction in ("positive_evidence", "negative_evidence"):
        for example in inference[direction]:
            example["source_title"] = movies[example["source_movie_id"] - 1].title
    if explicit_genres:
        for key in ("supported_negative_genres", "required_negative_genres"):
            inference[key] = sorted(set(inference[key]) | explicit_genres)
        for key in ("supported_positive_genres", "core_positive_genres", "secondary_positive_genres",
                    "mixed_genres", "insufficient_genres"):
            inference[key] = sorted(set(inference[key]) - explicit_genres)
        if inference["negative_evidence_status"] == "NONE":
            inference["negative_evidence_status"] = "STRONG"
    lines = [_render_online_inference_payload(separated)]
    if payload.disliked:
        lines.append(
            "EXPLICIT NEGATIVE EVIDENCE (participant supplied; keep claims narrow): "
            + json.dumps(payload.disliked)
        )
    if payload.context and payload.context.strip():
        lines.append("VIEWING CONTEXT: " + payload.context.strip())
    return lines, separated


def _render_online_inference_payload(evidence: dict[str, Any]) -> str:
    """Render only expressible evidence; keep omitted slots out of the prompt."""

    payload = evidence["inference_payload"]

    def examples(label: str, rows: list[dict[str, Any]]) -> str:
        rendered = [label + ":"]
        for index, row in enumerate(rows, 1):
            rendered.append(
                f"- E{index:02d}: {row.get('source_title', row['private_movie_title'])}\n"
                f"  MovieLens genres: {' | '.join(row['movielens_genres']) or '(none)'}"
            )
        return "\n".join(rendered)

    blocks: list[str] = []
    positive = list(payload["positive_evidence"])
    negative = list(payload["negative_evidence"])
    expressible_item_count = len(positive) + len(negative)
    if positive:
        blocks.append(examples("POSITIVE EVIDENCE (positive claims only)", positive))
        if payload["supported_positive_genres"]:
            blocks.append(
                "SUPPORTED POSITIVE GENRES: "
                + json.dumps(payload["supported_positive_genres"])
                + "\nCORE POSITIVE GENRES (required coverage): "
                + json.dumps(payload["core_positive_genres"])
                + "\nSECONDARY POSITIVE GENRES (optional coverage): "
                + json.dumps(payload["secondary_positive_genres"])
            )
        else:
            blocks.append(
                "POSITIVE STATUS: isolated; a narrow positive tendency may be phrased "
                "tentatively, but no categorical positive genre is licensed."
            )
    if negative or evidence.get("explicit_negative_evidence"):
        if negative:
            blocks.append(examples("NEGATIVE EVIDENCE (negative claims only)", negative))
        blocks.append("NEGATIVE EVIDENCE STATUS: " + payload["negative_evidence_status"])
        if payload["required_negative_genres"]:
            blocks.append(
                "SUPPORTED NEGATIVE GENRES: "
                + json.dumps(payload["supported_negative_genres"])
                + "\nREQUIRED NEGATIVE GENRES (complete coverage required): "
                + json.dumps(payload["required_negative_genres"])
            )
        else:
            blocks.append(
                "NEGATIVE STATUS RULE: the negative tendency is isolated; phrase it "
                "once and tentatively, never as a categorical genre dislike."
            )
    if payload["mixed_genres"]:
        blocks.append(
            "MIXED GENRES (describe only as selective or context-dependent): "
            + json.dumps(payload["mixed_genres"])
        )
    if negative and not positive and not payload["mixed_genres"]:
        blocks.append(
            "CASE OUTPUT RULE: This is a negative-only preference profile. Write only "
            "the narrow negative preference and its grounded scope, using cautious "
            "wording throughout, then stop. Do not discuss positive preferences or "
            "request more information. Avoid an extremely short one-clause result; when "
            "the supplied negative material supports it, use two natural sentences "
            "without repeating or broadening the claim."
        )
    elif positive and not negative and not evidence.get("explicit_negative_evidence") and not payload["mixed_genres"]:
        blocks.append(
            "CASE OUTPUT RULE: This is a positive-only preference profile. Write only "
            "the grounded positive preferences. Do not discuss negative preferences or "
            "request more information. Preserve all salient grounded positive details."
        )
    elif positive and negative and not payload["mixed_genres"]:
        blocks.append(
            "CASE OUTPUT RULE: Express the grounded positive preferences and grounded "
            "negative preferences, then stop. Do not discuss any other direction or slot, "
            "request more information, or comment on how complete the description is."
        )
    if not blocks:
        blocks.append(
            "NO PREFERENCE SIGNAL OF ANY KIND: return one short neutral statement asking "
            "for more preference information."
        )
    elif expressible_item_count >= 4:
        blocks.append(
            "RICHNESS TARGET (style only, not a validity gate): This input contains "
            "enough expressible preference material for a fuller natural profile. Aim "
            "for 3-4 sentences and roughly 90-150 words by developing only the supported "
            "genres, coherent title-grounded characteristics, and meaningful positive, "
            "negative, or mixed distinctions already supplied. Never pad, repeat, or "
            "invent content to reach the target."
        )
    else:
        blocks.append(
            "SPARSE STYLE TARGET (style only, not a validity gate): This input may "
            "produce fewer than 90 words. Prefer a focused, sufficiently contextualized "
            "profile, usually in 2 natural sentences when the supplied material supports "
            "that development. Titles and release years are private identifiers for "
            "grounding only: never repeat them in the summary. Describe recognizable "
            "narrative interests when confidently known, retaining the supplied genre "
            "scope in natural prose. Do not return a bare genre clause, repeat a claim, invent "
            "an opposite preference, or add unsupported material for length."
        )
    blocks.append(
        "OUTPUT OMISSION RULE: Do not mention absent directions, omitted classifications, "
        "or anything that cannot be expressed as a grounded preference."
    )
    return "\n\n".join(blocks)


def _summary_generation_prompt(
    history_lines: list[str], previous_errors: list[str] | None = None
) -> str:
    prompt = (
        "Create the participant-facing TEARS preference profile from this private "
        "evidence. Use the evidence only to infer preferences; none of its titles, "
        "numbers, or measurement language may appear in the profile.\n\n"
        + "\n".join(history_lines)
        + "\n\nBefore returning, check that every positive claim comes from positive "
        "evidence, every negative claim comes from negative evidence or an explicit "
        "dislike, mixed evidence is not stated categorically, no opposite preference "
        "has been inferred, and unsupported categories have been omitted silently. "
        "Preserve every salient grounded distinction. Follow the supplied RICHNESS "
        "TARGET or SPARSE STYLE TARGET as a stylistic aim only, without turning length "
        "into a validity condition. Use a short neutral request for more "
        "preference information only if the entire input has no preference signal."
    )
    if not previous_errors:
        return prompt
    rating_leakage_repair = ""
    if "rating_leakage" in previous_errors:
        rating_leakage_repair = (
            " The rating_leakage error means the previous draft exposed how a "
            "preference was measured. Preserve the evidence by expressing the underlying "
            "preference semantically as a like, dislike, or relative preference, or omit "
            "it when unsupported. Do not merely delete a grounded preference claim. The "
            "replacement must "
            "not mention the evidence measurement or use any form of rating, rated, star, "
            "score, scoring, scale, numeric value, or equivalent rating metadata."
        )
    grammar_repair = ""
    if "grammar_subject_verb_agreement" in previous_errors:
        grammar_repair = (
            " Correct third-person plural agreement throughout: write forms such as "
            "'they prefer,' 'they enjoy,' and 'they are,' never 'they prefers,' 'they "
            "enjoys,' or 'they is.'"
        )
    prefix_repair = ""
    if "format_prefix" in previous_errors:
        prefix_repair = (
            " Begin the summary with the literal characters 'Summary:' followed by the "
            "participant-facing preference text."
        )
    silent_abstention_repair = ""
    if "participant_facing_abstention" in previous_errors:
        silent_abstention_repair = (
            " Remove every sentence or clause about preferences that are absent, "
            "unknown, unclear, or not established. Return only the supported or "
            "properly tentative preference content; do not replace the removed text."
        )
    weak_grounding_repair = ""
    if any(
        error.startswith(
            ("grounding_negative_on_unsupported:", "grounding_weak_broad_dislike:")
        )
        for error in previous_errors
    ):
        weak_grounding_repair = (
            " For an insufficient genre under WEAK negative status, use a single narrow "
            "tentative sentence with cautious wording in every clause, following the form "
            "'the viewer may be less interested in [genre] films, particularly [grounded "
            "content].' Substitute only the actual grounded genre and content. Do not say "
            "dislikes, avoids, has an aversion to, does not enjoy, is likely to avoid, or "
            "otherwise state a categorical negative preference."
        )
    audit_language_repair = ""
    if "audit_process_language" in previous_errors:
        audit_language_repair = (
            " Remove all audit and process language. Do not use the words evidence, "
            "data, support, supported, abstention, signal, status, input, record, "
            "observed, reported, missing, insufficient, indeterminate, unavailable, "
            "expressed, example, profile, note, isolated, tentative, categorical, or the "
            "phrase viewing history. State only the natural preference itself, using "
            "may, might, or seems when caution is needed."
        )
    unsupported_slot_repair = ""
    if any(error.startswith("unsupported_slot_mentioned:") for error in previous_errors):
        unsupported_slot_repair = (
            " Remove every mention of the genres named by unsupported_slot_mentioned, "
            "including neutral or qualified mentions. Do not replace those mentions."
        )
    missing_grounded_genre_repair = ""
    if any(error.startswith("grounding_missing_positive:") for error in previous_errors):
        missing_grounded_genre_repair += (
            " Restore natural positive coverage of every genre named by "
            "grounding_missing_positive. Those genres are required, positively grounded "
            "preferences; include each one without exposing the validator label."
        )
    if any(error.startswith("grounding_missing_negative:") for error in previous_errors):
        missing_grounded_genre_repair += (
            " Restore natural negative coverage of every genre named by "
            "grounding_missing_negative. Those genres are required, negatively grounded "
            "preferences; include each one without exposing the validator label or "
            "weakening a STRONG negative direction."
        )
    return (
        prompt
        + "\n\nThe previous draft was rejected by the TEARS privacy and format validator "
        + "for: "
        + ", ".join(previous_errors)
        + ". Generate a fresh replacement from the original private evidence above. "
        + "Preserve preference direction and do not add genres, themes, plot elements, "
        + "or viewing needs that the evidence does not support. Omit unsupported semantic "
        + "categories and every statement about their absence. Preserve all salient "
        + "grounded distinctions. Re-run the complete "
        + "grounding and privacy preflight."
        + rating_leakage_repair
        + grammar_repair
        + prefix_repair
        + silent_abstention_repair
        + weak_grounding_repair
        + audit_language_repair
        + unsupported_slot_repair
        + missing_grounded_genre_repair
    )


def _generate_valid_tears_summary(
    client: Any,
    history_lines: list[str],
    recommender: PilotHybridRecommender,
    titles: list[str],
    attempts: list[dict[str, Any]],
    grounding_evidence: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if grounding_evidence:
        classified = grounding_evidence["inference_payload"]
        if not (classified["positive_evidence"] or classified["negative_evidence"]
                or classified["mixed_genres"] or grounding_evidence.get("explicit_negative_evidence")
                or any("VIEWING CONTEXT:" in line for line in history_lines)):
            summary = "Summary: The viewer is still exploring their movie preferences."
            errors = _summary_validation_errors(summary, recommender, titles, grounding_evidence)
            if not errors:
                attempts.append({"attempt": 1, "kind": "validated_neutral_profile",
                    "word_count": len(summary.split()), "validator_errors": [],
                    "generation_errors": [], "retry_errors": []})
                return summary, attempts
    previous_errors: list[str] | None = None
    final_retry_errors: list[str] = []
    final_validator_errors: list[str] = []
    for attempt_number in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        response = client.responses.create(
            model=SUMMARY_MODEL,
            input=[
                {"role": "system", "content": ONLINE_TASK1A_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _summary_generation_prompt(
                        history_lines, previous_errors
                    ),
                },
            ],
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "viewer_profile",
                    "strict": True,
                    "schema": SUMMARY_SCHEMA,
                },
                "verbosity": "high",
            },
            max_output_tokens=2000,
            store=False,
        )
        parsed = json.loads(response.output_text)
        summary = str(parsed["summary"]).strip()
        final_validator_errors = _summary_validation_errors(
            summary, recommender, titles, grounding_evidence
        )
        word_count = len(summary.split())
        generation_errors: list[str] = []
        final_retry_errors = list(final_validator_errors)
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
                summary, recommender, titles, grounding_evidence
            )
            if final_validator_errors:
                final_retry_errors = list(final_validator_errors)
                previous_errors = final_retry_errors
                continue
            return summary, attempts
        previous_errors = final_retry_errors
    if grounding_evidence and not any("VIEWING CONTEXT:" in line for line in history_lines):
        classified = grounding_evidence["inference_payload"]
        if (len(classified["positive_evidence"]) > 1
                and classified["supported_positive_genres"]
                and not classified["negative_evidence"]
                and not classified["mixed_genres"]
                and not grounding_evidence.get("explicit_negative_evidence")):
            genres = ", ".join(genre.lower() for genre in classified["supported_positive_genres"])
            fallback = f"Summary: The viewer enjoys {genres} films."
            if not _summary_validation_errors(fallback, recommender, titles, grounding_evidence):
                attempts.append({"attempt": len(attempts) + 1, "kind": "validated_positive_genre_fallback",
                    "word_count": len(fallback.split()), "validator_errors": [],
                    "generation_errors": [], "retry_errors": []})
                return fallback, attempts
        fallback = sparse_positive_fallback({
            "classified_evidence": grounding_evidence["inference_payload"],
            "explicit_dislikes": grounding_evidence.get("explicit_negative_evidence", []),
        }, display=True)
        if (fallback and len(classified["positive_evidence"]) == 1
                and not classified["negative_evidence"]
                and not grounding_evidence.get("explicit_negative_evidence")):
            genres = set(classified["positive_evidence"][0]["movielens_genres"])
            if {"Animation", "Children"} <= genres:
                # Preserve the compound scope even when generation fails. A
                # bare list ("animation, children, comedy") loses that relation.
                other = sorted(genres - {"Animation", "Children"})
                detail = " with elements of " + ", ".join(g.lower() for g in other) if other else ""
                fallback = (
                    "Summary: The viewer may enjoy animated, family-friendly films"
                    + detail + ". Their interest could vary from one film to another."
                )
        if fallback and not _summary_validation_errors(fallback, recommender, titles, grounding_evidence):
            attempts.append({"attempt": len(attempts) + 1, "kind": "validated_sparse_fallback",
                "word_count": len(fallback.split()), "validator_errors": [],
                "generation_errors": [], "retry_errors": []})
            return fallback, attempts
    raise HTTPException(
        status_code=422,
        detail=(
            "Generated TEARS summary failed the online privacy and format validator after "
            f"{SUMMARY_MAX_ATTEMPTS} attempts: " + ", ".join(final_retry_errors)
        ),
    )


def _recommendation_summary(payload: TEARSRequest, request: Request):
    source_id = payload.summary_source_request_id
    if source_id is None:
        # Compatibility for saved studies and independently authored profiles.
        return payload.summary, []
    source = request.app.state.study_store.event_by_request(source_id, "summary_result")
    if source is None or any(source[key] != getattr(payload.study, key)
                             for key in ("participant_id", "session_id", "system")):
        raise HTTPException(409, "The source summary does not belong to this study session.")
    saved = source["payload"]
    backend = saved.get("backend_summary")
    if not backend:
        raise HTTPException(409, "Regenerate this profile to enable synchronized summary edits.")
    display = saved["representation"]
    if saved.get("summary_policy") == SUMMARY_POLICY:
        # One inspectable representation: edits are the model input verbatim.
        # Legacy dual-view sources below still preserve their saved hidden text.
        return payload.summary, []
    if payload.summary.strip() == display.strip():
        return backend, []
    cached = request.app.state.study_store.synchronized_summary(
        payload.study, source_id, payload.summary
    )
    if cached:
        return cached["summary"], cached["edit_patches"]
    from openai import OpenAI

    return synchronize_edit(OpenAI(), SUMMARY_MODEL, backend, display, payload.summary)


def _verify_frozen_serving_settings(
    payload: TEARSRequest | GERSRequest, system: str
) -> None:
    if payload.min_release_year != DEFAULT_MIN_RECOMMENDATION_YEAR:
        raise HTTPException(
            status_code=409,
            detail=(
                "The current candidate policy includes all release years; reload and get a new baseline"
            ),
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


def _summary_cache_key(payload: SummaryRequest) -> str:
    return signature({
        "input": summary_input_snapshot(payload),
        "model": SUMMARY_MODEL,
        "summary_policy": SUMMARY_POLICY,
        "prompt_sha256": ONLINE_TASK1A_PROMPT_SHA256,
        "views_sha256": SUMMARY_VIEWS_SHA256,
        # Includes the online evidence renderer, validators, and decoding settings.
        "api_source_sha256": PILOT_API_SOURCE_SHA256,
    })


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
    backend_attempts: list[dict[str, Any]] = []
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
    cache_key = _summary_cache_key(payload)
    cached_summary = request.app.state.study_store.generated_summary(
        payload.study.participant_id, cache_key
    )
    if cached_summary is None and not os.getenv("OPENAI_API_KEY"):
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
    generation_evidence, grounding_evidence = _online_generation_evidence(payload)

    try:
        if cached_summary is None:
            from openai import OpenAI

            summary, generation_attempts = _generate_valid_tears_summary(
                OpenAI(),
                generation_evidence,
                request.app.state.recommender,
                [movie.title for movie in payload.movies],
                generation_attempts,
                grounding_evidence=grounding_evidence,
            )
            cached_summary = request.app.state.study_store.remember_generated_summary(
                payload.study.participant_id, cache_key, summary, payload.study.request_id
            )
        summary = cached_summary["summary"]
        backend_summary = summary
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
                    "generation_evidence": generation_evidence,
                    "summary_model": SUMMARY_MODEL,
                    "backend_prompt_sha256": ONLINE_TASK1A_PROMPT_SHA256,
                    "summary_views_sha256": SUMMARY_VIEWS_SHA256,
                    "max_attempts": SUMMARY_MAX_ATTEMPTS,
                    "initial_verbosity": "high",
                    "generation_length_policy": (
                        "evidence_proportional_rich_supported_coverage_soft_targets"
                    ),
                },
                "representation": summary,
                "backend_summary": backend_summary,
                "backend_generation_attempts": backend_attempts,
                "summary_policy": SUMMARY_POLICY,
                "generation_cache": {
                    "key": cache_key,
                    "reused": cached_summary["source_request_id"] != payload.study.request_id,
                    "source_request_id": cached_summary["source_request_id"],
                },
                "validator": {"valid": True, "errors": []},
                "generation_attempts": generation_attempts,
            },
            latency_ms=latency_ms,
        )
        return {
            "summary": summary,
            "summary_source_request_id": payload.study.request_id,
            "validator": {"valid": True, "errors": []},
            "generation": {
                "attempt_count": len(generation_attempts),
                "retry_count": max(0, len(generation_attempts) - 1),
                "reused": cached_summary["source_request_id"] != payload.study.request_id,
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
                "backend_generation_attempts": backend_attempts,
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
            backend_summary, edit_patches = _recommendation_summary(payload, request)
            if _manual_edit_validation_errors(backend_summary):
                raise HTTPException(422, "The synchronized summary contains invalid text; please retry.")
            effective_input = effective_tears_input(recommender, backend_summary)
            effective_input["display_summary"] = payload.summary
            effective_input["summary_source_request_id"] = payload.summary_source_request_id
            effective_input["edit_patches"] = edit_patches
            effective_input["ranking_policy"] = {
                "id": TEARS_RANKING_POLICY,
                "score_semantics": "model_logit_plus_explicit_compound_genre_priority",
                **explicit_genre_preferences(backend_summary),
            }
            items = recommender.recommend_tears(
                summary=backend_summary,
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
