/* global globalThis */

export const PROTOCOL_ID = "tears-gers-human-study-tasks";
export const PROTOCOL_VERSION = "1.0.0";
export const MAX_PROFILE_EDIT_ATTEMPTS = 5;
export const CANONICAL_CONTEXT = "I want to unwind";

export function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export async function inputSignature(value) {
  const bytes = new TextEncoder().encode(stableStringify(value));
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function newStudyId(prefix) {
  const value = globalThis.crypto?.randomUUID?.();
  if (value) return `${prefix}-${value}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function recommendationInputSnapshot(system, payload) {
  return {
    system,
    liked_movie_ids: [...payload.liked_movie_ids],
    preference_evidence: payload.preference_evidence.map((movie) => ({
      movie_id: movie.movie_id,
      rating: movie.rating,
    })),
    excluded_movie_ids: [...payload.excluded_movie_ids],
    catalog_fingerprint: payload.catalog_fingerprint,
    onboarding_fingerprint: payload.onboarding_fingerprint,
    context: (payload.context || "").trim(),
    alpha: payload.alpha,
    top_k: payload.top_k,
    min_release_year: payload.min_release_year,
    representation:
      system === "TEARS" ? payload.summary : [...payload.genres],
  };
}

export function summaryInputSnapshot(payload) {
  return {
    system: "TEARS",
    movies: payload.movies.map((movie) => ({
      title: movie.title,
      rating: movie.rating,
      genres: [...movie.genres],
    })),
    disliked: [...payload.disliked],
    context: (payload.context || "").trim(),
  };
}

export function responseMatchesCurrent(active, responseStudy, currentSignature) {
  return Boolean(
    active &&
      responseStudy &&
      active.request_id === responseStudy.request_id &&
      active.input_signature === responseStudy.input_signature &&
      currentSignature === responseStudy.input_signature
  );
}

export function genreNamesWithFrequencies({
  selectedMovies,
  removedMovieGenres,
  chosenGenres,
  genreList,
}) {
  const movieGenreIds = [];
  for (const movie of selectedMovies) {
    for (const genreId of movie.genre_ids || []) {
      if (!removedMovieGenres.includes(genreId)) movieGenreIds.push(genreId);
    }
  }
  // Duplicates are intentional: each occurrence is one unit in _genre_vector.
  return [...movieGenreIds, ...chosenGenres]
    .map((id) => genreList.find((genre) => genre.id === id)?.name)
    .filter(Boolean);
}

export function targetStatus(trial) {
  if (!trial) return null;
  const latest = trial.history?.at(-1);
  if (!latest) {
    return {
      baseline_rank: trial.baseline_rank,
      baseline_score: trial.baseline_score,
      current_state: "baseline",
      current_rank: trial.baseline_rank,
      current_score: trial.baseline_score,
    };
  }
  return {
    baseline_rank: trial.baseline_rank,
    baseline_score: trial.baseline_score,
    current_state: latest.target_state || latest.status,
    current_rank: latest.target_rank,
    current_score: latest.target_score,
  };
}
