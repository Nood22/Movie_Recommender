import assert from "node:assert/strict";
import test from "node:test";

import {
  genreNamesWithFrequencies,
  inputSignature,
  recommendationInputSnapshot,
  responseMatchesCurrent,
  targetStatus,
} from "../src/study/studyProtocol.mjs";

test("GERS movie-derived genre frequencies survive the frontend payload", () => {
  const genres = genreNamesWithFrequencies({
    selectedMovies: [
      { genre_ids: [18, 35] },
      { genre_ids: [18, 12] },
      { genre_ids: [18] },
    ],
    removedMovieGenres: [35],
    chosenGenres: [12],
    genreList: [
      { id: 12, name: "Adventure" },
      { id: 18, name: "Drama" },
      { id: 35, name: "Comedy" },
    ],
  });

  assert.deepEqual(genres, ["Drama", "Drama", "Adventure", "Drama", "Adventure"]);
});

test("recommendation signatures cover representation and immutable settings", async () => {
  const payload = {
    summary: "Summary: representation",
    liked_movie_ids: [1],
    preference_evidence: [{ movie_id: 1, rating: 5 }],
    excluded_movie_ids: [2],
    catalog_fingerprint: "a".repeat(64),
    onboarding_fingerprint: "b".repeat(64),
    context: "",
    alpha: 0.5,
    top_k: 12,
    min_release_year: 2020,
  };
  const first = await inputSignature(recommendationInputSnapshot("TEARS", payload));
  const second = await inputSignature(
    recommendationInputSnapshot("TEARS", { ...payload, top_k: 13 })
  );
  assert.notEqual(first, second);
  assert.equal(first.length, 64);
});

test("stale responses are rejected even when their request finishes later", () => {
  const current = { request_id: "new", input_signature: "b".repeat(64) };
  const stale = { request_id: "old", input_signature: "a".repeat(64) };
  assert.equal(responseMatchesCurrent(current, stale, current.input_signature), false);
  assert.equal(responseMatchesCurrent(current, current, current.input_signature), true);
  assert.equal(responseMatchesCurrent(current, current, "dirty"), false);
});

test("target history preserves explicit not_returned observations", () => {
  assert.deepEqual(
    targetStatus({
      baseline_rank: 4,
      baseline_score: 0.7,
      history: [
        {
          attempt_number: 1,
          target_state: "not_returned",
          target_rank: null,
          target_score: null,
        },
      ],
    }),
    {
      baseline_rank: 4,
      baseline_score: 0.7,
      current_state: "not_returned",
      current_rank: null,
      current_score: null,
    }
  );
});

test("participant apps do not render the legacy study workspace", async () => {
  const { readFile } = await import("node:fs/promises");
  const participantApps = await Promise.all(
    ["LandingPage.jsx", "TearsApp.jsx", "GersApp.jsx"].map((fileName) =>
      readFile(new URL(`../src/components/${fileName}`, import.meta.url), "utf8")
    )
  );
  const source = participantApps.join("\n");
  assert.doesNotMatch(source, /StudyWorkspace|Study Activities/);
  assert.doesNotMatch(source, /Try moving it higher|Try moving it lower/);
  assert.doesNotMatch(source, /questionnaire|instrument unavailable/i);
});

test("logging remains durable without questionnaire or trial readiness dependencies", async () => {
  const { readFile } = await import("node:fs/promises");
  const session = await readFile(
    new URL("../src/study/useStudySession.js", import.meta.url),
    "utf8"
  );
  const tears = await readFile(
    new URL("../src/components/TearsApp.jsx", import.meta.url),
    "utf8"
  );
  const gers = await readFile(
    new URL("../src/components/GersApp.jsx", import.meta.url),
    "utf8"
  );
  assert.match(session, /durableId\("localStorage"/);
  assert.match(session, /durableId\("sessionStorage"/);
  assert.match(session, /input_signature: await inputSignature\(input\)/);
  assert.match(session, /\/study\/render/);
  assert.doesNotMatch(session, /\/study\/instrument|persistEvaluation|refreshTrial/);
  assert.match(tears, /taskId: "1a"/);
  assert.match(tears, /taskId: "1b"/);
  assert.doesNotMatch(tears, /CANONICAL_CONTEXT|Choose a context/);
  assert.match(gers, /taskId: "1b"/);
  assert.match(gers, /const context = ""/);
  assert.doesNotMatch(gers, /Choose a context|StudyWorkspace/);
});
