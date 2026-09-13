import assert from "node:assert/strict";
import test from "node:test";

import {
  clearTMDBMetadataCache,
  filterRecommendationRecords,
  recommendationRankMovement,
  resolveMovieLensRecommendation,
  resolveVerifiedTMDBMetadata,
} from "../src/utils/tmdbMetadata.mjs";

test("reports exact up/down rank movement for retained recommendations", () => {
  const previous = [
    { movie_id: 10, title: "First (2020)", rank: 8 },
    { movie_id: 20, title: "Second (2020)", rank: 3 },
  ];
  assert.deepEqual(
    recommendationRankMovement({ movie_id: 10, rank: 2 }, previous),
    { direction: "up", symbol: "↑", positions: 6 }
  );
  assert.deepEqual(
    recommendationRankMovement({ movie_id: 20, rank: 7 }, previous),
    { direction: "down", symbol: "↓", positions: 4 }
  );
  assert.equal(
    recommendationRankMovement({ movie_id: 30, rank: 1 }, previous),
    null
  );
});

test("defensively excludes the complete onboarding catalog", () => {
  const catalog = [{ movieId: 4993, title: "Example (2001)" }];
  const recommendations = [
    { movie_id: 4993, title: "Wrong title (2001)" },
    { movie_id: 999, title: "Example (2001)" },
    { movie_id: 318, title: "Shawshank Redemption, The (1994)" },
  ];
  assert.deepEqual(
    filterRecommendationRecords(recommendations, [], catalog).map(
      (movie) => movie.movie_id
    ),
    [318]
  );
});

test("uses the ML-32M linked TMDB ID before title search", async () => {
  clearTMDBMetadataCache();
  const calls = [];
  const httpClient = {
    async get(url) {
      calls.push(url);
      assert.ok(url.endsWith("/movie/120"));
      return {
        data: {
          id: 120,
          title: "The Lord of the Rings: The Fellowship of the Ring",
          release_date: "2001-12-18",
          release_dates: { results: [] },
          poster_path: "/linked.jpg",
          overview: "Linked metadata",
          vote_average: 8.4,
        },
      };
    },
  };
  const metadata = await resolveVerifiedTMDBMetadata({
    movieId: 4993,
    tmdbId: 120,
    canonicalTitle: "Lord of the Rings: The Fellowship of the Ring, The (2001)",
    apiKey: "test",
    httpClient,
  });
  assert.equal(metadata.tmdb_id, 120);
  assert.equal(metadata.poster_path, "/linked.jpg");
  assert.equal(metadata.rating, 8.4);
  assert.equal(calls.length, 1);
});

test("recommendation cards keep canonical ML-32M and link IDs", () => {
  const card = resolveMovieLensRecommendation(
    {
      movie_id: 4993,
      imdb_id: "0120737",
      tmdb_id: 120,
      title: "Lord of the Rings: The Fellowship of the Ring, The (2001)",
      genres: ["Adventure", "Fantasy"],
      score: 1,
      rank: 1,
      rank_label: "#1",
    },
    { tmdb_id: 120, release_year: "2001", poster_url: "/linked.jpg" }
  );
  assert.equal(card.movie_id, 4993);
  assert.equal(card.imdb_id, "0120737");
  assert.equal(card.tmdb_id, 120);
  assert.equal(card.tmdb_metadata.tmdb_id, 120);
});
