import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  clearTMDBMetadataCache,
  resolveVerifiedTMDBMetadata,
} from "../src/utils/tmdbMetadata.mjs";

const deployment = JSON.parse(
  await readFile(
    new URL("../src/data/serving_deployment.json", import.meta.url),
    "utf8"
  )
);

test("TEARS and GERS share the all-years selected-only candidate policy", async () => {
  assert.equal(deployment.candidate_filter.value, null);
  assert.equal(deployment.candidate_filter.enabled, false);
  assert.equal(deployment.candidate_policy_id, "all-years-selected-only-v2");
  for (const component of ["TearsApp.jsx", "GersApp.jsx"]) {
    const source = await readFile(
      new URL(`../src/components/${component}`, import.meta.url),
      "utf8"
    );
    assert.match(
      source,
      /const MIN_RECOMMENDATION_YEAR = servingDeployment\.candidate_filter\.value;/
    );
    assert.doesNotMatch(source, /MIN_RECOMMENDATION_YEAR = 20\d\d/);
    assert.match(source, /excluded_movie_ids: \[\]/);
    assert.doesNotMatch(source, /onboardingCatalogMovieIds\.has/);
  }
});

test("poster metadata still resolves by exact title and year without a linked TMDB ID", async () => {
  clearTMDBMetadataCache();
  const httpClient = {
    async get(url, options) {
      assert.ok(url.endsWith("/search/movie"));
      assert.equal(options.params.year, "2015");
      return {
        data: {
          results: [
            {
              id: 12345,
              title: "Fallback Candidate",
              original_title: "Fallback Candidate",
              release_date: "2015-04-03",
              poster_path: "/fallback.jpg",
              overview: "Verified fallback metadata",
              vote_average: 7.2,
            },
          ],
        },
      };
    },
  };

  const metadata = await resolveVerifiedTMDBMetadata({
    movieId: 999999,
    tmdbId: null,
    canonicalTitle: "Fallback Candidate (2015)",
    apiKey: "test",
    httpClient,
  });

  assert.equal(metadata.tmdb_id, 12345);
  assert.equal(metadata.release_year, "2015");
  assert.match(metadata.poster_url, /fallback\.jpg$/);
});
