import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalMovieTitleYearKey,
  clearTMDBMetadataCache,
  filterRecommendationRecords,
  normalizeTMDBTitle,
  resolveMovieLensRecommendation,
  resolveVerifiedTMDBMetadata,
  tmdbSearchVariants,
} from "../src/utils/tmdbMetadata.mjs";

test("render filtering excludes only selections and exact duplicate IDs", () => {
  const selected = [
    { movieId: 356, title: "Forrest Gump (1994)" },
    { movieId: 480, title: "Jurassic Park (1993)" },
  ];
  const recommendations = [
    { movie_id: 356, title: "Wrong backend title (2000)" },
    { movie_id: 9999, title: "Forrest Gump (1994)" },
    { movie_id: 1544, title: "The Lost World: Jurassic Park (1997)" },
    { movie_id: 2858, title: "American Beauty (1999)" },
    { movie_id: 2858, title: "American Beauty (1999)" },
  ];

  assert.deepEqual(
    filterRecommendationRecords(recommendations, selected).map(
      (movie) => movie.movie_id
    ),
    [1544, 2858]
  );
});

test("render filtering defensively excludes the onboarding catalog", () => {
  const recommendations = [
    { movie_id: 2028, title: "Saving Private Ryan (1998)" },
    { movie_id: 9998, title: "Saving Private Ryan (1998)" },
    { movie_id: 4000, title: "Outside Movie (2001)" },
    { movie_id: 4000, title: "Outside Movie (2001)" },
  ];
  const catalog = [
    { movieId: 2028, title: "Saving Private Ryan (1998)" },
  ];

  assert.deepEqual(
    filterRecommendationRecords(recommendations, [], catalog).map(
      (movie) => movie.movie_id
    ),
    [4000]
  );
});

test("canonical title/year safeguard is exact", () => {
  assert.equal(
    canonicalMovieTitleYearKey("Silence of the Lambs, The (1991)"),
    canonicalMovieTitleYearKey("The Silence of the Lambs (1991)")
  );
  assert.notEqual(
    canonicalMovieTitleYearKey("The Silence of the Lambs (1991)"),
    canonicalMovieTitleYearKey("The Silence of the Lambs (2001)")
  );
  assert.notEqual(
    canonicalMovieTitleYearKey("Jurassic Park (1993)"),
    canonicalMovieTitleYearKey("The Lost World: Jurassic Park (1997)")
  );
});

test("resolves a card atomically from its MovieLens recommendation", () => {
  const resolved = resolveMovieLensRecommendation(
    {
      movie_id: 2571,
      title: "Matrix, The (1999)",
      genres: ["Action", "Sci-Fi"],
      score: 0.75,
      rank: 1,
      rank_label: "#1",
    },
    {
      tmdb_id: 603,
      release_year: "1999",
      poster_url: "https://image.test/matrix.jpg",
      overview: "Verified overview",
    }
  );

  assert.deepEqual(
    {
      id: resolved.movie_id,
      title: resolved.title,
      year: resolved.year,
      genres: resolved.genres,
      tmdbId: resolved.tmdb_metadata.tmdb_id,
    },
    {
      id: 2571,
      title: "Matrix, The (1999)",
      year: "1999",
      genres: ["Action", "Sci-Fi"],
      tmdbId: 603,
    }
  );
});

test("rejects TMDB metadata whose verified year belongs to another record", () => {
  const resolved = resolveMovieLensRecommendation(
    {
      movie_id: 480,
      title: "Jurassic Park (1993)",
      genres: ["Adventure"],
      score: 1,
      rank: 1,
      rank_label: "#1",
    },
    { tmdb_id: 999, release_year: "1997", poster_url: "/wrong.jpg" }
  );
  assert.equal(resolved.tmdb_metadata, null);
  assert.equal(resolved.poster_url, null);
});

test("normalizes MovieLens punctuation and trailing articles", () => {
  assert.equal(normalizeTMDBTitle("Matrix, The (1999)"), "the matrix");
  assert.equal(
    normalizeTMDBTitle("Star Wars: Episode V \u2013 The Empire Strikes Back"),
    "star wars episode v the empire strikes back"
  );
});

test("adds conservative Star Wars episode variants", () => {
  assert.deepEqual(
    tmdbSearchVariants(
      "Star Wars: Episode V - The Empire Strikes Back (1980)"
    ),
    [
      "Star Wars: Episode V - The Empire Strikes Back",
      "The Empire Strikes Back",
      "Star Wars",
    ]
  );
});

test("adds a search variant without MovieLens parenthetical aliases", () => {
  assert.deepEqual(tmdbSearchVariants("Independence Day (ID4) (1996)"), [
    "Independence Day (ID4)",
    "Independence Day",
  ]);
});

test("requires exact accepted title and exact release year", async () => {
  clearTMDBMetadataCache();
  const calls = [];
  const httpClient = {
    async get(url, options) {
      if (!url.endsWith("/search/movie")) {
        return {
          data: {
            id: 999,
            title: "The Empire Strikes Back",
            release_date: "1981-01-01",
            poster_path: "/wrong-year.jpg",
            release_dates: { results: [] },
          },
        };
      }
      calls.push(options.params.query);
      return {
        data: {
          results: [
            {
              id: 999,
              title: "The Empire Strikes Back",
              release_date: "1981-01-01",
              poster_path: "/wrong-year.jpg",
            },
            {
              id: 1891,
              title: "The Empire Strikes Back",
              release_date: "1980-05-17",
              poster_path: "/verified.jpg",
              vote_average: 8.4,
            },
          ],
        },
      };
    },
  };
  const result = await resolveVerifiedTMDBMetadata({
    movieId: 1196,
    canonicalTitle: "Star Wars: Episode V - The Empire Strikes Back (1980)",
    apiKey: "test",
    httpClient,
  });
  assert.equal(result.tmdb_id, 1891);
  assert.equal(result.poster_path, "/verified.jpg");
  assert.equal(result.rating, 8.4);
  assert.deepEqual(calls, ["Star Wars: Episode V - The Empire Strikes Back"]);
});

test("caches confirmed matches by MovieLens ID", async () => {
  clearTMDBMetadataCache();
  let calls = 0;
  const httpClient = {
    async get() {
      calls += 1;
      return {
        data: {
          results: [
            {
              id: 348,
              title: "Alien",
              release_date: "1979-05-25",
              poster_path: "/alien.jpg",
            },
          ],
        },
      };
    },
  };
  const request = {
    movieId: 1214,
    canonicalTitle: "Alien (1979)",
    apiKey: "test",
    httpClient,
  };
  await resolveVerifiedTMDBMetadata(request);
  await resolveVerifiedTMDBMetadata(request);
  assert.equal(calls, 1);
});

test("does not cache transient failures", async () => {
  clearTMDBMetadataCache();
  let calls = 0;
  const httpClient = {
    async get() {
      calls += 1;
      if (calls === 1) throw new Error("temporary");
      return {
        data: {
          results: [
            {
              id: 408,
              title: "Snow White and the Seven Dwarfs",
              release_date: "1937-12-21",
              poster_path: "/snow-white.jpg",
            },
          ],
        },
      };
    },
  };
  const request = {
    movieId: 594,
    canonicalTitle: "Snow White and the Seven Dwarfs (1937)",
    apiKey: "test",
    httpClient,
  };
  assert.equal(await resolveVerifiedTMDBMetadata(request), null);
  assert.equal((await resolveVerifiedTMDBMetadata(request)).tmdb_id, 408);
  assert.equal(calls, 2);
});

test("verifies Snow White 1937 through authoritative release dates", async () => {
  clearTMDBMetadataCache();
  const httpClient = {
    async get(url) {
      if (url.endsWith("/search/movie")) {
        return {
          data: {
            results: [
              {
                id: 408,
                title: "Snow White and the Seven Dwarfs",
                release_date: "1938-01-22",
                poster_path: "/search.jpg",
              },
            ],
          },
        };
      }
      assert.ok(url.endsWith("/movie/408"));
      return {
        data: {
          id: 408,
          title: "Snow White and the Seven Dwarfs",
          original_title: "Snow White and the Seven Dwarfs",
          release_date: "1938-01-22",
          poster_path: "/verified-snow-white.jpg",
          release_dates: {
            results: [
              {
                iso_3166_1: "US",
                release_dates: [{ release_date: "1937-12-21T00:00:00.000Z" }],
              },
            ],
          },
        },
      };
    },
  };
  const result = await resolveVerifiedTMDBMetadata({
    movieId: 594,
    canonicalTitle: "Snow White and the Seven Dwarfs (1937)",
    apiKey: "test",
    httpClient,
  });
  assert.equal(result.tmdb_id, 408);
  assert.equal(result.release_year, "1937");
  assert.equal(result.poster_path, "/verified-snow-white.jpg");
});
