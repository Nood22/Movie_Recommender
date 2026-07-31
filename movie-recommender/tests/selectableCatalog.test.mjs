import assert from "node:assert/strict";
import test from "node:test";

import {
  deduplicateSelectableCatalog,
  selectableCatalogMovieIds,
} from "../src/utils/selectableCatalog.mjs";

test("deduplicates selectable movies by canonical MovieLens ID", () => {
  const firstForrestGump = {
    movieId: 356,
    title: "Forrest Gump (1994)",
    genres: ["Drama", "Romance"],
  };
  const catalog = [
    firstForrestGump,
    {
      movieId: "356",
      title: "Forrest Gump (1994)",
      genres: ["Comedy", "Romance", "War"],
    },
    { movieId: 1544, title: "Lost World, The (1997)", genres: [] },
    { movieId: 480, title: "Jurassic Park (1993)", genres: [] },
    { movieId: 9991, title: "The Example (1990)", genres: [] },
    { movieId: 9992, title: "The Example (2000)", genres: [] },
    { movieId: 9993, title: "The Example (1990)", genres: [] },
  ];

  const result = deduplicateSelectableCatalog(catalog);

  assert.equal(result.filter((movie) => Number(movie.movieId) === 356).length, 1);
  assert.equal(result[0], firstForrestGump);
  assert.deepEqual(
    result.map((movie) => Number(movie.movieId)),
    [356, 1544, 480, 9991, 9992, 9993]
  );
});

test("returns one canonical numeric exclusion ID per selectable movie", () => {
  assert.deepEqual(
    selectableCatalogMovieIds([
      { movieId: 356, title: "Forrest Gump (1994)" },
      { movieId: "356", title: "Forrest Gump (1994)" },
      { movieId: 2028, title: "Saving Private Ryan (1998)" },
    ]),
    [356, 2028]
  );
});

test("uses exact canonical title and year only when MovieLens ID is missing", () => {
  const firstMissingId = {
    title: "Example, The (1990)",
    genres: ["Drama"],
  };
  const result = deduplicateSelectableCatalog([
    firstMissingId,
    { title: "The Example (1990)", genres: ["Comedy"] },
    { title: "The Example (2000)", genres: ["Drama"] },
  ]);

  assert.deepEqual(result, [
    firstMissingId,
    { title: "The Example (2000)", genres: ["Drama"] },
  ]);
});
