import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

import {
  deduplicateSelectableCatalog,
  selectableCatalogMovieIds,
} from "../src/utils/selectableCatalog.mjs";

const records = JSON.parse(
  await readFile(
    new URL("../src/data/pilot_support20_onboarding.json", import.meta.url),
    "utf8"
  )
);
const manifest = JSON.parse(
  await readFile(
    new URL(
      "../src/data/pilot_support20_onboarding.manifest.json",
      import.meta.url
    ),
    "utf8"
  )
);

test("pilot onboarding has 100 unique selectable IDs and titles", () => {
  assert.equal(records.length, 100);
  assert.equal(deduplicateSelectableCatalog(records).length, 100);
  assert.equal(new Set(records.map((movie) => movie.movieId)).size, 100);
  assert.equal(new Set(records.map((movie) => movie.title)).size, 100);
  assert.deepEqual(
    selectableCatalogMovieIds(records),
    records.map((movie) => movie.movieId)
  );
  assert.deepEqual(manifest.movie_ids, records.map((movie) => movie.movieId));
  assert.equal(manifest.catalog_size, 100);
});

test("shared onboarding follows exact year quotas and separates same-year neighbors", () => {
  const quotas = {
    2023: 13, 2022: 13, 2021: 12, 2020: 12,
    2019: 5, 2018: 5, 2017: 5, 2016: 5, 2015: 5,
  };
  assert.deepEqual(manifest.year_quotas, quotas);
  assert.deepEqual(manifest.older_movies, { before_year: 2015, count: 25 });
  assert.equal(records.filter(movie => movie.releaseYear < 2015).length, 25);
  assert.ok(records.every(movie => movie.releaseYear <= 2023));
  for (let index = 1; index < records.length; index += 1) {
    assert.notEqual(records[index - 1].releaseYear, records[index].releaseYear);
  }
  for (const [year, count] of Object.entries(quotas)) {
    const rows = records.filter(movie => movie.releaseYear === Number(year));
    assert.equal(rows.length, count, year);
    assert.deepEqual(rows, [...rows].sort((a, b) =>
      b.frozenPopularity - a.frozenPopularity ||
      b.pilotTrainRatingCount - a.pilotTrainRatingCount ||
      a.movieId - b.movieId || a.title.localeCompare(b.title)
    ));
  }
  for (let offset = 0; offset < records.length; offset += 20) {
    const olderCount = records.slice(offset, offset + 20).filter(movie => movie.releaseYear < 2015).length;
    assert.ok(olderCount >= 4 && olderCount <= 6, `Older titles are unevenly clustered near ${offset}`);
  }
  assert.equal(manifest.selection_policy, "popular-year-quotas-v1");
  assert.match(manifest.display_order, /never place equal release years next to each other/);
  assert.match(manifest.popularity_source, /Frozen pilot train_observed/);
  assert.match(manifest.popularity_source, /no live TMDB popularity is used$/);
});

test("recommendation requests are fingerprint-bound to this catalog", async () => {
  for (const relativePath of [
    "../src/components/TearsApp.jsx",
    "../src/components/GersApp.jsx",
  ]) {
    const component = await readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.match(component, /catalog_fingerprint: servingDeployment\.matrix_fingerprint/);
    assert.match(component, /onboarding_fingerprint: pilotManifest\.fingerprint/);
    assert.match(component, /min_release_year: MIN_RECOMMENDATION_YEAR/);
  }
  assert.equal(manifest.matrix_fingerprint.length, 64);
  assert.equal(manifest.fingerprint.length, 64);
});

test("TEARS and GERS share onboarding without excluding unselected catalog entries", async () => {
  const tears = await readFile(
    new URL("../src/components/TearsApp.jsx", import.meta.url),
    "utf8"
  );
  const gers = await readFile(
    new URL("../src/components/GersApp.jsx", import.meta.url),
    "utf8"
  );
  for (const component of [tears, gers]) {
    assert.match(component, /import pilotMovies from "\.\.\/data\/pilot_support20_onboarding\.json"/);
    assert.match(component, /excluded_movie_ids:/);
  }
  assert.match(tears, /excluded_movie_ids: \[\]/);
  assert.match(gers, /excluded_movie_ids: \[\]/);
  assert.deepEqual(
    selectableCatalogMovieIds(records),
    deduplicateSelectableCatalog(records).map((movie) => Number(movie.movieId))
  );
});

test("landing wall is catalog-bound and has no original hard-coded poster list", async () => {
  const landing = await readFile(
    new URL("../src/components/LandingPage.jsx", import.meta.url),
    "utf8"
  );
  assert.match(landing, /import pilotMovies from/);
  assert.match(landing, /LANDING_MOVIES/);
  assert.match(landing, /pilot-posters\/\$\{movie\.movieId\}\.jpg/);
  for (const legacyFilename of [
    "Amelie.jpg",
    "Her.jpg",
    "Interstellar.jpg",
    "Whiplash.jpg",
    "Arrival.jpg",
    "Drive.jpg",
    "ShapeofWater.jpg",
    "GoneGirl.jpg",
    "Dune.jpg",
    "BlackSwan.jpg",
    "lalaland.jpg",
  ]) {
    assert.ok(!landing.includes(legacyFilename), legacyFilename);
  }
});

test("landing poster wall cannot move or displace neighbors on hover", async () => {
  const landing = await readFile(
    new URL("../src/components/LandingPage.jsx", import.meta.url),
    "utf8"
  );
  const desktopWall = landing.slice(
    landing.indexOf("POSTER WALL"),
    landing.indexOf("md:hidden")
  );
  assert.match(desktopWall, /pointer-events-none/);
  assert.match(desktopWall, /<img/);
  assert.doesNotMatch(desktopWall, /whileHover|hover:|translate|scale|rotate[XY]|transform/);
  assert.match(landing, /grid grid-cols-3 gap-3 px-4 opacity-35 md:hidden/);
});

test("all catalog-bound landing posters are bundled locally", async () => {
  const landing = await readFile(new URL("../src/components/LandingPage.jsx", import.meta.url), "utf8");
  assert.match(landing, /const LANDING_MOVIES = pilotMovies\.slice\(0, 11\)/);
  const landingMovies = records.slice(0, 11);
  for (const movie of landingMovies) {
    const poster = await stat(
      new URL(`../public/pilot-posters/${movie.movieId}.jpg`, import.meta.url)
    );
    assert.ok(poster.size > 10_000, `${movie.movieId} poster is unexpectedly small`);
  }
});

test("pilot subpath never falls through to original host-root assets", async () => {
  for (const relativePath of [
    "../src/components/LandingPage.jsx",
    "../src/components/TearsApp.jsx",
    "../src/components/GersApp.jsx",
  ]) {
    const component = await readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.ok(!component.includes('src="/logo.png"'));
    assert.ok(!component.includes('"/placeholder_poster.png"'));
    assert.ok(!component.includes("`/Posters/"));
  }
});

test("catalog metadata uses canonical ML-32M link identifiers", () => {
  for (const movie of records) {
    assert.ok(Number.isInteger(movie.movieId));
    assert.ok(Number.isInteger(movie.tmdbId));
    assert.match(movie.imdbId, /^\d+$/);
    assert.match(movie.title, /\(\d{4}\)$/);
  }
});
