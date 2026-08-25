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

test("pilot onboarding IDs are unique and all 50 are excluded", () => {
  assert.equal(records.length, 50);
  assert.equal(deduplicateSelectableCatalog(records).length, 50);
  assert.deepEqual(
    selectableCatalogMovieIds(records),
    records.map((movie) => movie.movieId)
  );
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

test("all catalog-bound landing posters are bundled locally", async () => {
  const releaseYear = (movie) =>
    Number(movie.title.match(/\((\d{4})\)\s*$/)?.[1] || 0);
  const landingMovies = [...records]
    .sort(
      (first, second) =>
        releaseYear(second) - releaseYear(first) ||
        second.pilotTrainPositiveCount - first.pilotTrainPositiveCount ||
        first.movieId - second.movieId
    )
    .slice(0, 11);
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
