import { canonicalMovieTitleYearKey } from "./tmdbMetadata.mjs";

export function canonicalMovieLensId(movieId) {
  const numericId = Number(movieId);
  return Number.isInteger(numericId) && numericId > 0 ? numericId : null;
}

export function deduplicateSelectableCatalog(records) {
  const seenMovieIds = new Set();
  const seenFallbackTitleYears = new Set();

  return records.filter((record) => {
    const movieId = canonicalMovieLensId(record?.movieId);
    if (movieId !== null) {
      if (seenMovieIds.has(movieId)) return false;
      seenMovieIds.add(movieId);
      return true;
    }

    const titleYear = canonicalMovieTitleYearKey(String(record?.title || ""));
    if (seenFallbackTitleYears.has(titleYear)) return false;
    seenFallbackTitleYears.add(titleYear);
    return true;
  });
}

export function selectableCatalogMovieIds(records) {
  return deduplicateSelectableCatalog(records)
    .map((record) => canonicalMovieLensId(record.movieId))
    .filter((movieId) => movieId !== null);
}
