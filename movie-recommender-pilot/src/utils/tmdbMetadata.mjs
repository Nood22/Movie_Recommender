import axios from "axios";

const TMDB_API_BASE = "https://api.themoviedb.org/3";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500";
// Bump whenever matching semantics change so hot-reloaded, pre-fix entries cannot
// be reused. Including the canonical title/year also prevents ID-only poisoning.
const METADATA_CACHE_VERSION = "verified-title-year-v3";
const metadataCache = new Map();

export function splitMovieTitleYear(title) {
  const match = title.match(/\s*\((\d{4})\)\s*$/);
  return {
    title: match ? title.slice(0, match.index).trim() : title.trim(),
    year: match?.[1] || "",
  };
}

export function canonicalSearchTitle(title) {
  const { title: withoutYear } = splitMovieTitleYear(title);
  const articleMatch = withoutYear.match(/^(.*),\s*(The|A|An)$/i);
  return articleMatch
    ? `${articleMatch[2]} ${articleMatch[1]}`
    : withoutYear;
}

export function normalizeTMDBTitle(title) {
  return canonicalSearchTitle(title)
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[\u2018\u2019\u02bc]/g, "'")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function canonicalMovieTitleYearKey(title) {
  const { year } = splitMovieTitleYear(title);
  return `${normalizeTMDBTitle(title)}\u0000${year}`;
}

export function filterRecommendationRecords(
  recommendations,
  selectedMovies,
  excludedCatalogMovies = []
) {
  const excludedMovies = [...selectedMovies, ...excludedCatalogMovies];
  const excludedIds = new Set(
    excludedMovies.map((movie) => String(Number(movie.movieId)))
  );
  const excludedTitleYears = new Set(
    excludedMovies.map((movie) => canonicalMovieTitleYearKey(movie.title))
  );
  const seenIds = new Set();

  return recommendations.filter((recommendation) => {
    const movieId = String(Number(recommendation.movie_id));
    if (
      excludedIds.has(movieId) ||
      excludedTitleYears.has(canonicalMovieTitleYearKey(recommendation.title)) ||
      seenIds.has(movieId)
    ) {
      return false;
    }
    seenIds.add(movieId);
    return true;
  });
}

export function resolveMovieLensRecommendation(item, verifiedTMDBMetadata) {
  const movieId = Number(item?.movie_id);
  if (!Number.isInteger(movieId)) {
    throw new TypeError("Recommendation is missing a canonical MovieLens ID");
  }

  const canonicalTitle = String(item?.title || "").trim();
  const { year } = splitMovieTitleYear(canonicalTitle);
  if (!canonicalTitle || !year) {
    throw new TypeError("Recommendation is missing a canonical MovieLens title/year");
  }

  const metadata =
    verifiedTMDBMetadata?.release_year === year ? verifiedTMDBMetadata : null;
  return {
    movie_id: movieId,
    imdb_id: item.imdb_id || null,
    tmdb_id: Number.isInteger(Number(item.tmdb_id))
      ? Number(item.tmdb_id)
      : null,
    title: canonicalTitle,
    year,
    genres: Array.isArray(item.genres) ? [...item.genres] : [],
    score: item.score,
    rank: item.rank,
    rank_label: item.rank_label,
    score_fmt: Number(item.score).toFixed(2),
    tmdb_metadata: metadata,
    poster_url: metadata?.poster_url || null,
    overview: metadata?.overview || "",
  };
}

export function tmdbSearchVariants(title) {
  const fullTitle = canonicalSearchTitle(title)
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\s*-\s*/g, " - ")
    .replace(/\s+/g, " ")
    .trim();
  const variants = [fullTitle];
  const withoutAliases = fullTitle.replace(/\s*\([^)]*\)\s*/g, " ").trim();
  if (withoutAliases && withoutAliases !== fullTitle) {
    variants.push(withoutAliases);
  }
  const episodeMatch = fullTitle.match(
    /^(.*?):\s*Episode\s+[IVXLCDM]+\s*-\s*(.+)$/i
  );
  if (episodeMatch) {
    variants.push(canonicalSearchTitle(episodeMatch[2]));
    variants.push(canonicalSearchTitle(episodeMatch[1]));
  }

  const seen = new Set();
  return variants.filter((variant) => {
    const normalized = normalizeTMDBTitle(variant);
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

function resultMatches(result, acceptedTitles, expectedYear) {
  const resultYear = (result.release_date || "").split("-")[0];
  if (resultYear !== expectedYear) {
    return { matches: false, reason: `year ${resultYear || "missing"}` };
  }
  const resultTitles = [result.title, result.original_title]
    .filter(Boolean)
    .map(normalizeTMDBTitle);
  if (!resultTitles.some((title) => acceptedTitles.has(title))) {
    return { matches: false, reason: "title variant mismatch" };
  }
  return { matches: true, reason: null };
}

function hasExactReleaseYear(details, expectedYear) {
  if ((details.release_date || "").split("-")[0] === expectedYear) {
    return true;
  }
  return (details.release_dates?.results || []).some((country) =>
    (country.release_dates || []).some(
      (release) => (release.release_date || "").slice(0, 4) === expectedYear
    )
  );
}

function metadataFromTMDBRecord(record, expectedYear) {
  return {
    tmdb_id: Number(record.id),
    tmdb_title: record.title,
    release_year: expectedYear,
    poster_path: record.poster_path || null,
    poster_url: record.poster_path
      ? `${TMDB_IMAGE_BASE}${record.poster_path}`
      : null,
    overview: record.overview || "",
    rating: Number.isFinite(Number(record.vote_average))
      ? Number(record.vote_average)
      : null,
  };
}

export async function resolveVerifiedTMDBMetadata({
  movieId,
  tmdbId = null,
  canonicalTitle,
  apiKey,
  httpClient = axios,
}) {
  const cacheKey = [
    METADATA_CACHE_VERSION,
    String(Number(movieId)),
    String(Number(tmdbId) || "search"),
    canonicalMovieTitleYearKey(canonicalTitle),
  ].join(":");
  if (metadataCache.has(cacheKey)) {
    return metadataCache.get(cacheKey);
  }

  const lookupPromise = (async () => {
    const { year } = splitMovieTitleYear(canonicalTitle);
    const variants = tmdbSearchVariants(canonicalTitle);
    const acceptedTitles = new Set(variants.map(normalizeTMDBTitle));
    let completedSearch = false;
    let transientError = null;

    const linkedTMDBId = Number(tmdbId);
    if (Number.isInteger(linkedTMDBId) && linkedTMDBId > 0) {
      try {
        const detailsResponse = await httpClient.get(
          `${TMDB_API_BASE}/movie/${linkedTMDBId}`,
          {
            params: {
              api_key: apiKey,
              append_to_response: "release_dates",
            },
            timeout: 5000,
          }
        );
        const details = detailsResponse.data;
        if (Number(details.id) === linkedTMDBId && hasExactReleaseYear(details, year)) {
          return metadataFromTMDBRecord(details, year);
        }
      } catch (error) {
        transientError = error;
        console.warn("TEARS linked TMDB lookup error", {
          movieId,
          tmdbId: linkedTMDBId,
          message: error.message,
        });
      }
    }

    for (const query of variants) {
      let results;
      try {
        const response = await httpClient.get(`${TMDB_API_BASE}/search/movie`, {
          params: {
            api_key: apiKey,
            query,
            year,
            include_adult: false,
          },
          timeout: 5000,
        });
        completedSearch = true;
        results = response.data.results || [];
      } catch (error) {
        transientError = error;
        console.warn("TEARS TMDB transient lookup error", {
          movieId,
          canonicalTitle,
          query,
          message: error.message,
        });
        continue;
      }

      for (const result of results) {
        const verification = resultMatches(result, acceptedTitles, year);
        if (!verification.matches) {
          const resultTitles = [result.title, result.original_title]
            .filter(Boolean)
            .map(normalizeTMDBTitle);
          const titleMatches = resultTitles.some((title) =>
            acceptedTitles.has(title)
          );
          if (titleMatches && verification.reason.startsWith("year ")) {
            try {
              const detailsResponse = await httpClient.get(
                `${TMDB_API_BASE}/movie/${result.id}`,
                {
                  params: {
                    api_key: apiKey,
                    append_to_response: "release_dates",
                  },
                  timeout: 5000,
                }
              );
              const details = detailsResponse.data;
              const detailTitles = [details.title, details.original_title]
                .filter(Boolean)
                .map(normalizeTMDBTitle);
              if (
                detailTitles.some((title) => acceptedTitles.has(title)) &&
                hasExactReleaseYear(details, year)
              ) {
                result.title = details.title;
                result.original_title = details.original_title;
                result.poster_path = details.poster_path;
                result.overview = details.overview;
                result.vote_average = details.vote_average;
                result.release_date = `${year}-01-01`;
              }
            } catch (error) {
              transientError = error;
              console.warn("TEARS TMDB details verification error", {
                movieId,
                tmdbId: result.id,
                message: error.message,
              });
            }
          }
        }

        const finalVerification = resultMatches(result, acceptedTitles, year);
        if (!finalVerification.matches) {
          continue;
        }

        return metadataFromTMDBRecord(result, year);
      }
    }

    if (completedSearch) {
      if (transientError) throw transientError;
      return null;
    }
    throw transientError || new Error("TMDB search did not complete");
  })();

  metadataCache.set(cacheKey, lookupPromise);
  try {
    return await lookupPromise;
  } catch (error) {
    metadataCache.delete(cacheKey);
    return null;
  }
}

export function clearTMDBMetadataCache() {
  metadataCache.clear();
}
