import pilotMovies from "../data/pilot_support20_onboarding.json";
import pilotManifest from "../data/pilot_support20_onboarding.manifest.json";
import servingDeployment from "../data/serving_deployment.json";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import {
  filterRecommendationRecords,
  resolveMovieLensRecommendation,
  resolveVerifiedTMDBMetadata,
} from "../utils/tmdbMetadata.mjs";
import {
  canonicalMovieLensId,
  deduplicateSelectableCatalog,
  selectableCatalogMovieIds,
} from "../utils/selectableCatalog.mjs";
import { publicAsset } from "../utils/publicAsset.mjs";
import { apiErrorMessage } from "../utils/apiError.mjs";
import { PILOT_API_URL } from "../utils/apiConfig";
const TEARS_ALPHA = 0.5;
const MIN_RECOMMENDATION_YEAR = 2020;
const TMDB_KEY = process.env.REACT_APP_TMDB_API_KEY || "";
const SELECTABLE_MOVIES = deduplicateSelectableCatalog(pilotMovies);
const ONBOARDING_CATALOG_MOVIE_IDS = selectableCatalogMovieIds(pilotMovies);

function normalizeMovieId(movieId) {
  const numericId = Number(movieId);
  return Number.isInteger(numericId)
    ? String(numericId)
    : String(movieId).trim();
}

/* ---------------------------------------------------------
    TMDB SEARCH + POSTER
----------------------------------------------------------*/
async function fetchPoster(movieId, title, tmdbId = null) {
  return resolveVerifiedTMDBMetadata({
    movieId,
    tmdbId,
    canonicalTitle: title,
    apiKey: TMDB_KEY,
  });
}
/* ---------------------------------------------------------
    MAIN TEARS COMPONENT
----------------------------------------------------------*/
export default function TearsApp({ goBack }) {
  const [movies, setMovies] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);

  const [selected, setSelected] = useState([]);
  const [movieRatings, setMovieRatings] = useState({});
  const [summary, setSummary] = useState("");
  const [summaryError, setSummaryError] = useState("");
  const [recommendationError, setRecommendationError] = useState("");
  const [context, setContext] = useState("");
  const [dislikedGenres] = useState("");
  const [topK, setTopK] = useState(12);

  const [recommendations, setRecommendations] = useState([]);
  const [previousRecommendations, setPrevious] = useState([]);

  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const summaryRequestId = useRef(0);
  const recommendationRequestId = useRef(0);
  const lastRenderDiagnostic = useRef("");
  const completedRecommendationResponse = useRef(null);

  const contextList = [
    "I want a cozy movie for tonight",
    "I want something romantic",
    "I want something thrilling",
    "I want a family-friendly movie",
    "I want something emotional",
    "I want something fun and light",
  ];
  /* ---------------------------------------------------------
      LOAD MOVIES
  ----------------------------------------------------------*/
  useEffect(() => {
  async function loadMoviesWithPosters() {
    const enriched = await Promise.all(
      SELECTABLE_MOVIES.map(async (m) => {
        const poster = await fetchPoster(m.movieId, m.title, m.tmdbId);

        return {
          ...m,
          poster: poster?.poster_url || publicAsset("placeholder_poster.png"),
          overview: poster?.overview || "",
          rating: poster?.rating ?? null,
          year: poster?.release_year || m.title.match(/\((\d{4})\)\s*$/)?.[1] || "",
        };
      })
    );

    setMovies(enriched);
    setLoadingMovies(false);
  }

  loadMoviesWithPosters();
}, []);

  /* ---------------------------------------------------------
      SELECT MOVIES → GPT SUMMARIZER (PATCHED)
  ----------------------------------------------------------*/
const requestSummary = async (
  likedMovies,
  nextContext = context,
  nextRatings = movieRatings
) => {
  const requestId = ++summaryRequestId.current;
  if (likedMovies.length === 0) {
    setSummary("");
    setSummaryError("");
    setSummaryLoading(false);
    return;
  }

  const unratedMovies = likedMovies.filter(
    (movie) => !Number.isFinite(nextRatings[movie.movieId])
  );
  if (unratedMovies.length > 0) {
    setSummary("");
    setSummaryError(
      `All selected movies must be rated. Unrated: ${unratedMovies
        .map((movie) => movie.title)
        .join(", ")}`
    );
    setSummaryLoading(false);
    return;
  }

  setSummary("");
  setSummaryError("");
  setSummaryLoading(true);
  try {
    const res = await axios.post(`${PILOT_API_URL}/summarize`, {
      movies: likedMovies.map((movie) => ({
        title: movie.title,
        rating: nextRatings[movie.movieId],
        genres: movie.genres || [],
      })),
      disliked: dislikedGenres.split(",").map((g) => g.trim()).filter(Boolean),
      context: nextContext,
    });
    if (requestId === summaryRequestId.current) {
      setSummary(res.data.summary || "");
    }
  } catch (err) {
    console.error("TEARS summarizer error", err);
    if (requestId === summaryRequestId.current) {
      setSummaryError(
        err.response?.data?.detail || "Summary generation failed. Please try again."
      );
    }
  } finally {
    if (requestId === summaryRequestId.current) {
      setSummaryLoading(false);
    }
  }
};

const toggleSelect = async (movie) => {
  let updated;
  let nextRatings = movieRatings;
  const selectedMovieId = canonicalMovieLensId(movie.movieId);

  if (
    selected.find(
      (m) => canonicalMovieLensId(m.movieId) === selectedMovieId
    )
  ) {
    updated = selected.filter(
      (m) => canonicalMovieLensId(m.movieId) !== selectedMovieId
    );
    nextRatings = { ...movieRatings };
    delete nextRatings[movie.movieId];
    setMovieRatings(nextRatings);
  } else {
    updated = [...selected, movie];
  }

  setSelected(updated);
  // A selection change makes both visible results and in-flight responses stale.
  recommendationRequestId.current += 1;
  completedRecommendationResponse.current = null;
  setRecommendations([]);
  setPrevious([]);
  setLoading(false);
  await requestSummary(updated, context, nextRatings);
};

const handleRatingChange = async (movieId, value) => {
  const nextRatings = {
    ...movieRatings,
    [movieId]: Number(value),
  };
  setMovieRatings(nextRatings);
  await requestSummary(selected, context, nextRatings);
};

  /* ---------------------------------------------------------
      CONTEXT CHANGE → GPT SUMMARIZER (PATCHED)
  ----------------------------------------------------------*/
  const handleContextChange = async (value) => {
    setContext(value);
    await requestSummary(selected, value, movieRatings);
  };

  /* ---------------------------------------------------------
      REQUEST TEARS RECOMMENDATIONS
  ----------------------------------------------------------*/
  const handleRecommend = async () => {
    if (!summary.trim()) {
      setRecommendationError(
        summaryLoading
          ? "Please wait while your taste summary is being prepared."
          : "Select and rate at least one movie before requesting recommendations."
      );
      return;
    }

    const requestId = ++recommendationRequestId.current;
    setLoading(true);
    setRecommendationError("");
    setRecommendations([]);
    completedRecommendationResponse.current = null;

    try {
      setPrevious(recommendations);

      const payload = {
        summary: summary.trim(),
        liked_movie_ids: selected.map((movie) => Number(movie.movieId)),
        excluded_movie_ids: ONBOARDING_CATALOG_MOVIE_IDS,
        catalog_fingerprint: servingDeployment.matrix_fingerprint,
        onboarding_fingerprint: pilotManifest.fingerprint,
        alpha: TEARS_ALPHA,
        top_k: topK,
        min_release_year: MIN_RECOMMENDATION_YEAR,
      };

      const res = await axios.post(`${PILOT_API_URL}/recommend`, payload);
      const rawItems = res.data.items || [];
      const selectedMovieIds = new Set(
        selected.map((movie) => normalizeMovieId(movie.movieId))
      );
      const onboardingCatalogMovieIds = new Set(
        ONBOARDING_CATALOG_MOVIE_IDS.map(normalizeMovieId)
      );
      const seenRecommendationIds = new Set();
      const items = rawItems.filter((item) => {
        const movieId = normalizeMovieId(item.movie_id);
        if (
          selectedMovieIds.has(movieId) ||
          onboardingCatalogMovieIds.has(movieId) ||
          seenRecommendationIds.has(movieId)
        ) {
          return false;
        }
        seenRecommendationIds.add(movieId);
        return true;
      });

      const enriched = [];
      for (const item of items) {
        const metadata = await fetchPoster(
          item.movie_id,
          item.title,
          item.tmdb_id
        );
        enriched.push(resolveMovieLensRecommendation(item, metadata));
      }

      if (requestId === recommendationRequestId.current) {
        // Replace the previous result set; recommendations are never appended.
        completedRecommendationResponse.current = {
          requestId,
          backendReturned: rawItems.map((movie) => ({
            movie_id: Number(movie.movie_id),
            title: movie.title,
          })),
        };
        setRecommendations(enriched);
      }

    } catch (err) {
      if (requestId === recommendationRequestId.current) {
        console.error("TEARS recommendation error", err);
        setRecommendationError(
          apiErrorMessage(
            err,
            "Recommendations are temporarily unavailable. Please try again."
          )
        );
      }
    }

    if (requestId === recommendationRequestId.current) {
      setLoading(false);
    }
  };

  /* ---------------------------------------------------------
      RANK CHANGE
  ----------------------------------------------------------*/
  function getRankChange(movie) {
    if (!previousRecommendations.length) return null;

    const matchesMovie = (candidate) =>
      movie.movie_id != null && candidate.movie_id != null
        ? candidate.movie_id === movie.movie_id
        : candidate.title === movie.title;
    const prev = previousRecommendations.find(matchesMovie);
    const now = recommendations.find(matchesMovie);

    if (!prev || !now) return null;

    const diff = prev.rank - now.rank;
    if (diff > 0) return `⬆ +${diff}`;
    if (diff < 0) return `⬇ ${diff}`;
    return "–";
  }

  const renderedRecommendations = filterRecommendationRecords(
    recommendations,
    selected,
    SELECTABLE_MOVIES
  );

  if (process.env.NODE_ENV === "development") {
    const completedResponse = completedRecommendationResponse.current;
    const rendered = renderedRecommendations.map((movie) => ({
      movie_id: movie.movie_id,
      title: movie.title,
    }));
    const onboardingIdsForDiagnostic = new Set(
      ONBOARDING_CATALOG_MOVIE_IDS.map(normalizeMovieId)
    );
    const diagnostic = completedResponse && {
      selectedIds: selected.map((movie) => Number(movie.movieId)),
      excludedOnboardingCatalogCount: ONBOARDING_CATALOG_MOVIE_IDS.length,
      backendReturnedIds: completedResponse.backendReturned.map(
        (movie) => movie.movie_id
      ),
      renderedIds: rendered.map((movie) => movie.movie_id),
      selectableCatalogOverlap: rendered
        .filter((movie) =>
          onboardingIdsForDiagnostic.has(normalizeMovieId(movie.movie_id))
        )
        .map((movie) => movie.movie_id),
    };
    const diagnosticSignature = diagnostic
      ? `${completedResponse.requestId}:${JSON.stringify(diagnostic)}`
      : "";
    if (diagnostic && diagnosticSignature !== lastRenderDiagnostic.current) {
      lastRenderDiagnostic.current = diagnosticSignature;
      console.log("TEARS render", diagnostic);
    }
  }

  /* ---------------------------------------------------------
      UI RENDER
  ----------------------------------------------------------*/
  return (
    <div
      className="min-h-screen relative overflow-x-hidden bg-gradient-to-br
      from-[#0d0017] via-[#0b0012] to-black text-white px-4 sm:px-6 py-6 sm:py-8"
    >
      {/* LOGO */}
      <div className="absolute top-4 left-4 sm:top-6 sm:left-8 z-20">
        <motion.img
          src={publicAsset("logo.png")}
          alt="Studio Auréa"
          className="h-14 sm:h-20 drop-shadow-[0_0_35px_#00C8FF55]"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
        />
      </div>

      {/* BACK BUTTON */}
      <button
        onClick={goBack}
        className="absolute top-4 right-4 sm:top-6 sm:right-6 z-20 bg-white/5 border border-[#00C8FF55]
          px-4 sm:px-5 py-2 rounded-2xl text-sm sm:text-lg backdrop-blur-xl
          shadow-[0_0_20px_#00C8FF55] hover:shadow-[0_0_35px_#00C8FFAA]
          hover:border-[#00C8FF]"
      >
        ← Back
      </button>

      {/* TITLE */}
      <div className="max-w-[1600px] mx-auto mt-28 sm:mt-32 mb-8 sm:mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#80E7FF]/80 mb-3">
          Full 200,948-profile dataset · 180,948 training users
        </p>
        <h1 className="text-3xl sm:text-4xl font-bold text-[#A7E8FF] drop-shadow-[0_0_20px_#00C8FF55]">
          TEARS – Summary-Based Recommender
        </h1>
      </div>

      <div className="max-w-[1600px] mx-auto flex flex-col xl:flex-row gap-8 xl:gap-10">
        {/* ------------------------------------------------------
            LEFT PANEL — MOVIE SELECTION
        --------------------------------------------------------*/}
        <div className="w-full xl:w-3/5">
          <h2 className="text-lg font-semibold mb-3">Select movies you like:</h2>

          {loadingMovies ? (
            <div className="text-center text-gray-400 py-10">Loading…</div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3 sm:gap-4">
              {movies.map((m) => (
                <motion.button
                  key={String(m.movieId)}
                  type="button"
                  aria-pressed={selected.some(
                    (s) =>
                      canonicalMovieLensId(s.movieId) ===
                      canonicalMovieLensId(m.movieId)
                  )}
                  onClick={() => toggleSelect(m)}
                  whileHover={{ scale: 1.08 }}
                  className={`relative cursor-pointer rounded-xl p-[3px] text-left transition-all
                    ${
                      selected.find(
                        (s) =>
                          canonicalMovieLensId(s.movieId) ===
                          canonicalMovieLensId(m.movieId)
                      )
                        ? "ring-2 ring-[#00C8FF] shadow-[0_0_20px_#00C8FF99]"
                        : "ring-1 ring-white/10 hover:ring-[#00C8FF88] hover:shadow-[0_0_20px_#00C8FF55]"
                    }`}
                >
                  <img
                    src={m.poster}
                    alt={m.title}
                    onError={(event) => {
                      event.currentTarget.onerror = null;
                      event.currentTarget.src = publicAsset("placeholder_poster.png");
                    }}
                    className="w-full aspect-[2/3] object-cover rounded-lg"
                  />

                  <p className="text-sm mt-1 text-center md:line-clamp-3">{m.title}</p>

                  {/* HOVER CARD */}
                  <div
                    className="absolute inset-0 opacity-0 hover:opacity-100
                      bg-black/80 backdrop-blur-md rounded-xl flex flex-col
                      justify-between p-3 z-20 transition-all duration-300
                      hover:-translate-y-1"
                  >
                    <div>
                      <h3 className="font-bold text-sm mb-1">{m.title}</h3>
                      <p className="text-[11px] text-yellow-300">
                        ⭐ {m.rating || "N/A"}
                      </p>
                      <p className="text-[10px] text-gray-300 mt-1">{m.year}</p>

                      <div className="flex flex-wrap gap-1 mt-2">
                        {m.genres?.slice(0, 3).map((g, idx) => (
                          <span
                            key={idx}
                            className="text-[9px] px-2 py-[2px] bg-white/10
                              rounded-full border border-white/10"
                          >
                            {g}
                          </span>
                        ))}
                      </div>
                    </div>

                    <p className="text-[10px] text-gray-200 line-clamp-4 mt-2">
                      {m.overview}
                    </p>
                  </div>
                </motion.button>
              ))}
            </div>
          )}
        </div>

        {/* ------------------------------------------------------
            RIGHT PANEL — SUMMARY + RECOMMENDATIONS
        --------------------------------------------------------*/}
        <div className="w-full xl:w-2/5 xl:sticky xl:top-6 self-start bg-white/5 border border-[#00C8FF] rounded-3xl p-4 sm:p-5
            backdrop-blur-xl shadow-[0_0_30px_#00C8FF55]">

          {/* SUMMARY */}
          {selected.length > 0 && (
            <div className="mb-4">
              <h2 className="text-lg font-semibold mb-2">Rate selected movies:</h2>
              {selected.map((movie) => (
                <label
                  key={movie.movieId}
                  className="flex items-center justify-between gap-3 mb-2 text-sm"
                >
                  <span>{movie.title}</span>
                  <select
                    aria-label={`Rating for ${movie.title}`}
                    value={movieRatings[movie.movieId] ?? ""}
                    onChange={(event) =>
                      handleRatingChange(movie.movieId, event.target.value)
                    }
                    className="bg-white/10 border border-[#00C8FF55] rounded-lg px-2 py-1 text-white"
                  >
                    <option value="" disabled className="text-black">
                      Unrated
                    </option>
                    {[1, 2, 3, 4, 5].map((rating) => (
                      <option key={rating} value={rating} className="text-black">
                        {rating} {rating === 1 ? "star" : "stars"}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          )}
          {summaryError && (
            <p role="alert" className="text-sm text-red-300 mb-3">
              {summaryError}
            </p>
          )}
          <h2 className="text-lg font-semibold mb-2">Your Summary</h2>
          {summaryLoading && (
            <p className="text-xs text-[#80E7FF] mb-2">
              Refining your instant summary in the background…
            </p>
          )}
          <textarea
            className="w-full h-64 bg-white/5 border border-white/10 p-3
              rounded-lg text-sm mb-4"
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
          />

          {/* Genre exclusion UI is intentionally deferred. Keep the state and
              request field in place so the feature can be restored later. */}

          {/* CONTEXT */}
          <h2 className="text-lg font-semibold mb-2">Choose a context:</h2>
          <select
            value={context}
            onChange={(e) => handleContextChange(e.target.value)}
            className="w-full py-3 px-3 rounded-lg mb-6 bg-white/10
              border border-[#00C8FF55] text-white transition-all
              focus:border-[#00C8FF] focus:shadow-[0_0_25px_#00C8FF88]"
          >
            <option value="" className="text-black">
              Select a context…
            </option>
            {contextList.map((c) => (
              <option key={c} value={c} className="text-black">
                {c}
              </option>
            ))}
          </select>

          <div className="mb-6">
            <label htmlFor="tears-top-k" className="text-sm font-semibold">
              Number of recommendations: {topK}
            </label>
            <input
              id="tears-top-k"
              type="range"
              min="1"
              max="25"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full mt-2"
            />
          </div>

          {/* BUTTON */}
          <button
            onClick={handleRecommend}
            disabled={loading || summaryLoading || !summary.trim()}
            className="w-full py-3 rounded-xl font-semibold text-lg bg-[#0A0A0A]
              border border-[#00C8FF] shadow-[0_0_25px_#00C8FF55]
              hover:shadow-[0_0_40px_#00C8FFAA] transition-all
              disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            {loading
              ? "Finding your movies…"
              : summaryLoading
                ? "Preparing your summary…"
                : "Get Recommendations"}
          </button>

          {recommendationError && (
            <p role="alert" className="mt-3 text-sm text-red-300">
              {recommendationError}
            </p>
          )}

          {/* RESULTS */}
          {renderedRecommendations.length > 0 && (
            <div className="mt-6">
              <h2 className="text-md font-bold mb-3 text-[#80E7FF]">
                Recommended Movies
              </h2>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
                {renderedRecommendations.map((m) => {
                  const resolvedMovieId = m.movie_id;
                  return (
                  <motion.div
                    key={String(resolvedMovieId)}
                    whileHover={{ scale: 1.07 }}
                    className="relative bg-white/10 backdrop-blur-lg rounded-xl p-2
                      hover:shadow-[0_0_20px_#00C8FF55] transition-all"
                  >
                    {m.poster_url ? (
                      <img
                        src={m.poster_url}
                        alt={m.title}
                        onError={(event) => {
                          event.currentTarget.onerror = null;
                          event.currentTarget.src = publicAsset("placeholder_poster.png");
                        }}
                        className="w-full aspect-[2/3] object-cover rounded-lg"
                      />
                    ) : (
                      <div className="aspect-[2/3] flex items-center justify-center bg-gray-700
                        rounded-lg text-3xl">
                        🎬
                      </div>
                    )}

                    <p className="mt-2 font-semibold text-sm">{m.title}</p>

                    {/* Hover Details */}
                    <div
                      className="absolute inset-0 opacity-0 hover:opacity-100
                        bg-black/85 backdrop-blur-md text-white p-3 flex flex-col
                        justify-between rounded-xl transition-all duration-300
                        hover:-translate-y-1"
                    >
                      <div>
                        <h3 className="font-bold text-sm mb-1">{m.title}</h3>

                        <p className="text-[12px] text-green-300 font-bold">
                          {m.rank_label}{" "}
                          {getRankChange(m) && (
                            <span className="ml-1 text-purple-300">
                              ({getRankChange(m)})
                            </span>
                          )}
                        </p>

                        <p className="text-[11px] text-yellow-300 mt-1">
                          Score: {m.score_fmt}
                        </p>

                        <p className="text-[10px] text-gray-300 mt-1">
                          {m.year}
                        </p>

                        <div className="flex flex-wrap gap-1 mt-2">
                          {m.genres?.slice(0, 3).map((g, idx) => (
                            <span
                              key={idx}
                              className="text-[9px] px-2 py-[2px] bg-white/10
                                rounded-full border border-white/10"
                            >
                              {g}
                            </span>
                          ))}
                        </div>
                      </div>

                      <p className="text-[10px] text-gray-200 line-clamp-4 mt-2">
                        {m.overview}
                      </p>
                    </div>
                  </motion.div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
