import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import pilotMovies from "../data/pilot_support20_onboarding.json";
import pilotManifest from "../data/pilot_support20_onboarding.manifest.json";
import servingDeployment from "../data/serving_deployment.json";
import {
  resolveMovieLensRecommendation,
  resolveVerifiedTMDBMetadata,
} from "../utils/tmdbMetadata.mjs";
import { deduplicateSelectableCatalog } from "../utils/selectableCatalog.mjs";
import { publicAsset } from "../utils/publicAsset.mjs";
import { apiErrorMessage } from "../utils/apiError.mjs";
import { PILOT_API_URL } from "../utils/apiConfig";
import { useStudySession } from "../study/useStudySession";
import {
  genreNamesWithFrequencies,
  recommendationInputSnapshot,
  responseMatchesCurrent,
} from "../study/studyProtocol.mjs";


const TMDB_KEY = process.env.REACT_APP_TMDB_API_KEY || "";
const MIN_RECOMMENDATION_YEAR = servingDeployment.candidate_filter.value;
const GERS_ALPHA = 0.5;
const FIXED_MOVIELENS_CATALOG = deduplicateSelectableCatalog(pilotMovies);

function normalizeGenre(g) {
  return g
    .replace(/-/g, " ")
    .replace(/science fiction/i, "Science Fiction")
    .replace(/sci fi/i, "Sci-Fi")
    .replace(/sci-fi/i, "Sci-Fi")
    .trim();
}

/* ---------------------------------------------------------
    TMDB GENRE MAP
----------------------------------------------------------*/
const GENRE_MAP = {
  "Action": 28,
  "Adventure": 12,
  "Animation": 16,
  "Comedy": 35,
  "Crime": 80,
  "Documentary": 99,
  "Drama": 18,
  "Family": 10751,
  "Fantasy": 14,
  "History": 36,
  "Horror": 27,
  "Music": 10402,
  "Mystery": 9648,
  "Romance": 10749,
  "Science Fiction": 878,
  "Sci-Fi": 878,
  "Sci Fi": 878,
  "SciFi": 878,
  "Thriller": 53,
  "War": 10752,
  "Western": 37,
  "TV Movie": 10770
};

/* -----------------------------------------------
   TMDB POSTER FETCHER
------------------------------------------------ */
async function fetchPoster(movieId, title, tmdbId = null) {
  return resolveVerifiedTMDBMetadata({
    movieId,
    tmdbId,
    canonicalTitle: title,
    apiKey: TMDB_KEY,
  });
}

/* -----------------------------------------------
   SAFE GENRE NAME
------------------------------------------------ */
function safeGenreName(gid, genreList) {
  const item = genreList.find((g) => g.id === gid);
  return item ? item.name : "Unknown";
}

/* -----------------------------------------------
   MAIN GERS APP
------------------------------------------------ */
export default function GersApp({ goBack }) {
  const study = useStudySession("GERS");
  const [movies, setMovies] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);

  const [selected, setSelected] = useState([]);
  const [chosenGenres, setChosenGenres] = useState([]);
  const [removedMovieGenres, setRemovedMovieGenres] = useState([]);

  const context = "";
  const [recommendations, setRecommendations] = useState([]);
  const [topK, setTopK] = useState(12);
  const [loading, setLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");
  const representationRevisionRef = useRef(0);
  const recommendationRequestId = useRef(0);
  const activeRecommendationRequest = useRef(null);
  const currentRecommendationSignature = useRef("dirty-initial");

  const bumpRepresentationRevision = () => {
    representationRevisionRef.current += 1;
  };

  const invalidateRecommendations = () => {
    recommendationRequestId.current += 1;
    activeRecommendationRequest.current = null;
    currentRecommendationSignature.current = `dirty-${recommendationRequestId.current}`;
    setRecommendations([]);
    setLoading(false);
  };

  /* FIXED GENRE SET */
  const genreList = [
  { id: 28, name: "Action" },
  { id: 12, name: "Adventure" },
  { id: 16, name: "Animation" },
  { id: 35, name: "Comedy" },
  { id: 80, name: "Crime" },
  { id: 99, name: "Documentary" },
  { id: 18, name: "Drama" },
  { id: 10751, name: "Family" },
  { id: 14, name: "Fantasy" },
  { id: 36, name: "History" },
  { id: 27, name: "Horror" },
  { id: 10402, name: "Music" },
  { id: 9648, name: "Mystery" },
  { id: 10749, name: "Romance" },
  { id: 878, name: "Science Fiction" },
  { id: 53, name: "Thriller" },
  { id: 10752, name: "War" },
  { id: 37, name: "Western" },
  { id: 10770, name: "TV Movie" }
];


  /* -----------------------------------------------
     LOAD MOVIES
  ------------------------------------------------ */

/* -----------------------------------------------
   LOAD MOVIES — FIXED LIST WITH GENRE IDS
------------------------------------------------ */
useEffect(() => {
  let cancelled = false;

  async function loadFixedMovieLensCatalog() {
    setLoadingMovies(true);
    const mapped = await Promise.all(
      FIXED_MOVIELENS_CATALOG.map(async (movie) => {
        const metadata = await fetchPoster(
          movie.movieId,
          movie.title,
          movie.tmdbId
        );
        return {
          id: movie.movieId,
          movieId: movie.movieId,
          title: movie.title,
          poster: metadata?.poster_url || publicAsset("placeholder_poster.png"),
          overview: metadata?.overview || "No summary available.",
          rating: metadata?.rating ?? null,
          year:
            metadata?.release_year ||
            movie.title.match(/\((\d{4})\)\s*$/)?.[1] ||
            "",
          genre_ids: movie.genres
            .map((genre) => GENRE_MAP[normalizeGenre(genre)] || null)
            .filter(Boolean),
        };
      })
    );

    if (!cancelled) {
      setMovies(mapped);
      setLoadingMovies(false);
    }
  }

  loadFixedMovieLensCatalog();
  return () => {
    cancelled = true;
  };
}, []);


  /* -----------------------------------------------
     SELECT MOVIE
  ------------------------------------------------ */
  const toggleSelect = (movie) => {
    let updated = selected.find((s) => s.id === movie.id)
      ? selected.filter((s) => s.id !== movie.id)
      : [...selected, movie];

    const remainingGenreIds = new Set(
      updated.flatMap((selectedMovie) => selectedMovie.genre_ids || [])
    );
    setSelected(updated);
    setRemovedMovieGenres((removed) =>
      removed.filter((genreId) => remainingGenreIds.has(genreId))
    );
    invalidateRecommendations();
    bumpRepresentationRevision();
  };

  const removeMovieGenre = (genreId) => {
    setRemovedMovieGenres((removed) =>
      removed.includes(genreId) ? removed : [...removed, genreId]
    );
    invalidateRecommendations();
    bumpRepresentationRevision();
  };

  /* -----------------------------------------------
     SELECT GENRE
  ------------------------------------------------ */
  const toggleGenre = (id) => {
    let updated = chosenGenres.includes(id)
      ? chosenGenres.filter((g) => g !== id)
      : [...chosenGenres, id];

    setChosenGenres(updated);
    invalidateRecommendations();
    bumpRepresentationRevision();
  };

  /* -----------------------------------------------
     GET RECOMMENDATIONS
  ------------------------------------------------ */
  const handleRecommend = async () => {
    const combinedGenreNames = genreNamesWithFrequencies({
      selectedMovies: selected,
      removedMovieGenres,
      chosenGenres,
      genreList,
    });

    if (combinedGenreNames.length === 0) {
      setRecommendationError("Select at least one movie or genre first.");
      return;
    }

    const requestId = ++recommendationRequestId.current;
    setLoading(true);
    setRecommendationError("");

    try {
      const payload = {
        genres: combinedGenreNames,
        // The frozen hybrid also needs the RecVAE interaction side.
        liked_movie_ids: selected.map((movie) => Number(movie.movieId)),
        // Participant rating semantics are intentionally unresolved for GERS;
        // null is logged and the frozen serving model keeps its fixed 5.0 input.
        preference_evidence: selected.map((movie) => ({
          movie_id: Number(movie.movieId),
          rating: null,
        })),
        // The onboarding catalog is for preference elicitation only. Neither
        // selected nor unselected catalog titles may reappear as results.
        excluded_movie_ids: [],
        catalog_fingerprint: servingDeployment.matrix_fingerprint,
        onboarding_fingerprint: pilotManifest.fingerprint,
        context,
        alpha: GERS_ALPHA,
        top_k: topK,
        min_release_year: MIN_RECOMMENDATION_YEAR,
      };
      const input = recommendationInputSnapshot("GERS", payload);
      const meta = await study.metadata({
        taskId: "1b",
        input,
        representationRevision: representationRevisionRef.current,
      });
      payload.study = meta;
      activeRecommendationRequest.current = meta;
      currentRecommendationSignature.current = meta.input_signature;
      const res = await axios.post(`${PILOT_API_URL}/gers`, payload);

      if (
        requestId !== recommendationRequestId.current ||
        !responseMatchesCurrent(
          activeRecommendationRequest.current,
          res.data.study,
          currentRecommendationSignature.current
        )
      ) {
        return;
      }

      const items = res.data.items || [];

      const posters = await Promise.all(
        items.map(async (it) => {
          const metadata = await fetchPoster(
            it.movie_id,
            it.title,
            it.tmdb_id
          );
          const movie = resolveMovieLensRecommendation(it, metadata);
          return {
            ...movie,
            rating: metadata?.rating ?? null,
          };
        })
      );

      if (
        requestId === recommendationRequestId.current &&
        responseMatchesCurrent(
          activeRecommendationRequest.current,
          res.data.study,
          currentRecommendationSignature.current
        )
      ) {
        await study.persistRender(meta, posters);
      }
      if (
        requestId === recommendationRequestId.current &&
        responseMatchesCurrent(
          activeRecommendationRequest.current,
          res.data.study,
          currentRecommendationSignature.current
        )
      ) {
        setRecommendations(posters);
      }
    } catch (err) {
      if (requestId === recommendationRequestId.current) {
        console.error("GERS ERROR:", err);
        setRecommendationError(
          apiErrorMessage(
            err,
            "Recommendations are temporarily unavailable. Please try again."
          )
        );
      }
    }

    if (requestId === recommendationRequestId.current) setLoading(false);
  };

  const selectedMovieGenreCounts = selected.reduce((counts, movie) => {
    movie.genre_ids?.forEach((genreId) => {
      counts.set(genreId, (counts.get(genreId) || 0) + 1);
    });
    return counts;
  }, new Map());
  const selectedMovieGenres = genreList
    .map((genre, originalIndex) => ({
      ...genre,
      count: selectedMovieGenreCounts.get(genre.id) || 0,
      originalIndex,
    }))
    .filter(
      (genre) =>
        genre.count > 0 && !removedMovieGenres.includes(genre.id)
    )
    .sort(
      (first, second) =>
        second.count - first.count || first.originalIndex - second.originalIndex
    );
  /* -----------------------------------------------
     UI
  ------------------------------------------------ */
  return (
    <div className="min-h-screen overflow-x-hidden bg-gradient-to-br from-[#0d0017] via-[#0b0012] to-black text-white px-4 sm:px-6 py-6 sm:py-8">

      {/* LOGO */}
      <div className="absolute top-4 left-4 sm:top-6 sm:left-8 z-20">
        <motion.img
          src={publicAsset("logo.png")}
          alt="Studio Auréa"
          className="h-14 sm:h-20 drop-shadow-[0_0_35px_#00ff99aa]"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
        />
      </div>

      {/* BACK BUTTON */}
      <button
        onClick={goBack}
        className="absolute top-4 right-4 sm:top-6 sm:right-6 px-4 sm:px-5 py-2 rounded-2xl bg-white/5
          text-sm sm:text-base
          border border-green-500 text-white backdrop-blur-xl
          shadow-[0_0_25px_#00ff99aa] hover:shadow-[0_0_40px_#00ff99ff]"
      >
        ← Back
      </button>

      {/* TITLE */}
      <div className="max-w-[1600px] mx-auto mt-24 sm:mt-28 mb-8 sm:mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-green-300/80 mb-3">
          Build your taste from movies and genres
        </p>
        <h1 className="text-3xl sm:text-4xl font-bold text-green-300 drop-shadow-[0_0_25px_#00ff99aa]">
          GERS – Genre-Based Recommender
        </h1>
      </div>

      <div className="max-w-[1600px] mx-auto flex flex-col xl:flex-row gap-8 xl:gap-10">

        {/* LEFT SIDE */}
        <div className="w-full xl:w-3/5">
          <h2 className="text-lg font-semibold mb-3">Select movies you like:</h2>

          {loadingMovies ? (
            <div className="text-center text-gray-400 py-10">
              Loading movies...
            </div>
          ) : (

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3 sm:gap-4 relative z-10 overflow-visible">

              {movies.map((m) => (
                <motion.button
                  key={m.id}
                  type="button"
                  aria-pressed={selected.some((s) => s.id === m.id)}
                  whileHover={{ scale: 1.08 }}
                  onClick={() => toggleSelect(m)}
                  className={`relative cursor-pointer rounded-xl p-[3px] text-left transition-all
                    ${
                      selected.find((s) => s.id === m.id)
                        ? "ring-2 ring-green-400 shadow-[0_0_20px_#00ff99aa]"
                        : "ring-1 ring-white/10 hover:ring-green-400 hover:shadow-[0_0_20px_#00ff9955]"
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
                    className="absolute inset-0 opacity-0 hover:opacity-100 transition-all
                      duration-300 bg-black/80 backdrop-blur-md rounded-xl flex flex-col
                      justify-between p-3 z-20 hover:-translate-y-1"
                  >
                    <div>
                      <h3 className="font-bold text-sm mb-1">{m.title}</h3>

                      <p className="text-[11px] text-green-300">
                        ⭐ {m.rating || "N/A"}
                      </p>

                      <p className="text-[10px] text-gray-300 mt-1">
                        {m.year || ""}
                      </p>

                      <div className="flex flex-wrap gap-1 mt-2">
                        {m.genre_ids?.slice(0, 3).map((gid, idx) => (
                          <span
                            key={idx}
                            className="text-[9px] px-2 py-[2px] bg-white/10 border border-white/10
                              rounded-full"
                          >
                            {safeGenreName(gid, genreList)}
                          </span>
                        ))}
                      </div>
                    </div>

                    <p className="text-[10px] line-clamp-4 text-gray-200 mt-2">
                      {m.overview}
                    </p>

                    {/* NEW GENRE SUMMARY */}

                  </div>
                </motion.button>
              ))}
            </div>
          )}
        </div>

        {/* RIGHT SIDE */}
        <div className="w-full xl:w-2/5 xl:sticky xl:top-6 self-start bg-white/5 border border-green-400 rounded-3xl p-4 sm:p-5 backdrop-blur-xl shadow-[0_0_30px_#00ff9933]">

          <div className="mb-5 rounded-2xl border border-green-400/50 bg-black/25 p-4">
            <h2 className="text-sm font-semibold text-green-300 mb-2">
              Genres from selected movies
            </h2>

            {selectedMovieGenres.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {selectedMovieGenres.map((genre) => (
                  <span
                    key={genre.id}
                    className={`inline-flex items-center gap-1 rounded-full border pl-3 pr-1 py-1 text-xs ${
                      genre.count > 1
                        ? "order-first border-green-300 bg-green-700/60 text-white shadow-[0_0_12px_#00ff9966]"
                        : "border-white/20 bg-white/10 text-gray-200"
                    }`}
                  >
                    {genre.name}
                    {genre.count > 1 && (
                      <span className="ml-1 font-bold text-green-200">
                        ×{genre.count}
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => removeMovieGenre(genre.id)}
                      aria-label={`Remove ${genre.name} from selected movie genres`}
                      title={`Remove ${genre.name}`}
                      className="ml-1 flex h-5 w-5 items-center justify-center rounded-full
                        text-sm leading-none text-gray-300 transition-colors
                        hover:bg-red-500/30 hover:text-white focus:outline-none
                        focus:ring-1 focus:ring-red-300"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400">
                {selected.length > 0
                  ? "No movie-derived genres are active."
                  : "Select movies to see their genres here."}
              </p>
            )}
          </div>

          {/* GENRE BUTTONS */}
          <h2 className="text-lg font-semibold mb-3">Select genres:</h2>

          <div className="flex flex-wrap gap-2 mb-5">
            {genreList.map((g) => (
              <button
                key={g.id}
                type="button"
                aria-pressed={chosenGenres.includes(g.id)}
                onClick={() => toggleGenre(g.id)}
                className={`px-3 py-1 rounded text-xs border transition-all ${
                  chosenGenres.includes(g.id)
                    ? "bg-green-700 border-green-400 shadow-[0_0_15px_#00ff99aa]"
                    : "bg-white/10 border-white/20 hover:bg-white/20"
                }`}
              >
                {g.name}
              </button>
            ))}
          </div>

          <div className="mb-6">
            <label htmlFor="gers-top-k" className="text-sm font-semibold">
              Number of recommendations: {topK}
            </label>
            <input
              id="gers-top-k"
              type="range"
              min="1"
              max="25"
              value={topK}
              onChange={(event) => {
                setTopK(Number(event.target.value));
                invalidateRecommendations();
              }}
              className="w-full mt-2 accent-green-400"
            />
          </div>

          {/* BUTTON */}
          <button
            onClick={handleRecommend}
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-lg
              bg-[#121212] border border-green-500 shadow-[0_0_25px_#00ff99aa]
              hover:shadow-[0_0_40px_#00ff99dd]
              disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            {loading ? "Finding your movies…" : "Get Recommendations"}
          </button>

          {recommendationError && (
            <p role="alert" className="mt-3 text-sm text-red-300">
              {recommendationError}
            </p>
          )}

          {/* RESULTS */}
          {recommendations.length > 0 && (
            <div className="mt-6">
              <h2 className="text-md font-bold mb-3 text-green-300">
                Recommended Movies:
              </h2>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
                {recommendations.map((m) => (
                  <motion.div
                    key={`${m.movie_id}-${m.rank}`}
                    whileHover={{ scale: 1.08 }}
                    className="relative cursor-pointer rounded-xl p-[3px]
                      transition-all ring-1 ring-white/10 hover:ring-green-400
                      hover:shadow-[0_0_20px_#00ff9955] z-10 hover:z-20"
                  >
                    <img
                      src={m.poster_url || publicAsset("placeholder_poster.png")}
                      alt={m.title}
                      onError={(event) => {
                        event.currentTarget.onerror = null;
                        event.currentTarget.src = publicAsset("placeholder_poster.png");
                      }}
                      className="w-full aspect-[2/3] object-cover rounded-lg"
                    />

                    <p className="text-sm mt-1 text-center">
                      {m.rank_label || `#${m.rank}`} {m.title}
                    </p>

                    <div
                      className="absolute inset-0 opacity-0 hover:opacity-100
                        bg-black/80 backdrop-blur-md text-white p-3 flex flex-col
                        justify-between rounded-xl transition-all duration-300
                        hover:-translate-y-1 z-20"
                    >
                      <div>
                        <h3 className="font-bold text-sm mb-1">{m.title}</h3>

                        <p className="text-[12px] text-green-300 font-bold">
                          {m.rank_label || `#${m.rank}`}
                        </p>

                        <p className="text-[11px] text-green-300 mt-1">
                          ⭐ {m.rating ?? "N/A"}
                        </p>

                        <p className="text-[10px] text-gray-300 mt-1">
                          {m.year}
                        </p>

                        <div className="flex flex-wrap gap-1 mt-2">
                          {m.genres?.slice(0, 3).map((genre) => (
                            <span
                              key={genre}
                              className="text-[9px] px-2 py-[2px] bg-white/10
                                rounded-full border border-white/10"
                            >
                              {genre}
                            </span>
                          ))}
                        </div>
                      </div>

                      <p className="text-[10px] text-gray-200 line-clamp-4 mt-2">
                        {m.overview || "No summary available."}
                      </p>
                    </div>
                  </motion.div>

                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
