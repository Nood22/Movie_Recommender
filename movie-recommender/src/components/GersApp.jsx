import { useEffect, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FIXED_MOVIES } from "../utils/fixedMovies";


const TMDB_KEY = "fc4a0ec3fa9d745f0b94e417da01cd26";
const QUALITY_API_URL =
  process.env.REACT_APP_QUALITY_API_URL || "http://127.0.0.1:8001";

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
async function fetchPoster(title) {
  try {
    const cleaned = title
      .replace(/\(.*?\)/g, "")
      .replace(/[:,]/g, "")
      .replace(/\./g, "")
      .replace(/\s+-\s+.*/g, "")
      .replace(/  +/g, " ")
      .trim();

    const search = async (q) =>
      axios.get(
        `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${encodeURIComponent(
          q
        )}&include_adult=false`
      );

    let q1 = await search(cleaned);
    if (q1.data.results.length > 0) return q1.data.results[0];

    let q2 = await search(title);
    if (q2.data.results.length > 0) return q2.data.results[0];

    if (title.includes("-")) {
      let short = title.split("-")[0].trim();
      let q3 = await search(short);
      if (q3.data.results.length > 0) return q3.data.results[0];
    }

    return null;
  } catch {
    return null;
  }
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
  const [movies, setMovies] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);

  const [selected, setSelected] = useState([]);
  const [chosenGenres, setChosenGenres] = useState([]);

  const [context, setContext] = useState("");
  const [summary, setSummary] = useState("");

  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

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


  const contextList = [
    "I want a cozy movie for tonight",
    "I want something romantic",
    "I want something thrilling",
    "I want a family-friendly movie",
    "I want something emotional",
    "I want something fun and light",
  ];

  /* -----------------------------------------------
     LOAD MOVIES
  ------------------------------------------------ */

/* -----------------------------------------------
   LOAD MOVIES — FIXED LIST WITH GENRE IDS
------------------------------------------------ */
useEffect(() => {
  setLoadingMovies(true);

  const mapped = FIXED_MOVIES.map((m) => ({
    id: m.tmdb_id,
    title: m.title,
    poster: m.poster,
    overview: m.overview,
    rating: m.rating,
    year: m.year,

    // 🔥 Convert string genres → TMDB IDs
    genre_ids: m.genres
      .map((g) => GENRE_MAP[normalizeGenre(g)] || null)
      .filter(Boolean),
  }));

  setMovies(mapped);
  setLoadingMovies(false);
}, []);


  /* -----------------------------------------------
     GPT GENRE SUMMARY (FAST-DEBOUNCED)
  ------------------------------------------------ */
 const generateSummary = (moviesArr, manualGenreIds) => {
  // 1) extract genre IDs
  const movieGenreIds = [];
  moviesArr.forEach((m) =>
    m.genre_ids?.forEach((g) => movieGenreIds.push(g))
  );

  const combined = [...new Set([...movieGenreIds, ...manualGenreIds])];

  // ⭐⭐ نمایش فوری و بدون تأخیر ⭐⭐
  const instantNames = combined
    .map((id) => genreList.find((g) => g.id === id)?.name)
    .filter(Boolean);

  setSummary("You seem to enjoy: " + instantNames.join(", "));
};


  /* -----------------------------------------------
     SELECT MOVIE
  ------------------------------------------------ */
  const toggleSelect = (movie) => {
    let updated = selected.find((s) => s.id === movie.id)
      ? selected.filter((s) => s.id !== movie.id)
      : [...selected, movie];

    setSelected(updated);

    generateSummary(updated, chosenGenres);
  };

  /* -----------------------------------------------
     SELECT GENRE
  ------------------------------------------------ */
  const toggleGenre = (id) => {
    let updated = chosenGenres.includes(id)
      ? chosenGenres.filter((g) => g !== id)
      : [...chosenGenres, id];

    setChosenGenres(updated);
    generateSummary(selected, updated);
  };

  /* -----------------------------------------------
     CONTEXT
  ------------------------------------------------ */
  const handleContextChange = (value) => {
    setContext(value);
  };

  /* -----------------------------------------------
     GET RECOMMENDATIONS
  ------------------------------------------------ */
  const handleRecommend = async () => {
    const movieGenreIds = [];
    selected.forEach((m) => {
      if (Array.isArray(m.genre_ids)) {
        m.genre_ids.forEach((id) => movieGenreIds.push(id));
      }
    });

    const combinedGenreNames = [...new Set([...movieGenreIds, ...chosenGenres])]
      .map((id) => genreList.find((g) => g.id === id)?.name)
      .filter(Boolean);

    if (combinedGenreNames.length === 0) {
      alert("Please select at least one movie or genre.");
      return;
    }

    setLoading(true);

    try {
      const description = [
        summary || `Preferred genres: ${combinedGenreNames.join(", ")}`,
        context,
      ].filter(Boolean).join("\nContext: ");

      const res = await axios.post(`${QUALITY_API_URL}/recommend`, {
        description,
        liked: selected.map((movie) => movie.title),
        disliked_genres: [],
        top_k: 9,
      });

      const items = res.data.items || [];

      const posters = await Promise.all(
        items.map(async (it) => {
          const result = await fetchPoster(it.title);

          return {
            ...it,
            poster_path: result?.poster_path || null,
            overview: result?.overview || "No overview",
            rating: result?.vote_average || null,
            year: result?.release_date?.split("-")[0] || "",
            genres: it.genres || [],
          };
        })
      );

      setRecommendations(posters);
    } catch (err) {
      console.error("GERS ERROR:", err);
      alert("Backend error — check logs.");
    }

    setLoading(false);
  };

  /* -----------------------------------------------
     UI
  ------------------------------------------------ */
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0d0017] via-[#0b0012] to-black text-white px-6 py-8">

      {/* LOGO */}
      <div className="absolute top-6 left-8 z-20">
        <motion.img
          src="/logo.png"
          className="h-20 drop-shadow-[0_0_35px_#00ff99aa]"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
        />
      </div>

      {/* BACK BUTTON */}
      <button
        onClick={goBack}
        className="absolute top-6 right-6 px-5 py-2 rounded-2xl bg-white/5 
          border border-green-500 text-white backdrop-blur-xl
          shadow-[0_0_25px_#00ff99aa] hover:shadow-[0_0_40px_#00ff99ff]"
      >
        ← Back
      </button>

      {/* TITLE */}
      <div className="mt-24 mb-10">
        <h1 className="text-4xl font-bold text-green-300 drop-shadow-[0_0_25px_#00ff99aa]">
          GERS – Genre-Based Recommender
        </h1>
      </div>

      <div className="flex gap-10">
        
        {/* LEFT SIDE */}
        <div className="w-3/5">
          <h2 className="text-lg font-semibold mb-3">Select movies you like:</h2>

          {loadingMovies ? (
            <div className="text-center text-gray-400 py-10">
              Loading movies...
            </div>
          ) : (
            
              <div className="grid grid-cols-5 gap-4 relative z-10 overflow-visible">

              {movies.map((m) => (
                <motion.div
                  key={m.id}
                  whileHover={{ scale: 1.08 }}
                  onClick={() => toggleSelect(m)}
                  className={`relative cursor-pointer rounded-xl p-[3px] transition-all
                    ${
                      selected.find((s) => s.id === m.id)
                        ? "ring-2 ring-green-400 shadow-[0_0_20px_#00ff99aa]"
                        : "ring-1 ring-white/10 hover:ring-green-400 hover:shadow-[0_0_20px_#00ff9955]"
                    }`}
                >
                  <img
                    src={m.poster}
                    alt={m.title}
                    className="w-full h-48 object-cover rounded-lg"
                  />

                  <p className="text-sm mt-1 text-center">{m.title}</p>

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
                </motion.div>
              ))}
            </div>
          )}
        </div>

        {/* RIGHT SIDE */}
        <div className="w-2/5 bg-white/5 border border-green-400 rounded-3xl p-5 backdrop-blur-xl">

          {/* SUMMARY */}
          <h2 className="text-lg font-semibold mb-2">Your Preference Genres!</h2>

          <textarea
            className="w-full h-32 bg-white/5 border border-white/10 p-3 rounded-lg text-sm mb-4"
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
          />

          {/* GENRE BUTTONS */}
          <h2 className="text-lg font-semibold mb-3">Select genres:</h2>

          <div className="flex flex-wrap gap-2 mb-5">
            {genreList.map((g) => (
              <button
                key={g.id}
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

          {/* CONTEXT */}
          <h2 className="text-lg font-semibold mb-2">Choose a context:</h2>

          <select
            className="w-full py-3 px-3 rounded-lg mb-6 bg-white/10 border border-white/20
              focus:border-green-400 focus:shadow-[0_0_20px_#00ff99aa]"
            value={context}
            onChange={(e) => handleContextChange(e.target.value)}
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

          {/* BUTTON */}
          <button
            onClick={handleRecommend}
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-lg
              bg-[#121212] border border-green-500 shadow-[0_0_25px_#00ff99aa]
              hover:shadow-[0_0_40px_#00ff99dd]"
          >
            {loading ? "Loading..." : "Get Recommendations"}
          </button>

          {/* RESULTS */}
          {recommendations.length > 0 && (
            <div className="mt-6">
              <h2 className="text-md font-bold mb-3 text-green-300">
                Recommended Movies:
              </h2>

              <div className="grid grid-cols-3 gap-4">
                {recommendations.map((m) => (
                  <motion.div
                    key={`${m.movie_id}-${m.rank}`}
                    whileHover={{ scale: 1.3 }}
                    style={{ zIndex: 999 }}
                    onHoverStart={(event) => {
                      if (event?.currentTarget) {
                        event.currentTarget.style.zIndex = "9999";
                      }
                    }}
                    onHoverEnd={(event) => {
                      if (event?.currentTarget) {
                        event.currentTarget.style.zIndex = "1";
                      }
                    }}
                    className="relative cursor-pointer rounded-xl p-[3px]
                      transition-all duration-300 ring-1 ring-white/10
                      hover:ring-green-300 hover:shadow-[0_0_30px_#00ff99aa]"
                  >
                    <img
                      src={m.poster_path
                        ? `https://image.tmdb.org/t/p/w500${m.poster_path}`
                        : "/placeholder_poster.png"}
                      alt={m.title}
                      onError={(event) => {
                        event.currentTarget.onerror = null;
                        event.currentTarget.src = "/placeholder_poster.png";
                      }}
                      className="w-full h-48 object-cover rounded-lg"
                    />

                    <p className="text-sm mt-1 text-center">
                      #{m.rank} {m.title}
                    </p>
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
