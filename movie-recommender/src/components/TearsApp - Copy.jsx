/* ---------------------------------------------------------
    TEARSApp.jsx — FULL GPT-SMART VERSION
    Nova Edition — 2025-11 — For UX Thesis Study
----------------------------------------------------------*/
import fixedMovies from "../data/fixed_50_movies_ml1m.json";
import { useEffect, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
//import { FIXED_MOVIES_TEARS } from "../utils/fixedMovies_tears";


const API_URL = process.env.REACT_APP_API_URL;
const TMDB_KEY = "fc4a0ec3fa9d745f0b94e417da01cd26";

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


/* ---------------------------------------------------------
    TMDB SEARCH + POSTER
----------------------------------------------------------*/
async function fetchPoster(title) {
  try {
    let clean = title
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

    let q1 = await search(clean);
    if (q1.data.results.length > 0) return q1.data.results[0];

    let q2 = await search(title);
    if (q2.data.results.length > 0) return q2.data.results[0];

    if (title.includes("-")) {
      let shorter = title.split("-")[0].trim();
      let q3 = await search(shorter);
      if (q3.data.results.length > 0) return q3.data.results[0];
    }

    return null;
  } catch {
    return null;
  }
}

/* ---------------------------------------------------------
    MAIN TEARS COMPONENT
----------------------------------------------------------*/
export default function TearsApp({ goBack }) {
  const [movies, setMovies] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);

  const [selected, setSelected] = useState([]);
  const [summary, setSummary] = useState("");
  const [context, setContext] = useState("");

  const [recommendations, setRecommendations] = useState([]);
  const [previousRecommendations, setPrevious] = useState([]);

  const [loading, setLoading] = useState(false);

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
  setMovies(fixedMovies);
  console.log("🔥 Loaded movies from ML-1M:", fixedMovies.length);
  console.log("🔥 First movie:", fixedMovies[0]);
  setLoadingMovies(false);
}, []);

//useEffect(() => {
  //setMovies(FIXED_MOVIES_TEARS);
  //console.log("🔥 Loaded movies:", FIXED_MOVIES_TEARS.length);
  //console.log("🔥 First movie:", FIXED_MOVIES_TEARS[0]);
  //setLoadingMovies(false);
//}, []);



  /* ---------------------------------------------------------
      SELECT MOVIES → GPT SUMMARIZER (PATCHED)
  ----------------------------------------------------------*/
  const toggleSelect = async (movie) => {
    let updated;

    if (selected.find((m) => m.tmdb_id === movie.tmdb_id)) {
      updated = selected.filter((m) => m.tmdb_id !== movie.tmdb_id);
    } else {
      updated = [...selected, movie];
    }

    setSelected(updated);

    if (updated.length > 0) {
      try {
        const payload = {
          liked: updated.map((m) => m.title),

          disliked: [],
          context,
        };

        const res = await axios.post(`${API_URL}/summarize`, payload);
        setSummary(res.data.summary || "");

      } catch (err) {
        console.error("🔥 GPT Summarizer Error:", err);
      }
    } else {
      setSummary("");
    }
  };

  /* ---------------------------------------------------------
      CONTEXT CHANGE → GPT SUMMARIZER (PATCHED)
  ----------------------------------------------------------*/
  const handleContextChange = async (value) => {
    setContext(value);

    if (selected.length === 0) return;

    try {
      const payload = {
        liked: selected.map((m) => m.title),

        disliked: [],
        context: value,
      };

      const res = await axios.post(`${API_URL}/summarize`, payload);
      setSummary(res.data.summary || "");

    } catch (err) {
      console.error("🔥 Context Summarizer Error:", err);
    }
  };

  /* ---------------------------------------------------------
      EXTRACT DISLIKES FROM SUMMARY (NLP)
  ----------------------------------------------------------*/
  function extractDislikes(summaryText) {
    const t = summaryText.toLowerCase();
    const disliked = [];

    if (/dislike.*horror|no horror|hate horror/.test(t)) disliked.push("Horror");
    if (/dislike.*comedy|no comedy|hate comedy/.test(t)) disliked.push("Comedy");
    if (/dislike.*drama|no drama/.test(t)) disliked.push("Drama");
    if (/dislike.*thriller/.test(t)) disliked.push("Thriller");
    if (/dislike.*action/.test(t)) disliked.push("Action");

    return disliked;
  }

  /* ---------------------------------------------------------
      REQUEST TEARS RECOMMENDATIONS
  ----------------------------------------------------------*/
  const handleRecommend = async () => {
    if (!summary.trim()) return;

    setLoading(true);

    try {
      setPrevious(recommendations);

      const disliked = extractDislikes(summary);

      const payload = {
        summary,
        context,
        liked: selected.map((m) => m.title),
        disliked,
        top_k: 12,
        alpha: 1.0,
      };

      const res = await axios.post(`${API_URL}/recommend`, payload);
      const items = res.data.items || [];

      const enriched = [];
      for (let it of items) {
        const poster = await fetchPoster(it.title);

        enriched.push({
          ...it,
          poster_path: poster?.poster_path || null,
          overview: poster?.overview || "",
          year: poster?.release_date ? poster.release_date.split("-")[0] : "",
          rating: poster?.vote_average || "N/A",
          genres: poster?.genre_ids?.map((id) => GENRE_MAP[id]) || [],
          score_fmt: it.score.toFixed(2),
        });
      }

      setRecommendations(enriched);

    } catch (err) {
      console.error("🔥 FRONTEND ERROR", err);
      alert("Backend error — check server logs.");
    }

    setLoading(false);
  };

  /* ---------------------------------------------------------
      RANK CHANGE
  ----------------------------------------------------------*/
  function getRankChange(title) {
    if (!previousRecommendations.length) return null;

    const prev = previousRecommendations.find((m) => m.title === title);
    const now = recommendations.find((m) => m.title === title);

    if (!prev || !now) return null;

    const diff = prev.rank - now.rank;
    if (diff > 0) return `⬆ +${diff}`;
    if (diff < 0) return `⬇ ${diff}`;
    return "–";
  }

  /* ---------------------------------------------------------
      UI RENDER
  ----------------------------------------------------------*/
  return (
    <div
      className="min-h-screen relative bg-gradient_to_br 
      from-[#0d0017] via-[#0b0012] to-black text-white px-6 py-8"
    >
      {/* LOGO */}
      <div className="absolute top-6 left-8 z-20">
        <motion.img
          src="/logo.png"
          className="h-20 drop-shadow-[0_0_35px_#00C8FF55]"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
        />
      </div>

      {/* BACK BUTTON */}
      <button
        onClick={goBack}
        className="absolute top-6 right-6 z-20 bg-white/5 border border-[#00C8FF55]
          px-5 py-2 rounded-2xl text-lg backdrop-blur-xl
          shadow-[0_0_20px_#00C8FF55] hover:shadow-[0_0_35px_#00C8FFAA]
          hover:border-[#00C8FF]"
      >
        ← Back
      </button>

      {/* TITLE */}
      <div className="mt-32 mb-10">
        <h1 className="text-4xl font-bold text-[#A7E8FF] drop-shadow-[0_0_20px_#00C8FF55]">
          TEARS – Summary-Based Recommender
        </h1>
      </div>

      <div className="flex gap-10">
        {/* ------------------------------------------------------
            LEFT PANEL — MOVIE SELECTION
        --------------------------------------------------------*/}
        <div className="w-3/5">
          <h2 className="text-lg font-semibold mb-3">Select movies you like:</h2>

          {loadingMovies ? (
            <div className="text-center text-gray-400 py-10">Loading…</div>
          ) : (
            <div className="grid grid-cols-5 gap-4">
              {movies.map((m) => (
                <motion.div
                  key={m.tmdb_id}
                  onClick={() => toggleSelect(m)}
                  whileHover={{ scale: 1.08 }}
                  className={`relative cursor-pointer rounded-xl p-[3px] transition-all
                    ${
                      selected.find((s) => s.tmdb_id === m.tmdb_id)
                        ? "ring-2 ring-[#00C8FF] shadow-[0_0_20px_#00C8FF99]"
                        : "ring-1 ring-white/10 hover:ring-[#00C8FF88] hover:shadow-[0_0_20px_#00C8FF55]"
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
                </motion.div>
              ))}
            </div>
          )}
        </div>

        {/* ------------------------------------------------------
            RIGHT PANEL — SUMMARY + RECOMMENDATIONS
        --------------------------------------------------------*/}
        <div className="w-2/5 bg-white/5 border border-[#00C8FF] rounded-3xl p-5 
            backdrop-blur-xl shadow-[0_0_30px_#00C8FF55]">
          
          {/* SUMMARY */}
          <h2 className="text-lg font-semibold mb-2">Your Summary</h2>
          <textarea
            className="w-full h-32 bg-white/5 border border-white/10 p-3 
              rounded-lg text-sm mb-4"
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
          />

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

          {/* BUTTON */}
          <button
            onClick={handleRecommend}
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-lg bg-[#0A0A0A]
              border border-[#00C8FF] shadow-[0_0_25px_#00C8FF55]
              hover:shadow-[0_0_40px_#00C8FFAA] transition-all"
          >
            {loading ? "Loading..." : "Get Recommendations"}
          </button>

          {/* RESULTS */}
          {recommendations.length > 0 && (
            <div className="mt-6">
              <h2 className="text-md font-bold mb-3 text-[#80E7FF]">
                Recommended Movies
              </h2>

              <div className="grid grid-cols-3 gap-4">
                {recommendations.map((m) => (
                  <motion.div
                    key={m.title}
                    whileHover={{ scale: 1.07 }}
                    className="relative bg-white/10 backdrop-blur-lg rounded-xl p-2
                      hover:shadow-[0_0_20px_#00C8FF55] transition-all"
                  >
                    {m.poster_path ? (
                      <img
                        src={`https://image.tmdb.org/t/p/w500${m.poster_path}`}
                        alt={m.title}
                        className="w-full h-44 object-cover rounded-lg"
                      />
                    ) : (
                      <div className="h-44 flex items-center justify_center bg-gray-700 
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
                          {getRankChange(m.title) && (
                            <span className="ml-1 text-purple-300">
                              ({getRankChange(m.title)})
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
                              className="text-[9px] px-2 py-[2px] bg_white/10
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
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
