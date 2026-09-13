import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import pilotMovies from "../data/pilot_support20_onboarding.json";
import { publicAsset } from "../utils/publicAsset.mjs";

// Match the same interleaved catalog order shown on both recommendation screens.
const LANDING_MOVIES = pilotMovies.slice(0, 11);
const MotionLink = motion(Link);

export default function LandingPage() {
  // Keep the original poster-wall layout, but bind its content to the frozen
  // ML-32M pilot onboarding artifact instead of the original static filenames.
  const posters = LANDING_MOVIES.map((movie) =>
    publicAsset(`pilot-posters/${movie.movieId}.jpg`)
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen relative overflow-hidden bg-[#0d0714] text-white"
    >
      <div className="absolute -left-32 top-1/3 h-80 w-80 rounded-full bg-blue-500/10 blur-[120px]" />
      <div className="absolute right-0 bottom-0 h-96 w-96 rounded-full bg-emerald-500/10 blur-[140px]" />
      {/* Vignette */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-black/40 via-transparent to-black/70 z-[5]" />

      {/* LOGO */}
      <div className="absolute top-4 left-4 sm:top-6 sm:left-8 z-[40]">
        <motion.img
          src={publicAsset("logo.png")}
          alt="Studio Aurea"
          className="h-20 sm:h-24 md:h-32 drop-shadow-[0_0_35px_#B497BD66]"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1 }}
        />
      </div>

      {/* POSTER WALL (ONLY ONE ROW, higher & more visible) */}
      <div className="absolute inset-0 pointer-events-none hidden md:block z-[15]">
        <div
          className="absolute right-[-120px] top-[-180px] rotate-[-30deg]
                     grid grid-cols-3 gap-10"
        >
          {posters.map((p, i) => (
            <img
              key={LANDING_MOVIES[i].movieId}
              src={p}
              alt={LANDING_MOVIES[i].title}
              className="w-64 h-96 object-cover rounded-2xl shadow-2xl opacity-[0.92] border border-white/10"
            />
          ))}
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 grid grid-cols-3 gap-3 px-4 opacity-35 md:hidden z-[15]">
        {posters.slice(0, 3).map((poster, index) => (
          <motion.img
            key={LANDING_MOVIES[index].movieId}
            src={poster}
            alt={LANDING_MOVIES[index].title}
            className="w-full aspect-[2/3] object-cover rounded-t-xl shadow-2xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 * index }}
          />
        ))}
      </div>

      {/* MAIN CONTENT (higher) */}
      <main className="relative z-[30] flex min-h-screen max-w-2xl flex-col items-start justify-center px-6 pb-12 pt-32 text-left sm:px-12">

        <motion.p
          className="mb-5 rounded-full border border-white/15 bg-white/[0.06] px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-[#e7d2ff] backdrop-blur-xl"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          Your movies, your taste profile
        </motion.p>

        <motion.h1
          className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-5"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            backgroundImage: "linear-gradient(to right, #e7d2ff, #b497bd, #ffffff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          AI Movie Recommender
        </motion.h1>

        <p className="max-w-xl text-lg sm:text-xl md:text-2xl text-gray-300 mb-4 leading-relaxed">
          Choose between <span className="text-blue-400 font-bold">TEARS</span>{" "}
          (summary-based) or{" "}
          <span className="text-green-400 font-bold">GERS</span>{" "}
          (genre-based) models.
        </p>

        <p className="mb-8 sm:mb-10 max-w-lg text-sm leading-relaxed text-gray-400">
          Pick a few movies, shape your taste profile, and explore recommendations
          in two different ways.
        </p>

        {/* BUTTONS */}
        <div className="flex w-full flex-col gap-4 sm:w-auto sm:flex-row sm:gap-6">
          {/* TEARS */}
          <MotionLink
            to="/tears"
            whileHover={{ scale: 1.05 }}
            className="w-full sm:w-auto px-10 py-4 rounded-xl bg-[#120d1f]/60 border border-blue-400/40
                       text-center text-blue-200 text-lg font-semibold backdrop-blur-md
                       shadow-[0_0_25px_#8ab4ff55] hover:shadow-[0_0_35px_#8ab4ffaa]
                       hover:bg-[#1b1f45]/70 transition-all"
          >
            <span className="block">Explore with TEARS</span>
            <span className="mt-1 block text-xs font-normal text-blue-200/60">
              Summary-based
            </span>
          </MotionLink>

          {/* GERS */}
          <MotionLink
            to="/gers"
            whileHover={{ scale: 1.05 }}
            className="w-full sm:w-auto px-10 py-4 rounded-xl bg-[#120d1f]/60 border border-green-400/40
                       text-center text-green-200 text-lg font-semibold backdrop-blur-md
                       shadow-[0_0_25px_#78ffaa55] hover:shadow-[0_0_35px_#78ffaaaa]
                       hover:bg-[#0f3823]/70 transition-all"
          >
            <span className="block">Explore with GERS</span>
            <span className="mt-1 block text-xs font-normal text-green-200/60">
              Genre-based
            </span>
          </MotionLink>
        </div>
      </main>
    </motion.div>
  );
}
