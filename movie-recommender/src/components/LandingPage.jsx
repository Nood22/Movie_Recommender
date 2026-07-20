import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { useState, useEffect } from "react";

export default function LandingPage() {

  // تمام پوسترهایی که داری بزار اینجا
  const posters = [
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
  ];

  // parallax
  const [scrollY, setScrollY] = useState(0);
  useEffect(() => {
    const f = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", f);
    return () => window.removeEventListener("scroll", f);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen relative overflow-hidden bg-[#0d0714] text-white"
    >
      {/* Vignette */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-black/40 via-transparent to-black/70 z-[5]" />

      {/* LOGO */}
      <div className="absolute top-6 left-8 z-[40]">
        <motion.img
          src="/logo.png"
          alt="Studio Aurea"
          className="h-32 drop-shadow-[0_0_35px_#B497BD66]"
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
            <motion.img
              key={i}
              src={`/Posters/${p}`}
              alt=""
              className="w-64 h-96 object-cover rounded-2xl shadow-2xl opacity-[0.92] border border-white/10"
              animate={{
                rotateY: scrollY * 0.02,
                rotateX: scrollY * 0.01,
              }}
              style={{
                transform: `translateY(${-(scrollY * 0.1)}px)`,
              }}
            />
          ))}
        </div>
      </div>

      {/* MAIN CONTENT (higher) */}
      <div className="relative z-[30] flex flex-col items-start text-left pt-[24vh] px-12 max-w-xl">

        <motion.h1
          className="text-5xl md:text-6xl font-extrabold mb-5"
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

        <p className="text-2xl text-gray-300 mb-12 leading-relaxed">
          Choose between <span className="text-blue-400 font-bold">TEARS</span>{" "}
          (summary-based) or{" "}
          <span className="text-green-400 font-bold">GERS</span>{" "}
          (genre-based) models.
        </p>

        {/* BUTTONS */}
        <div className="flex gap-6">
          {/* TEARS */}
          <Link to="/tears">
            <motion.button
              whileHover={{ scale: 1.08 }}
              className="px-10 py-4 rounded-xl bg-[#120d1f]/60 border border-blue-400/40
                         text-blue-200 text-lg font-semibold backdrop-blur-md
                         shadow-[0_0_25px_#8ab4ff55] hover:shadow-[0_0_35px_#8ab4ffaa]
                         hover:bg-[#1b1f45]/70 transition-all"
            >
              TEARS
            </motion.button>
          </Link>

          {/* GERS */}
          <Link to="/gers">
            <motion.button
              whileHover={{ scale: 1.08 }}
              className="px-10 py-4 rounded-xl bg-[#120d1f]/60 border border-green-400/40
                         text-green-200 text-lg font-semibold backdrop-blur-md
                         shadow-[0_0_25px_#78ffaa55] hover:shadow-[0_0_35px_#78ffaaaa]
                         hover:bg-[#0f3823]/70 transition-all"
            >
              GERS
            </motion.button>
          </Link>
        </div>
      </div>
    </motion.div>
  );
}
