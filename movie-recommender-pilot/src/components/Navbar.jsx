// src/components/Navbar.jsx
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="w-full px-8 py-4 flex items-center justify-between bg-black/70 backdrop-blur-md fixed top-0 left-0 z-50 border-b border-white/10">

      {/* Logo */}
      <Link to="/" className="flex items-center gap-3">
        <span className="text-3xl">🎥</span>
        <h1
          className="text-2xl font-extrabold tracking-wide"
          style={{
            backgroundImage: "linear-gradient(to right, #B497BD, #E9D5FF, #ffffff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "1px",
          }}
        >
          STUDIO AURÉA
        </h1>
      </Link>

      {/* Navigation Links */}
      <div className="flex items-center gap-8 text-lg font-semibold">
        <Link
          to="/tears"
          className="text-gray-300 hover:text-white transition"
        >
          TEARS
        </Link>

        <Link
          to="/gers"
          className="text-gray-300 hover:text-white transition"
        >
          GERS
        </Link>
      </div>
    </nav>
  );
}
