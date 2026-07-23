// buildFixedMovies.js
import fs from "fs";
import axios from "axios";

// ---- YOUR TMDB API KEY ----
const TMDB_KEY = "fc4a0ec3fa9d745f0b94e417da01cd26";

// ---- FIXED 50 MOVIES ----
const MOVIE_TITLES = [
  "Spider-Man: Across the Spider-Verse",
  "Dune: Part Two",
  "Dune",
  "La La Land",
  "The Batman",
  "Joker",
  "Everything Everywhere All at Once",
  "Top Gun: Maverick",
  "Arrival",
  "Spider-Man: No Way Home",
  "Avengers: Infinity War",
  "Avengers: Endgame",
  "Blade Runner 2049",
  "Mission: Impossible – Fallout",
  "Baby Driver",
  "The Whale",
  "Barbie",
  "Oppenheimer",
  "Saltburn",
  "Poor Things",
  "The Holdovers",
  "Parasite",
  "The Zone of Interest",
  "Past Lives",
  "Green Book",
  "Soul",
  "Coco",
  "Your Name",
  "Suzume",
  "1917",
  "Jojo Rabbit",
  "Bohemian Rhapsody",
  "Ford v Ferrari",
  "The Menu",
  "Knives Out",
  "Glass Onion",
  "Get Out",
  "Us",
  "The Northman",
  "The Lighthouse",
  "The Revenant",
  "Inside Out",
  "Inside Out 2",
  "A Star Is Born",
  "Room",
  "The Father",
  "Challengers",
  "Civil War"
];

// ---- SIMPLIFIED GENRE MAPPING ----
const SIMPLE_GENRES = {
  Action: "Action",
  Adventure: "Adventure",
  Animation: "Animation",
  Comedy: "Comedy",
  Crime: "Crime",
  Drama: "Drama",
  Fantasy: "Fantasy",
  Horror: "Horror",
  Mystery: "Mystery",
  Romance: "Romance",
  "Science Fiction": "Sci-Fi",
  Thriller: "Thriller",
  Family: "Family",
};

// ---- Generate short 2-sentence overview ----
function shortOverview(text) {
  if (!text) return "";
  const sentences = text.split(".");
  return (sentences.slice(0, 2).join(".") + ".").trim();
}

// ---- Convert TMDB genres to simple genres ----
function mapGenres(genres) {
  return genres
    .map((g) => SIMPLE_GENRES[g.name] || null)
    .filter(Boolean);
}

// ---- Search by title to find TMDB ID ----
async function getTMDBId(title) {
  const res = await axios.get(
    `https://api.themoviedb.org/3/search/movie`,
    {
      params: {
        api_key: TMDB_KEY,
        query: title
      }
    }
  );

  if (!res.data.results.length) return null;
  return res.data.results[0].id;
}

// ---- Main Build Function ----
async function build() {
  const finalMovies = [];

  for (const title of MOVIE_TITLES) {
    console.log("📌 Fetching:", title);

    const id = await getTMDBId(title);
    if (!id) {
      console.log("❌ TMDB ID NOT FOUND:", title);
      continue;
    }

    const detail = await axios.get(
      `https://api.themoviedb.org/3/movie/${id}`,
      { params: { api_key: TMDB_KEY } }
    );

    const m = detail.data;

    finalMovies.push({
      tmdb_id: id,
      title: m.title,
      poster: m.poster_path ? `https://image.tmdb.org/t/p/w500${m.poster_path}` : null,
      overview: shortOverview(m.overview),
      rating: m.vote_average,
      popularity: m.popularity,
      genres: mapGenres(m.genres),
      year: m.release_date?.split("-")[0] || "Unknown"
    });
  }

  const output = `export const FIXED_MOVIES = ${JSON.stringify(finalMovies, null, 2)};`;

  fs.writeFileSync("./src/utils/fixedMovies.js", output);
  console.log("🎉 DONE! fixedMovies.js ساخته شد.");
}

build();
