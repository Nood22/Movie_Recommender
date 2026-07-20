import { FIXED_MOVIES } from "./FIXED_MOVIES_TEARS";

export async function fetchLeftPanelMovies() {
  console.log("🔥 FETCHING FIXED MOVIES (SHOULD BE 50)", FIXED_MOVIES.length);
  return FIXED_MOVIES;
}
