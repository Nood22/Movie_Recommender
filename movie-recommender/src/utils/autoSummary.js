// src/utils/autoSummary.js

export function autoSummaryFromMovies(list) {
  if (!list || list.length === 0) return "";

  // 1) Collect all genres
  let allGenres = [];
  list.forEach(m => {
    if (m.genres) allGenres.push(...m.genres);
  });

  // Count genres
  const counts = {};
  allGenres.forEach(g => {
    if (!g) return;
    counts[g] = (counts[g] || 0) + 1;
  });

  // Sort by frequency
  const sortedGenres = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(x => x[0]);

  const topGenres = sortedGenres.slice(0, 3);

  // 2) Movie titles
  const titles = list.map(m => m.title).slice(0, 5);

  // Build summary
  let text = "";

  if (topGenres.length > 0) {
    text += `User seems to enjoy ${topGenres.join(", ")} movies. `;
  }

  if (titles.length > 0) {
    text += `Selected titles include: ${titles.join(", ")}.`;
  }

  return text.trim();
}
