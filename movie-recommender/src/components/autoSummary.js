// ---------------------------------------------
// autoSummary.js  (Basic & Stable Version)
// ایجاد خلاصه سریع بر اساس انتخاب فیلم‌ها
// ---------------------------------------------

export function autoSummaryFromMovies(list) {
  if (!list || list.length === 0) return "";

  // 1) جمع‌آوری تمام ژانرها
  let allGenres = [];
  list.forEach(m => {
    if (m.genres && Array.isArray(m.genres)) {
      allGenres.push(...m.genres);
    }
  });

  // 2) شمارش فراوانی ژانرها
  const counts = {};
  allGenres.forEach(g => {
    if (!g) return;
    counts[g] = (counts[g] || 0) + 1;
  });

  // 3) مرتب‌سازی از پر تکرارترین تا کمترین
  const sortedGenres = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(x => x[0]);

  const topGenres = sortedGenres.slice(0, 3);

  // 4) عناوین فیلم‌ها
  const titles = list.map(m => m.title).slice(0, 5);

  // ---------------------------------------------
  // 5) ساخت Summary کوتاه و REB-safe
  // ---------------------------------------------
  let text = "";

  if (topGenres.length > 0) {
    text += `User seems to enjoy ${topGenres.join(", ")} movies. `;
  }

  if (titles.length > 0) {
    text += `Selected titles include: ${titles.join(", ")}.`;
  }

  return text.trim();
}
