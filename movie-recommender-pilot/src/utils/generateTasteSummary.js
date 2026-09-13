// src/utils/generateTasteSummary.js

// No API, No OpenAI — simple taste summary builder

export function generateTasteSummary(selectedMovies, context = "") {
  if (!selectedMovies || selectedMovies.length === 0) {
    return context ? `Context: ${context}` : "";
  }

  const titles = selectedMovies.map((m) => m.title);

  let summary = `User likes: ${titles.join(", ")}`;

  if (context) summary += `\nContext: ${context}`;

  return summary;
}
