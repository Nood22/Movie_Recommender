import onboardingMovies from "./data/pilot_support20_onboarding.json";

test("uses 50 unique ML-32M onboarding movies with linked metadata", () => {
  expect(onboardingMovies).toHaveLength(50);
  expect(new Set(onboardingMovies.map((movie) => movie.movieId)).size).toBe(50);
  onboardingMovies.forEach((movie) => {
    expect(Number.isInteger(movie.movieId)).toBe(true);
    expect(Number.isInteger(movie.modelItemId)).toBe(true);
    expect(Number.isInteger(movie.tmdbId)).toBe(true);
    expect(movie.imdbId).toMatch(/^\d+$/);
    expect(movie.title).toMatch(/\(\d{4}\)$/);
    expect(movie.genres.length).toBeGreaterThan(0);
  });
});
