import onboardingMovies from "./data/pilot_support20_onboarding.json";

test("uses 100 unique ordered ML-32M onboarding movies with linked metadata", () => {
  expect(onboardingMovies).toHaveLength(100);
  expect(new Set(onboardingMovies.map((movie) => movie.movieId)).size).toBe(100);
  expect(new Set(onboardingMovies.map((movie) => movie.title)).size).toBe(100);
  onboardingMovies.forEach((movie) => {
    expect(Number.isInteger(movie.movieId)).toBe(true);
    expect(Number.isInteger(movie.modelItemId)).toBe(true);
    expect(Number.isInteger(movie.tmdbId)).toBe(true);
    expect(movie.imdbId).toMatch(/^\d+$/);
    expect(movie.title).toMatch(/\(\d{4}\)$/);
    expect(movie.genres.length).toBeGreaterThan(0);
    expect(movie.releaseYear).toBe(Number(movie.title.match(/\((\d{4})\)$/)[1]));
    expect(movie.frozenPopularity).toBe(movie.pilotTrainPositiveCount);
  });
});
