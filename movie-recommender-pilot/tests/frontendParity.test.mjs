import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const originalRoot = new URL("../../movie-recommender/src/", import.meta.url);
const pilotRoot = new URL("../src/", import.meta.url);

async function source(root, relativePath) {
  return (await readFile(new URL(relativePath, root), "utf8"))
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+$/gm, "");
}

function section(sourceText, startMarker, endMarker) {
  const start = sourceText.indexOf(startMarker);
  const end = sourceText.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return sourceText.slice(start, end);
}

test("pilot keeps the original routes and shared Studio Auréa support code", async () => {
  for (const relativePath of [
    "index.js",
    "components/MovieCard.jsx",
    "components/Navbar.jsx",
    "components/autoSummary.js",
  ]) {
    assert.equal(
      await source(pilotRoot, relativePath),
      await source(originalRoot, relativePath),
      `${relativePath} differs from the original frontend`
    );
  }

  const pilotAppSource = await source(pilotRoot, "App.js");
  assert.match(pilotAppSource, /window\.location\.pathname\.startsWith\("\/pilot\/"\)/);
  const pilotApp = pilotAppSource
    .replace(
      /  \/\/ The same build[\s\S]*?  return \(/,
      "  return ("
    )
    .replace('<Router basename={basename}>', "<Router>");
  assert.equal(pilotApp, await source(originalRoot, "App.js"));
});

test("pilot preserves the original screen identity without exposing provenance", async () => {
  const screenContracts = [
    ["components/LandingPage.jsx", ["AI Movie Recommender", "TEARS", "GERS", "#0d0714"]],
    ["components/TearsApp.jsx", ["TEARS – Summary-Based Recommender", "#00C8FF", "Your Summary"]],
    ["components/GersApp.jsx", ["GERS – Genre-Based Recommender", "green-300", "Select genres:"]],
  ];

  for (const [relativePath, markers] of screenContracts) {
    const original = await source(originalRoot, relativePath);
    const pilot = await source(pilotRoot, relativePath);
    for (const marker of markers) {
      assert.match(original, new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
      assert.match(pilot, new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }
    assert.match(pilot, /publicAsset\("logo\.png"\)/);
  }

  const participantScreens = await Promise.all(
    ["components/LandingPage.jsx", "components/TearsApp.jsx", "components/GersApp.jsx"]
      .map((relativePath) => source(pilotRoot, relativePath))
  );
  assert.match(participantScreens[0], /Your movies, your taste profile/);
  assert.match(participantScreens[1], /Describe and refine your movie taste/);
  assert.match(participantScreens[2], /Build your taste from movies and genres/);
  assert.doesNotMatch(
    participantScreens.join("\n"),
    /200,948|180,948|9,763|trained on|training users|checkpoint|seed|epochs?|Score:/i
  );
});

test("pilot screens include responsive layouts and accessible error feedback", async () => {
  const landing = await source(pilotRoot, "components/LandingPage.jsx");
  const tears = await source(pilotRoot, "components/TearsApp.jsx");
  const gers = await source(pilotRoot, "components/GersApp.jsx");
  const styles = await source(pilotRoot, "index.css");

  assert.match(landing, /flex-col gap-4 sm:w-auto sm:flex-row/);
  for (const screen of [tears, gers]) {
    assert.match(screen, /flex flex-col xl:flex-row/);
    assert.match(screen, /grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5/);
    assert.match(screen, /role="alert"/);
    assert.doesNotMatch(screen, /alert\(/);
  }
  assert.match(styles, /prefers-reduced-motion: reduce/);
  assert.match(styles, /focus-visible:ring-2/);
});

test("TEARS gives participant edits a spacious, vertically resizable textarea", async () => {
  const tears = await source(pilotRoot, "components/TearsApp.jsx");
  assert.match(
    tears,
    /h-72 min-h-\[18rem\] sm:h-80 xl:h-\[22rem\] resize-y overflow-y-auto/
  );
  assert.match(tears, /aria-label="Editable movie taste summary"/);
  assert.match(tears, /Participant edits are signed and submitted verbatim\.\n\s*summary,/);
  assert.doesNotMatch(tears, /summary: summary\.trim\(\)/);
});

test("pilot frontend derives the API prefix from its deployment base", async () => {
  const config = await source(pilotRoot, "utils/apiConfig.js");
  const tears = await source(pilotRoot, "components/TearsApp.jsx");
  const gers = await source(pilotRoot, "components/GersApp.jsx");
  const deployment = await source(pilotRoot, "data/serving_deployment.json");

  assert.match(config, /process\.env\.PUBLIC_URL/);
  assert.match(config, /REACT_APP_QUALITY_API_URL/);
  assert.match(config, /`\$\{publicBase\}\/api`/);
  assert.match(tears, /`\$\{PILOT_API_URL\}\/summarize`/);
  assert.match(tears, /`\$\{PILOT_API_URL\}\/recommend`/);
  assert.match(gers, /`\$\{PILOT_API_URL\}\/gers`/);
  assert.match(tears, /servingDeployment\.matrix_fingerprint/);
  assert.match(gers, /servingDeployment\.matrix_fingerprint/);
  assert.match(
    deployment,
    /27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce/
  );
  assert.doesNotMatch(`${tears}\n${gers}`, /127\.0\.0\.1:8001/);
});

test("participant selection keeps catalog semantics and invisible request provenance", async () => {
  const pilotTears = await source(pilotRoot, "components/TearsApp.jsx");
  const pilotGers = await source(pilotRoot, "components/GersApp.jsx");
  const sessionLogging = await source(pilotRoot, "study/useStudySession.js");
  assert.match(pilotTears, /canonicalMovieLensId\(movie\.movieId\)/);
  assert.match(pilotTears, /selected\.map\(\(movie\) => Number\(movie\.movieId\)\)/);
  assert.match(pilotTears, /responseMatchesCurrent/);
  assert.match(pilotGers, /FIXED_MOVIELENS_CATALOG/);
  assert.match(pilotGers, /genreNamesWithFrequencies/);
  assert.doesNotMatch(pilotGers, /new Set\(\[\.\.\.movieGenreIds/);
  assert.match(sessionLogging, /participant_id: participantId/);
  assert.match(sessionLogging, /session_id: sessionId/);
  assert.match(sessionLogging, /input_signature: await inputSignature\(input\)/);
  assert.match(sessionLogging, /\/study\/render/);
});

test("participant screens omit study activities, questionnaires, trials, and context work", async () => {
  const participantScreens = await Promise.all(
    ["components/LandingPage.jsx", "components/TearsApp.jsx", "components/GersApp.jsx"]
      .map((relativePath) => source(pilotRoot, relativePath))
  );
  const renderedSource = participantScreens.join("\n");

  assert.doesNotMatch(renderedSource, /StudyWorkspace|Study Activities/i);
  assert.doesNotMatch(
    renderedSource,
    /questionnaire|instrument|trial|attempt|researcher|debug|Task 4/i
  );
  assert.doesNotMatch(renderedSource, /Choose a context|Select a context|CANONICAL_CONTEXT/);
});

test("pilot Tailwind classes do not contain the known separator typos", async () => {
  const tears = await source(pilotRoot, "components/TearsApp.jsx");
  assert.doesNotMatch(tears, /bg-gradient_to_br|justify_center|bg_white/);
});
