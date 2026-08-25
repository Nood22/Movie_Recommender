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

  const pilotApp = (await source(pilotRoot, "App.js")).replace(
    '<Router basename={process.env.PUBLIC_URL || "/"}>',
    "<Router>"
  );
  assert.equal(pilotApp, await source(originalRoot, "App.js"));
});

test("pilot preserves the original screen identity while identifying the new model", async () => {
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

  assert.match(
    await source(pilotRoot, "components/LandingPage.jsx"),
    /Full TEARS · 200,948 profiles · GERS scientific pilot/
  );
  assert.match(
    await source(pilotRoot, "components/TearsApp.jsx"),
    /Full 200,948-profile dataset · 180,948 training users/
  );
  assert.match(
    await source(pilotRoot, "components/GersApp.jsx"),
    /Scientific pilot · trained on 9,763 user profiles/
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

test("participant selection behavior remains aligned with the original", async () => {
  const originalTears = await source(originalRoot, "components/TearsApp.jsx");
  const pilotTears = await source(pilotRoot, "components/TearsApp.jsx");
  const normalizedPilotTears = section(
    pilotTears,
    "const requestSummary",
    "      REQUEST TEARS RECOMMENDATIONS"
  ).replaceAll("PILOT_API_URL", "QUALITY_API_URL");
  assert.equal(
    normalizedPilotTears,
    section(
      originalTears,
      "const requestSummary",
      "      REQUEST TEARS RECOMMENDATIONS"
    )
  );

  const originalGers = await source(originalRoot, "components/GersApp.jsx");
  const pilotGers = await source(pilotRoot, "components/GersApp.jsx");
  assert.equal(
    section(pilotGers, "  const toggleSelect", "     GET RECOMMENDATIONS"),
    section(originalGers, "  const toggleSelect", "     GET RECOMMENDATIONS")
  );
});

test("pilot Tailwind classes do not contain the known separator typos", async () => {
  const tears = await source(pilotRoot, "components/TearsApp.jsx");
  assert.doesNotMatch(tears, /bg-gradient_to_br|justify_center|bg_white/);
});
