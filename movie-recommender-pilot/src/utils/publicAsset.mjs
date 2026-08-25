export function publicAsset(path) {
  const base = String(process.env.PUBLIC_URL || "").replace(/\/$/, "");
  const relativePath = String(path).replace(/^\/+/, "");
  return `${base}/${relativePath}`;
}
