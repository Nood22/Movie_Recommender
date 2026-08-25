function detailEntryMessage(entry) {
  if (typeof entry === "string") return entry.trim();
  if (!entry || typeof entry !== "object") return "";

  const message = typeof entry.msg === "string" ? entry.msg.trim() : "";
  if (!message) return "";

  const location = Array.isArray(entry.loc)
    ? entry.loc.filter((part) => part !== "body").join(".")
    : "";
  return location ? `${location}: ${message}` : message;
}

export function apiErrorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }

  if (Array.isArray(detail)) {
    const message = detail.map(detailEntryMessage).filter(Boolean).join("; ");
    if (message) return message;
  }

  if (detail && typeof detail === "object") {
    const message = detailEntryMessage(detail);
    if (message) return message;
  }

  return fallback;
}
