const publicBase = (process.env.PUBLIC_URL || "").replace(/\/$/, "");

// A public /pilot build and a local root build can both use the API served by
// pilot_api.py without duplicating deployment-specific URLs in components.
export const PILOT_API_URL =
  process.env.REACT_APP_QUALITY_API_URL || `${publicBase}/api`;
