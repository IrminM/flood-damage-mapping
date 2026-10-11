/* Global namespace + app configuration. Loaded first. */
window.FI = {
  config: {
    USE_MOCK: false,         // connected to FastAPI backend
    API_BASE: (typeof window !== "undefined" && window.location && (window.location.protocol === "http:" || window.location.protocol === "https:") && window.location.port === "8000")
      ? "/api"
      : "http://127.0.0.1:8000/api",
    ENDPOINTS: {             // backend compatibility routes
      assess: "/assess"
    },
    WEATHER: {               // Open-Meteo: free, no API key
      GEOCODE_URL: "https://geocoding-api.open-meteo.com/v1/search",
      FORECAST_URL: "https://api.open-meteo.com/v1/forecast",
      ARCHIVE_URL: "https://archive-api.open-meteo.com/v1/archive",
      ARCHIVE_AFTER_DAYS: 90 // events older than this use the archive service
    }
  },
  state: { result: null, period: "post", source: "s1", layers: { flood: 1, roads: 1, bridges: 1, settl: 1, flow: 1, debris: 1, buildings: 1 } },
  api: {},
  ui: {}
};
