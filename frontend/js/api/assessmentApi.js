/* Flood assessment API client. The only place that talks to the backend for assessments. */
(function (FI) {
  FI.api.getCases = async function () {
    const cfg = FI.config;
    const endpoint = cfg.API_BASE + "/cases";
    try {
      const res = await fetch(endpoint);
      if (res.ok) return await res.json();
    } catch (_) {}
    return null;
  };

  FI.api.runAssessment = async function ({ case_id, area, date }) {
    const cfg = FI.config;
    if (cfg.USE_MOCK) {
      await new Promise((r) => setTimeout(r, 1800));
      return FI.mockAssessment(area, date);
    }

    const endpoint = cfg.API_BASE + cfg.ENDPOINTS.assess;
    let res;
    try {
      res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_id: case_id || area, area, date })
      });
    } catch (networkErr) {
      throw new Error(`Unable to connect to backend service at ${endpoint}. Please ensure the FastAPI backend is running (open a terminal, run 'python scripts/run_server.py', and keep it open). Note: 127.0.0.1 refers to this computer.`);
    }

    if (!res.ok) {
      let detailMsg = "";
      try {
        const errorData = await res.json();
        detailMsg = errorData.detail || errorData.message || "";
      } catch (_) {
        // Fall back to status code
      }
      throw new Error(detailMsg || `Backend returned HTTP error ${res.status}`);
    }

    return await res.json();
  };
})(window.FI);
