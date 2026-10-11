/* FloodIntel — Application Entry Point & Orchestration. */
(function (FI) {
  const { $ } = FI;

  async function run() {
    const caseSelect = $("case-select");
    const caseId = caseSelect ? caseSelect.value : ($("area")?.value || "trishuli_basin");
    const area = $("area")?.value || caseId;
    const date = $("date").value, time = $("time").value || "12:00";
    const err = FI.validate(area, date);
    if ($("err")) $("err").textContent = err || "";
    if (err) return;

    const runBtn = $("run");
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.innerHTML = "<span>⏳</span> Processing Satellite Data…";
    }

    if ($("banner")) $("banner").innerHTML = "";

    const ov = $("ov");
    if (ov) {
      ov.style.display = "grid";
      ov.innerHTML = `
        <div style="background:rgba(255,255,255,0.95);padding:18px 24px;border-radius:8px;box-shadow:var(--shadow-md);border:1px solid var(--border-medium);text-align:center;max-width:380px">
          <div style="font-size:14px;font-weight:700;color:var(--navy-900);margin-bottom:6px">Synthesizing Satellite &amp; Connectivity Data…</div>
          <div style="font-size:12px;color:var(--ink-secondary);line-height:1.4">Analyzing Sentinel-1 SAR change, Attention U-Net water candidates, and OpenStreetMap road graph reachability.</div>
        </div>
      `;
    }

    FI.ui.renderSkeleton();

    try {
      FI.state.result = await FI.api.runAssessment({ case_id: caseId, area, date });
      if (ov) ov.style.display = "none";
      FI.ui.drawMap();
      FI.ui.renderFindings(FI.state.result);
      FI.ui.renderReport(FI.state.result);

      if (FI.state.result.status === "data_not_prepared") {
        if ($("sys-status")) {
          $("sys-status").textContent = "Data Pending";
        }
        if ($("banner")) {
          $("banner").innerHTML = `
            <div class="banner-alert warn">
              <span>ℹ</span>
              <span><b>Data Not Prepared:</b> ${FI.state.result.area} (${FI.state.result.date}) · Pipeline inputs pending ingestion</span>
            </div>
          `;
        }
      } else {
        if ($("sys-status")) {
          $("sys-status").textContent = "Assessment Active";
        }
        if ($("banner") && FI.state.result.validation_status) {
          $("banner").innerHTML = `
            <div class="banner-alert info">
              <span>✓</span>
              <span><b>Assessment Active:</b> ${FI.state.result.area} (${FI.state.result.date}) · AI: <code>${FI.state.result.validation_status}</code></span>
            </div>
          `;
        }
      }
    } catch (e) {
      FI.state.result = null;
      if (typeof FI.ui.clearMapLayers === "function") {
        FI.ui.clearMapLayers();
      }
      if (typeof FI.ui.clearReport === "function") {
        FI.ui.clearReport();
      }
      if (ov) {
        ov.style.display = "grid";
        ov.innerHTML = `
          <div style="background:rgba(255,255,255,0.95);padding:18px 24px;border-radius:8px;box-shadow:var(--shadow-md);border:1px solid #fecaca;text-align:center;max-width:380px">
            <div style="font-size:14px;font-weight:700;color:#b91c1c;margin-bottom:6px">Assessment Request Failed</div>
            <div style="font-size:12px;color:var(--ink-secondary)">${e.message || "Unknown error."}</div>
          </div>
        `;
      }
      FI.ui.clearFindings();
      if ($("banner")) {
        $("banner").innerHTML = `
          <div class="banner-alert bad" style="margin-top:8px">
            <span>⚠</span>
            <span><b>Assessment Failed:</b> ${e.message || "Unknown error"}. Ensure FastAPI backend is active on http://127.0.0.1:8000.</span>
          </div>
        `;
      }
    }

    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = "<span>▶</span> Run Flood Assessment";
    }
  }

  FI.runAssessment = run;

  document.addEventListener("DOMContentLoaded", () => {
    FI.ui.initControls();
    FI.ui.drawMap();
    if ($("run")) $("run").onclick = run;
    FI.ui.refreshWeather();

    if (FI.api && typeof FI.api.getCases === "function") {
      FI.api.getCases().then((cases) => {
        if (cases && Array.isArray(cases)) {
          FI.state.cases = {};
          cases.forEach((c) => {
            FI.state.cases[c.id] = c;
          });
        }
      }).catch(() => {});
    }

    const params = new URLSearchParams(window.location.search);
    if (params.get("case") && $("case-select")) {
      FI.ui.switchCase(params.get("case"));
    } else {
      if (params.get("area") && $("area")) {
        $("area").value = params.get("area");
      }
      if (params.get("date") && $("date")) {
        $("date").value = params.get("date");
      }
    }
    if (params.get("autorun") === "1" || params.get("autorun") === "true") {
      setTimeout(async () => {
        await run();
        if (params.get("basemap") === "satellite") {
          FI.ui.toggleBasemap("satellite");
        }
        if (params.get("review") === "1") {
          const btn = $("btn-toggle-review");
          if (btn) btn.click();
          if (params.get("panel") && $("modal-panel-select")) {
            $("modal-panel-select").value = params.get("panel");
            $("modal-panel-select").dispatchEvent(new Event("change"));
          }
        }
        if (params.get("tab")) {
          const tBtn = $(`tab-btn-${params.get("tab")}`);
          if (tBtn) tBtn.click();
        }
      }, 300);
    }
  });
})(window.FI);
