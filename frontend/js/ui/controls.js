/* Wires up tab navigation, map toolbar, floating legend, presets, review modal, and print actions. */
(function (FI) {
  const { $ } = FI;

  function resolveUrl(url) {
    if (!url) return "";
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    const base = (FI.config.API_BASE || "").replace(/\/api\/?$/, "");
    return base + (url.startsWith("/") ? url : "/" + url);
  }

  function weatherNow() {
    const area = $("area").value, date = $("date").value, err = FI.validate(area, date);
    if ($("err")) $("err").textContent = err || "";
    if (!err) FI.ui.loadWeather(area, date, $("time").value || "12:00");
  }
  FI.ui.refreshWeather = weatherNow;

  const DEFAULT_CASES = {
    trishuli_basin: {
      id: "trishuli_basin",
      title: "Trishuli River Basin & Upper Gorge",
      region: "Rasuwa District & Trishuli Corridor",
      country: "Nepal",
      event_date: "2026-08-26",
      status: "ready",
      bbox: [85.17, 27.95, 85.36, 28.18],
      center: [27.97, 85.20],
      default_zoom: 11,
      validation_status: "not_validated_on_Trishuli",
      sensors_summary: "S1 (Aug 16/28) · S2 (Aug 27) · GLO-30 DEM · OSM"
    },
    emilia_romagna: {
      id: "emilia_romagna",
      title: "Emilia-Romagna River Basins",
      region: "Ravenna / Forlì / Faenza Corridor",
      country: "Italy",
      event_date: "2023-05-17",
      status: "data_not_prepared",
      bbox: [11.70, 44.15, 12.35, 44.55],
      center: [44.35, 12.02],
      default_zoom: 11,
      validation_status: "uncalibrated_external_aoi",
      sensors_summary: "S1 IW (Track 44) · S2B (Tile 32TPQ) · GLO-30 DEM · OSM"
    },
    valencia_dana: {
      id: "valencia_dana",
      title: "Valencia DANA Flash Flooding",
      region: "Turia River Basin & L'Horta Sud",
      country: "Spain",
      event_date: "2024-10-29",
      status: "data_not_prepared",
      bbox: [-0.60, 39.25, -0.20, 39.60],
      center: [39.42, -0.40],
      default_zoom: 11,
      validation_status: "uncalibrated_external_aoi",
      sensors_summary: "S1 IW (Track 8) · S2 (Oct 31) · GLO-30 DEM · OSM"
    }
  };

  FI.ui.switchCase = function (caseId) {
    const registry = FI.state.cases || DEFAULT_CASES;
    const caseObj = registry[caseId] || DEFAULT_CASES[caseId];
    if (!caseObj) return;

    if ($("case-select")) $("case-select").value = caseId;
    if ($("area")) $("area").value = caseId;
    if ($("date")) $("date").value = caseObj.event_date;
    if ($("brand-subtitle")) $("brand-subtitle").textContent = "— " + caseObj.title;
    if ($("sensor-chip-text")) $("sensor-chip-text").textContent = "🛰 " + caseObj.sensors_summary;
    if ($("val-pill-text")) $("val-pill-text").textContent = "AI Status: " + caseObj.validation_status;

    const valPill = $("val-pill");
    if (valPill) {
      if (caseObj.status === "data_not_prepared") {
        valPill.title = "Data not prepared for this geographic basin. Model uncalibrated for external AOI.";
      } else {
        valPill.title = "Model uncalibrated on Himalayan terrain (trained on Sen1Floods11 lowland plains)";
      }
    }

    if (typeof FI.ui.setCaseView === "function") {
      FI.ui.setCaseView(caseObj);
    }

    if (typeof FI.ui.loadWeather === "function") {
      const coordStr = `${caseObj.bbox[0]},${caseObj.bbox[1]},${caseObj.bbox[2]},${caseObj.bbox[3]}`;
      FI.ui.loadWeather(coordStr, caseObj.event_date, $("time")?.value || "12:00");
    }

    if (typeof FI.runAssessment === "function") {
      FI.runAssessment();
    }
  };

  FI.ui.initControls = function () {
    // 1. Sidebar Tab Navigation
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach((btn) => {
      btn.onclick = () => {
        const targetId = btn.getAttribute("data-tab");
        if (!targetId) return;

        tabBtns.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        tabPanes.forEach((p) => p.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const targetPane = $(targetId);
        if (targetPane) targetPane.classList.add("active");
      };
    });

    // 2. Disaster Case Study Selector
    const caseSelect = $("case-select");
    if (caseSelect) {
      caseSelect.onchange = () => {
        FI.ui.switchCase(caseSelect.value);
      };
    }

    document.querySelectorAll(".btn-preset-pill[data-val]").forEach((btn) => {
      btn.onclick = () => {
        const val = btn.getAttribute("data-val");
        if (val) {
          FI.ui.switchCase(val);
        }
      };
    });

    // 3. Basemap Controls
    $("bm-carto") && ($("bm-carto").onclick = () => FI.ui.toggleBasemap("carto"));
    $("bm-sat") && ($("bm-sat").onclick = () => FI.ui.toggleBasemap("satellite"));

    // 4. Zoom to AOI
    $("btn-zoom-fit") && ($("btn-zoom-fit").onclick = () => FI.ui.zoomToAoi());

    // 5. Floating Legend Collapse / Expand Toggle
    const legendBox = $("map-layer-legend");
    const legendToggleBtn = $("btn-collapse-legend");
    if (legendToggleBtn && legendBox) {
      let isCollapsed = false;
      legendToggleBtn.onclick = () => {
        isCollapsed = !isCollapsed;
        const body = legendBox.querySelector(".legend-body");
        if (body) body.style.display = isCollapsed ? "none" : "block";
        legendToggleBtn.textContent = isCollapsed ? "▼" : "▲";
        legendToggleBtn.title = isCollapsed ? "Expand legend" : "Collapse legend";
      };
    }

    // 6. AI Review Modal Lightbox
    const modal = $("review-modal");
    const reviewImg = $("review-img");
    const modalPanelSelect = $("modal-panel-select");
    const modalTitle = $("modal-title");
    const btnModalOverview = $("btn-modal-overview");
    const btnModalOverlap = $("btn-modal-overlap");
    const btnModalZoom = $("btn-modal-zoom");
    const btnCloseModal = $("btn-close-modal");

    const PANEL_MAP = {
      overview: {
        layerKey: "review_full_aoi_png",
        fallback: "/api/layers/review_visualizations_full_aoi",
        title: "Multi-Panel AI & SAR Spatial Evidence Review (Full Scene)",
        alt: "Multi-Panel Full Scene AI Review",
      },
      zoom1: {
        fallback: "/api/layers/review_visualizations_zoom_central_valley",
        title: "Spatial Evidence Review — Zoom 1: Central Trishuli Valley & Confluence",
        alt: "Zoom 1 Central Valley Review",
      },
      zoom2: {
        fallback: "/api/layers/review_visualizations_zoom_north_gorge",
        title: "Spatial Evidence Review — Zoom 2: Northern Trishuli Gorge Corridor",
        alt: "Zoom 2 North Gorge Review",
      },
      panel_a: {
        fallback: "/api/layers/map_01_s2_rgb_context",
        title: "Panel A: Sentinel-2 L1C True Color RGB Context (August 2026)",
        alt: "Panel A Sentinel-2 RGB Context",
      },
      panel_b: {
        fallback: "/api/layers/map_02_dem_slope_context",
        title: "Panel B: Copernicus DEM 30m Slope & Valleys",
        alt: "Panel B DEM Terrain Slope",
      },
      panel_c: {
        fallback: "/api/layers/map_03_raw_model_probability",
        title: "Panel C: Raw Attention U-Net Continuous Flood Probability (Threshold = 0.40)",
        alt: "Panel C Raw AI Probability Heatmap",
      },
      panel_d: {
        fallback: "/api/layers/map_04_terrain_gated_candidates",
        title: "Panel D: Terrain-Gated Candidate Water Mask (3.75 km² Gated vs 475 km² Rejected)",
        alt: "Panel D Terrain-Gated Candidate Water",
      },
      panel_e: {
        fallback: "/api/layers/map_05_stage2_sar_change_evidence",
        title: "Panel E: Sentinel-1 SAR Multi-Temporal Change Evidence (Baseline SAR)",
        alt: "Panel E Sentinel-1 SAR Change Evidence",
      },
      overlap: {
        layerKey: "review_overlap_png",
        fallback: "/api/layers/map_06_evidence_overlap_and_disagreement",
        title: "Panel F: Multi-Evidence Spatial Overlap & Disagreement (HOT + AI + SAR)",
        alt: "Evidence Disagreement Map",
      },
      panel_f: {
        layerKey: "review_overlap_png",
        fallback: "/api/layers/map_06_evidence_overlap_and_disagreement",
        title: "Panel F: Multi-Evidence Spatial Overlap & Disagreement (HOT + AI + SAR)",
        alt: "Evidence Disagreement Map",
      },
      agreement: {
        layerKey: "agreement_map",
        fallback: "assets/agreement_map.png",
        title: "Toolkit Multi-Source Spatial Agreement Map (IWM + TASA + HOT vs AI Water)",
        alt: "Validation Toolkit Multi-Source Agreement Map",
      },
    };

    function resetZoom() {
      if (reviewImg) reviewImg.classList.remove("zoomed");
      if (btnModalZoom) {
        btnModalZoom.textContent = "🔍 100% Zoom";
        btnModalZoom.classList.remove("active");
      }
    }

    function toggleZoom() {
      if (!reviewImg) return;
      const isZoomed = reviewImg.classList.toggle("zoomed");
      if (btnModalZoom) {
        btnModalZoom.textContent = isZoomed ? "⤢ Fit Window" : "🔍 100% Zoom";
        btnModalZoom.classList.toggle("active", isZoomed);
      }
    }

    function setModalImage(kind) {
      resetZoom();
      const info = PANEL_MAP[kind] || PANEL_MAP.overview;
      const R = FI.state.result;
      const rawUrl = (info.layerKey && R?.layers?.[info.layerKey]) ? R.layers[info.layerKey] : info.fallback;
      const resolved = resolveUrl(rawUrl);
      const sep = resolved.includes("?") ? "&" : "?";
      const bustUrl = `${resolved}${sep}v=${Date.now()}`;

      if (btnModalOverview) {
        btnModalOverview.classList.toggle("active", kind === "overview");
      }
      if (btnModalOverlap) {
        btnModalOverlap.classList.toggle("active", kind === "overlap" || kind === "panel_f");
      }
      const btnModalVal = $("btn-modal-val");
      if (btnModalVal) {
        btnModalVal.classList.toggle("active", kind === "agreement");
      }
      if (modalPanelSelect) {
        modalPanelSelect.value = (kind === "overlap") ? "panel_f" : kind;
      }
      if (modalTitle) {
        modalTitle.textContent = info.title;
      }
      if (reviewImg) {
        reviewImg.src = bustUrl;
        reviewImg.alt = info.alt;
      }
    }

    function openReviewModal(kind) {
      if (!modal) return;
      modal.classList.add("open");
      setModalImage(kind || "overview");
    }

    function closeReviewModal() {
      if (!modal) return;
      modal.classList.remove("open");
      resetZoom();
    }

    $("btn-toggle-review") && ($("btn-toggle-review").onclick = () => openReviewModal("overview"));
    $("btn-open-review-overview") && ($("btn-open-review-overview").onclick = () => openReviewModal("overview"));
    $("btn-open-review-overlap") && ($("btn-open-review-overlap").onclick = () => openReviewModal("panel_f"));
    $("btn-open-toolkit-agreement") && ($("btn-open-toolkit-agreement").onclick = () => openReviewModal("agreement"));

    btnModalOverview && (btnModalOverview.onclick = () => setModalImage("overview"));
    btnModalOverlap && (btnModalOverlap.onclick = () => setModalImage("panel_f"));
    const btnModalVal = $("btn-modal-val");
    btnModalVal && (btnModalVal.onclick = () => setModalImage("agreement"));
    modalPanelSelect && (modalPanelSelect.onchange = (e) => setModalImage(e.target.value));
    btnModalZoom && (btnModalZoom.onclick = toggleZoom);
    reviewImg && (reviewImg.onclick = toggleZoom);
    btnCloseModal && (btnCloseModal.onclick = closeReviewModal);

    if (modal) {
      modal.onclick = (e) => {
        if (e.target === modal) closeReviewModal();
      };
    }

    // 7. Situation Report Print Button
    const printBtn = $("print");
    if (printBtn) {
      printBtn.onclick = () => {
        // Activate report tab first for proper rendering
        const reportTabBtn = $("tab-btn-report");
        if (reportTabBtn) reportTabBtn.click();
        window.print();
      };
    }

    // 8. Weather Refresh Trigger
    const wxRefreshBtn = $("wxb");
    if (wxRefreshBtn) {
      wxRefreshBtn.onclick = (e) => {
        e.stopPropagation();
        weatherNow();
      };
    }
  };
})(window.FI);
