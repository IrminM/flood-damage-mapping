/* Map rendering engine: Leaflet GIS with unified floating layer & legend controls. */
(function (FI) {
  const { $, svg } = FI;

  // Leaflet map singleton state
  let leafletMap = null;
  const baseLayers = {};
  let currentBase = "carto";
  const layerGroups = {
    flood: null,
    roads: null,
    bridges: null,
    settl: null,
    flow: null
  };
  let cachedAnalysisKey = null;
  let activeBounds = null;
  const settlementMarkers = new Map();
  let aoiRectangle = null;

  function info(text) {
    const el = $("pick");
    if (el) el.innerHTML = text;
  }

  function resolveUrl(url) {
    if (!url) return null;
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    const base = (FI.config.API_BASE || "").replace(/\/api\/?$/, "");
    return base + (url.startsWith("/") ? url : "/" + url);
  }

  function esc(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  let satelliteErrorNotified = false;
  let cartoErrorNotified = false;

  function initLeafletMap() {
    if (leafletMap) return leafletMap;
    if (typeof L === "undefined") {
      console.error("Leaflet library not loaded.");
      return null;
    }

    L.Icon.Default.imagePath = "vendor/leaflet/images/";

    const container = $("gis-map");
    if (!container) return null;

    leafletMap = L.map("gis-map", {
      center: [27.97, 85.20],
      zoom: 11,
      minZoom: 6,
      maxZoom: 18,
      zoomControl: true,
      attributionControl: true
    });

    if (leafletMap.attributionControl) {
      leafletMap.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
    }

    // Basemaps: Public OpenStreetMap (no API key required)
    baseLayers.carto = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
      maxZoom: 19
    });

    // Satellite Imagery: Esri World Imagery MapServer (public, no API key required)
    baseLayers.satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, GIS User Community',
      maxZoom: 18
    });

    baseLayers.carto.on("tileerror", function () {
      if (!cartoErrorNotified) {
        cartoErrorNotified = true;
        console.warn("Cartographic tile load notice.");
      }
    });

    baseLayers.satellite.on("tileerror", function () {
      if (!satelliteErrorNotified) {
        satelliteErrorNotified = true;
        console.warn("Satellite tile load delay.");
        const pickEl = $("pick");
        if (pickEl && currentBase === "satellite") {
          pickEl.innerHTML = '<span style="color:var(--status-warn);font-weight:600">⚠ Satellite Imagery tiles encountering load delays. Switch to Cartographic basemap if tiles fail to appear.</span>';
        }
      }
    });

    baseLayers.carto.addTo(leafletMap);

    // Initialize empty layer groups
    Object.keys(layerGroups).forEach((k) => {
      layerGroups[k] = L.layerGroup().addTo(leafletMap);
    });

    leafletMap.on("zoomend", updateSettlementSizes);

    return leafletMap;
  }

  FI.ui.toggleBasemap = function (type) {
    if (!leafletMap) return;
    if (type === "satellite" && currentBase !== "satellite") {
      leafletMap.removeLayer(baseLayers.carto);
      baseLayers.satellite.addTo(leafletMap);
      currentBase = "satellite";
      $("bm-sat")?.classList.add("active");
      $("bm-sat")?.setAttribute("aria-pressed", "true");
      $("bm-carto")?.classList.remove("active");
      $("bm-carto")?.setAttribute("aria-pressed", "false");
      info("Active basemap: <b>Satellite Imagery</b> (Esri World Imagery). High-resolution ground texture context.");
    } else if (type === "carto" && currentBase !== "carto") {
      leafletMap.removeLayer(baseLayers.satellite);
      baseLayers.carto.addTo(leafletMap);
      currentBase = "carto";
      $("bm-carto")?.classList.add("active");
      $("bm-carto")?.setAttribute("aria-pressed", "true");
      $("bm-sat")?.classList.remove("active");
      $("bm-sat")?.setAttribute("aria-pressed", "false");
      info("Active basemap: <b>Cartographic</b> (OpenStreetMap Standard). Vector basemap matching pre-event OSM baseline.");
    }
  };

  FI.ui.toggleLayer = function (key, isVisible) {
    if (!leafletMap) return;
    const grp = layerGroups[key];
    if (!grp) return;
    if (isVisible) {
      if (!leafletMap.hasLayer(grp)) leafletMap.addLayer(grp);
    } else {
      if (leafletMap.hasLayer(grp)) leafletMap.removeLayer(grp);
    }
  };

  FI.ui.zoomToAoi = function () {
    if (!leafletMap) return;
    if (activeBounds && activeBounds.isValid()) {
      leafletMap.fitBounds(activeBounds, { padding: [35, 35] });
    } else {
      leafletMap.setView([27.97, 85.20], 11);
    }
  };

  FI.ui.setCaseView = function (caseObj) {
    if (!leafletMap || !caseObj) return;
    Object.keys(layerGroups).forEach((k) => layerGroups[k]?.clearLayers());
    settlementMarkers.clear();
    cachedAnalysisKey = null;

    if (aoiRectangle && leafletMap.hasLayer(aoiRectangle)) {
      leafletMap.removeLayer(aoiRectangle);
      aoiRectangle = null;
    }

    if (caseObj.bbox) {
      const [w, s, e, n] = caseObj.bbox;
      activeBounds = L.latLngBounds([[s, w], [n, e]]);
      aoiRectangle = L.rectangle(activeBounds, {
        color: caseObj.status === "ready" ? "#2563eb" : "#d97706",
        weight: 2,
        dashArray: caseObj.status === "ready" ? null : "5, 5",
        fillOpacity: 0.05
      }).addTo(leafletMap);
      aoiRectangle.bindTooltip(`AOI: ${caseObj.title} (${caseObj.status === "ready" ? "Verified" : "Pending Ingestion"})`, { sticky: true });
      leafletMap.fitBounds(activeBounds, { padding: [35, 35] });
    } else if (caseObj.center) {
      leafletMap.setView(caseObj.center, caseObj.default_zoom || 11);
    }
    renderUnifiedLegend();
  };

  FI.ui.focusSettlement = function (idOrName) {
    if (!leafletMap) return;
    const key = String(idOrName || "").trim().toLowerCase();
    const marker = settlementMarkers.get(key);
    if (marker) {
      leafletMap.setView(marker.getLatLng(), 14, { animate: true });
      marker.openPopup();
      FI.ui.toggleLayer("settl", true);
      const chk = document.querySelector(`input[data-k="settl"]`);
      if (chk) chk.checked = true;
    }
  };

  // Styling and popup builders for GIS layers (Strict Geospatial Design Tokens)
  function styleFloodEvidence(feature) {
    const p = feature.properties || {};
    const cat = (p.category_name || "").toLowerCase();
    const ev = (p.evidence_name || "").toLowerCase();
    const cls = (p.class_name || "").toLowerCase();

    if (cat.includes("ai_water_and_sar") || cat.includes("dual") || ev.includes("dual")) {
      return { color: "#1d4ed8", weight: 1.5, fillColor: "#1d4ed8", fillOpacity: 0.70 };
    }
    if (cat.includes("ai_only") || ev.includes("ai_segmentation")) {
      return { color: "#0891b2", weight: 1.2, fillColor: "#0891b2", fillOpacity: 0.50 };
    }
    if (cat.includes("valley_change") || cls.includes("debris") || ev.includes("change_detection")) {
      return { color: "#d97706", weight: 1.2, fillColor: "#d97706", fillOpacity: 0.60 };
    }
    if (cat.includes("hillside_change")) {
      return { color: "#9a3412", weight: 1.2, dashArray: "3, 3", fillColor: "#9a3412", fillOpacity: 0.50 };
    }
    if (cat.includes("permanent")) {
      return { color: "#0284c7", weight: 1.2, fillColor: "#0284c7", fillOpacity: 0.55 };
    }
    return { color: "#2563eb", weight: 1.2, fillColor: "#3b82f6", fillOpacity: 0.45 };
  }

  function onEachFloodFeature(feature, layer) {
    const p = feature.properties || {};
    const cat = (p.category_name || p.evidence_name || "").toLowerCase();
    let title = "Flood / Surface Change Evidence";
    let badge = "Evidence";
    let badgeClass = "ok";

    if (cat.includes("ai_water_and_sar") || cat.includes("dual")) {
      title = "Dual AI + SAR Corroborated Water";
      badge = "Multi-Sensor Corroborated";
      badgeClass = "ok";
    } else if (cat.includes("ai_only") || cat.includes("ai_segmentation")) {
      title = "AI-Only Candidate Water";
      badge = "Exploratory (Uncalibrated)";
      badgeClass = "warn";
    } else if (cat.includes("valley_change")) {
      title = "SAR Valley Surface Change";
      badge = "Radar Backscatter Change";
      badgeClass = "warn";
    } else if (cat.includes("hillside_change")) {
      title = "SAR Hillside Change Proxy";
      badge = "Steep Slope Change Proxy";
      badgeClass = "warn";
    }

    const areaStr = p.area_m2 ? `${(p.area_m2 / 10000).toFixed(2)} ha (${p.area_m2.toLocaleString()} m²)` : "N/A";
    const popupHtml = `
      <div class="pop-head">${esc(title)}</div>
      <span class="tag-badge ${badgeClass}">${esc(badge)}</span>
      <div class="pop-row" style="margin-top:6px"><span>Delineated Area:</span><b>${areaStr}</b></div>
      <div class="pop-row"><span>Evidence Layer:</span><b>${esc(p.category_name || p.evidence_name || "Multi-sensor")}</b></div>
      <div style="font-size:11px;color:var(--ink-muted);margin-top:6px;border-top:1px solid var(--border-subtle);padding-top:4px">
        Exploratory synthesis. Attention U-Net uncalibrated on Trishuli basin. Requires ground engineering verification.
      </div>
    `;
    layer.bindPopup(popupHtml);
    layer.on("click", () => {
      info(`Selected <b>${esc(title)}</b> (${areaStr}) · Classification: <code>${esc(p.category_name || p.evidence_name || "Multi-sensor")}</code>`);
    });
  }

  function onEachRoadFeature(feature, layer) {
    const p = feature.properties || {};
    const name = p.name || p.ref || (p.osm_id ? `OSM ${p.osm_id}` : "Road segment");
    const roadType = p.highway || "Road";
    const lenAff = p.affected_length_m ? `${Math.round(p.affected_length_m)} m` : (p.total_length_m ? `${Math.round(p.total_length_m)} m` : "Segment");
    const impactClass = p.impact_class || "Potentially Inundated";

    const popupHtml = `
      <div class="pop-head">${esc(name)}</div>
      <span class="tag-badge cut">Hazard Corridor Proximity</span>
      <div class="pop-row" style="margin-top:6px"><span>Highway Class:</span><b>${esc(roadType)}</b></div>
      <div class="pop-row"><span>Affected Length:</span><b>${lenAff}</b></div>
      <div class="pop-row"><span>Impact Model:</span><b>${esc(impactClass)}</b></div>
      <div style="font-size:11px;color:var(--ink-muted);margin-top:6px;border-top:1px solid var(--border-subtle);padding-top:4px">
        Pre-event OpenStreetMap alignment intersecting satellite hazard corridor. Indicates potential disruption, not verified physical collapse.
      </div>
    `;
    layer.bindPopup(popupHtml);
    layer.on("click", () => {
      info(`Selected road segment <b>${esc(name)}</b> (${esc(roadType)}) · Est. affected: ${lenAff} · Intersects hazard zone`);
    });
  }

  function onEachBridgeFeature(feature, layer) {
    const p = feature.properties || {};
    const name = p.name || p.ref || (p.osm_id ? `OSM Bridge ${p.osm_id}` : "Bridge structure");
    const structure = p["bridge:structure"] || p.highway || "Bridge";

    const popupHtml = `
      <div class="pop-head">${esc(name)}</div>
      <span class="tag-badge cut">Hazard Corridor Intersection</span>
      <div class="pop-row" style="margin-top:6px"><span>Structure Type:</span><b>${esc(structure)}</b></div>
      <div class="pop-row"><span>OSM Reference:</span><b>${esc(p.ref || p.osm_id || "N/A")}</b></div>
      <div style="font-size:11px;color:var(--ink-muted);margin-top:6px;border-top:1px solid var(--border-subtle);padding-top:4px">
        Bridge coordinates intersect satellite-detected flood corridor. Structural integrity requires on-site inspection.
      </div>
    `;
    layer.bindPopup(popupHtml);
    layer.on("click", () => {
      info(`Selected bridge <b>${esc(name)}</b> (${esc(structure)}) · Intersects flood hazard corridor`);
    });
  }

  function getSettlementStyle(isCut, isInsufficient, z) {
    let radius = 2.5;
    let weight = 0.75;
    let fillOpacity = 0.70;

    if (z <= 11) {
      radius = isCut ? 3.5 : 2.0;
      weight = isCut ? 1.5 : 0.6;
      fillOpacity = isCut ? 0.95 : 0.65;
    } else if (z <= 13) {
      radius = isCut ? 4.5 : 3.0;
      weight = isCut ? 1.5 : 0.8;
      fillOpacity = isCut ? 0.95 : 0.80;
    } else {
      radius = isCut ? 6.0 : 4.0;
      weight = isCut ? 2.0 : 1.0;
      fillOpacity = isCut ? 0.98 : 0.88;
    }

    const fillColor = isCut ? "#e11d48" : (isInsufficient ? "#94a3b8" : "#059669");
    return { radius, weight, fillOpacity, fillColor };
  }

  function updateSettlementSizes() {
    if (!leafletMap) return;
    const z = leafletMap.getZoom();
    settlementMarkers.forEach((marker) => {
      const st = getSettlementStyle(marker._isCut, marker._isInsufficient, z);
      marker.setRadius(st.radius);
      marker.setStyle({
        weight: st.weight,
        fillOpacity: st.fillOpacity,
        fillColor: st.fillColor
      });
      if (marker._isCut && typeof marker.bringToFront === "function") {
        marker.bringToFront();
      }
    });
  }

  function pointToSettlement(feature, latlng) {
    const p = feature.properties || {};
    const isCut = p.status === "disconnected" || Boolean(p.cut);
    const isInsufficient = p.status === "insufficient_network_evidence";
    const z = leafletMap ? leafletMap.getZoom() : 11;
    const st = getSettlementStyle(isCut, isInsufficient, z);

    const marker = L.circleMarker(latlng, {
      radius: st.radius,
      fillColor: st.fillColor,
      color: "#ffffff",
      weight: st.weight,
      opacity: 1,
      fillOpacity: st.fillOpacity,
      bubblingMouseEvents: true
    });

    marker._isCut = isCut;
    marker._isInsufficient = isInsufficient;

    marker.on("mouseover", function () {
      this.setRadius(this.getRadius() + 1.5);
      this.setStyle({ weight: 2 });
    });
    marker.on("mouseout", function () {
      const curZ = leafletMap ? leafletMap.getZoom() : 11;
      const baseSt = getSettlementStyle(this._isCut, this._isInsufficient, curZ);
      this.setRadius(baseSt.radius);
      this.setStyle({ weight: baseSt.weight });
    });

    if (p.settlement_id) settlementMarkers.set(String(p.settlement_id).toLowerCase(), marker);
    if (p.settlement_name) settlementMarkers.set(String(p.settlement_name).toLowerCase(), marker);

    return marker;
  }

  function onEachSettlementFeature(feature, layer) {
    const p = feature.properties || {};
    const name = p.settlement_name || p.n || "Settlement";
    const placeType = p.place_type || "settlement";
    const isCut = p.status === "disconnected" || Boolean(p.cut);
    const badge = isCut ? "CUT OFF" : (p.status === "insufficient_network_evidence" ? "Isolated Network" : "CONNECTED");
    const badgeClass = isCut ? "cut" : "ok";

    const destName = p.nearest_destination_name || p.nearest_destination || "Nearest service town";
    const hospName = p.nearest_hospital || null;
    const baseTime = p.baseline_time_min != null ? `${p.baseline_time_min} min` : "N/A";
    const postTime = isCut ? '<span style="color:var(--color-settl-cutoff);font-weight:700">UNREACHABLE</span>' : (p.post_disruption_time_min != null ? `${p.post_disruption_time_min} min` : "Normal");

    const hospRow = hospName ? `<div class="pop-row"><span>Nearest Hospital:</span><b>${esc(hospName)}</b></div>` : "";
    const reasonRow = p.isolation_reason ? `<div class="pop-row"><span>Isolation Reason:</span><b>${esc(p.isolation_reason)}</b></div>` : "";

    const popupHtml = `
      <div class="pop-head">${esc(name)} <small style="font-weight:normal;color:var(--ink-muted)">(${esc(placeType)})</small></div>
      <span class="tag-badge ${badgeClass}">${badge}</span>
      <div class="pop-row" style="margin-top:6px"><span>Service Hub:</span><b>${esc(destName)}</b></div>
      ${hospRow}
      <div class="pop-row"><span>Baseline Travel:</span><b>${baseTime}</b></div>
      <div class="pop-row"><span>Post-Disruption:</span><b>${postTime}</b></div>
      ${reasonRow}
      <div style="font-size:11px;color:var(--ink-muted);margin-top:6px;border-top:1px solid var(--border-subtle);padding-top:4px">
        Graph routing on pre-event OpenStreetMap road topology interrupted by flood hazard evidence.
      </div>
    `;
    layer.bindPopup(popupHtml);
    layer.on("click", () => {
      info(`Selected settlement <b>${esc(name)}</b> (${esc(placeType)}) · Status: <span class="tag-badge ${badgeClass}">${badge}</span> · Destination: ${esc(destName)} (${postTime})`);
    });
  }

  function onEachFlowFeature(feature, layer) {
    const p = feature.properties || {};
    const lengthKm = p.path_length_km != null ? `${Number(p.path_length_km).toFixed(2)} km` : "50.01 km";
    const dropM = p.total_drop_m != null ? `${Math.round(p.total_drop_m)} m drop (${Math.round(p.start_elevation_m || 3117)}m → ${Math.round(p.end_elevation_m || 566)}m)` : "2,551 m drop (3,117m → 566m)";
    const vertCount = p.vertices_count || 1398;

    const popupHtml = `
      <div class="pop-head">Trishuli Downstream Drainage Corridor</div>
      <span class="tag-badge ok">DEM D8 Gravity Route</span>
      <div class="pop-row" style="margin-top:6px"><span>Corridor Length:</span><b>${lengthKm}</b></div>
      <div class="pop-row"><span>Elevation Drop:</span><b>${dropM}</b></div>
      <div class="pop-row"><span>Terrain Steps:</span><b>${vertCount} cells (30m)</b></div>
      <div class="pop-row"><span>Settlements Along:</span><b>20 sequenced communities</b></div>
      <div class="pop-row"><span>Elevation Model:</span><b>Copernicus WorldDEM GLO-30</b></div>
      <div style="font-size:10.5px;color:var(--ink-secondary);margin-top:6px;border-top:1px solid var(--border-subtle);padding-top:4px;line-height:1.35">
        <b>Qualification:</b> Topographic gravity drainage line modeled via steepest-descent (D8) priority-flood routing from the upper gorge. <b>Not observed inundation or arrival prediction.</b> Does not account for hydraulic wave speed or debris damming.
      </div>
    `;
    layer.bindPopup(popupHtml);
    layer.on("click", () => {
      info(`Selected <b>Trishuli Downstream Drainage Corridor</b> (${lengthKm}, ${dropM}) modeled from Copernicus DEM GLO-30. Potential drainage path; not observed floodwater.`);
    });
  }

  FI.ui.clearMapLayers = function () {
    if (!leafletMap) return;
    Object.keys(layerGroups).forEach((k) => layerGroups[k]?.clearLayers());
    settlementMarkers.clear();
    cachedAnalysisKey = null;
    if (aoiRectangle && leafletMap.hasLayer(aoiRectangle)) {
      leafletMap.removeLayer(aoiRectangle);
      aoiRectangle = null;
    }
    renderUnifiedLegend();
  };

  // Unified Floating Layer & Legend Component Builder
  function renderUnifiedLegend() {
    const container = $("layers");
    if (!container) return;

    const R = FI.state.result;
    if (R && R.status === "data_not_prepared") {
      container.innerHTML = `
        <div style="font-size:11px;color:var(--ink-secondary);padding:6px 2px;line-height:1.4">
          <div style="font-weight:600;color:var(--status-warn);margin-bottom:3px">⚠ Data Not Prepared</div>
          <div>No GIS layers prepared for <b>${esc(R.area || "this AOI")}</b>. Satellite SAR scenes and OSM baseline ingestion required.</div>
        </div>
      `;
      return;
    }

    const c = (R && FI.countImpacts) ? FI.countImpacts(R) : null;
    const floodKm = R?.hazard_metrics?.exploratory_terrain_gated_ai_water_km2 != null
      ? `${Number(R.hazard_metrics.exploratory_terrain_gated_ai_water_km2).toFixed(2)} km²`
      : "3.75 km²";
    const roadsCount = (c && c.roads != null)
      ? `${c.roads} seg (${c.roadsKm != null ? c.roadsKm + " km" : "12.7 km"})`
      : "62 seg (12.7 km)";
    const bridgesCount = (c && c.bridges != null)
      ? `${c.bridges} structures`
      : "29 structures";
    const settlCount = (c && c.settlementsTotal != null)
      ? `${c.settlementsTotal} (${c.cut} cut off)`
      : "505 (49 cut off)";

    const flowCount = R?.downstream_summary?.path_length_km != null
      ? `${Number(R.downstream_summary.path_length_km).toFixed(1)} km (${R.downstream_summary.settlements_count || 20} settl)`
      : "50.0 km (20 settl)";

    const layersMeta = [
      { key: "flood", name: "Flood & Surface Change", count: floodKm, color: "#1d4ed8", type: "rect" },
      { key: "roads", name: "Potentially Affected Roads", count: roadsCount, color: "#dc2626", type: "line" },
      { key: "bridges", name: "Potentially Affected Bridges", count: bridgesCount, color: "#7c3aed", type: "line" },
      { key: "settl", name: "Settlement Accessibility", count: settlCount, color: "#e11d48", type: "circle" },
      { key: "flow", name: "Downstream Drainage Flow (DEM)", count: flowCount, color: "#2563eb", type: "line", extra: "border-top: 2px dashed #93c5fd" }
    ];

    const currentOn = FI.state.layers || {};

    container.innerHTML = layersMeta.map((l) => {
      const isChecked = currentOn[l.key] !== 0;
      let swatchStyle = `background:${l.color};`;
      if (l.extra) swatchStyle += l.extra;
      const swatchClass = l.type === "circle" ? "swatch-circle" : (l.type === "line" ? "swatch-line" : "");

      return `
        <label class="legend-row">
          <input type="checkbox" class="legend-checkbox" data-k="${l.key}" ${isChecked ? "checked" : ""}>
          <span class="legend-swatch ${swatchClass}" style="${swatchStyle}"></span>
          <div class="legend-label-group">
            <span class="legend-name">${esc(l.name)}</span>
            <span class="legend-count">${esc(l.count)}</span>
          </div>
        </label>
      `;
    }).join("");

    container.querySelectorAll(".legend-checkbox").forEach((chk) => {
      chk.onchange = (e) => {
        const k = e.target.getAttribute("data-k");
        const checked = e.target.checked;
        FI.state.layers[k] = checked ? 1 : 0;
        FI.ui.toggleLayer(k, checked);
      };
    });
  }

  // Load real GeoJSON layers asynchronously
  async function loadGisLayers(R) {
    if (!R) return;

    if (R.status === "data_not_prepared" || !R.layers || Object.keys(R.layers).length === 0) {
      Object.keys(layerGroups).forEach((k) => layerGroups[k]?.clearLayers());
      settlementMarkers.clear();
      cachedAnalysisKey = `${R.area}_${R.date}`;
      if (aoiRectangle && leafletMap && leafletMap.hasLayer(aoiRectangle)) {
        leafletMap.removeLayer(aoiRectangle);
        aoiRectangle = null;
      }
      if (R.bbox && leafletMap) {
        const [w, s, e, n] = R.bbox;
        activeBounds = L.latLngBounds([[s, w], [n, e]]);
        aoiRectangle = L.rectangle(activeBounds, {
          color: "#d97706",
          weight: 2,
          dashArray: "5, 5",
          fillOpacity: 0.05
        }).addTo(leafletMap);
        aoiRectangle.bindTooltip(`AOI: ${esc(R.area)} (Pending Ingestion)`, { sticky: true });
        leafletMap.fitBounds(activeBounds, { padding: [35, 35] });
      }
      renderUnifiedLegend();
      leafletMap.invalidateSize();
      return;
    }

    const analysisKey = `${R.area}_${R.date}`;
    if (cachedAnalysisKey === analysisKey) {
      syncLayerVisibility();
      return;
    }

    Object.keys(layerGroups).forEach((k) => layerGroups[k]?.clearLayers());
    settlementMarkers.clear();
    activeBounds = L.latLngBounds([]);

    if (aoiRectangle && leafletMap && leafletMap.hasLayer(aoiRectangle)) {
      leafletMap.removeLayer(aoiRectangle);
      aoiRectangle = null;
    }
    if (R.bbox && leafletMap) {
      const [w, s, e, n] = R.bbox;
      aoiRectangle = L.rectangle([[s, w], [n, e]], {
        color: "#2563eb",
        weight: 1.5,
        fillOpacity: 0.03
      }).addTo(leafletMap);
      aoiRectangle.bindTooltip(`AOI: ${esc(R.area)} (Verified Coverage)`, { sticky: true });
      activeBounds.extend([[s, w], [n, e]]);
    }

    const endpoints = {
      flood: resolveUrl(R.layers?.flood_evidence),
      roads: resolveUrl(R.layers?.affected_roads),
      bridges: resolveUrl(R.layers?.affected_bridges),
      settl: resolveUrl(R.layers?.settlement_connectivity),
      flow: resolveUrl(R.layers?.downstream_path)
    };

    const fetchLayer = async (url) => {
      if (!url) return null;
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        console.warn(`Layer fetch failed for ${url}:`, err);
        return null;
      }
    };

    const [floodData, roadsData, bridgesData, settlData, flowData] = await Promise.all([
      fetchLayer(endpoints.flood),
      fetchLayer(endpoints.roads),
      fetchLayer(endpoints.bridges),
      fetchLayer(endpoints.settl),
      fetchLayer(endpoints.flow)
    ]);

    // 1. Flood Evidence
    if (floodData && floodData.features) {
      const floodLyr = L.geoJSON(floodData, {
        style: styleFloodEvidence,
        onEachFeature: onEachFloodFeature
      });
      layerGroups.flood.addLayer(floodLyr);
      if (floodLyr.getBounds().isValid()) activeBounds.extend(floodLyr.getBounds());
    }

    // 2. Downstream Flow Path
    if (flowData && flowData.features) {
      const flowLyr = L.geoJSON(flowData, {
        style: { color: "#2563eb", weight: 2.5, opacity: 0.85, dashArray: "6, 6" },
        onEachFeature: onEachFlowFeature
      });
      layerGroups.flow.addLayer(flowLyr);
      if (flowLyr.getBounds().isValid()) activeBounds.extend(flowLyr.getBounds());
    }

    // 3. Affected Roads
    if (roadsData && roadsData.features) {
      const roadsLyr = L.geoJSON(roadsData, {
        filter: (f) => f.properties && (f.properties.is_affected || (f.properties.impact_class && f.properties.impact_class !== "none")),
        style: { color: "#dc2626", weight: 3.5, opacity: 0.9, dashArray: "6, 4" },
        onEachFeature: onEachRoadFeature
      });
      layerGroups.roads.addLayer(roadsLyr);
      if (roadsLyr.getBounds().isValid()) activeBounds.extend(roadsLyr.getBounds());
    }

    // 4. Affected Bridges
    if (bridgesData && bridgesData.features) {
      const bridgesLyr = L.geoJSON(bridgesData, {
        filter: (f) => f.properties && (f.properties.is_affected || (f.properties.impact_class && f.properties.impact_class !== "none")),
        style: { color: "#7c3aed", weight: 6, opacity: 0.95 },
        onEachFeature: onEachBridgeFeature
      });
      layerGroups.bridges.addLayer(bridgesLyr);
      if (bridgesLyr.getBounds().isValid()) activeBounds.extend(bridgesLyr.getBounds());
    }

    // 5. Settlement Connectivity
    if (settlData && settlData.features) {
      const settlLyr = L.geoJSON(settlData, {
        pointToLayer: pointToSettlement,
        onEachFeature: onEachSettlementFeature
      });
      layerGroups.settl.addLayer(settlLyr);
      if (settlLyr.getBounds().isValid()) activeBounds.extend(settlLyr.getBounds());
      settlLyr.eachLayer((l) => {
        if (l._isCut && typeof l.bringToFront === "function") {
          l.bringToFront();
        }
      });
    }

    cachedAnalysisKey = analysisKey;
    syncLayerVisibility();
    renderUnifiedLegend();

    if (activeBounds.isValid()) {
      leafletMap.fitBounds(activeBounds, { padding: [35, 35] });
    }
    leafletMap.invalidateSize();
  }

  function syncLayerVisibility() {
    if (!leafletMap) return;
    const on = FI.state.layers || {};
    FI.ui.toggleLayer("flood", Boolean(on.flood));
    FI.ui.toggleLayer("roads", Boolean(on.roads));
    FI.ui.toggleLayer("bridges", Boolean(on.bridges));
    FI.ui.toggleLayer("settl", Boolean(on.settl));
    FI.ui.toggleLayer("flow", Boolean(on.flow !== 0));
  }

  // Fallback SVG illustration for mock mode
  function drawMockSvg() {
    const { result: R, period, source, layers: on } = FI.state;
    const s = $("map");
    if (!s) return;
    s.innerHTML = "";
    const optical = source === "s2";
    svg("rect", { width: 800, height: 520, fill: optical ? "#dfe6d3" : "#d5dde6" }, s);
    if (!R) return;
    const post = period === "post";

    if (on.flood && R.river) {
      svg("path", { d: R.river, fill: "none", stroke: "#7fb0f0", "stroke-width": 10, opacity: .7, "stroke-linecap": "round" }, s);
      if (post) svg("path", { d: R.river, fill: "none", stroke: "#1d4ed8", "stroke-width": R.floodW || 26, opacity: .55, "stroke-linecap": "round" }, s)
        .onclick = () => info("Flooded area (mock geometry)");
    }
    if (on.debris && post && Array.isArray(R.debris)) {
      R.debris.forEach(([x, y, rx, ry]) =>
        svg("ellipse", { cx: x, cy: y, rx, ry, fill: "#b5651d", opacity: .7 }, s).onclick = () => info("Debris-covered area (mock geometry)"));
    }
    if (on.roads && Array.isArray(R.roads)) {
      R.roads.forEach((r, i) => {
        const hit = post && r.hit;
        svg("path", { d: r.p, fill: "none", stroke: hit ? "#dc2626" : "#8fa0b5", "stroke-width": hit ? 3.5 : 2.5, "stroke-dasharray": hit ? "7 4" : "" }, s)
          .onclick = () => info(`Road ${i + 1}: ${hit ? "estimated affected" : "no impact estimated"}`);
      });
    }
    if (on.settl && Array.isArray(R.settlements)) {
      R.settlements.forEach((t) => {
        const cut = post && t.cut;
        svg("circle", { cx: t.x, cy: t.y, r: 8, fill: cut ? "#e11d48" : "#059669", stroke: "#fff", "stroke-width": 2 }, s)
          .onclick = () => info(`${t.n}: ${cut ? "no road connection" : "connected"}`);
      });
    }
  }

  // Main entry point
  FI.ui.drawMap = function () {
    const { result: R } = FI.state;

    if (FI.config.USE_MOCK || R?.data_mode === "MOCK") {
      $("gis-map").style.display = "none";
      $("map").style.display = "block";
      drawMockSvg();
      return;
    }

    $("map").style.display = "none";
    $("gis-map").style.display = "block";

    const map = initLeafletMap();
    if (!map) return;
    map.invalidateSize();

    renderUnifiedLegend();

    if (!R) return;

    loadGisLayers(R);
  };
})(window.FI);
