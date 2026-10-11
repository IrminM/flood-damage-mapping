/* Shared helpers: DOM access, SVG creation, input validation. */
(function (FI) {
  const SVG_NS = "http://www.w3.org/2000/svg";

  FI.$ = (id) => document.getElementById(id);

  FI.svg = function (tag, attrs, parent) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    parent.appendChild(e);
    return e;
  };

  /** Returns an error message, or "" when inputs are valid. */
  FI.validate = function (area, date) {
    if (!area.trim()) return "Enter an area name or bounding box.";
    if (/^[-\d.\s]+,/.test(area)) {
      const p = area.split(",").map(Number);
      const bad = p.length !== 4 || p.some(isNaN) || p[0] >= p[2] || p[1] >= p[3] ||
        Math.abs(p[0]) > 180 || Math.abs(p[2]) > 180 || Math.abs(p[1]) > 90 || Math.abs(p[3]) > 90;
      if (bad) return "Bounding box must be west,south,east,north with valid coordinates.";
    }
    if (!date) return "Choose the flood / event date.";
    if (new Date(date) > new Date()) return "Event date cannot be in the future.";
    return "";
  };
})(window.FI);
