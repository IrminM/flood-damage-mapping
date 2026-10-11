/* Temporary sample data used while the backend is not connected.
   Delete this file (and its <script> tag) once real data is wired in.
   Coordinates are SVG space (0–800 × 0–520); real data should be GeoJSON drawn by your map layer. */
(function (FI) {
  FI.mockAssessment = function (area, date) {
    return {
      area, date,
      river: "M60 40 C200 120 150 220 300 250 S420 330 520 360 S700 470 760 500",
      floodW: 26,
      floodAreaKm2: null,
      debrisAreaKm2: null,
      debris: [[190, 170, 30, 16], [330, 262, 38, 18], [470, 335, 34, 15]],
      roads: [
        { p: "M20 260 L300 250 L560 300 L790 280", hit: true },
        { p: "M120 80 L260 150 L330 262 L420 430", hit: false },
        { p: "M560 300 L640 190 L780 150", hit: false },
        { p: "M420 430 L560 480 L790 440", hit: true }
      ],
      bridges: [[300, 250, true], [520, 360, true], [640, 190, false]],
      buildings: Array.from({ length: 34 }, (_, i) => [40 + (i * 53) % 720, 60 + (i * 97) % 420, i % 5 === 0 || i % 7 === 0]),
      town: [640, 190],
      hospital: [560, 300],
      settlements: [
        { n: "Settlement A", x: 210, y: 200, cut: false },
        { n: "Settlement B", x: 380, y: 300, cut: true },
        { n: "Settlement C", x: 480, y: 410, cut: true },
        { n: "Settlement D", x: 700, y: 260, cut: false },
        { n: "Settlement E", x: 120, y: 300, cut: false }
      ]
    };
  };
})(window.FI);
