/* Weather panel: calls Open-Meteo API and renders an inline operational summary. */
(function (FI) {
  const { $ } = FI;

  FI.ui.loadWeather = async function (area, date, time) {
    const box = $("wx");
    if (!box) return;
    box.textContent = "Loading weather…";
    try {
      const w = await FI.api.getWeather(area, date, time);
      const tempStr = w.temperature != null ? `${w.temperature} °C` : "";
      const condStr = w.conditions || "Partly cloudy";
      const rainStr = w.dayRain != null ? `${w.dayRain} mm rain/24h` : "";
      
      const summaryText = [w.place || "Trishuli Basin", tempStr, condStr, rainStr].filter(Boolean).join(" · ");
      box.textContent = summaryText;

      const chip = $("wx-chip");
      if (chip) {
        chip.title = `${w.place} (${w.date} ${String(w.hour || 12).padStart(2, "0")}:00)\n` +
          `Temp: ${w.temperature} °C (Min/Max: ${w.dayMin}/${w.dayMax} °C)\n` +
          `Conditions: ${w.conditions}\n` +
          `Rain (1h): ${w.rain || 0} mm | Rain (24h): ${w.dayRain || 0} mm\n` +
          `Humidity: ${w.humidity || "N/A"}% | Wind: ${w.wind || "N/A"} km/h`;
      }
    } catch (e) {
      box.textContent = "Weather unavailable (offline)";
    }
  };
})(window.FI);
