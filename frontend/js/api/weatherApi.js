/* Weather client (Open-Meteo, free, no API key). Returns plain data; no DOM access. */
(function (FI) {
  const CODES = {0:"Clear",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",80:"Rain showers",81:"Rain showers",82:"Violent showers",95:"Thunderstorm",96:"Thunderstorm, hail",99:"Thunderstorm, hail"};

  async function locate(area) {
    const cfg = FI.config.WEATHER;
    const m = area.split(",").map(Number);
    if (m.length === 4 && !m.some(isNaN)) {
      const lat = (m[1] + m[3]) / 2, lon = (m[0] + m[2]) / 2;
      return { lat, lon, name: `Area centre (${lat.toFixed(2)}, ${lon.toFixed(2)})` };
    }
    const queries = [area, ...area.split(",").map((s) => s.trim())];
    for (const q of queries) {
      if (!q) continue;
      const res = await fetch(`${cfg.GEOCODE_URL}?count=1&name=${encodeURIComponent(q)}`);
      const g = await res.json();
      if (g.results && g.results[0]) {
        const r = g.results[0];
        return { lat: r.latitude, lon: r.longitude, name: [r.name, r.admin1, r.country].filter(Boolean).join(", ") };
      }
    }
    throw new Error("location not found");
  }

  FI.api.getWeather = async function (area, date, time) {
    const cfg = FI.config.WEATHER;
    const place = await locate(area);
    const useArchive = (new Date(date) - new Date()) / 864e5 < -cfg.ARCHIVE_AFTER_DAYS;
    const url = (useArchive ? cfg.ARCHIVE_URL : cfg.FORECAST_URL) +
      `?latitude=${place.lat}&longitude=${place.lon}&start_date=${date}&end_date=${date}` +
      "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code" +
      "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto";
    const res = await fetch(url);
    if (!res.ok) throw new Error("weather service error " + res.status);
    const d = await res.json();
    const i = Math.min(23, parseInt(time) || 0);
    if (!d.hourly || d.hourly.temperature_2m[i] == null) throw new Error("no data for that date");
    return {
      place: place.name, date, hour: i,
      temperature: d.hourly.temperature_2m[i],
      conditions: CODES[d.hourly.weather_code[i]] || "—",
      rain: d.hourly.precipitation[i],
      humidity: d.hourly.relative_humidity_2m[i],
      wind: d.hourly.wind_speed_10m[i],
      dayMin: d.daily.temperature_2m_min[0],
      dayMax: d.daily.temperature_2m_max[0],
      dayRain: d.daily.precipitation_sum[0]
    };
  };
})(window.FI);
