import React, { useEffect, useState } from "react";
import { useData } from "../context/DataContext";
import { StorageKeys, useStorage } from "../context/StorageContext";

type WeatherPoint = {
  temp: number;
  time: string;
  symbol: string;
};

type WeatherUnit = "F" | "C";

const MAX_DAILY_AVERAGES = 30;

const getPrincetonDateKey = () => {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
};

const getPreviousDateKey = (dateKey: string) => {
  const [year, month, day] = dateKey.split("-").map(Number);
  const base = new Date(Date.UTC(year, month - 1, day, 12));
  base.setUTCDate(base.getUTCDate() - 1);
  return base.toISOString().slice(0, 10);
};

function WeatherTable() {
  const data = useData();
  const storage = useStorage();
  const weather: WeatherPoint[] = data?.weather || [];
  const [comparisonText, setComparisonText] = useState("Collecting weather baseline...");
  const [unit, setUnit] = useState<WeatherUnit>(() => {
    const saved = storage.getLocalStorage(StorageKeys.WEATHER_UNIT);
    return saved === "C" ? "C" : "F";
  });

  useEffect(() => {
    storage.setLocalStorage(StorageKeys.WEATHER_UNIT, unit);
  }, [unit]);

  useEffect(() => {
    if (weather.length === 0) return;

    const todayAverage =
      weather.reduce((sum, point) => sum + point.temp, 0) / weather.length;
    const todayKey = getPrincetonDateKey();
    const yesterdayKey = getPreviousDateKey(todayKey);

    let averages: Record<string, number> = {};
    const savedAverages = storage.getLocalStorage(StorageKeys.WEATHER_DAILY_AVERAGES);
    if (savedAverages) {
      try {
        const parsed = JSON.parse(savedAverages);
        if (parsed && typeof parsed === "object") {
          averages = parsed as Record<string, number>;
        }
      } catch {
        averages = {};
      }
    }

    averages[todayKey] = todayAverage;

    const sortedDates = Object.keys(averages).sort();
    if (sortedDates.length > MAX_DAILY_AVERAGES) {
      const toRemove = sortedDates.slice(0, sortedDates.length - MAX_DAILY_AVERAGES);
      toRemove.forEach((key) => delete averages[key]);
    }
    storage.setLocalStorage(StorageKeys.WEATHER_DAILY_AVERAGES, JSON.stringify(averages));

    const yesterdayAverage = averages[yesterdayKey];
    if (typeof yesterdayAverage !== "number") {
      setComparisonText("No yesterday comparison yet.");
      return;
    }

    const delta = todayAverage - yesterdayAverage;
    if (delta > 1) {
      setComparisonText("Warmer than yesterday");
    } else if (delta < -1) {
      setComparisonText("Colder than yesterday");
    } else {
      setComparisonText("Same as yesterday");
    }
  }, [weather, storage]);

  const formatTemp = (tempF: number) => {
    if (unit === "C") {
      return `${Math.round(((tempF - 32) * 5) / 9)}°`;
    }
    return `${Math.round(tempF)}°`;
  };

  return (
    <div className="widget-card weather">
      <div className="weather-card-header">
        <h3 className="weather-title">Weather</h3>
        <div className="weather-unit-switch" role="group" aria-label="Temperature unit">
          <button
            type="button"
            className={`weather-unit-btn ${unit === "F" ? "active" : ""}`}
            onClick={() => setUnit("F")}
          >
            °F
          </button>
          <button
            type="button"
            className={`weather-unit-btn ${unit === "C" ? "active" : ""}`}
            onClick={() => setUnit("C")}
          >
            °C
          </button>
        </div>
      </div>
      <div className="weather-grid">
        {weather.map((point, i) => (
          <div key={i} className="weather-col">
            <div className="weather-time">{point.time}</div>
            <div className="weather-temp">{formatTemp(point.temp)}</div>
          </div>
        ))}
      </div>
      <div className="weather-comparison">{comparisonText}</div>
    </div>
  );
}

export default WeatherTable;
