from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DB_PATH = PROCESSED_DIR / "weather.db"


def city_name_from_file(file_path: Path) -> str:
    """Convert a filename like new_york_current.json into a readable city name."""
    stem = file_path.stem
    if stem.endswith("_current"):
        stem = stem.replace("_current", "")
    elif stem.endswith("_forecast"):
        stem = stem.replace("_forecast", "")
    elif stem.endswith("_historical_archive"):
        stem = stem.replace("_historical_archive", "")
    elif stem.endswith("_historical"):
        stem = stem.replace("_historical", "")
    return stem.replace("_", " ").title()


def normalize_daily_payload(city_name: str, payload: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """Flatten Open-Meteo daily weather response rows into one record per date."""
    daily = payload.get("daily", {})
    time_values = daily.get("time", [])
    rows: List[Dict[str, Any]] = []

    for index, timestamp in enumerate(time_values):
        row = {
            "city": city_name,
            "source": source,
            "date": timestamp,
            "temperature_2m_max_c": daily.get("temperature_2m_max", [None] * len(time_values))[index],
            "temperature_2m_min_c": daily.get("temperature_2m_min", [None] * len(time_values))[index],
            "precipitation_sum_mm": daily.get("precipitation_sum", [None] * len(time_values))[index],
            "relative_humidity_2m_mean_pct": daily.get("relative_humidity_2m_mean", [None] * len(time_values))[index],
            "wind_speed_10m_max_kmh": daily.get("wind_speed_10m_max", [None] * len(time_values))[index],
            "precipitation_probability_max_pct": daily.get("precipitation_probability_max", [None] * len(time_values))[index],
            "weather_code": daily.get("weather_code", [None] * len(time_values))[index],
        }
        rows.append(row)

    return rows


def normalize_current_payload(city_name: str, payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Convert a current weather payload into a single row with the important metrics."""
    current = payload.get("current", {})
    timestamp = current.get("time")
    return {
        "city": city_name,
        "source": source,
        "timestamp": timestamp,
        "date": timestamp[:10] if isinstance(timestamp, str) and len(timestamp) >= 10 else None,
        "temperature_2m_c": current.get("temperature_2m"),
        "relative_humidity_2m_pct": current.get("relative_humidity_2m"),
        "apparent_temperature_c": current.get("apparent_temperature"),
        "precipitation_mm": current.get("precipitation"),
        "weather_code": current.get("weather_code"),
        "wind_speed_10m_kmh": current.get("wind_speed_10m"),
    }


def load_raw_weather_files() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read all JSON files in the raw folder and normalize them into current and daily records."""
    current_rows: List[Dict[str, Any]] = []
    daily_rows: List[Dict[str, Any]] = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    # Archive files load first so the fresher last-365-days rows replace any overlapping dates.
    files = sorted(RAW_DIR.glob("*.json"), key=lambda path: ("archive" not in path.name, path.name))
    if not files:
        raise FileNotFoundError(f"No raw JSON files found in {RAW_DIR}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        city_name = city_name_from_file(file_path)
        file_name = file_path.name

        if "current" in file_name:
            current_rows.append(normalize_current_payload(city_name, payload, "current"))
        elif "forecast" in file_name or "historical" in file_name:
            daily_rows.extend(
                normalize_daily_payload(
                    city_name,
                    payload,
                    "forecast" if "forecast" in file_name else "historical",
                )
            )

    deduplicated = {(row["city"], row["source"], row["date"]): row for row in daily_rows}
    return current_rows, list(deduplicated.values())


def save_to_sqlite(current_records: Iterable[Dict[str, Any]], daily_records: Iterable[Dict[str, Any]]) -> None:
    """Write current and daily weather records to separate SQLite tables."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    current_list = list(current_records)
    daily_list = list(daily_records)

    if not current_list and not daily_list:
        raise ValueError("No weather records were available to save.")

    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute("DROP TABLE IF EXISTS weather_records")
        connection.execute("DROP TABLE IF EXISTS weather_current")
        connection.execute("DROP TABLE IF EXISTS weather_daily")

        if current_list:
            current_df = pd.DataFrame(current_list)
            current_df.to_sql("weather_current", connection, index=False, if_exists="replace")
        if daily_list:
            daily_df = pd.DataFrame(daily_list)
            daily_df.to_sql("weather_daily", connection, index=False, if_exists="replace")
        connection.commit()
        print(f"Saved {len(current_list)} current rows and {len(daily_list)} daily rows to {DB_PATH}")
    finally:
        connection.close()


def main() -> None:
    """Load raw weather payloads, normalize them, and persist them to SQLite."""
    current_records, daily_records = load_raw_weather_files()
    save_to_sqlite(current_records, daily_records)


if __name__ == "__main__":
    main()
