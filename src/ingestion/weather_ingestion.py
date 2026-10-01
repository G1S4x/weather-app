from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

# Full calendar years of history used as the "normal" baseline.
ARCHIVE_YEARS = 5

CITIES: Dict[str, Dict[str, float | str]] = {
    # Northeast
    "New York": {"latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York", "region": "Northeast"},
    "Boston": {"latitude": 42.3601, "longitude": -71.0589, "timezone": "America/New_York", "region": "Northeast"},
    "Philadelphia": {"latitude": 39.9526, "longitude": -75.1652, "timezone": "America/New_York", "region": "Northeast"},
    "Pittsburgh": {"latitude": 40.4406, "longitude": -79.9959, "timezone": "America/New_York", "region": "Northeast"},
    "Washington": {"latitude": 38.9072, "longitude": -77.0369, "timezone": "America/New_York", "region": "Northeast"},
    # Midwest
    "Chicago": {"latitude": 41.8781, "longitude": -87.6298, "timezone": "America/Chicago", "region": "Midwest"},
    "Detroit": {"latitude": 42.3314, "longitude": -83.0458, "timezone": "America/Detroit", "region": "Midwest"},
    "Minneapolis": {"latitude": 44.9778, "longitude": -93.2650, "timezone": "America/Chicago", "region": "Midwest"},
    "Milwaukee": {"latitude": 43.0389, "longitude": -87.9065, "timezone": "America/Chicago", "region": "Midwest"},
    "Columbus": {"latitude": 39.9612, "longitude": -82.9988, "timezone": "America/New_York", "region": "Midwest"},
    "Cleveland": {"latitude": 41.4993, "longitude": -81.6944, "timezone": "America/New_York", "region": "Midwest"},
    "Cincinnati": {"latitude": 39.1031, "longitude": -84.5120, "timezone": "America/New_York", "region": "Midwest"},
    "Indianapolis": {"latitude": 39.7684, "longitude": -86.1581, "timezone": "America/Indiana/Indianapolis", "region": "Midwest"},
    "St. Louis": {"latitude": 38.6270, "longitude": -90.1994, "timezone": "America/Chicago", "region": "Midwest"},
    "Kansas City": {"latitude": 39.0997, "longitude": -94.5786, "timezone": "America/Chicago", "region": "Midwest"},
    "Omaha": {"latitude": 41.2565, "longitude": -95.9345, "timezone": "America/Chicago", "region": "Midwest"},
    # South
    "Houston": {"latitude": 29.7604, "longitude": -95.3698, "timezone": "America/Chicago", "region": "South"},
    "Dallas": {"latitude": 32.7767, "longitude": -96.7970, "timezone": "America/Chicago", "region": "South"},
    "Austin": {"latitude": 30.2672, "longitude": -97.7431, "timezone": "America/Chicago", "region": "South"},
    "San Antonio": {"latitude": 29.4241, "longitude": -98.4936, "timezone": "America/Chicago", "region": "South"},
    "Atlanta": {"latitude": 33.7490, "longitude": -84.3880, "timezone": "America/New_York", "region": "South"},
    "Miami": {"latitude": 25.7617, "longitude": -80.1918, "timezone": "America/New_York", "region": "South"},
    "Tampa": {"latitude": 27.9506, "longitude": -82.4572, "timezone": "America/New_York", "region": "South"},
    "Orlando": {"latitude": 28.5383, "longitude": -81.3792, "timezone": "America/New_York", "region": "South"},
    "Charlotte": {"latitude": 35.2271, "longitude": -80.8431, "timezone": "America/New_York", "region": "South"},
    "Nashville": {"latitude": 36.1627, "longitude": -86.7816, "timezone": "America/Chicago", "region": "South"},
    "New Orleans": {"latitude": 29.9511, "longitude": -90.0715, "timezone": "America/Chicago", "region": "South"},
    # West
    "Los Angeles": {"latitude": 34.0522, "longitude": -118.2437, "timezone": "America/Los_Angeles", "region": "West"},
    "San Francisco": {"latitude": 37.7749, "longitude": -122.4194, "timezone": "America/Los_Angeles", "region": "West"},
    "San Diego": {"latitude": 32.7157, "longitude": -117.1611, "timezone": "America/Los_Angeles", "region": "West"},
    "Seattle": {"latitude": 47.6062, "longitude": -122.3321, "timezone": "America/Los_Angeles", "region": "West"},
    "Portland": {"latitude": 45.5152, "longitude": -122.6784, "timezone": "America/Los_Angeles", "region": "West"},
    "Denver": {"latitude": 39.7392, "longitude": -104.9903, "timezone": "America/Denver", "region": "West"},
    "Phoenix": {"latitude": 33.4484, "longitude": -112.0740, "timezone": "America/Phoenix", "region": "West"},
    "Las Vegas": {"latitude": 36.1699, "longitude": -115.1398, "timezone": "America/Los_Angeles", "region": "West"},
    "Salt Lake City": {"latitude": 40.7608, "longitude": -111.8910, "timezone": "America/Denver", "region": "West"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("weather_ingestion")


def disable_proxy_settings() -> None:
    """Prevent environment-level proxy variables from forcing blocked outbound requests."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    logger.info("Disabled proxy environment variables before outbound API requests.")


def fetch_json(url: str, params: Dict[str, Any], timeout: int = 60, max_retries: int = 6) -> Dict[str, Any]:
    """Fetch a JSON payload from the API and raise a clear error if the request fails."""
    try:
        for attempt in range(max_retries + 1):
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                proxies={"http": None, "https": None},
            )
            # Open-Meteo enforces a per-minute limit; wait it out instead of failing the city.
            if response.status_code != 429 or attempt == max_retries:
                break
            wait_seconds = min(60, 20 * (attempt + 1))
            logger.warning("Rate limited by %s, retrying in %s seconds", url, wait_seconds)
            time.sleep(wait_seconds)
        response.raise_for_status()
        payload = response.json()
        logger.info("Fetched data from %s", url)
        return payload
    except requests.exceptions.RequestException as exc:
        logger.exception("Request failed for %s with params %s", url, params)
        raise RuntimeError(f"Failed to fetch data from Open-Meteo: {exc}") from exc


def save_json(payload: Dict[str, Any], filename: str) -> Path:
    """Persist a raw JSON payload to the local raw data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / filename
    with destination.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    logger.info("Saved raw data to %s", destination)
    return destination


def fetch_current_weather(city_name: str, latitude: float, longitude: float, timezone: str) -> Dict[str, Any]:
    """Request current weather conditions for a city."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "timezone": timezone,
    }
    url = "https://api.open-meteo.com/v1/forecast"
    return fetch_json(url, params=params)


def fetch_daily_forecast(city_name: str, latitude: float, longitude: float, timezone: str, days: int = 7) -> Dict[str, Any]:
    """Request daily weather forecast for the next several days."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,relative_humidity_2m_mean",
        "timezone": timezone,
        "forecast_days": days,
    }
    url = "https://api.open-meteo.com/v1/forecast"
    return fetch_json(url, params=params)


def fetch_historical_weather(
    city_name: str,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, Any]:
    """Request historical daily weather data; defaults to the last 365 days."""
    end_date = end_date or date.today().isoformat()
    start_date = start_date or (date.today() - timedelta(days=365)).isoformat()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max",
        "timezone": timezone,
    }
    url = "https://archive-api.open-meteo.com/v1/archive"
    return fetch_json(url, params=params)


def archive_is_current(path: Path, end_date: str) -> bool:
    """True when a saved archive file already runs through the expected end date."""
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as file_handle:
        times = json.load(file_handle).get("daily", {}).get("time", [])
    return bool(times) and times[-1] == end_date


def ingest_archive(city_name: str, slug: str, latitude: float, longitude: float, timezone: str) -> None:
    """Fetch the multi-year baseline once; it only changes when a new year starts."""
    last_year = date.today().year - 1
    start_date = f"{last_year - ARCHIVE_YEARS + 1}-01-01"
    end_date = f"{last_year}-12-31"
    filename = f"{slug}_historical_archive.json"
    if archive_is_current(RAW_DIR / filename, end_date):
        logger.info("Archive for %s already covers %s to %s", city_name, start_date, end_date)
        return
    archive_data = fetch_historical_weather(city_name, latitude, longitude, timezone, start_date, end_date)
    save_json(archive_data, filename)


def ingest_city(city_name: str, config: Dict[str, float | str]) -> None:
    """Fetch and save current, forecast, and historical weather data for one city."""
    latitude = float(config["latitude"])
    longitude = float(config["longitude"])
    timezone = str(config["timezone"])

    logger.info("Starting ingestion for %s", city_name)

    current_data = fetch_current_weather(city_name, latitude, longitude, timezone)
    forecast_data = fetch_daily_forecast(city_name, latitude, longitude, timezone, days=7)
    historical_data = fetch_historical_weather(city_name, latitude, longitude, timezone)

    slug = city_name.lower().replace(" ", "_")
    save_json(current_data, f"{slug}_current.json")
    save_json(forecast_data, f"{slug}_forecast.json")
    save_json(historical_data, f"{slug}_historical.json")
    ingest_archive(city_name, slug, latitude, longitude, timezone)

    logger.info("Completed ingestion for %s", city_name)


def main() -> None:
    """Run ingestion for the configured U.S. cities."""
    disable_proxy_settings()
    logger.info("Weather ingestion started.")
    for city_name, config in CITIES.items():
        try:
            ingest_city(city_name, config)
        except RuntimeError as exc:
            logger.error("Skipping %s because of an ingestion error: %s", city_name, exc)

    logger.info("Weather ingestion finished.")


if __name__ == "__main__":
    main()
