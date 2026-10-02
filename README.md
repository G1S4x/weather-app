# Weather Forecast Dashboard

A Streamlit dashboard for comparing U.S. city weather against seasonal normals, with current conditions, a 7-day forecast, and historical trends by region.

## Overview

This project ingests Open-Meteo weather data for major U.S. cities, normalizes it into a SQLite database, and exposes the results in a dashboard with:

- current conditions by city
- 7-day forecast ranges and trends
- region comparisons against long-term normals
- anomaly views for forecast highs relative to historical averages
- simple refresh controls for re-running data ingestion and transformations

## Project structure

```text
weather_forecast/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   └── processing/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── .env.example
├── requirements.txt
├── README.md
└── weather_analysis.ipynb
```

## Data sources

- Open-Meteo Forecast API for current conditions and upcoming daily forecasts
- Open-Meteo Archive API for historical baseline data
- Local raw JSON files saved into `data/raw/`
- Processed SQLite table saved in `data/processed/weather.db`

## Local setup

```bash
cd weather_forecast
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To regenerate the processed database from the raw JSON files:

```bash
python -m src.processing.transform_weather
```

To launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## GitHub + Streamlit Cloud deployment

1. Initialize and push this repo to GitHub:

```bash
git init
git add .
git commit -m "Initial weather dashboard commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. In Streamlit Community Cloud, click New app.
3. Connect the GitHub repo you just pushed.
4. Set the app path to `app/streamlit_app.py`.
5. Keep the default Python install and use the repo root as the working directory.
6. Streamlit will install dependencies from `requirements.txt` automatically.

## Notes

- This project does not require an API key.
- The app reads from the processed SQLite database, so it is fast and reliable once the data pipeline has run.
- The raw JSON archive files are intentionally kept in the repo because they are expensive to refresh and only change yearly.
