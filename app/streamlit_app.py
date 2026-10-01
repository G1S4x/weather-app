"""Streamlit dashboard for the U.S. weather pipeline.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion import weather_ingestion  # noqa: E402
from src.processing import transform_weather  # noqa: E402

DB_PATH = transform_weather.DB_PATH
CITIES = list(weather_ingestion.CITIES.keys())
CITY_REGION = {city: str(config["region"]) for city, config in weather_ingestion.CITIES.items()}
REGIONS = ["Northeast", "Midwest", "South", "West"]

# The five most populous U.S. cities; a pinned home city takes the first spot.
TOP_CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
NO_HOME = "None"

# Categorical slots 1-4 in fixed order, one per region (validated for line charts in both modes).
REGION_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
SINGLE_SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
NORMAL_INK = "#898781"
LABEL_INK = {"light": "#52514e", "dark": "#c3c2b7"}
# Diverging: cooler (blue) <- neutral gray -> warmer (red).
DIVERGING = {
    "light": ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b0ab", "#e34948", "#a32a2a"],
    "dark": ["#104281", "#256abf", "#5598e7", "#383835", "#e66767", "#c93d3d", "#8f2222"],
}
WARM, COOL = "#e34948", "#2a78d6"

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Dense drizzle", "🌦️"),
    56: ("Freezing drizzle", "🌧️"), 57: ("Freezing drizzle", "🌧️"),
    61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "🌨️"),
    80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌦️"), 82: ("Violent rain showers", "⛈️"),
    85: ("Snow showers", "🌨️"), 86: ("Snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with hail", "⛈️"),
}


def describe(code: float | None) -> tuple[str, str]:
    if code is None or pd.isna(code):
        return "Unknown", "🌡️"
    return WEATHER_CODES.get(int(code), ("Unknown", "🌡️"))


# ---------- data ----------

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the processed current and daily tables from SQLite."""
    with sqlite3.connect(DB_PATH) as connection:
        current_df = pd.read_sql_query("SELECT * FROM weather_current", connection)
        daily_df = pd.read_sql_query("SELECT * FROM weather_daily", connection)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["region"] = daily_df["city"].map(CITY_REGION)
    daily_df["temp_avg_c"] = (daily_df["temperature_2m_max_c"] + daily_df["temperature_2m_min_c"]) / 2
    daily_df["doy"] = daily_df["date"].dt.dayofyear
    current_df["region"] = current_df["city"].map(CITY_REGION)
    return current_df, daily_df


def baseline_years(daily_df: pd.DataFrame) -> list[int]:
    """Complete calendar years of history, used to define 'normal'."""
    historical = daily_df[daily_df["source"] == "historical"]
    days_per_year = historical.groupby(historical["date"].dt.year)["date"].nunique()
    return sorted(int(year) for year, days in days_per_year.items() if days >= 365 and year < date.today().year)


@st.cache_data
def compute_normals(daily_df: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    """Per-city normal high/low/avg for each day of the year, smoothed over a 15-day window."""
    historical = daily_df[(daily_df["source"] == "historical") & daily_df["date"].dt.year.isin(years)]
    by_day = historical.groupby(["city", "doy"])[["temperature_2m_max_c", "temperature_2m_min_c", "temp_avg_c"]].mean()

    frames = []
    for city, frame in by_day.groupby(level="city"):
        frame = frame.droplevel("city").reindex(range(1, 367)).interpolate(limit_direction="both")
        # Wrap the year so smoothing near Jan 1 / Dec 31 borrows from the other end.
        wrapped = pd.concat([frame.iloc[-7:], frame, frame.iloc[:7]])
        smoothed = wrapped.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
        smoothed.index.name = "doy"
        frames.append(smoothed.reset_index().assign(city=city))

    return pd.concat(frames).rename(
        columns={
            "temperature_2m_max_c": "normal_high_c",
            "temperature_2m_min_c": "normal_low_c",
            "temp_avg_c": "normal_avg_c",
        }
    )


def refresh_data() -> None:
    """Re-run ingestion and transformation, then clear the cached tables."""
    weather_ingestion.main()
    transform_weather.main()
    st.cache_data.clear()


def to_units(df: pd.DataFrame, imperial: bool) -> pd.DataFrame:
    """Return a copy with temperature, precipitation, and wind in the chosen units."""
    df = df.copy()
    if not imperial:
        return df
    for col in [c for c in df.columns if c.endswith("_c")]:
        df[col] = df[col] * 9 / 5 + 32
    for col in [c for c in df.columns if c.endswith("_mm")]:
        df[col] = df[col] / 25.4
    for col in [c for c in df.columns if c.endswith("_kmh")]:
        df[col] = df[col] / 1.609344
    return df


def signed(value: float, unit: str, digits: int = 0) -> str:
    return f"{value:+.{digits}f}{unit}"


# ---------- charts ----------

def region_scale(mode: str) -> alt.Scale:
    return alt.Scale(domain=REGIONS, range=REGION_COLORS[mode])


def diverging_scale(values: pd.Series, mode: str) -> alt.Scale:
    """Symmetric scale around zero so gray always means 'normal'."""
    limit = max(float(values.abs().quantile(0.95)), 1.0)
    # Interpolate in RGB so the ramp passes cleanly through gray instead of drifting to brown/teal.
    return alt.Scale(domain=[-limit, limit], range=DIVERGING[mode], clamp=True, interpolate="rgb")


def spread_labels(ends: pd.DataFrame, y: str, y_span: float, height: int, min_gap_px: int = 16) -> pd.DataFrame:
    """Nudge line-end labels apart vertically so close lines don't print on top of each other."""
    ends = ends.sort_values(y).copy()
    # Convert the pixel gap into data units using the plotted value range.
    plot_height = max(height - 70, 100)  # minus legend and axis
    gap = (y_span or 1.0) * min_gap_px / plot_height
    positions = ends[y].tolist()
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + gap)
    ends["label_y"] = positions
    return ends


def region_line_chart(df: pd.DataFrame, x: str, y: str, mode: str, fmt: str, x_format: str, height: int = 320):
    """Four region lines with a crosshair, a per-x tooltip, and direct labels at the line ends."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format=x_format, grid=False, tickCount="day" if x_format == "%a" else alt.Undefined)),
        y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("region:N", scale=region_scale(mode), legend=alt.Legend(title=None, orient="top")),
    )
    lines = base.mark_line(strokeWidth=2)

    ends = spread_labels(df.loc[df.groupby("region")[x].idxmax()], y, df[y].max() - df[y].min(), height)
    labels = alt.Chart(ends).mark_text(align="left", dx=6, fontSize=12, color=LABEL_INK[mode]).encode(
        x=f"{x}:T", y=alt.Y("label_y:Q", scale=alt.Scale(zero=False)), text="region:N"
    )

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    wide = df.pivot_table(index=x, columns="region", values=y).reset_index()
    rule = (
        alt.Chart(wide)
        .mark_rule(color="gray", strokeWidth=1)
        .encode(
            x=f"{x}:T",
            opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
            tooltip=[alt.Tooltip(f"{x}:T", title="Date", format=x_format)]
            + [alt.Tooltip(f"{r}:Q", title=r, format=fmt) for r in REGIONS if r in wide.columns],
        )
        .add_params(nearest)
    )
    points = base.mark_point(filled=True, size=70).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    ).transform_filter(nearest)

    return alt.layer(lines, labels, rule, points).properties(height=height, padding={"right": 80})


def forecast_range_chart(df: pd.DataFrame, mode: str, unit: str) -> alt.LayerChart:
    """Low-to-high bar per day, with gray ticks marking the normal high and low."""
    base = alt.Chart(df).encode(x=alt.X("day_label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)))
    tooltip = [
        alt.Tooltip("day_label:N", title="Day"),
        alt.Tooltip("temperature_2m_max_c:Q", title=f"High ({unit})", format=".0f"),
        alt.Tooltip("temperature_2m_min_c:Q", title=f"Low ({unit})", format=".0f"),
        alt.Tooltip("normal_high_c:Q", title=f"Normal high ({unit})", format=".0f"),
        alt.Tooltip("normal_low_c:Q", title=f"Normal low ({unit})", format=".0f"),
        alt.Tooltip("precipitation_probability_max_pct:Q", title="Chance of precip (%)"),
    ]
    bars = base.mark_bar(width=28, cornerRadius=4, color=SINGLE_SERIES[mode]).encode(
        y=alt.Y("temperature_2m_min_c:Q", title=None, scale=alt.Scale(zero=False, padding=20)),
        y2="temperature_2m_max_c:Q",
        tooltip=tooltip,
    )
    normal_high = base.mark_tick(thickness=2, size=44, color=NORMAL_INK).encode(y="normal_high_c:Q", tooltip=tooltip)
    normal_low = base.mark_tick(thickness=2, size=44, color=NORMAL_INK).encode(y="normal_low_c:Q", tooltip=tooltip)
    high_text = base.mark_text(dy=-12, fontSize=13, fontWeight="bold", color=LABEL_INK[mode]).encode(
        y="temperature_2m_max_c:Q", text=alt.Text("temperature_2m_max_c:Q", format=".0f")
    )
    low_text = base.mark_text(dy=14, fontSize=12, color=LABEL_INK[mode]).encode(
        y="temperature_2m_min_c:Q", text=alt.Text("temperature_2m_min_c:Q", format=".0f")
    )
    return alt.layer(bars, normal_high, normal_low, high_text, low_text).properties(height=300)


def forecast_anomaly_heatmap(df: pd.DataFrame, mode: str, unit: str) -> alt.FacetChart:
    """City x day grid of forecast highs, colored by how far they sit from normal, grouped by region."""
    rect = alt.Chart().mark_rect(cornerRadius=3).encode(
        x=alt.X("weekday:N", sort=None, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelOverlap=False)),
        y=alt.Y("city:N", title=None),
        color=alt.Color(
            "high_vs_normal:Q",
            title=f"High vs normal ({unit})",
            scale=diverging_scale(df["high_vs_normal"], mode),
            legend=alt.Legend(orient="bottom", gradientLength=260),
        ),
        tooltip=[
            alt.Tooltip("city:N", title="City"),
            alt.Tooltip("day_label:N", title="Day"),
            alt.Tooltip("temperature_2m_max_c:Q", title=f"High ({unit})", format=".0f"),
            alt.Tooltip("normal_high_c:Q", title=f"Normal high ({unit})", format=".0f"),
            alt.Tooltip("high_vs_normal:Q", title=f"Difference ({unit})", format="+.0f"),
        ],
    )
    text = alt.Chart().mark_text(fontSize=11).encode(
        x=alt.X("weekday:N", sort=None),
        y="city:N",
        text=alt.Text("temperature_2m_max_c:Q", format=".0f"),
        color=(
            alt.value("white")
            if mode == "dark"
            else alt.condition(f"abs(datum.high_vs_normal) > {8 if unit == '°F' else 4.5}", alt.value("white"), alt.value("#0b0b0b"))
        ),
    )
    return (
        alt.layer(rect, text, data=df)
        .properties(width=alt.Step(52), height=alt.Step(22))
        .facet(row=alt.Row("region:N", sort=REGIONS, title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold")))
        .resolve_scale(y="independent")
    )


def regional_anomaly_bars(df: pd.DataFrame, unit: str) -> alt.FacetChart:
    """Small multiples: each region's monthly departure from normal; red is warmer, blue cooler."""
    bars = alt.Chart().mark_bar(cornerRadius=3, width={"band": 0.75}).encode(
        x=alt.X("yearmonth(month):O", title=None, axis=alt.Axis(format="%b", labelAngle=0)),
        y=alt.Y("anomaly:Q", title=f"vs normal ({unit})"),
        color=alt.condition("datum.anomaly > 0", alt.value(WARM), alt.value(COOL)),
        tooltip=[
            alt.Tooltip("region:N", title="Region"),
            alt.Tooltip("month:T", title="Month", format="%B %Y"),
            alt.Tooltip("anomaly:Q", title=f"vs normal ({unit})", format="+.1f"),
        ],
    )
    zero = alt.Chart().mark_rule(color="gray").encode(y=alt.datum(0))
    return (
        alt.layer(bars, zero, data=df)
        .properties(width=300, height=180)
        .facet(facet=alt.Facet("region:N", sort=REGIONS, title=None, header=alt.Header(labelFontWeight="bold")), columns=2)
    )


def calendar_heatmap(df: pd.DataFrame, mode: str, unit: str) -> alt.Chart:
    """GitHub-style calendar of daily departures from normal over the past year."""
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("week_start:T", title=None, axis=alt.Axis(format="%b", tickCount="month", grid=False)),
            x2="week_end:T",
            y=alt.Y("weekday:N", sort=weekdays, title=None),
            color=alt.Color(
                "anomaly:Q",
                title=f"Daily avg vs normal ({unit})",
                scale=diverging_scale(df["anomaly"], mode),
                legend=alt.Legend(orient="bottom", gradientLength=260),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%a %b %d, %Y"),
                alt.Tooltip("temperature_2m_max_c:Q", title=f"High ({unit})", format=".0f"),
                alt.Tooltip("temperature_2m_min_c:Q", title=f"Low ({unit})", format=".0f"),
                alt.Tooltip("anomaly:Q", title=f"vs normal ({unit})", format="+.1f"),
            ],
        )
        .properties(height=alt.Step(20))
    )


def ranking_bars(df: pd.DataFrame, value: str, title: str, fmt: str, mode: str) -> alt.ConcatChart:
    """One panel per region so each region keeps its own color without mixing four hues in one chart."""
    x_max = float(df[value].max()) * 1.12 or 1.0
    panels = []
    for region in REGIONS:
        subset = df[df["region"] == region].sort_values(value, ascending=False)
        order = subset["city"].tolist()
        base = alt.Chart(subset, title=alt.Title(region, fontSize=13, anchor="start")).encode(
            y=alt.Y("city:N", sort=order, title=None),
            x=alt.X(f"{value}:Q", title=title, scale=alt.Scale(domain=[0, x_max])),
        )
        bars = base.mark_bar(cornerRadiusEnd=4, height=14, color=REGION_COLORS[mode][REGIONS.index(region)]).encode(
            tooltip=[alt.Tooltip("city:N", title="City"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)]
        )
        text = base.mark_text(align="left", dx=4, fontSize=11, color=LABEL_INK[mode]).encode(
            text=alt.Text(f"{value}:Q", format=fmt)
        )
        panels.append(alt.layer(bars, text).properties(width=280, height=alt.Step(20)))
    return alt.concat(*panels, columns=2).resolve_scale(y="independent")


# ---------- page ----------

st.set_page_config(page_title="U.S. Weather Dashboard", page_icon="🌤️", layout="wide")

if not DB_PATH.exists():
    st.error("No processed data found. Click **Refresh from API** in the sidebar to build it.")

# The home city lives in the URL (?home=Detroit) so it survives reloads and bookmarks.
saved_home = st.query_params.get("home", NO_HOME)
home_options = [NO_HOME, *sorted(CITIES)]

with st.sidebar:
    st.header("Settings")
    home = st.selectbox(
        "📍 Home city",
        home_options,
        index=home_options.index(saved_home) if saved_home in home_options else 0,
        help="Pinned to the top of the dashboard. Bookmark the page to keep it.",
    )
    if home == NO_HOME:
        st.query_params.pop("home", None)
    else:
        st.query_params["home"] = home
    imperial = st.toggle("Imperial units (°F, in, mph)", value=True)
    st.divider()
    if st.button("🔄 Refresh from API", width="stretch"):
        with st.spinner(f"Fetching {len(CITIES)} cities from Open-Meteo…"):
            refresh_data()
        st.success("Data refreshed.")

if not DB_PATH.exists():
    st.stop()

home = None if home == NO_HOME else home
featured = (([home] if home else []) + [c for c in TOP_CITIES if c != home])[:5]

mode = "dark" if getattr(st.context, "theme", None) is not None and st.context.theme.type == "dark" else "light"
temp_unit, precip_unit, wind_unit = ("°F", "in", "mph") if imperial else ("°C", "mm", "km/h")
deg = "°"

current_raw, daily_raw = load_data()
years = baseline_years(daily_raw)
normals = to_units(compute_normals(daily_raw, tuple(years)), imperial)
current = to_units(current_raw, imperial)
daily = to_units(daily_raw, imperial).merge(normals, on=["city", "doy"], how="left")
historical = daily[daily["source"] == "historical"]
forecast = daily[daily["source"] == "forecast"].copy()
forecast["day_label"] = forecast["date"].dt.strftime("%a %-m/%-d")
forecast["weekday"] = forecast["date"].dt.strftime("%a")
forecast["high_vs_normal"] = forecast["temperature_2m_max_c"] - forecast["normal_high_c"]
baseline_label = f"{years[0]}–{years[-1]}" if years else "n/a"

st.title("🌤️ U.S. Weather Dashboard")
st.caption(
    f"{len(CITIES)} cities from Open-Meteo · normals = {baseline_label} average"
    f" · last updated {current_raw['timestamp'].max()}"
)

# ----- Right now: home + the biggest cities -----
st.subheader("Right now")
today = forecast[forecast["date"] == forecast["date"].min()].set_index("city")
current_by_city = current.set_index("city")
for col, city in zip(st.columns(len(featured)), featured):
    row = current_by_city.loc[city]
    day = today.loc[city]
    label, icon = describe(row["weather_code"])
    diff = day["high_vs_normal"]
    with col, st.container(border=True):
        st.markdown(f"**📍 {city}** · home" if city == home else f"**{city}**")
        st.metric(
            f"{icon} {label}",
            f"{row['temperature_2m_c']:.0f}{temp_unit}",
            f"{abs(diff):.0f}{deg} {'above' if diff >= 0 else 'below'} normal",
            delta_color="off",
            delta_arrow="up" if diff >= 0 else "down",
        )
        st.caption(
            f"High {day['temperature_2m_max_c']:.0f}{deg} · Low {day['temperature_2m_min_c']:.0f}{deg} · Feels {row['apparent_temperature_c']:.0f}{deg}  \n"
            f"💧 {row['relative_humidity_2m_pct']:.0f}% · 💨 {row['wind_speed_10m_kmh']:.0f} {wind_unit}"
        )

# ----- Next 7 days -----
st.subheader("Next 7 days")
forecast_city = st.segmented_control(
    "Forecast for", featured, default=featured[0], key=f"forecast_city_{featured[0]}", label_visibility="collapsed"
) or featured[0]
city_forecast = forecast[forecast["city"] == forecast_city].sort_values("date")

for col, (_, day) in zip(st.columns(len(city_forecast)), city_forecast.iterrows()):
    label, icon = describe(day["weather_code"])
    with col:
        st.markdown(
            f"<div style='text-align:center'><b>{day['date']:%a}</b><br>"
            f"<span style='font-size:2rem' title='{label}'>{icon}</span><br>"
            f"<b>{day['temperature_2m_max_c']:.0f}{deg}</b> / {day['temperature_2m_min_c']:.0f}{deg}<br>"
            f"<small>💧 {day['precipitation_probability_max_pct']:.0f}%</small></div>",
            unsafe_allow_html=True,
        )

st.markdown(f"**{forecast_city}: daily low to high ({temp_unit})** — gray ticks mark the normal high and low")
st.altair_chart(forecast_range_chart(city_forecast, mode, temp_unit), width="stretch")

region_forecast = forecast.groupby(["region", "date"], as_index=False)["temperature_2m_max_c"].mean()
left, right = st.columns([2, 3])
with left:
    st.markdown(f"**Average forecast high by region ({temp_unit})**")
    st.altair_chart(
        region_line_chart(region_forecast, "date", "temperature_2m_max_c", mode, ".0f", "%a", height=300),
        width="stretch",
    )
    by_region = forecast.groupby("region")["high_vs_normal"].mean().sort_values()
    st.caption(
        f"This week the **{by_region.index[-1]}** runs warmest relative to normal "
        f"({signed(by_region.iloc[-1], deg)}) and the **{by_region.index[0]}** coolest ({signed(by_region.iloc[0], deg)})."
    )
with right:
    st.markdown(f"**Every city's forecast high, colored by distance from normal ({temp_unit})**")
    st.altair_chart(forecast_anomaly_heatmap(forecast, mode, temp_unit))

# ----- Long-term trends -----
st.subheader("Long-term trends")
seasons_tab, normal_tab, calendar_tab, rankings_tab, data_tab = st.tabs(
    ["Seasons", "Past year vs normal", "Daily calendar", "City rankings", "Data"]
)

with seasons_tab:
    monthly = (
        historical.assign(month=historical["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["region", "month"], as_index=False)["temp_avg_c"].mean()
    )
    st.markdown(f"**Monthly average temperature by region since {historical['date'].min():%Y} ({temp_unit})**")
    st.altair_chart(region_line_chart(monthly, "month", "temp_avg_c", mode, ".1f", "%b %Y"), width="stretch")

    in_baseline = historical[historical["date"].dt.year.isin(years)]
    typical_precip = (
        in_baseline.assign(year=in_baseline["date"].dt.year, month_num=in_baseline["date"].dt.month)
        .groupby(["region", "city", "year", "month_num"], as_index=False)["precipitation_sum_mm"].sum()
        .groupby(["region", "month_num"], as_index=False)["precipitation_sum_mm"].mean()
    )
    typical_precip["month"] = pd.to_datetime(dict(year=2000, month=typical_precip["month_num"], day=1))
    st.markdown(f"**A typical year of precipitation, average per city ({precip_unit}/month, {baseline_label})**")
    st.altair_chart(
        region_line_chart(typical_precip, "month", "precipitation_sum_mm", mode, ".1f" if imperial else ".0f", "%b", height=260),
        width="stretch",
    )

with normal_tab:
    last_year = historical[historical["date"] > historical["date"].max() - pd.Timedelta(days=365)].copy()
    last_year["anomaly"] = last_year["temp_avg_c"] - last_year["normal_avg_c"]
    st.markdown(f"**Past 12 months compared with the {baseline_label} normal**")
    region_totals = last_year.groupby("region")["anomaly"].mean()
    for col, region in zip(st.columns(len(REGIONS)), REGIONS):
        value = region_totals.get(region, 0.0)
        col.metric(
            region,
            signed(value, temp_unit, 1),
            "warmer than normal" if value >= 0 else "cooler than normal",
            delta_color="off",
            delta_arrow="up" if value >= 0 else "down",
        )

    region_months = (
        last_year.assign(month=last_year["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["region", "month"], as_index=False)["anomaly"].mean()
    )
    st.altair_chart(regional_anomaly_bars(region_months, temp_unit))

    city_anomaly = last_year.groupby(["city", "region"], as_index=False)["anomaly"].mean().sort_values("anomaly")
    coolest, warmest = city_anomaly.iloc[0], city_anomaly.iloc[-1]
    st.caption(
        f"Biggest outliers: **{warmest['city']}** ran {signed(warmest['anomaly'], temp_unit, 1)} vs normal; "
        f"**{coolest['city']}** ran {signed(coolest['anomaly'], temp_unit, 1)}."
    )

with calendar_tab:
    city_options = sorted(CITIES)
    calendar_city = st.selectbox("City", city_options, index=city_options.index(featured[0]), key=f"calendar_city_{featured[0]}")
    cal = historical[
        (historical["city"] == calendar_city)
        & (historical["date"] > historical["date"].max() - pd.Timedelta(days=365))
    ].copy()
    cal["anomaly"] = cal["temp_avg_c"] - cal["normal_avg_c"]
    cal["weekday"] = cal["date"].dt.strftime("%a")
    cal["week_start"] = cal["date"] - pd.to_timedelta(cal["date"].dt.weekday, unit="D")
    cal["week_end"] = cal["week_start"] + pd.Timedelta(days=7)
    warm_days = int((cal["anomaly"] > 0).sum())
    st.markdown(
        f"**{calendar_city}: every day of the past year vs normal** — "
        f"{warm_days} of {len(cal)} days were warmer than the {baseline_label} average"
    )
    st.altair_chart(calendar_heatmap(cal, mode, temp_unit), width="stretch")

with rankings_tab:
    nice_low, nice_high, hot, freeze, wet = 18.3, 27.8, 32.2, 0.0, 1.0  # °C / mm thresholds
    base = daily_raw[(daily_raw["source"] == "historical") & daily_raw["date"].dt.year.isin(years)]
    flags = base.assign(
        nice=base["temperature_2m_max_c"].between(nice_low, nice_high) & (base["precipitation_sum_mm"] < wet),
        hot=base["temperature_2m_max_c"] >= hot,
        freeze=base["temperature_2m_min_c"] <= freeze,
        wet=base["precipitation_sum_mm"] >= wet,
    )
    per_year = flags.groupby(["city", "region"])[["nice", "hot", "freeze", "wet", "precipitation_sum_mm"]].sum() / max(len(years), 1)
    per_year = to_units(per_year.reset_index(), imperial)

    metrics = {
        f"Nice days (high {65 if imperial else 18}–{82 if imperial else 28}{temp_unit}, dry)": ("nice", ".0f"),
        f"Hot days (high ≥ {90 if imperial else 32}{temp_unit})": ("hot", ".0f"),
        f"Freezing nights (low ≤ {32 if imperial else 0}{temp_unit})": ("freeze", ".0f"),
        f"Wet days (≥ {0.04 if imperial else 1} {precip_unit})": ("wet", ".0f"),
        f"Annual precipitation ({precip_unit})": ("precipitation_sum_mm", ".1f" if imperial else ".0f"),
    }
    choice = st.radio("Rank cities by", list(metrics), horizontal=True)
    value_col, fmt = metrics[choice]
    st.markdown(f"**{choice}, per year ({baseline_label} average)**")
    st.altair_chart(ranking_bars(per_year, value_col, "per year", fmt, mode))
    leader = per_year.sort_values(value_col, ascending=False).iloc[0]
    st.caption(f"Leader: **{leader['city']}** ({leader['region']}) with {leader[value_col]:{fmt}} per year.")

with data_tab:
    region_filter = st.multiselect("Regions", REGIONS, default=REGIONS)
    source_choice = st.radio("Rows", ["Historical", "Forecast"], horizontal=True)
    table = daily[(daily["source"] == source_choice.lower()) & daily["region"].isin(region_filter)]
    table = table.drop(columns=["temp_avg_c", "doy"]).sort_values(["city", "date"])
    st.caption(f"{len(table):,} rows · values shown in {'imperial' if imperial else 'metric'} units (column names keep the metric suffix)")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"weather_{source_choice.lower()}.csv",
        mime="text/csv",
    )
