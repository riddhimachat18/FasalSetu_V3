"""
Weather Agent — plain Python functions, compatible with google-adk FunctionTool.
Uses OpenWeatherMap (free tier) if OPENWEATHER_API_KEY is set in .env;
falls back to seasonal estimates so the agent still works without the key.

Public tools (registered with ADK FunctionTool):
  get_weather_forecast        — current weather by city/state name or "lat,lon"
  check_spray_conditions      — wind/rain/temp/humidity gate for spraying
  get_farming_weather_advice  — weather + crop-specific farming actions today
  get_detailed_forecast       — 5-day daily forecast with per-day farming notes

Private helpers (prefixed _ — NOT exposed as tools):
  _get_coords, _parse_coords, _reverse_geocode, _get_weather_raw,
  _seasonal_fallback, _degrees_to_compass, _spray_issues
"""

import logging
import os
from datetime import datetime
from typing import Tuple

import requests
from dotenv import load_dotenv

load_dotenv()
logger  = logging.getLogger("agents.weather")
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# OWM free-tier endpoints only
_OWM_CURRENT  = "https://api.openweathermap.org/data/2.5/weather"
_OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
_GEO_URL      = "https://api.openweathermap.org/geo/1.0/direct"
_REV_GEO_URL  = "https://api.openweathermap.org/geo/1.0/reverse"

_STATE_COORDS: dict = {
    "punjab":           (31.15, 75.34),
    "haryana":          (29.06, 76.09),
    "uttar pradesh":    (26.85, 80.91),
    "madhya pradesh":   (23.47, 77.95),
    "maharashtra":      (19.75, 75.71),
    "rajasthan":        (27.02, 74.22),
    "gujarat":          (22.69, 71.58),
    "bihar":            (25.09, 85.31),
    "west bengal":      (23.68, 85.05),
    "karnataka":        (15.32, 75.71),
    "andhra pradesh":   (15.91, 79.74),
    "telangana":        (17.85, 79.11),
    "tamil nadu":       (11.13, 78.66),
    "odisha":           (20.25, 84.77),
    "jharkhand":        (23.61, 85.27),
    "delhi":            (28.61, 77.21),
    "chhattisgarh":     (21.27, 81.86),
    "himachal pradesh": (31.90, 77.11),
    "uttarakhand":      (30.07, 79.02),
    "assam":            (26.14, 91.74),
    "kerala":           (10.85, 76.27),
    "goa":              (15.30, 74.12),
}


# ── Private helpers ────────────────────────────────────────────────────────

def _degrees_to_compass(degrees: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(degrees / (360 / len(dirs))) % len(dirs)]


def _parse_coords(location: str):
    """Return (lat, lon) tuple if location is 'lat,lon' string, else None."""
    if "," not in location:
        return None
    parts = location.split(",", 1)
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None


def _get_coords(location: str):
    """Single source of truth for resolving location string to (lat, lon)."""
    coords = _parse_coords(location)
    if coords:
        return coords

    loc_lower = location.lower().strip()
    for state, c in _STATE_COORDS.items():
        if state in loc_lower:
            logger.debug("State lookup: %s", state)
            return c

    if OWM_KEY:
        try:
            r = requests.get(_GEO_URL,
                params={"q": f"{location},IN", "limit": 1, "appid": OWM_KEY},
                timeout=5)
            if r.ok and r.json():
                d = r.json()[0]
                return d["lat"], d["lon"]
        except Exception as e:
            logger.warning("Geocoding failed: %s", e)

    logger.warning("Falling back to India centre for: %s", location)
    return 22.5, 78.9


def _reverse_geocode(lat: float, lon: float) -> str:
    """Return readable location name from coordinates."""
    if not OWM_KEY:
        return f"{lat:.3f},{lon:.3f}"
    try:
        r = requests.get(_REV_GEO_URL,
            params={"lat": lat, "lon": lon, "limit": 1, "appid": OWM_KEY},
            timeout=5)
        if r.ok and r.json():
            d = r.json()[0]
            parts = [d.get("name",""), d.get("state","")]
            return ", ".join(p for p in parts if p)
    except Exception as e:
        logger.warning("Reverse geocode failed: %s", e)
    return f"{lat:.3f},{lon:.3f}"


def _seasonal_fallback(location: str) -> dict:
    month = datetime.now().month
    if   month in (3,4,5):     season, temp, hum = "Summer",       36, 40
    elif month in (6,7,8,9):   season, temp, hum = "Monsoon",      29, 85
    elif month in (10,11):     season, temp, hum = "Post-Monsoon", 27, 65
    else:                      season, temp, hum = "Winter",       18, 55
    return {
        "location":       location,
        "source":         "seasonal-estimate",
        "season":         season,
        "temperature_c":  temp,
        "humidity_pct":   hum,
        "wind_speed_kmh": 12,
        "wind_direction": "N/A",
        "condition":      season,
        "rain_expected":  month in (6,7,8,9),
        "note": "Live weather unavailable. Add OPENWEATHER_API_KEY to .env for real data.",
    }


def _get_weather_raw(lat: float, lon: float, label: str) -> dict:
    """One OWM current-weather call. All public tools funnel through this."""
    if not OWM_KEY:
        return _seasonal_fallback(label)
    try:
        r = requests.get(_OWM_CURRENT,
            params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"},
            timeout=8)
        r.raise_for_status()
        d = r.json()
        wind_deg = d["wind"].get("deg", 0)
        return {
            "location":       label,
            "coordinates":    {"lat": lat, "lon": lon},
            "source":         "openweathermap-live",
            "temperature_c":  round(d["main"]["temp"], 1),
            "feels_like_c":   round(d["main"]["feels_like"], 1),
            "humidity_pct":   d["main"]["humidity"],
            "pressure_hpa":   d["main"]["pressure"],
            "wind_speed_kmh": round(d["wind"]["speed"] * 3.6, 1),
            "wind_direction": _degrees_to_compass(wind_deg),
            "condition":      d["weather"][0]["description"].title(),
            "rain_expected":  "rain" in d,
            "clouds_pct":     d.get("clouds", {}).get("all", 0),
            "visibility_km":  round(d.get("visibility", 10000) / 1000, 1),
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        logger.error("OWM HTTP %d for %s", code, label)
        fb = _seasonal_fallback(label)
        fb["error"] = "Invalid API key — check OPENWEATHER_API_KEY in .env" if code == 401 else f"HTTP {code}"
        return fb
    except Exception as e:
        logger.error("OWM request failed: %s", e)
        return _seasonal_fallback(label)


def _spray_issues(wx: dict) -> list:
    """Return list of unsafe-spray reasons from a weather dict (empty = safe)."""
    issues = []
    wind  = wx.get("wind_speed_kmh", 12)
    humid = wx.get("humidity_pct",   65)
    temp  = wx.get("temperature_c",  28)
    rain  = wx.get("rain_expected", False)
    if wind > 15:  issues.append(f"Wind {wind} km/h exceeds 15 km/h limit — spray will drift")
    if rain:       issues.append("Rain expected — spray will wash off before absorption")
    if temp > 35:  issues.append(f"Temperature {temp}°C too high — evaporation and leaf burn risk")
    if temp < 10:  issues.append(f"Temperature {temp}°C too low — poor absorption")
    if humid < 40: issues.append(f"Humidity {humid}% too low — spray evaporates too quickly")
    return issues


# ── Public tools ───────────────────────────────────────────────────────────

def get_weather_forecast(location: str) -> dict:
    """Get current weather for a location in India.

    Args:
        location: City or state name in India (e.g. Ludhiana, Punjab, Pune)
                  or GPS coordinates as "lat,lon" string (e.g. "28.61,77.21")

    Returns:
        Current temperature, humidity, wind speed and direction, condition, rain status.
    """
    lat, lon = _get_coords(location)
    label = _reverse_geocode(lat, lon) if _parse_coords(location) else location
    return _get_weather_raw(lat, lon, label)


def check_spray_conditions(location: str) -> dict:
    """Check whether weather conditions are safe for pesticide or fertiliser spraying today.

    Args:
        location: City or state in India where spraying is planned,
                  or GPS coordinates as "lat,lon"

    Returns:
        Safe or unsafe verdict with specific reasons based on wind, rain, temperature, humidity.
    """
    wx     = get_weather_forecast(location)
    issues = _spray_issues(wx)
    return {
        "location":      location,
        "safe_to_spray": len(issues) == 0,
        "verdict":       "Safe to spray" if not issues else "Do NOT spray now",
        "reasons":       issues if issues else ["All conditions suitable for spraying"],
        "best_time":     "Early morning 6-9 AM or evening 5-7 PM gives best absorption",
        "weather":       wx,
    }


def get_farming_weather_advice(location: str, crop: str = "general") -> dict:
    """Get weather-based farming advice for a specific crop and location today.

    Args:
        location: City or state in India, or "lat,lon" coordinates
        crop: Crop being grown (e.g. wheat, rice, cotton, tomato, maize)

    Returns:
        Weather summary, list of farming actions to take today, and spray safety check.
    """
    wx     = get_weather_forecast(location)   # single API call
    issues = _spray_issues(wx)                # reuses wx — no second API call
    temp   = wx.get("temperature_c", 28)
    rain   = wx.get("rain_expected", False)
    humid  = wx.get("humidity_pct",  65)
    crop_l = crop.lower()
    actions = []

    if rain:
        actions.append("Hold off on irrigation — rain expected today")
        actions.append("Do not spray fertilisers or pesticides — rain will wash them off")
    if temp > 38:
        actions.append("Extreme heat — increase irrigation; use shade nets for seedlings")
    if temp < 12:
        actions.append("Cold weather — delay transplanting; cover seedlings overnight")

    if ("rice" in crop_l or "paddy" in crop_l) and humid > 80:
        actions.append("High humidity — monitor rice for blast and brown spot")
    if "tomato" in crop_l and humid > 75:
        actions.append("High humidity — check tomatoes for early or late blight; improve spacing")
    if "wheat" in crop_l and temp > 32:
        actions.append("Heat stress on wheat — irrigate in the evening; watch for yellow rust")
    if "cotton" in crop_l and rain:
        actions.append("Post-rain — check for bollworm egg hatching in 3 to 5 days")
    if ("maize" in crop_l or "corn" in crop_l) and temp > 35:
        actions.append("High temperature — ensure soil moisture at tasselling stage")

    if not actions:
        actions.append("Conditions are normal — continue regular crop management")

    return {
        "location":    location,
        "crop":        crop,
        "weather":     wx,
        "actions":     actions,
        "spray_check": {
            "safe_to_spray": len(issues) == 0,
            "verdict":       "Safe to spray" if not issues else "Do NOT spray now",
            "reasons":       issues if issues else ["All conditions suitable"],
        },
    }


def get_detailed_forecast(location: str, days: int = 5) -> dict:
    """Get a multi-day weather forecast for farming planning.

    Args:
        location: City or state in India, or "lat,lon" coordinates
        days: Number of forecast days, 1 to 5 (free OWM tier provides 5 days maximum)

    Returns:
        Daily forecast with min/max temperature, rainfall, humidity, and farming notes per day.
    """
    days = max(1, min(days, 5))   # free tier hard cap
    lat, lon = _get_coords(location)

    if not OWM_KEY:
        return {
            "location": location,
            "error":    "Detailed forecast requires OPENWEATHER_API_KEY in .env",
            "current":  _seasonal_fallback(location),
        }

    try:
        r = requests.get(
            _OWM_FORECAST,
            params={"lat": lat, "lon": lon, "appid": OWM_KEY,
                    "units": "metric", "cnt": min(days * 8, 40)},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        # Aggregate 3-hour slots into daily summaries
        daily: dict = {}
        for item in data["list"]:
            dt  = datetime.fromtimestamp(item["dt"])
            key = dt.strftime("%Y-%m-%d")
            if key not in daily:
                daily[key] = {"date": key, "day_name": dt.strftime("%A"),
                              "temps": [], "humidity": [], "wind_kmh": [],
                              "rain_mm": 0.0, "conditions": []}
            daily[key]["temps"].append(item["main"]["temp"])
            daily[key]["humidity"].append(item["main"]["humidity"])
            daily[key]["wind_kmh"].append(item["wind"]["speed"] * 3.6)
            daily[key]["conditions"].append(item["weather"][0]["description"])
            daily[key]["rain_mm"] += item.get("rain", {}).get("3h", 0.0)

        forecast = []
        for key in sorted(daily)[:days]:
            d    = daily[key]
            tmax = round(max(d["temps"]), 1)
            tmin = round(min(d["temps"]), 1)
            tavg = round(sum(d["temps"])    / len(d["temps"]), 1)
            havg = round(sum(d["humidity"]) / len(d["humidity"]))
            wavg = round(sum(d["wind_kmh"]) / len(d["wind_kmh"]), 1)
            rain = round(d["rain_mm"], 1)
            cond = max(set(d["conditions"]), key=d["conditions"].count).title()

            notes = []
            if rain > 5:   notes.append("Rain expected — avoid spraying and heavy field operations")
            if tmax > 35:  notes.append("Hot day — irrigate in the evening")
            if wavg > 15:  notes.append("Windy — avoid pesticide or fertiliser spraying")
            if havg > 80:  notes.append("High humidity — monitor for fungal diseases")
            if not notes:  notes.append("Good conditions for normal farm work")

            forecast.append({
                "date":            key,
                "day":             d["day_name"],
                "temperature":     {"max_c": tmax, "min_c": tmin, "avg_c": tavg},
                "humidity_pct":    havg,
                "wind_speed_kmh":  wavg,
                "rainfall_mm":     rain,
                "condition":       cond,
                "farming_notes":   notes,
            })

        return {
            "location":      data["city"]["name"],
            "coordinates":   {"lat": lat, "lon": lon},
            "forecast_days": len(forecast),
            "forecast":      forecast,
            "source":        "openweathermap-5day",
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        return {"location": location,
                "error": "Invalid API key." if code == 401 else f"HTTP {code}",
                "current": get_weather_forecast(location)}
    except Exception as e:
        logger.error("Detailed forecast failed: %s", e)
        return {"location": location, "error": str(e),
                "current": get_weather_forecast(location)}
