"""
Weather Agent — plain Python functions, compatible with google-adk FunctionTool.
Uses OpenWeatherMap if OPENWEATHER_API_KEY is set; falls back to seasonal estimates.
"""
import logging
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()
logger  = logging.getLogger("agents.weather")
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"

_STATE_COORDS = {
    "punjab": (31.15, 75.34), "haryana": (29.06, 76.09),
    "uttar pradesh": (26.85, 80.91), "madhya pradesh": (23.47, 77.95),
    "maharashtra": (19.75, 75.71), "rajasthan": (27.02, 74.22),
    "gujarat": (22.69, 71.58), "bihar": (25.09, 85.31),
    "west bengal": (23.68, 85.05), "karnataka": (15.32, 75.71),
    "andhra pradesh": (15.91, 79.74), "telangana": (17.85, 79.11),
    "tamil nadu": (11.13, 78.66), "odisha": (20.25, 84.77),
    "jharkhand": (23.61, 85.27), "delhi": (28.61, 77.21),
}


def _get_coords(location: str):
    loc_lower = location.lower().strip()
    for state, coords in _STATE_COORDS.items():
        if state in loc_lower:
            return coords
    if OWM_KEY:
        try:
            r = requests.get(GEO_URL, params={"q": f"{location},IN", "limit": 1, "appid": OWM_KEY}, timeout=5)
            if r.ok and r.json():
                d = r.json()[0]
                return d["lat"], d["lon"]
        except Exception as e:
            logger.warning("Geocoding failed: %s", e)
    return 22.5, 78.9


def _seasonal_fallback(location: str) -> dict:
    month = datetime.now().month
    if month in (3, 4, 5):   season, temp, humidity = "Summer",      36, 40
    elif month in (6, 7, 8, 9): season, temp, humidity = "Monsoon",   29, 85
    elif month in (10, 11):   season, temp, humidity = "Post-Monsoon", 27, 65
    else:                     season, temp, humidity = "Winter",       18, 55
    return {
        "location": location, "source": "seasonal-estimate", "season": season,
        "temperature_c": temp, "humidity_pct": humidity, "wind_speed_kmh": 12,
        "condition": season, "rain_expected": month in (6, 7, 8, 9),
        "note": "Live weather unavailable. Add OPENWEATHER_API_KEY to .env for real data.",
    }


def get_weather_forecast(location: str) -> dict:
    """Get current weather for a location in India.

    Args:
        location: City or state name in India (e.g. Ludhiana, Punjab, Pune)

    Returns:
        Current temperature, humidity, wind speed, and rain status.
    """
    if not OWM_KEY:
        logger.info("OPENWEATHER_API_KEY not set — using seasonal fallback")
        return _seasonal_fallback(location)
    lat, lon = _get_coords(location)
    logger.info("Weather request: %s at (%s, %s)", location, lat, lon)
    try:
        r = requests.get(OWM_URL, params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"}, timeout=8)
        logger.info("OpenWeather status: %d", r.status_code)
        r.raise_for_status()
        d = r.json()
        logger.info("Weather data: temp=%s, humidity=%s", d["main"]["temp"], d["main"]["humidity"])
        return {
            "location": location, "source": "openweathermap-live",
            "temperature_c": round(d["main"]["temp"], 1),
            "feels_like_c":  round(d["main"]["feels_like"], 1),
            "humidity_pct":  d["main"]["humidity"],
            "wind_speed_kmh": round(d["wind"]["speed"] * 3.6, 1),
            "condition": d["weather"][0]["description"].title(),
            "rain_expected": d.get("rain") is not None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except requests.exceptions.HTTPError as e:
        logger.error("OpenWeather HTTP error: %s", e)
        if e.response.status_code == 401:
            return {**_seasonal_fallback(location), "error": "Invalid OpenWeather API key."}
        return {**_seasonal_fallback(location), "error": str(e)}
    except Exception as e:
        logger.error("OpenWeather request failed: %s", e, exc_info=True)
        return _seasonal_fallback(location)
        logger.warning("OWM request failed: %s", e)
        return _seasonal_fallback(location)


def check_spray_conditions(location: str) -> dict:
    """Check whether weather conditions are suitable for pesticide or fertiliser spraying.

    Args:
        location: City or state in India where spraying is planned

    Returns:
        Safe or unsafe verdict with specific reasons based on wind, rain, temperature, and humidity.
    """
    wx    = get_weather_forecast(location)
    wind  = wx.get("wind_speed_kmh", 12)
    humid = wx.get("humidity_pct", 65)
    temp  = wx.get("temperature_c", 28)
    rain  = wx.get("rain_expected", False)
    issues = []
    if wind > 15:   issues.append(f"Wind {wind} km/h too high (safe limit under 15 km/h) — spray will drift")
    if rain:        issues.append("Rain expected — spray will wash off before absorption")
    if temp > 35:   issues.append(f"Temperature {temp}°C too high — causes rapid evaporation and leaf burn")
    if temp < 10:   issues.append(f"Temperature {temp}°C too low — poor absorption")
    if humid < 40:  issues.append(f"Humidity {humid}% too low — spray evaporates quickly")
    safe = len(issues) == 0
    return {
        "location": location, "safe_to_spray": safe,
        "verdict":  "Safe to spray" if safe else "Do NOT spray now",
        "reasons":  issues if issues else ["All conditions suitable for spraying"],
        "best_time": "Spray early morning 6 to 9 AM or evening 5 to 7 PM for best results",
        "weather":  wx,
    }


def get_farming_weather_advice(location: str, crop: str = "general") -> dict:
    """Get weather-based farming advice for a specific crop and location.

    Args:
        location: City or state in India
        crop: Crop being grown (e.g. wheat, rice, cotton, tomato)

    Returns:
        Weather summary and specific farming actions to take today.
    """
    wx    = get_weather_forecast(location)
    temp  = wx.get("temperature_c", 28)
    rain  = wx.get("rain_expected", False)
    humid = wx.get("humidity_pct", 65)
    actions = []
    crop_l  = crop.lower()

    if rain:
        actions.append("Hold off on irrigation — rain expected today")
        actions.append("Avoid spraying fertilisers or pesticides — rain will wash them off")
    if temp > 38:
        actions.append("Extreme heat: increase irrigation frequency; consider shade nets for seedlings")
    if temp < 12:
        actions.append("Cold weather: delay transplanting; cover seedlings overnight")
    if ("rice" in crop_l or "paddy" in crop_l) and humid > 80:
        actions.append("High humidity: watch for blast and brown spot disease in rice")
    if "tomato" in crop_l and humid > 75:
        actions.append("High humidity: inspect tomatoes for early or late blight; ensure air circulation")
    if "wheat" in crop_l and temp > 32:
        actions.append("Heat stress on wheat: irrigate in evening; monitor for rust")
    if "cotton" in crop_l and rain:
        actions.append("Post-rain: check for bollworm egg hatching; monitor in 3 to 5 days")
    if not actions:
        actions.append("Conditions are normal. Continue regular crop management.")

    return {
        "location": location, "crop": crop, "weather": wx,
        "actions": actions,
        "spray_check": check_spray_conditions(location),
    }
