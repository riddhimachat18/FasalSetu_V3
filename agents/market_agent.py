"""
Market Price Agent — plain Python functions, compatible with google-adk FunctionTool.
Calls local agmarknet proxy at http://127.0.0.1:5000; falls back to MSP data if unreachable.
"""
import logging
from datetime import datetime

import requests

logger = logging.getLogger("agents.market")

AGMARKNET_PROXY = "http://127.0.0.1:5000/request"

_MSP = {
    "wheat":          {"msp": 2275,  "unit": "₹/quintal"},
    "paddy_common":   {"msp": 2183,  "unit": "₹/quintal"},
    "paddy_grade_a":  {"msp": 2203,  "unit": "₹/quintal"},
    "jowar_hybrid":   {"msp": 3371,  "unit": "₹/quintal"},
    "bajra":          {"msp": 2500,  "unit": "₹/quintal"},
    "maize":          {"msp": 2090,  "unit": "₹/quintal"},
    "ragi":           {"msp": 3846,  "unit": "₹/quintal"},
    "tur_arhar":      {"msp": 7000,  "unit": "₹/quintal"},
    "moong":          {"msp": 8682,  "unit": "₹/quintal"},
    "urad":           {"msp": 7400,  "unit": "₹/quintal"},
    "groundnut":      {"msp": 6377,  "unit": "₹/quintal"},
    "sunflower":      {"msp": 7280,  "unit": "₹/quintal"},
    "soybean":        {"msp": 4600,  "unit": "₹/quintal"},
    "cotton_medium":  {"msp": 7121,  "unit": "₹/quintal"},
    "cotton_long":    {"msp": 7521,  "unit": "₹/quintal"},
    "sugarcane":      {"msp": 340,   "unit": "₹/quintal"},
}

_ALIASES = {
    "rice": "paddy_common", "paddy": "paddy_common",
    "corn": "maize", "tur": "tur_arhar", "arhar": "tur_arhar",
    "cotton": "cotton_medium", "soya": "soybean",
}


def _normalize_crop(crop: str) -> str:
    key = crop.lower().strip().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


def get_market_prices(crop: str, state: str = "Maharashtra", city: str = "Pune") -> dict:
    """Get current market prices for a crop and compare against MSP.

    Args:
        crop: Crop name in English (e.g. wheat, rice, cotton, soybean, tomato)
        state: Indian state name (e.g. Maharashtra, Punjab, Uttar Pradesh)
        city: City or mandi name within the state (e.g. Pune, Ludhiana, Kanpur)

    Returns:
        Dictionary with market price, MSP comparison, trend, and sell/hold recommendation.
    """
    normalized = _normalize_crop(crop)
    msp_data = _MSP.get(normalized)

    live_price = None
    source = "mock-fallback"
    try:
        logger.info("Calling agmarknet proxy: %s?commodity=%s&state=%s&market=%s", AGMARKNET_PROXY, crop, state, city)
        resp = requests.get(
            AGMARKNET_PROXY,
            params={"commodity": crop, "state": state, "market": city},
            timeout=5,
        )
        logger.info("Agmarknet response status: %d", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("Agmarknet response data: %s", data)
            for key in ("price", "modal_price", "max_price", "Price", "Modal_Price"):
                if key in data and data[key]:
                    try:
                        live_price = float(str(data[key]).replace(",", ""))
                        source = "agmarknet-live"
                        logger.info("Extracted price: %s from key: %s", live_price, key)
                        break
                    except (ValueError, TypeError) as e:
                        logger.warning("Failed to parse price from key %s: %s", key, e)
                        continue
    except requests.exceptions.ConnectionError as e:
        logger.warning("Agmarknet proxy not reachable at %s: %s", AGMARKNET_PROXY, e)
    except Exception as e:
        logger.error("Agmarknet API error: %s", e, exc_info=True)

    if live_price is not None:
        price = live_price
        trend = "unknown"
    elif msp_data:
        price = msp_data["msp"]
        trend = "stable"
        source = "msp-reference"
    else:
        return {
            "error": f"No price data available for '{crop}'.",
            "tip": f"Try one of: {', '.join(list(_MSP.keys())[:8])}",
        }

    msp = msp_data["msp"] if msp_data else None
    below_msp = msp is not None and price < msp

    if below_msp:
        advice = (f"⚠️ Market price (₹{price}/q) is BELOW MSP (₹{msp}/q). "
                  f"Sell through government procurement at nearest NAFED/FCI centre.")
    elif source == "msp-reference":
        advice = (f"Live price unavailable. MSP for {crop} is ₹{msp}/quintal. "
                  f"Check agmarknet.gov.in for current mandi price.")
    elif trend == "rising":
        advice = "Prices rising. Hold 1–2 weeks if storage available."
    elif trend == "falling":
        advice = "Prices falling. Sell soon to avoid further loss."
    else:
        advice = "Prices stable. Sell at your convenience."

    return {
        "crop": crop,
        "market": city,
        "state": state,
        "price": price,
        "unit": "₹/quintal",
        "msp": msp,
        "below_msp": below_msp,
        "trend": trend,
        "source": source,
        "advice": advice,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def list_supported_crops() -> dict:
    """List all crops for which MSP data is available.

    Returns:
        Dictionary with list of supported crop names and their 2024-25 MSP values.
    """
    return {
        "supported_crops": list(_MSP.keys()),
        "aliases": _ALIASES,
        "msp_year": "2024-25",
        "note": "MSP set by Cabinet Committee on Economic Affairs (CCEA), Government of India.",
    }
