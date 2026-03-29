"""
Soil / NPK Agent — plain Python functions, compatible with google-adk FunctionTool.
Models are lazy-loaded on first call so missing .pkl files don't crash startup.
"""
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger("agents.soil")
MODEL_DIR = Path(__file__).parent.parent / "models"

_models: dict = {}


def _load_models() -> bool:
    if _models:
        return True
    try:
        _models["nitrogen"]   = joblib.load(MODEL_DIR / "FINAL_nitrogen_regressor.pkl")
        _models["phosphorus"] = joblib.load(MODEL_DIR / "FINAL_phosphorus_regressor.pkl")
        _models["potassium"]  = joblib.load(MODEL_DIR / "FINAL_potassium_regressor.pkl")
        _models["sqi"]        = joblib.load(MODEL_DIR / "FINAL_sqi_classifier.pkl")
        logger.info("NPK models loaded from %s", MODEL_DIR)
        return True
    except FileNotFoundError as e:
        logger.warning("NPK model not found: %s — run train_npk_model.py first", e)
        return False


def _status(value: float, low: float, high: float) -> str:
    if value < low:  return "deficient"
    if value > high: return "excess"
    return "adequate"


def _npk_recommendation(n: float, p: float, k: float) -> str:
    tips = []
    if n < 24.93: tips.append("Apply urea (46-0-0) or compost to boost nitrogen")
    if p < 29.93: tips.append("Add single superphosphate (SSP) for phosphorus")
    if k < 199.93: tips.append("Apply muriate of potash (MOP) for potassium")
    return "; ".join(tips) if tips else "NPK levels are within normal range for this sensor"


def predict_npk(
    soil_conductivity: float,
    soil_humidity: float,
    soil_pH: float,
    soil_temperature: float,
    hour: int = 12,
    day_of_year: int = 180,
) -> dict:
    """Predict soil NPK levels from sensor readings.

    Args:
        soil_conductivity: Soil electrical conductivity (typical range 0.35–0.75)
        soil_humidity: Soil moisture percentage (typical range 69.8–70.2)
        soil_pH: Soil pH (typical range 6.55–6.95)
        soil_temperature: Soil temperature in Celsius (typical range 24.8–25.2)
        hour: Hour of day (0–23), default 12
        day_of_year: Day of year (1–365), default 180

    Returns:
        Dictionary with nitrogen, phosphorus, potassium predictions and status.
    """
    if not _load_models():
        return {
            "error": "NPK models not loaded. Run: python scripts/train_npk_model.py",
            "tip": "Ensure soil_data_final.csv is available and training completed.",
        }
    x = np.array([[soil_conductivity, soil_humidity, soil_pH, soil_temperature, hour, day_of_year]])
    n = float(_models["nitrogen"].predict(x)[0])
    p = float(_models["phosphorus"].predict(x)[0])
    k = float(_models["potassium"].predict(x)[0])
    return {
        "nitrogen":   {"value": round(n, 3), "unit": "mg/kg", "status": _status(n, 24.93, 25.05)},
        "phosphorus": {"value": round(p, 3), "unit": "mg/kg", "status": _status(p, 29.93, 30.05)},
        "potassium":  {"value": round(k, 3), "unit": "mg/kg", "status": _status(k, 199.93, 200.05)},
        "confidence": "MAE ±0.06 units (~16% of sensor range)",
        "recommendation": _npk_recommendation(n, p, k),
    }


def get_soil_health_report(
    soil_conductivity: float,
    soil_humidity: float,
    soil_pH: float,
    soil_temperature: float,
    crop: str = "general",
    hour: int = 12,
    day_of_year: int = 180,
) -> dict:
    """Get a full soil health report including NPK predictions, pH advice, and crop-specific tips.

    Args:
        soil_conductivity: Soil electrical conductivity
        soil_humidity: Soil moisture percentage
        soil_pH: Soil pH value
        soil_temperature: Soil temperature in Celsius
        crop: Crop being grown (e.g. wheat, rice, tomato)
        hour: Hour of day (0–23)
        day_of_year: Day of year (1–365)

    Returns:
        Full soil health report with NPK status, pH advice, and crop-specific recommendations.
    """
    npk = predict_npk(soil_conductivity, soil_humidity, soil_pH, soil_temperature, hour, day_of_year)
    if "error" in npk:
        return npk

    if soil_pH < 6.0:
        ph_advice = "Soil is acidic. Apply agricultural lime to raise pH."
    elif soil_pH > 7.5:
        ph_advice = "Soil is alkaline. Apply gypsum or sulfur to lower pH."
    else:
        ph_advice = "Soil pH is within optimal range (6.0–7.5)."

    salinity_note = (
        "High conductivity detected — possible salt stress. Flush soil with clean water."
        if soil_conductivity > 0.7 else "Conductivity within normal range."
    )

    crop_tips = {
        "wheat":   "Wheat prefers pH 6.0–7.0. Ensure adequate phosphorus at sowing.",
        "rice":    "Rice needs high moisture. Keep soil humidity above 70%.",
        "tomato":  "Tomatoes are sensitive to calcium deficiency — check conductivity.",
        "cotton":  "Cotton tolerates mild alkalinity but needs potassium for boll formation.",
        "maize":   "Maize is nitrogen-hungry — top-dress urea 3–4 weeks after sowing.",
        "general": "Consult your local Krishi Vigyan Kendra for crop-specific advice.",
    }

    return {
        "npk": npk,
        "ph_advice": ph_advice,
        "salinity_note": salinity_note,
        "crop_tip": crop_tips.get(crop.lower(), crop_tips["general"]),
        "kvk_referral": soil_pH < 5.5 or soil_pH > 8.0 or soil_conductivity > 0.7,
    }
