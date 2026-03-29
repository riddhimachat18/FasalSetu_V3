"""
agents/offline_agent.py — FasalSetu
======================================
Offline/low-connectivity mode for FasalSetu.
When the farmer has no internet or API access, this agent serves
cached advisory content and rule-based responses without any LLM calls.

Design principles:
  - Zero external calls — works on 2G or offline entirely
  - All decisions are rule-based lookups against local JSON files
  - Responses are short (SMS-length) for slow connections
  - Degrades gracefully: offline → limited LLM → full LLM

Activation:
  Set OFFLINE_MODE=1 in .env, or pass offline=True to run_query().
  The orchestrator checks this and bypasses the LLM if set.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from langchain.tools import tool

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Offline advisory database ──────────────────────────────────────────────
# Covers the most common farmer queries without any network call.

_OFFLINE_CROP_CALENDAR = {
    # crop: {month_range: advice}
    "wheat": {
        (10, 11): "Sowing time for wheat. Use certified seeds. Sow after first rain or irrigation.",
        (12, 1):  "Wheat vegetative stage. Apply urea top-dress at 30 days.",
        (2, 3):   "Wheat heading/grain fill. Irrigate at flowering. Watch for rust.",
        (4,):     "Wheat harvest time. Harvest when 80% grains turn golden.",
    },
    "paddy": {
        (6, 7):   "Kharif paddy season. Prepare nursery. Transplant after 21-25 days.",
        (8, 9):   "Paddy tillering/panicle stage. Apply potash at panicle initiation.",
        (10, 11): "Paddy harvest. Harvest when 80% grains turn straw-coloured.",
    },
    "tomato": {
        (10, 11): "Tomato sowing season (Rabi). Use F1 hybrid seeds for better yield.",
        (12, 1):  "Tomato vegetative. Stake plants. Watch for early blight.",
        (2, 3):   "Tomato fruiting. Avoid excessive nitrogen. Irrigate regularly.",
    },
    "cotton": {
        (5, 6):   "Cotton sowing season. Use BT cotton approved varieties only.",
        (7, 8):   "Cotton vegetative. Apply first irrigation after 40-45 days.",
        (9, 10):  "Cotton boll stage. Monitor for bollworm. Apply recommended pesticides.",
        (11, 12): "Cotton picking season. Pick in dry weather for quality.",
    },
    "chana": {
        (10, 11): "Chickpea (chana) sowing. Use rhizobium seed treatment for nitrogen fixation.",
        (12, 1):  "Chickpea vegetative. Avoid excess irrigation — leads to lodging.",
        (2, 3):   "Chickpea flowering/podding. Watch for pod borer.",
        (3, 4):   "Chickpea harvest. Harvest when leaves turn yellow and pods brown.",
    },
}

_OFFLINE_SOIL_GUIDE = {
    "low_ph": {
        "condition": "pH below 5.5 (acidic soil)",
        "symptoms":  "Poor germination, yellowing leaves, slow growth",
        "fix":       "Apply agricultural lime at 2-4 tonnes/hectare. Mix well before sowing.",
        "cost":      "₹3,000-6,000/hectare. Available at Sahakari Bhandar or agri-input shop.",
    },
    "high_ph": {
        "condition": "pH above 8.0 (alkaline/saline soil)",
        "symptoms":  "Stunted growth, white crust on soil surface",
        "fix":       "Apply gypsum (calcium sulphate) at 2-5 tonnes/hectare. Improve drainage.",
        "cost":      "₹2,500-5,000/hectare",
    },
    "n_deficiency": {
        "condition": "Nitrogen deficiency",
        "symptoms":  "Yellowing of older leaves starting from tip, pale green color overall",
        "fix":       "Apply urea (46% N) at 50-60 kg/hectare OR DAP at 40-50 kg/hectare",
        "organic":   "Apply farmyard manure 10-15 tonnes/hectare before sowing",
        "cost":      "Urea ~₹267/50kg bag (subsidised). DAP ~₹1,350/50kg bag.",
    },
    "p_deficiency": {
        "condition": "Phosphorus deficiency",
        "symptoms":  "Purple/reddish colour on older leaves, poor root development",
        "fix":       "Apply Single Superphosphate (SSP) at 100-125 kg/hectare as basal dose",
        "organic":   "Apply bone meal at 200 kg/hectare",
        "cost":      "SSP ~₹400/50kg bag",
    },
    "k_deficiency": {
        "condition": "Potassium deficiency",
        "symptoms":  "Scorching of leaf margins, weak stems, poor grain quality",
        "fix":       "Apply Muriate of Potash (MOP) at 50-60 kg/hectare",
        "organic":   "Apply wood ash at 1-2 tonnes/hectare",
        "cost":      "MOP ~₹1,200/50kg bag",
    },
}

_OFFLINE_DISEASE_QUICK = {
    "yellow leaves":     "Likely nitrogen deficiency or viral disease. Apply urea if soil is pale. Consult KVK if spots present.",
    "brown spots":       "Likely fungal blight. Apply Mancozeb 75% WP at 2g/litre. Remove infected leaves.",
    "wilting":           "Check root zone for waterlogging or root rot. Improve drainage. Check for white grub in soil.",
    "white powder":      "Powdery mildew. Spray sulphur 80% WP at 3g/litre or wettable sulphur.",
    "holes in leaves":   "Insect damage. Identify pest first. Spray neem oil 3mL/litre for general control.",
    "stunted growth":    "Multiple causes: soil pH, nutrient deficiency, root damage. Get soil tested at KVK.",
    "curling leaves":    "Viral disease spread by whitefly/aphid. Use yellow sticky traps. Apply imidacloprid if severe.",
    "black spots":       "Fungal infection. Apply copper-based fungicide (Copper Oxychloride 50% WP at 3g/litre).",
    "rotting stem":      "Stem rot/damping off. Apply carbendazim 1g/litre at base. Avoid waterlogging.",
    "orange pustules":   "Rust disease. Apply propiconazole 1mL/litre or mancozeb 2g/litre immediately.",
}

_OFFLINE_EMERGENCY_CONTACTS = {
    "KVK helpline":       "1800-180-1551 (Toll free)",
    "NAFED procurement":  "1800-180-1551",
    "PM-KISAN helpline":  "155261 or 011-24300606",
    "Kisan Call Centre":  "1551 or 1800-180-1551 (24x7, multilingual)",
    "PMFBY claims":       "14447 (crop insurance)",
    "Soil Health Card":   "Contact block agriculture officer",
    "eNAM support":       "1800-270-0224",
}


# ── Rule-based response engine ─────────────────────────────────────────────

def _get_crop_calendar_advice(crop: str) -> Optional[str]:
    """Get seasonal advice for a crop based on current month."""
    month = datetime.now().month
    crop_lower = crop.lower().strip()

    for crop_key, calendar in _OFFLINE_CROP_CALENDAR.items():
        if crop_key in crop_lower or crop_lower in crop_key:
            for months, advice in calendar.items():
                if isinstance(months, tuple) and month in months:
                    return advice
                elif isinstance(months, int) and month == months:
                    return advice
    return None


def _match_disease_symptom(symptom_text: str) -> Optional[str]:
    """Match symptom description to offline disease database."""
    symptom_lower = symptom_text.lower()
    for keyword, advice in _OFFLINE_DISEASE_QUICK.items():
        if keyword in symptom_lower:
            return advice
    return None


# ── LangChain Tools ────────────────────────────────────────────────────────

@tool
def offline_crop_advisory(crop: str, query: str = "") -> dict:
    """
    Get farming advisory for a crop without any internet connection.
    Works entirely from local data — suitable for 2G/offline use.

    Args:
        crop:  Crop name (wheat, paddy, tomato, cotton, chana, mustard, etc.)
        query: Optional symptom or question (yellowing, spots, wilting, etc.)

    Returns:
        Seasonal calendar advice + symptom matching if query provided.
        Response is kept short for slow connections.
    """
    response = {"crop": crop, "offline": True}

    # Seasonal advice
    calendar_advice = _get_crop_calendar_advice(crop)
    if calendar_advice:
        response["seasonal_advice"] = calendar_advice
    else:
        response["seasonal_advice"] = (
            f"No specific calendar advice for '{crop}' this month. "
            f"Contact KVK helpline: 1800-180-1551"
        )

    # Symptom matching
    if query:
        symptom_advice = _match_disease_symptom(query)
        if symptom_advice:
            response["symptom_advice"] = symptom_advice
        else:
            response["symptom_advice"] = (
                "Cannot identify issue without more detail. "
                "Take a clear photo of affected plant and visit nearest KVK."
            )

    response["emergency_helpline"] = "1800-180-1551 (Free, 24x7)"
    return response


@tool
def offline_soil_guide(symptom: str) -> dict:
    """
    Get soil problem diagnosis and fix without any internet connection.
    Works entirely from local data.

    Args:
        symptom: Describe what you see — 'yellow leaves', 'white crust on soil',
                 'poor growth', 'purple leaves', 'leaf margin burn', 'acidic soil'

    Returns:
        Soil problem identification, fix, and cost estimate.
    """
    symptom_lower = symptom.lower()

    matches = []
    for key, data in _OFFLINE_SOIL_GUIDE.items():
        condition_lower = data["condition"].lower()
        symptoms_lower  = data["symptoms"].lower()
        if any(word in symptom_lower for word in condition_lower.split()
               if len(word) > 3):
            matches.append(data)
        elif any(word in symptom_lower for word in symptoms_lower.split()
                 if len(word) > 3):
            matches.append(data)

    if matches:
        best = matches[0]
        return {
            "diagnosis":  best["condition"],
            "symptoms":   best["symptoms"],
            "fix":        best["fix"],
            "organic_alternative": best.get("organic", ""),
            "cost_estimate": best.get("cost", ""),
            "offline":    True,
            "note":       "Get soil tested at KVK for precise diagnosis.",
        }

    return {
        "diagnosis":  "Cannot determine from description",
        "advice":     "Describe: leaf colour, plant age, affected area size",
        "fix":        "Contact KVK: 1800-180-1551 for free soil testing",
        "offline":    True,
    }


@tool
def get_emergency_contacts() -> dict:
    """
    Get all important helpline numbers for farmers.
    Works fully offline — no internet needed.

    Returns:
        Government helplines for KVK, crop insurance, PM-KISAN, market prices.
    """
    month     = datetime.now().month
    season    = ("kharif" if 6 <= month <= 10 else
                 "rabi" if 11 <= month or month <= 3 else "summer")
    season_tip = {
        "kharif": "Kharif season: Register crops for PMFBY insurance before July 31.",
        "rabi":   "Rabi season: Register crops for PMFBY insurance before December 31.",
        "summer": "Summer: Check eNAM for post-harvest storage and price discovery.",
    }

    return {
        "helplines":       _OFFLINE_EMERGENCY_CONTACTS,
        "season":          season,
        "seasonal_tip":    season_tip[season],
        "offline":         True,
        "always_available": "Kisan Call Centre 1551 available 24x7 in all regional languages",
    }


@tool
def offline_fertiliser_calculator(crop: str, area_hectares: float,
                                   deficiency: str = "") -> dict:
    """
    Calculate fertiliser quantities needed for a field — fully offline.

    Args:
        crop:             Crop name (wheat, paddy, maize, cotton, etc.)
        area_hectares:    Field size in hectares
        deficiency:       Which nutrient is deficient (N, P, K, or leave blank)

    Returns:
        Fertiliser type, quantity in kg, bags needed, and estimated cost.
    """
    # Standard NPK recommendations (kg/hectare) from ICAR guidelines
    _NPK_RECOMMENDATIONS = {
        "wheat":    {"N": 120, "P": 60, "K": 40},
        "paddy":    {"N": 100, "P": 50, "K": 50},
        "maize":    {"N": 120, "P": 60, "K": 40},
        "cotton":   {"N": 100, "P": 50, "K": 50},
        "soybean":  {"N":  20, "P": 60, "K": 40},
        "chana":    {"N":  20, "P": 60, "K": 20},
        "tomato":   {"N": 120, "P": 60, "K": 60},
        "mustard":  {"N":  80, "P": 40, "K": 40},
        "sugarcane":{"N": 250, "P": 80, "K": 100},
        "default":  {"N": 100, "P": 50, "K": 40},
    }

    crop_lower = crop.lower().strip()
    rec = _NPK_RECOMMENDATIONS.get(crop_lower,
          _NPK_RECOMMENDATIONS.get(
              next((k for k in _NPK_RECOMMENDATIONS if k in crop_lower), "default"),
              _NPK_RECOMMENDATIONS["default"]
          ))

    area = max(0.01, area_hectares)

    # Filter to deficient nutrient if specified
    nutrients = (["N", "P", "K"] if not deficiency
                 else [d.strip().upper() for d in deficiency.replace(",", " ").split()
                       if d.strip().upper() in ("N", "P", "K")])

    results = {}
    total_cost = 0

    for nutrient in nutrients:
        kg_per_ha = rec.get(nutrient, 0)
        total_kg  = round(kg_per_ha * area, 1)

        if nutrient == "N":
            product     = "Urea (46% N)"
            product_kg  = round(total_kg / 0.46, 1)
            bag_kg      = 50
            price_per_bag = 267   # subsidised MRP
        elif nutrient == "P":
            product     = "Single Superphosphate / SSP (16% P₂O₅)"
            product_kg  = round(total_kg / 0.16, 1)
            bag_kg      = 50
            price_per_bag = 400
        else:  # K
            product     = "Muriate of Potash / MOP (60% K₂O)"
            product_kg  = round(total_kg / 0.60, 1)
            bag_kg      = 50
            price_per_bag = 1200

        bags       = round(product_kg / bag_kg, 1)
        cost       = round((bags) * price_per_bag)
        total_cost += cost

        results[nutrient] = {
            "nutrient":       nutrient,
            "fertiliser":     product,
            "quantity_kg":    product_kg,
            "bags_needed":    bags,
            "bag_size_kg":    bag_kg,
            "estimated_cost": f"₹{cost:,}",
        }

    return {
        "crop":             crop,
        "area_hectares":    area,
        "recommendations":  results,
        "total_cost_estimate": f"₹{total_cost:,}",
        "note":     "Quantities based on ICAR standard recommendations. Reduce by 25% if soil test shows adequate levels.",
        "subsidy":  "Urea is sold at subsidised rate. Carry Aadhaar to purchase.",
        "offline":  True,
    }
