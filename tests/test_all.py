"""
tests/test_all.py — FasalSetu
================================
Run with: pytest tests/ -v
Tests run without ANTHROPIC_API_KEY (no LLM calls) by testing tools directly.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Soil agent tests ───────────────────────────────────────────────────────
class TestSoilAgent:
    SAMPLE_SOIL = {
        "soil_conductivity": 0.55,
        "soil_humidity":     70.0,
        "soil_pH":           6.75,
        "soil_temperature":  25.0,
        "hour":              10,
        "day_of_year":       115,
    }

    def test_predict_npk_returns_expected_keys(self):
        """NPK tool returns all required fields when models are loaded."""
        try:
            from agents.soil_agent import predict_npk
            result = predict_npk.invoke(self.SAMPLE_SOIL)
            assert "npk" in result
            assert "nitrogen" in result["npk"]
            assert "phosphorus" in result["npk"]
            assert "potassium" in result["npk"]
            assert "fertiliser_plan" in result
            assert "soil_quality_index" in result
        except RuntimeError as e:
            pytest.skip(f"Models not trained yet: {e}")

    def test_npk_status_values(self):
        """NPK status field is one of expected values."""
        try:
            from agents.soil_agent import predict_npk
            result = predict_npk.invoke(self.SAMPLE_SOIL)
            for nutrient in ["nitrogen", "phosphorus", "potassium"]:
                assert result["npk"][nutrient]["status"] in ("deficient", "adequate", "excess")
        except RuntimeError:
            pytest.skip("Models not trained yet")

    def test_fertiliser_plan_structure(self):
        """Fertiliser plan has expected structure."""
        try:
            from agents.soil_agent import predict_npk
            result = predict_npk.invoke(self.SAMPLE_SOIL)
            plan = result["fertiliser_plan"]
            assert "summary" in plan or "urgent_actions" in plan
        except RuntimeError:
            pytest.skip("Models not trained yet")


# ── Market agent tests ─────────────────────────────────────────────────────
class TestMarketAgent:
    def test_wheat_price(self):
        from agents.market_agent import get_market_prices
        result = get_market_prices.invoke({"crop": "wheat"})
        assert "market_price" in result
        assert "msp_2024_25" in result
        assert result["msp_2024_25"] == 2275

    def test_paddy_alias(self):
        from agents.market_agent import get_market_prices
        result = get_market_prices.invoke({"crop": "rice"})
        assert "market_price" in result
        assert "sell_advice" in result

    def test_unknown_crop(self):
        from agents.market_agent import get_market_prices
        result = get_market_prices.invoke({"crop": "xyz_unknown_crop_123"})
        assert "error" in result

    def test_msp_lookup(self):
        from agents.market_agent import get_msp_for_crop
        result = get_msp_for_crop.invoke({"crop": "wheat"})
        assert result["msp_2024_25"] == 2275
        assert "your_rights" in result

    def test_msp_tomato_has_no_msp(self):
        from agents.market_agent import get_msp_for_crop
        result = get_msp_for_crop.invoke({"crop": "tomato"})
        assert result["msp"] is None

    def test_below_msp_triggers_government_procurement_advice(self):
        from agents.market_agent import _generate_sell_advice
        advice = _generate_sell_advice("wheat", 2000, 2275, "stable")
        assert advice["action"].startswith("SELL via government")
        assert advice["urgency"] == "high"

    def test_next_season_returns_3_crops(self):
        from agents.market_agent import compare_crops_for_next_season
        result = compare_crops_for_next_season.invoke({
            "soil_type": "loamy",
            "state": "UP",
            "season": "rabi",
            "water_availability": "moderate",
        })
        assert len(result["top_recommendations"]) == 3


# ── Scheme agent tests ─────────────────────────────────────────────────────
class TestSchemeAgent:
    def test_pm_kisan_in_seed_data(self):
        from agents.scheme_agent import get_scheme_details
        result = get_scheme_details.invoke({"scheme_name": "PM-KISAN"})
        assert result["found"] is True
        assert "6,000" in result["benefit"]

    def test_category_list_returns_insurance(self):
        from agents.scheme_agent import list_schemes_by_category
        result = list_schemes_by_category.invoke({"category": "insurance"})
        assert result["total"] >= 1
        names = [s["scheme_name"] for s in result["schemes"]]
        assert any("PMFBY" in n or "Fasal Bima" in n for n in names)

    def test_category_all_returns_all_seeds(self):
        from agents.scheme_agent import list_schemes_by_category
        result = list_schemes_by_category.invoke({"category": "all"})
        assert result["total"] >= 10

    def test_chroma_seeded_on_first_query(self):
        from agents.scheme_agent import find_govt_schemes, _seed_chroma_if_empty
        _seed_chroma_if_empty()
        result = find_govt_schemes.invoke({"query": "crop insurance flood", "state": "central"})
        assert "schemes" in result
        assert len(result["schemes"]) >= 1


# ── Compliance guardrail tests ─────────────────────────────────────────────
class TestCompliance:
    def test_banned_pesticide_raises_violation(self):
        from compliance.guardrail import check_and_gate, ComplianceViolation
        response = {"answer": "You should apply Endosulfan to control pests."}
        with pytest.raises(ComplianceViolation) as exc_info:
            check_and_gate("test_agent", response)
        assert "BANNED_SUBSTANCE" in str(exc_info.value)
        assert "Endosulfan" in str(exc_info.value).lower() or "endosulfan" in str(exc_info.value).lower()

    def test_ddT_banned(self):
        from compliance.guardrail import check_and_gate, ComplianceViolation
        with pytest.raises(ComplianceViolation):
            check_and_gate("test_agent", {"answer": "Use DDT for mosquito control."})

    def test_restricted_pesticide_adds_warning_not_block(self):
        from compliance.guardrail import check_and_gate
        response = {"answer": "You can use Glyphosate carefully on this field."}
        result = check_and_gate("test_agent", response)
        assert "_compliance_warnings" in result
        assert any("RESTRICTED" in w for w in result["_compliance_warnings"])

    def test_clean_response_passes_through(self):
        from compliance.guardrail import check_and_gate
        response = {"answer": "Apply neem oil to control aphids on your tomato crop."}
        result = check_and_gate("test_agent", response)
        assert "_compliance_warnings" not in result
        assert result["answer"] == "Apply neem oil to control aphids on your tomato crop."

    def test_audit_log_grows(self):
        from compliance.guardrail import check_and_gate, get_audit_log, ComplianceViolation
        before = len(get_audit_log())
        try:
            check_and_gate("test_agent", {"answer": "safe advice"})
        except Exception:
            pass
        try:
            check_and_gate("test_agent", {"answer": "use endosulfan"})
        except ComplianceViolation:
            pass
        after = len(get_audit_log())
        assert after >= before + 2

    def test_check_pesticide_tool_banned(self):
        from compliance.guardrail import check_pesticide_safety
        result = check_pesticide_safety.invoke({"pesticide_name": "Monocrotophos"})
        assert result["status"] == "BANNED"
        assert result["safe_to_use"] is False

    def test_check_pesticide_tool_permitted(self):
        from compliance.guardrail import check_pesticide_safety
        result = check_pesticide_safety.invoke({"pesticide_name": "Mancozeb"})
        assert result["status"] == "PERMITTED"
        assert result["safe_to_use"] is True

    def test_check_pesticide_tool_restricted(self):
        from compliance.guardrail import check_pesticide_safety
        result = check_pesticide_safety.invoke({"pesticide_name": "Glyphosate"})
        assert result["status"] == "RESTRICTED"

    def test_dosage_warning_triggered(self):
        from compliance.guardrail import check_and_gate
        response = {"answer": "Spray at 50 mL per litre of water for best results."}
        result = check_and_gate("test_agent", response)
        assert "_compliance_warnings" in result
        assert any("DOSAGE" in w for w in result["_compliance_warnings"])

    def test_audit_summary_structure(self):
        from compliance.guardrail import get_audit_summary
        summary = get_audit_summary()
        assert "session_total_calls" in summary
        assert "blocked" in summary
        assert "warned" in summary
        assert "clean" in summary


# ── Disease agent tests (no model required) ───────────────────────────────
class TestDiseaseAgent:
    def test_disease_info_known_disease(self):
        from agents.disease_agent import get_disease_info
        result = get_disease_info.invoke({"disease_name": "Tomato Late Blight"})
        assert result.get("severity") == "high"
        assert "treatment" in result or "recommendations" in result

    def test_disease_info_unknown_returns_kvk(self):
        from agents.disease_agent import get_disease_info
        result = get_disease_info.invoke({"disease_name": "Completely Unknown XYZ Disease 999"})
        assert "error" in result
        assert result.get("refer_to_kvk") is True

    def test_detect_crop_disease_missing_file(self):
        from agents.disease_agent import detect_crop_disease
        result = detect_crop_disease.invoke({"image_path": "/nonexistent/path/image.jpg"})
        assert "error" in result


# ── Weather agent tests ────────────────────────────────────────────────────
class TestWeatherAgent:
    def test_forecast_returns_5_days(self):
        from agents.weather_agent import get_weather_forecast
        result = get_weather_forecast.invoke({"location": "UP", "days": 5})
        assert "days" in result
        assert len(result["days"]) == 5

    def test_forecast_day_structure(self):
        from agents.weather_agent import get_weather_forecast
        result = get_weather_forecast.invoke({"location": "MH", "days": 3})
        day = result["days"][0]
        for key in ["date", "condition", "temp_max_c", "temp_min_c",
                    "humidity_pct", "rainfall_mm", "wind_kmh"]:
            assert key in day, f"Missing key: {key}"

    def test_state_code_resolves(self):
        from agents.weather_agent import _get_coords
        lat, lon = _get_coords("PB")
        assert 28 < lat < 34
        assert 73 < lon < 77

    def test_farming_advice_has_alerts(self):
        from agents.weather_agent import get_farming_weather_advice
        result = get_farming_weather_advice.invoke({"location": "UP", "crop": "wheat"})
        assert "alerts" in result
        assert isinstance(result["alerts"], list)
        assert len(result["alerts"]) >= 1

    def test_spray_conditions_structure(self):
        from agents.weather_agent import check_spray_conditions
        result = check_spray_conditions.invoke({"location": "HR"})
        assert "safe_to_spray" in result
        assert isinstance(result["safe_to_spray"], bool)
        assert "verdict" in result
        assert "conditions" in result

    def test_spray_verdict_matches_safe_flag(self):
        from agents.weather_agent import check_spray_conditions
        result = check_spray_conditions.invoke({"location": "RJ"})
        if result["safe_to_spray"]:
            assert "SAFE" in result["verdict"]
        else:
            assert "DO NOT" in result["verdict"]

    def test_unknown_location_defaults_gracefully(self):
        from agents.weather_agent import get_weather_forecast
        result = get_weather_forecast.invoke({"location": "XYZ_UNKNOWN_STATE"})
        assert "days" in result     # falls back to Delhi coords
        assert len(result["days"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])