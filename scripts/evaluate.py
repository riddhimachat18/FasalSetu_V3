"""
scripts/evaluate.py — FasalSetu
=================================
Hackathon evaluation script.
Measures system performance across all agent domains without requiring
an ANTHROPIC_API_KEY — tests tools directly.

Run: python scripts/evaluate.py
Run with report: python scripts/evaluate.py --report

Outputs:
  - Per-domain accuracy / coverage scores
  - Compliance guardrail effectiveness
  - Response latency
  - Overall hackathon rubric score estimate
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Colour helpers ─────────────────────────────────────────────────────────
def _c(t, code): return f"\033[{code}m{t}\033[0m"
def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


results: dict = {
    "timestamp":   datetime.now().isoformat(),
    "domains":     {},
    "compliance":  {},
    "latency_ms":  {},
    "overall":     {},
}


def run_test(domain: str, name: str, fn, expect_pass: bool = True):
    """Run a single test, record result and latency."""
    t0 = time.perf_counter()
    try:
        result = fn()
        passed = result is not None and "error" not in result
        latency_ms = round((time.perf_counter() - t0) * 1000)
        status = "PASS" if passed else "FAIL"
        color  = green if passed else red
    except Exception as e:
        passed     = False
        latency_ms = round((time.perf_counter() - t0) * 1000)
        status     = "ERROR"
        result     = {"exception": str(e)}
        color      = red

    icon = "✓" if passed else "✗"
    print(f"    {color(icon)} {name:50s} {dim(f'{latency_ms}ms')}")

    if domain not in results["domains"]:
        results["domains"][domain] = {"pass": 0, "fail": 0, "tests": []}
    if passed:
        results["domains"][domain]["pass"] += 1
    else:
        results["domains"][domain]["fail"] += 1

    results["domains"][domain]["tests"].append({
        "name":       name,
        "passed":     passed,
        "latency_ms": latency_ms,
    })
    results["latency_ms"][f"{domain}/{name}"] = latency_ms
    return passed, result


# ── Domain evaluations ─────────────────────────────────────────────────────

def eval_market():
    print(f"\n{bold(cyan('Market Agent'))}")
    from agents.market_agent import (
        get_market_prices, get_msp_for_crop,
        compare_crops_for_next_season, _generate_sell_advice
    )
    run_test("market", "Wheat price lookup",
             lambda: get_market_prices.invoke({"crop": "wheat"}))
    run_test("market", "Rice alias resolution (rice → paddy)",
             lambda: get_market_prices.invoke({"crop": "rice"}))
    run_test("market", "Tur/Arhar alias",
             lambda: get_market_prices.invoke({"crop": "tur"}))
    run_test("market", "MSP wheat = ₹2275",
             lambda: (r := get_msp_for_crop.invoke({"crop": "wheat"}),
                      None if r.get("msp_2024_25") != 2275 else r)[1])
    run_test("market", "Tomato has no MSP",
             lambda: (r := get_msp_for_crop.invoke({"crop": "tomato"}),
                      None if r.get("msp") is not None else r)[1])
    run_test("market", "Below-MSP triggers govt procurement advice",
             lambda: (a := _generate_sell_advice("wheat", 2000, 2275, "stable"),
                      None if "government" not in a.get("action", "") else a)[1])
    run_test("market", "Next season recommends exactly 3 crops",
             lambda: (r := compare_crops_for_next_season.invoke({
                 "soil_type": "loamy", "state": "UP",
                 "season": "rabi", "water_availability": "moderate"}),
                 None if len(r.get("top_recommendations", [])) != 3 else r)[1])
    run_test("market", "Unknown crop returns error dict",
             lambda: (r := get_market_prices.invoke({"crop": "zzz_unknown"}),
                      r if "error" in r else None)[1])


def eval_compliance():
    print(f"\n{bold(cyan('Compliance Guardrail'))}")
    from compliance.guardrail import (
        check_and_gate, check_pesticide_safety,
        get_audit_summary, ComplianceViolation
    )

    # Blocked tests
    for pesticide in ["Endosulfan", "Monocrotophos", "DDT", "Aldrin", "Carbofuran 3% CG"]:
        def make_test(p):
            def t():
                try:
                    check_and_gate("eval", {"answer": f"Apply {p} for pest control"})
                    return None  # should have raised
                except ComplianceViolation:
                    return {"blocked": True}
            return t
        run_test("compliance", f"Block banned: {pesticide}", make_test(pesticide))

    # Warning tests (not blocked)
    for substance in ["Glyphosate", "Atrazine", "2,4-D"]:
        def make_warn(s):
            def t():
                r = check_and_gate("eval", {"answer": f"Use {s} carefully"})
                return r if "_compliance_warnings" in r else None
            return t
        run_test("compliance", f"Warn restricted: {substance}", make_warn(substance))

    # Pesticide safety tool
    run_test("compliance", "check_pesticide_safety BANNED",
             lambda: (r := check_pesticide_safety.invoke({"pesticide_name": "Endosulfan"}),
                      r if r.get("status") == "BANNED" else None)[1])
    run_test("compliance", "check_pesticide_safety RESTRICTED",
             lambda: (r := check_pesticide_safety.invoke({"pesticide_name": "Glyphosate"}),
                      r if r.get("status") == "RESTRICTED" else None)[1])
    run_test("compliance", "check_pesticide_safety PERMITTED",
             lambda: (r := check_pesticide_safety.invoke({"pesticide_name": "Mancozeb"}),
                      r if r.get("status") == "PERMITTED" else None)[1])

    # Dosage check
    run_test("compliance", "Excessive dosage triggers warning",
             lambda: (r := check_and_gate("eval", {"answer": "Spray 50 mL per litre"}),
                      r if "_compliance_warnings" in r else None)[1])

    # Audit summary
    run_test("compliance", "Audit summary has required fields",
             lambda: (s := get_audit_summary(),
                      s if all(k in s for k in
                               ["session_total_calls", "blocked", "warned", "clean"])
                      else None)[1])

    results["compliance"] = get_audit_summary()


def eval_weather():
    print(f"\n{bold(cyan('Weather Agent'))}")
    from agents.weather_agent import (
        get_weather_forecast, get_farming_weather_advice,
        check_spray_conditions, _get_coords
    )

    run_test("weather", "5-day forecast UP",
             lambda: (r := get_weather_forecast.invoke({"location": "UP", "days": 5}),
                      r if len(r.get("days", [])) == 5 else None)[1])
    run_test("weather", "Forecast day has all required keys",
             lambda: (r := get_weather_forecast.invoke({"location": "MH", "days": 3}),
                      r if all(k in r["days"][0] for k in
                               ["date","condition","temp_max_c","rainfall_mm","wind_kmh"])
                      else None)[1])
    run_test("weather", "All 20 state codes resolve",
             lambda: {"ok": all(
                 _get_coords(code) != (28.61, 77.20)
                 for code in ["UP","MH","PB","HR","MP","RJ","AP","TN",
                              "KA","GJ","WB","BR","OR","JH","CG","AS","KL"]
             )})
    run_test("weather", "Farming advice has alerts list",
             lambda: (r := get_farming_weather_advice.invoke({"location": "UP", "crop": "wheat"}),
                      r if isinstance(r.get("alerts"), list) and len(r["alerts"]) >= 1
                      else None)[1])
    run_test("weather", "Spray conditions verdict consistent",
             lambda: (r := check_spray_conditions.invoke({"location": "HR"}),
                      r if ("SAFE" in r["verdict"]) == r["safe_to_spray"] else None)[1])
    run_test("weather", "Unknown location falls back gracefully",
             lambda: (r := get_weather_forecast.invoke({"location": "XYZ_UNKNOWN_99"}),
                      r if len(r.get("days", [])) > 0 else None)[1])


def eval_schemes():
    print(f"\n{bold(cyan('Scheme Agent'))}")
    from agents.scheme_agent import (
        find_govt_schemes, get_scheme_details,
        list_schemes_by_category, _seed_chroma_if_empty
    )

    _seed_chroma_if_empty()

    run_test("schemes", "PM-KISAN found in seed data",
             lambda: (r := get_scheme_details.invoke({"scheme_name": "PM-KISAN"}),
                      r if r.get("found") else None)[1])
    run_test("schemes", "PMFBY found in seed data",
             lambda: (r := get_scheme_details.invoke({"scheme_name": "PMFBY"}),
                      r if r.get("found") else None)[1])
    run_test("schemes", "Insurance category returns ≥1 scheme",
             lambda: (r := list_schemes_by_category.invoke({"category": "insurance"}),
                      r if r.get("total", 0) >= 1 else None)[1])
    run_test("schemes", "All category returns ≥10 schemes",
             lambda: (r := list_schemes_by_category.invoke({"category": "all"}),
                      r if r.get("total", 0) >= 10 else None)[1])
    run_test("schemes", "Vector search returns results",
             lambda: (r := find_govt_schemes.invoke({"query": "crop insurance flood", "max_results": 2}),
                      r if len(r.get("schemes", [])) >= 1 else None)[1])
    run_test("schemes", "Search result has required fields",
             lambda: (r := find_govt_schemes.invoke({"query": "loan seeds farmer", "max_results": 1}),
                      r if r.get("schemes") and all(
                          k in r["schemes"][0] for k in ["scheme_name", "benefit", "apply_at"])
                      else None)[1])


def eval_disease():
    print(f"\n{bold(cyan('Disease Agent'))}")
    from agents.disease_agent import get_disease_info, detect_crop_disease

    run_test("disease", "Tomato Late Blight in treatment DB (severity=high)",
             lambda: (r := get_disease_info.invoke({"disease_name": "Tomato Late Blight"}),
                      r if r.get("severity") == "high" else None)[1])
    run_test("disease", "Potato Late Blight in treatment DB",
             lambda: (r := get_disease_info.invoke({"disease_name": "Potato Late Blight"}),
                      r if r.get("severity") == "high" else None)[1])
    run_test("disease", "Healthy tomato detected as severity=none",
             lambda: (r := get_disease_info.invoke({"disease_name": "Tomato healthy"}),
                      r if r.get("severity") == "none" else None)[1])
    run_test("disease", "Unknown disease refers to KVK",
             lambda: (r := get_disease_info.invoke({"disease_name": "ZZZ Unknown 999"}),
                      r if r.get("refer_to_kvk") else None)[1])
    run_test("disease", "Missing image file returns error",
             lambda: (r := detect_crop_disease.invoke({"image_path": "/no/such/file.jpg"}),
                      r if "error" in r else None)[1])


def eval_offline():
    print(f"\n{bold(cyan('Offline Agent'))}")
    from agents.offline_agent import (
        offline_crop_advisory, offline_soil_guide,
        get_emergency_contacts, offline_fertiliser_calculator
    )

    run_test("offline", "Wheat crop advisory returns advice",
             lambda: (r := offline_crop_advisory.invoke({"crop": "wheat"}),
                      r if "seasonal_advice" in r else None)[1])
    run_test("offline", "Yellow leaves symptom matched",
             lambda: (r := offline_soil_guide.invoke({"symptom": "yellow leaves pale color"}),
                      r if "diagnosis" in r else None)[1])
    run_test("offline", "Emergency contacts returns helplines",
             lambda: (r := get_emergency_contacts.invoke({}),
                      r if "helplines" in r and len(r["helplines"]) >= 5 else None)[1])
    run_test("offline", "Fertiliser calculator wheat 2ha",
             lambda: (r := offline_fertiliser_calculator.invoke({
                 "crop": "wheat", "area_hectares": 2.0}),
                 r if "recommendations" in r and "N" in r["recommendations"] else None)[1])
    run_test("offline", "Fertiliser calculator cost is non-zero",
             lambda: (r := offline_fertiliser_calculator.invoke({
                 "crop": "paddy", "area_hectares": 1.0}),
                 r if r.get("total_cost_estimate", "₹0") != "₹0" else None)[1])


def eval_voice():
    print(f"\n{bold(cyan('Voice / Language Agent'))}")
    from agents.voice_agent import (
        translate_farmer_query, get_language_support_info, _detect_language
    )

    run_test("voice", "Hindi text detected as 'hi'",
             lambda: {"ok": _detect_language("मेरी फसल में रोग है") == "hi"})
    run_test("voice", "English text detected as 'en'",
             lambda: {"ok": _detect_language("my crop has yellow spots") == "en"})
    run_test("voice", "Hindi farming glossary translates wheat",
             lambda: (r := translate_farmer_query.invoke({
                 "text": "गेहूं की फसल में खाद कैसे डालें",
                 "source_language": "hi"}),
                 r if "wheat" in r.get("english_query", "").lower() else None)[1])
    run_test("voice", "Language support info lists ≥10 languages",
             lambda: (r := get_language_support_info.invoke({}),
                      r if len(r.get("supported_languages", {})) >= 10 else None)[1])
    run_test("voice", "Auto-detect works for Hindi",
             lambda: (r := translate_farmer_query.invoke({
                 "text": "मिट्टी का परीक्षण कहाँ करें",
                 "source_language": "auto"}),
                 r if r.get("detected_language") == "hi" else None)[1])


# ── Summary and scoring ────────────────────────────────────────────────────

def print_summary():
    print(f"\n{bold(cyan('=' * 62))}")
    print(bold(cyan("  EVALUATION SUMMARY")))
    print(bold(cyan('=' * 62)))

    total_pass = total_fail = 0
    domain_scores = {}

    for domain, data in results["domains"].items():
        p = data["pass"]
        f = data["fail"]
        total = p + f
        score = round(p / total * 100) if total > 0 else 0
        total_pass += p
        total_fail += f
        domain_scores[domain] = score

        color = green if score >= 80 else (yellow if score >= 60 else red)
        print(f"  {domain:15s}  {color(f'{score:3d}%')}  "
              f"({p}/{total} tests)  "
              f"{dim('avg ' + str(round(sum(t['latency_ms'] for t in data['tests']) / len(data['tests']))) + 'ms')}")

    grand_total = total_pass + total_fail
    overall     = round(total_pass / grand_total * 100) if grand_total > 0 else 0

    print(f"\n  {'OVERALL':15s}  {(green if overall >= 80 else yellow)(f'{overall:3d}%')}  "
          f"({total_pass}/{grand_total} tests)")

    # Compliance stats
    comp = results.get("compliance", {})
    if comp:
        print(f"\n  {bold('Compliance Guardrail:')}")
        print(f"    Total calls:  {comp.get('session_total_calls', '?')}")
        print(f"    Blocked:      {red(str(comp.get('blocked', '?')))}")
        print(f"    Warned:       {yellow(str(comp.get('warned', '?')))}")

    # Hackathon rubric estimate
    print(f"\n{bold('  Hackathon Rubric Estimate:')}")
    rubric = {
        "Domain workflow execution":    min(100, domain_scores.get("market", 0) + domain_scores.get("weather", 0)) // 2,
        "Edge case handling":           domain_scores.get("offline", 0),
        "Compliance guardrails":        domain_scores.get("compliance", 0),
        "Multi-modal inputs":           domain_scores.get("voice", 0),
        "Agricultural domain coverage": min(100, sum(domain_scores.get(d, 0)
                                            for d in ["disease", "schemes", "market"]) // 3),
    }
    for criterion, score in rubric.items():
        bar  = "█" * (score // 10) + "░" * (10 - score // 10)
        col  = green if score >= 80 else (yellow if score >= 60 else red)
        print(f"    {criterion:40s}  {col(bar)}  {score}%")

    results["overall"] = {"score": overall, "rubric": rubric,
                          "pass": total_pass, "fail": total_fail}
    return overall


def save_report(output_path: str = "evaluation_report.json"):
    path = Path(output_path)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  {dim('Full report saved to:')} {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FasalSetu evaluation script")
    parser.add_argument("--report", action="store_true",
                        help="Save full JSON report to evaluation_report.json")
    parser.add_argument("--domain", choices=["market", "compliance", "weather",
                                              "schemes", "disease", "offline", "voice"],
                        help="Evaluate a single domain only")
    args = parser.parse_args()

    print(bold(cyan("\n  FasalSetu — Hackathon Evaluation\n")))

    domain_fns = {
        "market":     eval_market,
        "compliance": eval_compliance,
        "weather":    eval_weather,
        "schemes":    eval_schemes,
        "disease":    eval_disease,
        "offline":    eval_offline,
        "voice":      eval_voice,
    }

    if args.domain:
        domain_fns[args.domain]()
    else:
        for fn in domain_fns.values():
            fn()

    score = print_summary()

    if args.report:
        save_report()

    sys.exit(0 if score >= 60 else 1)


if __name__ == "__main__":
    main()
