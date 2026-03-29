"""
scripts/demo_cli.py — FasalSetu
================================
Interactive CLI for hackathon demos and local testing.
Runs without a web server — directly invokes the orchestrator.

Usage:
    python scripts/demo_cli.py                    # interactive mode
    python scripts/demo_cli.py --scenario all     # run all demo scenarios
    python scripts/demo_cli.py --scenario soil    # run one scenario
    python scripts/demo_cli.py --quick            # non-interactive quick test

Scenarios: soil | disease | market | scheme | weather | compliance | full
"""

import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Colour helpers ─────────────────────────────────────────────────────────
def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


def print_header(title: str):
    width = 62
    print()
    print(cyan("═" * width))
    print(cyan("║") + bold(f"  {title}".ljust(width - 2)) + cyan("║"))
    print(cyan("═" * width))


def print_result(result: dict, indent: int = 2):
    pad = " " * indent
    if "answer" in result:
        answer = result["answer"]
        wrapped = textwrap.fill(answer, width=70, initial_indent=pad, subsequent_indent=pad)
        print(green(wrapped))
    else:
        # Pretty-print dict without answer key
        for k, v in result.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                print(f"{pad}{bold(k)}:")
                for kk, vv in v.items():
                    print(f"{pad}  {kk}: {vv}")
            elif isinstance(v, list):
                print(f"{pad}{bold(k)}: {', '.join(str(i) for i in v[:3])}")
            else:
                print(f"{pad}{bold(k)}: {v}")

    # Show compliance info
    if result.get("_compliance_warnings"):
        print()
        for w in result["_compliance_warnings"]:
            print(yellow(f"  ⚠  {w}"))

    if result.get("blocked"):
        print()
        print(red("  ✗  BLOCKED BY COMPLIANCE GUARDRAIL"))
        for v in result.get("violations", []):
            print(red(f"      {v}"))

    if result.get("_audit_id") is not None:
        print(dim(f"\n  [audit #{result['_audit_id']}]"))


# ── Demo scenarios ─────────────────────────────────────────────────────────

def demo_soil(quick: bool = False):
    """Soil NPK analysis scenario."""
    print_header("SCENARIO 1 — Soil NPK Analysis")

    # Test without LLM: invoke tool directly
    try:
        from agents.soil_agent import predict_npk
        soil_input = {
            "soil_conductivity": 0.55,
            "soil_humidity":     70.0,
            "soil_pH":           6.4,
            "soil_temperature":  25.0,
            "hour":              10,
            "day_of_year":       115,
        }
        print(f"\n{dim('Input sensor readings:')}")
        for k, v in soil_input.items():
            print(dim(f"  {k}: {v}"))

        print(f"\n{bold('Running soil analysis...')}")
        result = predict_npk.invoke(soil_input)

        print(f"\n{bold('NPK Results:')}")
        for nutrient, data in result.get("npk", {}).items():
            status_color = green if data["status"] == "adequate" else (
                red if data["status"] == "deficient" else yellow
            )
            print(f"  {nutrient.upper():12s} {data['value']} {data['unit']}  "
                  f"[{status_color(data['status'].upper())}]")

        print(f"\n{bold('Fertiliser Plan:')}")
        plan = result.get("fertiliser_plan", {})
        print(f"  {plan.get('summary', '')}")
        for action in plan.get("urgent_actions", []):
            print(red(f"  ! {action}"))
        for opt in plan.get("organic_options", []):
            print(green(f"  ✓ {opt}"))

        print(f"\n{bold('Soil Quality Index:')}")
        sqi = result.get("soil_quality_index", {})
        print(f"  Category: {sqi.get('category')}  "
              f"(confidence: {sqi.get('confidence', 0):.0%})")

    except RuntimeError:
        print(yellow("  Models not trained yet. Run: python scripts/train_npk_model.py --data <csv>"))
        print(green("  (Showing mock output for demo)"))
        print(green("  N: 24.97 mg/kg [ADEQUATE]"))
        print(green("  P: 29.96 mg/kg [ADEQUATE]"))
        print(red("  K: 199.91 mg/kg [DEFICIENT]"))
        print(f"\n  Apply Muriate of Potash (MOP) at 40–60 kg/hectare")


def demo_compliance(quick: bool = False):
    """Compliance guardrail demo — the hackathon showstopper."""
    print_header("SCENARIO 2 — Compliance Guardrail (SHOWSTOPPER)")

    from compliance.guardrail import check_and_gate, check_pesticide_safety, ComplianceViolation

    tests = [
        ("Banned substance", {"answer": "Apply Endosulfan at 2 mL/L for pest control."}, True),
        ("Restricted substance", {"answer": "Use Glyphosate carefully on this crop."}, False),
        ("Safe advice", {"answer": "Apply neem oil at 3 mL/L for aphid control."}, False),
        ("Excessive dosage", {"answer": "Spray mancozeb at 50 mL per litre of water."}, False),
    ]

    for label, response, expect_block in tests:
        print(f"\n  {bold(label)}")
        print(dim(f"  Response: \"{response['answer'][:60]}...\""))
        try:
            result = check_and_gate("demo", response.copy())
            if result.get("_compliance_warnings"):
                print(yellow(f"  ⚠  WARNING: {result['_compliance_warnings'][0][:80]}"))
            else:
                print(green("  ✓  PASSED — response is compliant"))
        except ComplianceViolation as e:
            print(red(f"  ✗  BLOCKED: {str(e)[:100]}"))

    print(f"\n{bold('Direct pesticide safety checks:')}")
    for pesticide in ["Endosulfan", "Glyphosate", "Mancozeb", "Monocrotophos"]:
        result = check_pesticide_safety.invoke({"pesticide_name": pesticide})
        status = result["status"]
        color  = red if status == "BANNED" else (yellow if status == "RESTRICTED" else green)
        print(f"  {pesticide:20s}  {color(status)}")

    from compliance.guardrail import get_audit_summary
    summary = get_audit_summary()
    print(f"\n{bold('Audit summary this session:')}")
    print(f"  Total calls: {summary['session_total_calls']}")
    print(f"  Blocked:     {red(str(summary['blocked']))}")
    print(f"  Warned:      {yellow(str(summary['warned']))}")
    print(f"  Clean:       {green(str(summary['clean']))}")


def demo_market(quick: bool = False):
    """Market price and MSP scenario."""
    print_header("SCENARIO 3 — Market Prices & MSP Advisory")

    from agents.market_agent import get_market_prices, get_msp_for_crop

    crops = ["wheat", "tur", "tomato", "cotton"]
    for crop in crops:
        result = get_market_prices.invoke({"crop": crop})
        if "error" in result:
            continue

        msp    = result.get("msp_2024_25")
        price  = result.get("market_price")
        trend  = result.get("price_trend", "unknown")
        below  = msp and price and price < msp

        trend_icon = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
        msp_str    = f"MSP ₹{msp}" if msp else "No MSP"
        alert      = red("  ← BELOW MSP!") if below else ""

        print(f"\n  {bold(result['crop'])}")
        print(f"    Market: ₹{price}/qtl  {trend_icon} {trend}  |  {msp_str}{alert}")

        advice = result.get("sell_advice", {})
        action = advice.get("action", "")
        urgency_color = red if advice.get("urgency") == "high" else (
            yellow if advice.get("urgency") == "medium" else green
        )
        print(f"    Advice: {urgency_color(action)}")


def demo_weather(quick: bool = False):
    """Weather forecast and spray conditions."""
    print_header("SCENARIO 4 — Weather & Spray Conditions")

    from agents.weather_agent import get_farming_weather_advice, check_spray_conditions

    result = get_farming_weather_advice.invoke({"location": "UP", "crop": "wheat"})

    print(f"\n  {bold('Location:')} {result['location']}  |  Crop: {result['crop']}")
    print(f"\n  {bold('5-Day Forecast:')}")
    for d in result.get("forecast_summary", {}).get("days", [])[:5]:
        rain_str = f"  Rain: {d['rain_mm']}mm" if d["rain_mm"] > 0 else ""
        print(f"    {d['date']}  {d['condition']:20s}  {d['temp_max']}°C{rain_str}")

    print(f"\n  {bold('Alerts:')}")
    for alert in result.get("alerts", []):
        print(f"    {yellow('⚠')}  {alert}")

    print(f"\n  {bold('Recommended Actions:')}")
    for action in result.get("recommended_actions", []):
        print(f"    • {action}")

    spray_days = result.get("good_spray_days", [])
    if spray_days:
        print(f"\n  {bold('Good days to spray:')} {green(', '.join(spray_days))}")

    print(f"\n  {bold('Today spray check:')}")
    spray = check_spray_conditions.invoke({"location": "UP"})
    verdict_color = green if spray["safe_to_spray"] else red
    print(f"    {verdict_color(spray['verdict'])}")
    for issue in spray.get("issues", []):
        print(f"    {red('✗')} {issue}")


def demo_schemes(quick: bool = False):
    """Government scheme search scenario."""
    print_header("SCENARIO 5 — Government Schemes")

    from agents.scheme_agent import find_govt_schemes, _seed_chroma_if_empty
    _seed_chroma_if_empty()

    queries = [
        ("crop insurance", "MH"),
        ("low interest loan for seeds", "UP"),
        ("drip irrigation subsidy", "RJ"),
    ]

    for query, state in queries:
        print(f"\n  {bold('Query:')} \"{query}\" ({state})")
        result = find_govt_schemes.invoke({"query": query, "state": state, "max_results": 2})
        for scheme in result.get("schemes", []):
            print(f"    {green('►')} {scheme['scheme_name']}")
            print(f"      {dim(scheme['benefit'][:80])}")
            print(f"      Apply: {scheme['apply_at'][:60]}")


def demo_full_query(quick: bool = False):
    """Full orchestrator query — requires GEMINI_API_KEY."""
    print_header("SCENARIO 6 — Full Agent Query (requires API key)")

    import os
    if not os.getenv("GEMINI_API_KEY"):
        print(yellow("  GEMINI_API_KEY not set — skipping LLM scenario"))
        print(dim("  Set it in .env to run full agent queries"))
        return

    from agents.orchestrator import run_query

    queries = [
        (
            "My wheat leaves are turning yellow and I see small spots. What disease is this and how do I treat it?",
            {"location": "UP", "crop": "wheat", "growth_stage": "vegetative"},
        ),
        (
            "Should I sell my paddy now or wait? What is the MSP?",
            {"location": "PB", "crop": "paddy"},
        ),
        (
            "I want to apply Endosulfan on my cotton crop for pest control.",
            {"crop": "cotton"},
        ),
    ]

    for query, context in queries:
        print(f"\n  {bold('Farmer:')}")
        print(f"  \"{query}\"")
        print(f"  {dim('Context:')} {context}")
        print(f"\n  {bold('FasalSetu:')}")

        result = run_query(query, context)
        print_result(result, indent=2)
        print()


# ── Interactive mode ───────────────────────────────────────────────────────

def interactive_mode():
    import os
    if not os.getenv("GEMINI_API_KEY"):
        print(red("GEMINI_API_KEY not set in .env — interactive mode unavailable"))
        print("Run with --scenario to test individual agents without API key")
        return

    from agents.orchestrator import run_query

    print_header("FasalSetu Interactive Mode")
    print(dim("  Type your question. Type 'quit' to exit."))
    print(dim("  You can provide context like: [crop=wheat] [state=UP] [stage=vegetative]"))
    print()

    import re
    context_pattern = re.compile(r"\[(\w+)=([^\]]+)\]")

    while True:
        try:
            user_input = input(cyan("Farmer: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # Extract inline context tags
        context = {}
        for match in context_pattern.finditer(user_input):
            context[match.group(1)] = match.group(2)
        clean_input = context_pattern.sub("", user_input).strip()

        print(f"\n{bold('FasalSetu:')}")
        result = run_query(clean_input, context or None)
        print_result(result, indent=2)
        print()


# ── Main ───────────────────────────────────────────────────────────────────

SCENARIO_MAP = {
    "soil":       demo_soil,
    "compliance": demo_compliance,
    "market":     demo_market,
    "weather":    demo_weather,
    "scheme":     demo_schemes,
    "schemes":    demo_schemes,
    "full":       demo_full_query,
}

def main():
    parser = argparse.ArgumentParser(description="FasalSetu demo CLI")
    parser.add_argument("--scenario", choices=list(SCENARIO_MAP.keys()) + ["all"],
                        help="Demo scenario to run")
    parser.add_argument("--quick", action="store_true",
                        help="Skip pauses and prompts")
    args = parser.parse_args()

    print(bold(cyan("\n  FasalSetu — Agricultural AI Agent")))
    print(dim("  Soil · Disease · Market · Schemes · Weather · Compliance\n"))

    if args.scenario == "all":
        for name, fn in SCENARIO_MAP.items():
            if name == "schemes":
                continue  # skip alias
            fn(quick=True)
    elif args.scenario:
        SCENARIO_MAP[args.scenario](quick=args.quick)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
