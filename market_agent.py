"""
market_agent.py
──────────────────────────────────────────────────────────────────────────────
Conversational AI agent that fetches real market price data and gives
actionable, personalised guidance to farmers.

The agent uses Gemini (via Google Generative AI) with function calling to:
  1. Fetch live prices from Agmarknet (via market_api_fetcher.py)
  2. Compare current price vs MSP
  3. Identify best mandis to sell at
  4. Analyse 7-day price trends (rising/falling/stable)
  5. Give plain-language "should I sell now?" guidance
  6. Estimate revenue for their harvest

Requirements:
    pip install google-generativeai requests pandas python-dotenv rich

Usage:
    python market_agent.py                      # interactive chat loop
    python market_agent.py --crop Wheat --state Punjab --qty 50
    python market_agent.py --demo              # run with mock data (no API key needed)
"""

import os
import json
import argparse
import textwrap
from datetime import datetime, timedelta
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    console = Console()
    RICH = True
except ImportError:
    RICH = False

# Import our fetcher module
from market_api_fetcher import (
    fetch_agmarknet,
    fetch_price_history,
    analyse_prices,
    get_best_market,
    MSP_REFERENCE,
)

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
MODEL = "gemini-2.5-flash"

# ── Tool definitions (Gemini function calling) ────────────────────────────────

# Convert to Gemini function declaration format
def get_current_prices_func(commodity: str, state: str = None, market: str = None):
    """Fetch today's wholesale market prices for a commodity from Indian mandis.
    Returns min/max/modal price per quintal across multiple markets, states, and districts.
    Use this when the farmer asks about current prices.
    
    Args:
        commodity: Crop name e.g. Wheat, Tomato, Onion, Paddy, Gram, Tur, Potato, Maize, Cotton
        state: Indian state name e.g. Punjab, Maharashtra, Uttar Pradesh. Omit for all-India data.
        market: Specific mandi/market name. Omit to search all markets.
    """
    records = fetch_agmarknet(commodity=commodity, state=state, market=market)
    analysis = analyse_prices(records, commodity)
    if not analysis:
        return {"error": "No price data found. The crop may not be traded today or the name may be misspelled."}
    return analysis


def get_price_trend_func(commodity: str, state: str = None, days: int = 7):
    """Fetch price data for the last 7 days for a commodity to identify if prices are
    rising, falling, or stable. Use this before giving sell/hold advice.
    
    Args:
        commodity: Crop name
        state: State name (optional)
        days: Number of past days (default 7, max 30)
    """
    days = min(days, 30)
    records = fetch_price_history(commodity=commodity, state=state, days=days)
    if not records:
        return {"error": "No trend data available."}

    # Group by date, compute daily average
    from collections import defaultdict
    daily = defaultdict(list)
    for r in records:
        if r.get("modal_price") and r.get("arrival_date"):
            daily[r["arrival_date"]].append(r["modal_price"])

    daily_avg = {
        date: round(sum(prices) / len(prices))
        for date, prices in sorted(daily.items())
    }

    prices_list = list(daily_avg.values())
    if len(prices_list) >= 2:
        change     = prices_list[-1] - prices_list[0]
        pct_change = round((change / prices_list[0]) * 100, 1)
        if pct_change > 3:
            trend = "Rising"
        elif pct_change < -3:
            trend = "Falling"
        else:
            trend = "Stable"
    else:
        trend, pct_change = "Insufficient data", 0

    return {
        "commodity":   commodity,
        "days":        days,
        "trend":       trend,
        "pct_change":  pct_change,
        "daily_prices": daily_avg,
        "latest_avg":  prices_list[-1] if prices_list else None,
        "earliest_avg": prices_list[0] if prices_list else None,
    }


def find_best_mandi_func(commodity: str, state: str = None, quantity_qtl: float = 1.0):
    """Find the top 5 mandis currently paying the highest prices for a commodity,
    optionally filtered by state. Returns market name, price, and estimated revenue.
    
    Args:
        commodity: Crop name
        state: Filter by state (optional)
        quantity_qtl: Farmer's harvest in quintals (for revenue estimate)
    """
    records = fetch_agmarknet(commodity=commodity, state=state)
    if not records:
        return {"error": "No mandi data found."}

    import pandas as pd
    df = pd.DataFrame(records)
    df = df[df["modal_price"].notna()]

    top5 = (
        df.groupby(["market", "state"])["modal_price"]
          .mean()
          .reset_index()
          .sort_values("modal_price", ascending=False)
          .head(5)
    )
    result = []
    for _, row in top5.iterrows():
        result.append({
            "market":       row["market"],
            "state":        row["state"],
            "avg_price":    round(row["modal_price"]),
            "est_revenue":  round(row["modal_price"] * quantity_qtl),
            "quantity_qtl": quantity_qtl,
        })
    return {"top_mandis": result, "commodity": commodity}


def compare_with_msp_func(commodity: str, state: str = None):
    """Compare the current market price of a crop against its government MSP
    (Minimum Support Price). Tells the farmer if they should sell in the open market
    or seek government procurement.
    
    Args:
        commodity: Crop name
        state: State (optional, for local prices)
    """
    records  = fetch_agmarknet(commodity=commodity, state=state)
    analysis = analyse_prices(records, commodity)
    msp      = MSP_REFERENCE.get(commodity)
    market_avg = analysis.get("modal_avg") if analysis else None

    if not msp:
        return {
            "commodity":    commodity,
            "msp":          None,
            "market_avg":   market_avg,
            "note": "No MSP declared for this crop. Sell in open market."
        }

    if not market_avg:
        return {"error": "Could not fetch current market prices."}

    diff = market_avg - msp
    pct  = round((diff / msp) * 100, 1)
    return {
        "commodity":    commodity,
        "msp":          msp,
        "market_avg":   market_avg,
        "difference":   diff,
        "pct_above_msp": pct,
        "recommendation": (
            "Open market price is above MSP — sell at mandi for better returns."
            if diff > 0 else
            "Market price is BELOW MSP — sell through government procurement (APMC/FCI/state agency) to claim MSP."
        ),
    }


def estimate_revenue_func(commodity: str, quantity_qtl: float, state: str = None):
    """Estimate how much a farmer will earn selling their produce at current market prices.
    
    Args:
        commodity: Crop name
        quantity_qtl: Harvest quantity in quintals
        state: State to find local prices (optional)
    """
    records  = fetch_agmarknet(commodity=commodity, state=state)
    analysis = analyse_prices(records, commodity)
    msp      = MSP_REFERENCE.get(commodity)

    if not analysis or not analysis.get("modal_avg"):
        return {"error": "Cannot fetch current prices."}

    avg_price   = analysis["modal_avg"]
    best_market = get_best_market(records, quantity_qtl)

    result = {
        "commodity":           commodity,
        "quantity_qtl":        quantity_qtl,
        "avg_revenue":         round(avg_price * quantity_qtl),
        "best_market_revenue": round(best_market.get("modal_price", avg_price) * quantity_qtl) if best_market else None,
        "best_market":         best_market.get("market", ""),
        "avg_price":           avg_price,
        "best_price":          best_market.get("modal_price"),
    }
    if msp:
        result["msp_revenue"]    = round(msp * quantity_qtl)
        result["gain_over_msp"]  = round((avg_price - msp) * quantity_qtl)
    return result


# Register functions for Gemini
TOOL_FUNCTIONS = {
    "get_current_prices": get_current_prices_func,
    "get_price_trend": get_price_trend_func,
    "find_best_mandi": find_best_mandi_func,
    "compare_with_msp": compare_with_msp_func,
    "estimate_revenue": estimate_revenue_func,
}


# ── Agent loop ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FasalSetuBazaar, a friendly and knowledgeable market advisor for Indian farmers.
Your job is to give clear, actionable, and personalised guidance about when and where to sell crops.

PERSONALITY:
- Speak simply and directly, as if talking to a farmer in a village
- Always give a clear recommendation (sell now / wait / go to this mandi)
- Explain the reasoning in 2-3 sentences
- Quote actual prices in ₹/quintal
- Compare with MSP when relevant
- Mention transport costs as a consideration for distant mandis
- Be honest — if prices are bad, say so and explain options

GUIDANCE FRAMEWORK — always cover:
1. Current price vs MSP (is the farmer getting a fair deal?)
2. Price trend (rising = wait a few days, falling = sell quickly)
3. Best nearby mandi vs best overall mandi (tradeoff)
4. Revenue estimate for their harvest
5. Specific action step ("Go to X mandi this week" or "Register with FCI for procurement")

TOOLS:
- Use get_current_prices first to understand the market
- Use get_price_trend to decide whether to advise "sell now" or "hold"
- Use find_best_mandi when the farmer needs to know WHERE to sell
- Use compare_with_msp for any notified crop
- Use estimate_revenue when the farmer mentions their harvest quantity

If the farmer hasn't told you their crop, state, or quantity — ask these three questions together upfront before fetching data.
Never invent prices. Always fetch real data using tools before giving advice.
Respond in the language the farmer uses (Hindi or English)."""


def run_agent(user_message: str, conversation_history: list) -> tuple[str, list]:
    """
    Run one turn of the agent using Gemini.
    Returns (assistant_response_text, updated_conversation_history)
    """
    if not GEMINI_KEY:
        return "Error: GEMINI_API_KEY not set. Add it to your .env file.", conversation_history
    
    # Create model with function calling
    model = genai.GenerativeModel(
        model_name=MODEL,
        tools=list(TOOL_FUNCTIONS.values()),
        system_instruction=SYSTEM_PROMPT,
    )
    
    # Build chat history for Gemini
    chat = model.start_chat(history=[])
    
    # Add previous conversation history
    for msg in conversation_history:
        if msg["role"] == "user":
            chat.history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            chat.history.append({"role": "model", "parts": [msg["content"]]})
    
    # Send user message
    response = chat.send_message(user_message)
    
    # Handle function calls
    max_iterations = 5
    iteration = 0
    
    while response.candidates[0].content.parts and iteration < max_iterations:
        iteration += 1
        
        # Check if there are function calls
        function_calls = [
            part for part in response.candidates[0].content.parts 
            if hasattr(part, 'function_call') and part.function_call
        ]
        
        if not function_calls:
            # No more function calls, return the text response
            break
        
        # Execute all function calls
        function_responses = []
        for part in function_calls:
            func_call = part.function_call
            func_name = func_call.name
            func_args = dict(func_call.args)
            
            _log_tool_call(func_name, func_args)
            
            # Execute the function
            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=func_name,
                                response={"result": result}
                            )
                        )
                    )
                except Exception as e:
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=func_name,
                                response={"error": str(e)}
                            )
                        )
                    )
        
        # Send function responses back to model
        if function_responses:
            response = chat.send_message(function_responses)
    
    # Extract final text response
    final_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            final_text += part.text
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": final_text})
    
    return final_text, conversation_history


def _log_tool_call(name: str, inp: dict):
    if RICH:
        console.print(f"  [dim]→ {name}({', '.join(f'{k}={v}' for k,v in inp.items())})[/dim]")
    else:
        print(f"  → {name}({inp})")


# ── Demo mode (mock data when no API key) ─────────────────────────────────────

MOCK_RESPONSE = """
Based on current market data for **Wheat in Punjab**:

**Current Price**: ₹2,450/quintal (modal average across 12 mandis)
**MSP 2024-25**: ₹2,425/quintal
✅ Market price is **₹25 above MSP (+1%)** — open market is slightly better than govt procurement right now.

**7-Day Trend**: Prices have risen 4% over the last week (from ₹2,356 to ₹2,450). This is a **rising trend** — holding a few more days could be beneficial.

**Top Mandis (Punjab)**:
1. Ludhiana Grain Mandi — ₹2,510/qtl
2. Amritsar Mandi — ₹2,490/qtl
3. Patiala Mandi — ₹2,460/qtl

**Revenue Estimate (50 quintals)**:
- Average mandi: ₹1,22,500
- Ludhiana (best): ₹1,25,500
- If sold at MSP (FCI): ₹1,21,250

**My Recommendation:**
Prices are rising. If you can hold for 3–5 more days, you may get ₹2,500+. If you need to sell now, go to **Ludhiana Grain Mandi** — it's paying ₹2,510/qtl which is ₹3,250 more than the state average for your 50 quintals. Factor in transport cost before deciding between Ludhiana and your local mandi.
"""


# ── Interactive chat portal ────────────────────────────────────────────────────

def run_chat_portal():
    if RICH:
        console.print()
        console.print(Panel.fit(
            "[bold green]🌾 FasalSetuBazaar — Market Price Advisor[/bold green]\n"
            "[dim]Ask about prices, best mandis, and when to sell your crop[/dim]",
            border_style="green",
            padding=(1, 4),
        ))
        console.print("  [dim]Type 'quit' to exit. Type 'clear' to start over.[/dim]\n")
    else:
        print("\n" + "═"*60)
        print("   🌾  FASALSETUBAZAAR — MARKET PRICE ADVISOR")
        print("   Ask about prices, best mandis, when to sell")
        print("═"*60 + "\n  Type 'quit' to exit.\n")

    conversation = []

    while True:
        if RICH:
            user_input = Prompt.ask("\n  [bold]You[/bold]").strip()
        else:
            user_input = input("\n  You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            if RICH:
                console.print("\n[green]  Jai Kisan! 🌾[/green]\n")
            break
        if user_input.lower() == "clear":
            conversation.clear()
            print("  [Conversation cleared]\n")
            continue

        if RICH:
            with console.status("[green]  Checking market data...[/green]", spinner="dots"):
                response, conversation = run_agent(user_input, conversation)
        else:
            print("  Checking market data...")
            response, conversation = run_agent(user_input, conversation)

        if RICH:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]FasalSetuBazaar[/bold green]",
                border_style="green",
                padding=(0, 2),
            ))
        else:
            print(f"\n  FasalSetuBazaar:\n")
            for line in response.split("\n"):
                print(f"  {line}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FasalSetuBazaar Market Price Agent")
    parser.add_argument("--crop",  help="Crop name (e.g. Wheat)")
    parser.add_argument("--state", help="Your state (e.g. Punjab)")
    parser.add_argument("--qty",   type=float, help="Your harvest in quintals")
    parser.add_argument("--demo",  action="store_true", help="Run demo without API calls")
    args = parser.parse_args()

    if not GEMINI_KEY:
        print("[ERROR] Set GEMINI_API_KEY in your .env file")
        print("Get one at: https://aistudio.google.com/app/apikey")
        return

    if args.demo:
        print("\n[DEMO MODE — mock data]\n")
        print(MOCK_RESPONSE)
        return

    if args.crop:
        # Single-shot mode: build an initial message from CLI args
        parts = [f"I want to sell my {args.crop}"]
        if args.state:   parts.append(f"I am in {args.state}")
        if args.qty:     parts.append(f"I have {args.qty} quintals")
        parts.append("Should I sell now? Where should I sell?")
        query = ". ".join(parts) + "."

        conversation = []
        if RICH:
            with console.status("[green]Checking market data...[/green]", spinner="dots"):
                response, _ = run_agent(query, conversation)
            console.print(Panel(Markdown(response), title="FasalSetuBazaar", border_style="green"))
        else:
            print(f"\nQuery: {query}\n")
            response, _ = run_agent(query, conversation)
            print(response)
    else:
        run_chat_portal()


if __name__ == "__main__":
    main()
