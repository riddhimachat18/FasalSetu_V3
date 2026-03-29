"""
market_api_fetcher.py
──────────────────────────────────────────────────────────────────────────────
Fetches Indian agricultural market prices from all available sources,
caches them locally, and exposes a unified interface for the agent.

DATA SOURCES (in priority order):
──────────────────────────────────────────────────────────────────────────────
1. data.gov.in / Agmarknet API  [PRIMARY — FREE, OFFICIAL]
   - Government of India: Ministry of Agriculture & Farmers Welfare
   - Data: daily min/max/modal price per commodity per mandi
   - Coverage: 3,000+ mandis, 300+ commodities, 2M+ records/month
   - Registration: https://data.gov.in  (free, instant API key)
   - Resource IDs:
       Current prices : 9ef84268-d588-465a-a308-a864a43d0070
       Variety-wise   : 35985678-0d79-46b4-9ed6-6f13308a1d24
   - Endpoint: https://api.data.gov.in/resource/{resource_id}
   - Params  : api-key, format=json, filters[*], limit, offset
   - Rate    : 1,000 req/day free tier; request higher via portal

2. eNAM API  [SECONDARY — FREE, OFFICIAL]
   - National Agriculture Market (electronic mandi)
   - Data: live trade data from eNAM-connected mandis
   - Coverage: 1,000+ mandis across 22 states
   - Registration: https://enam.gov.in (requires approval)
   - Endpoint: https://enam.gov.in/web/trades/getTradeDataDetails
   - Best for: real-time auction prices, not just daily summaries

3. Agmarknet Direct  [TERTIARY — OFFICIAL, scrape-only]
   - https://agmarknet.gov.in — no public REST API as of 2025
   - Use Selenium scraper for bulk historical pulls only
   - Respect rate limits; govt servers are not robust

4. CEDA Ashoka  [ANALYTICS — FREE, ACADEMIC]
   - https://agmarknet.ceda.ashoka.edu.in
   - Cleaned, deduplicated Agmarknet data with trend visualisations
   - No formal API; CSV bulk download available
   - Best for: historical trend data for training/analysis

5. Farmonaut API  [PAID — COMMERCIAL]
   - https://farmonaut.com/api-development
   - Cleaned Agmarknet + satellite crop data bundled
   - Pricing: tiered (starts ~₹999/month)
   - Best for: production apps needing SLA guarantees

SETUP:
──────────────────────────────────────────────────────────────────────────────
    pip install requests pandas python-dotenv rich schedule

    Create .env file:
        DATA_GOV_IN_API_KEY=your_key_here      # from data.gov.in
        ENAM_API_KEY=your_key_here             # from enam.gov.in (optional)

    Register for free key: https://data.gov.in/user/register
    After login → Profile → API Key (instant, no approval needed)

USAGE:
──────────────────────────────────────────────────────────────────────────────
    python market_api_fetcher.py                        # fetch + cache today's prices
    python market_api_fetcher.py --commodity Wheat      # filter commodity
    python market_api_fetcher.py --state Punjab         # filter state
    python market_api_fetcher.py --days 7               # fetch last 7 days
    python market_api_fetcher.py --serve                # start scheduler (auto-refresh daily)
"""

import os
import json
import time
import logging
import argparse
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH = True
except ImportError:
    RICH = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("market_fetcher")

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR         = Path("./market_cache")
CACHE_TTL_HOURS   = 6          # refresh cache after 6 hours
MAX_RECORDS       = 500        # per API call
REQUEST_TIMEOUT   = 15         # seconds

DATA_GOV_IN_KEY   = os.getenv("DATA_GOV_IN_API_KEY", "/resource/35985678-0d79-46b4-9ed6-6f13308a1d24")

# data.gov.in resource IDs
RESOURCE_CURRENT  = "9ef84268-d588-465a-a308-a864a43d0070"  # daily current prices
RESOURCE_VARIETY  = "35985678-0d79-46b4-9ed6-6f13308a1d24"  # variety-wise prices

BASE_URL = "https://api.data.gov.in/resource/{resource_id}"

# ── Commodity → MSP reference (₹/quintal, 2024-25)
MSP_REFERENCE = {
    "Wheat":         2425,  "Paddy":         2300,  "Rice":          2300,
    "Maize":         2090,  "Jowar":         3371,  "Bajra":         2625,
    "Ragi":          4290,  "Barley":        1735,  "Gram":          5440,
    "Tur":           7550,  "Moong":         8682,  "Urad":          7400,
    "Groundnut":     6783,  "Sunflower":     7280,  "Soybean":       4892,
    "Sesame":       9267,   "Rapeseed":      5950,  "Cotton":       7121,
    "Jute":         5335,   "Sugarcane":      340,   "Tomato":         None,
    "Onion":         None,  "Potato":         None,
}

# ── Cache helpers ──────────────────────────────────────────────────────────────

CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(params: dict) -> str:
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"

def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_TTL_HOURS * 3600

def cache_read(params: dict) -> Optional[list]:
    path = _cache_path(_cache_key(params))
    if _cache_valid(path):
        with open(path) as f:
            data = json.load(f)
        log.info(f"Cache hit: {path.name}")
        return data
    return None

def cache_write(params: dict, records: list):
    path = _cache_path(_cache_key(params))
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# ── data.gov.in Agmarknet API ─────────────────────────────────────────────────

def fetch_agmarknet(
    commodity: Optional[str] = None,
    state:     Optional[str] = None,
    market:    Optional[str] = None,
    date:      Optional[str] = None,     # DD/MM/YYYY
    limit:     int           = MAX_RECORDS,
    use_cache: bool          = True,
) -> list[dict]:
    """
    Fetch from data.gov.in Agmarknet API.
    Returns list of price records.
    """
    params = {
        "commodity": commodity, "state": state,
        "market": market, "date": date, "limit": limit,
    }

    if use_cache:
        cached = cache_read(params)
        if cached is not None:
            return cached

    query = {
        "api-key":  DATA_GOV_IN_KEY,
        "format":   "json",
        "limit":    limit,
        "offset":   0,
    }
    if commodity: query["filters[Commodity]"]    = commodity
    if state:     query["filters[State]"]        = state
    if market:    query["filters[Market]"]       = market
    if date:      query["filters[Arrival_Date]"] = date

    url = BASE_URL.format(resource_id=RESOURCE_CURRENT)

    try:
        log.info(f"Fetching Agmarknet: commodity={commodity}, state={state}, market={market}")
        resp = requests.get(url, params=query, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records", [])
        # Normalise field names
        normalised = [_normalise_record(r) for r in records]
        log.info(f"Got {len(normalised)} records from data.gov.in")

        cache_write(params, normalised)
        return normalised

    except requests.exceptions.RequestException as e:
        log.error(f"Agmarknet API error: {e}")
        # Try returning stale cache if available
        path = _cache_path(_cache_key(params))
        if path.exists():
            log.warning("Using stale cache as fallback")
            with open(path) as f:
                return json.load(f)
        return []

def _normalise_record(raw: dict) -> dict:
    """Normalise Agmarknet field names to a consistent schema."""
    return {
        "commodity":    raw.get("Commodity")    or raw.get("commodity", ""),
        "variety":      raw.get("Variety")      or raw.get("variety", ""),
        "state":        raw.get("State")        or raw.get("state", ""),
        "district":     raw.get("District")     or raw.get("district", ""),
        "market":       raw.get("Market")       or raw.get("market", ""),
        "arrival_date": raw.get("Arrival_Date") or raw.get("arrival_date", ""),
        "min_price":    _to_int(raw.get("Min_x0020_Price") or raw.get("min_price") or raw.get("Min Price")),
        "max_price":    _to_int(raw.get("Max_x0020_Price") or raw.get("max_price") or raw.get("Max Price")),
        "modal_price":  _to_int(raw.get("Modal_x0020_Price") or raw.get("modal_price") or raw.get("Modal Price")),
        "source":       "data.gov.in/agmarknet",
    }

def _to_int(val) -> Optional[int]:
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return None


# ── Multi-day fetch ────────────────────────────────────────────────────────────

def fetch_price_history(
    commodity: str,
    state:     Optional[str] = None,
    days:      int           = 7,
) -> list[dict]:
    """Fetch price records for last N days, combine and return."""
    all_records = []
    for i in range(days):
        d = datetime.now() - timedelta(days=i)
        date_str = d.strftime("%d/%m/%Y")
        records = fetch_agmarknet(commodity=commodity, state=state, date=date_str)
        all_records.extend(records)
        time.sleep(0.5)  # gentle on gov servers
    return all_records


# ── Analytics helpers ──────────────────────────────────────────────────────────

def analyse_prices(records: list[dict], commodity: str) -> dict:
    """
    Given a list of price records, return summary analytics
    used by the agent for guidance generation.
    """
    if not records:
        return {}

    df = pd.DataFrame(records)
    df = df[df["modal_price"].notna()]
    if df.empty:
        return {}

    msp = MSP_REFERENCE.get(commodity)
    modal_mean  = df["modal_price"].mean()
    modal_max   = df["modal_price"].max()
    modal_min   = df["modal_price"].min()
    top_markets = (
        df.groupby("market")["modal_price"].mean()
          .sort_values(ascending=False)
          .head(5)
          .to_dict()
    )
    state_avg = (
        df.groupby("state")["modal_price"].mean()
          .sort_values(ascending=False)
          .to_dict()
    ) if "state" in df.columns else {}

    # Price vs MSP comparison
    msp_comparison = None
    if msp:
        pct = ((modal_mean - msp) / msp) * 100
        msp_comparison = {
            "msp": msp,
            "market_avg": round(modal_mean),
            "difference": round(modal_mean - msp),
            "pct_above_msp": round(pct, 1),
            "verdict": "above MSP" if pct >= 0 else "below MSP",
        }

    return {
        "commodity":      commodity,
        "records_count":  len(df),
        "modal_avg":      round(modal_mean),
        "modal_max":      int(modal_max),
        "modal_min":      int(modal_min),
        "price_range":    int(modal_max - modal_min),
        "top_markets":    {k: round(v) for k, v in top_markets.items()},
        "state_averages": {k: round(v) for k, v in state_avg.items()},
        "msp_comparison": msp_comparison,
        "as_of":          datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def get_best_market(records: list[dict], quantity_qtl: float = 1.0) -> dict:
    """Find the highest-paying market for a given commodity."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    df = df[df["modal_price"].notna()]
    best_row = df.loc[df["modal_price"].idxmax()].to_dict()
    revenue  = round(best_row["modal_price"] * quantity_qtl)
    return {
        "market":       best_row.get("market", ""),
        "state":        best_row.get("state", ""),
        "district":     best_row.get("district", ""),
        "modal_price":  best_row["modal_price"],
        "date":         best_row.get("arrival_date", ""),
        "est_revenue":  revenue,
        "quantity_qtl": quantity_qtl,
    }


# ── Display ────────────────────────────────────────────────────────────────────

def display_summary(analysis: dict):
    if not analysis:
        print("No data available.")
        return

    if RICH:
        t = Table(title=f"Market Price Summary — {analysis['commodity']}", show_header=True, header_style="bold green")
        t.add_column("Metric", style="dim", width=24)
        t.add_column("Value", justify="right")
        t.add_row("Records",        str(analysis["records_count"]))
        t.add_row("Modal Avg",      f"₹{analysis['modal_avg']:,}/qtl")
        t.add_row("Modal Max",      f"₹{analysis['modal_max']:,}/qtl")
        t.add_row("Modal Min",      f"₹{analysis['modal_min']:,}/qtl")
        t.add_row("Price Spread",   f"₹{analysis['price_range']:,}/qtl")
        if analysis.get("msp_comparison"):
            m = analysis["msp_comparison"]
            color = "green" if m["pct_above_msp"] >= 0 else "red"
            t.add_row("MSP",        f"₹{m['msp']:,}/qtl")
            t.add_row("vs MSP",     f"[{color}]{m['pct_above_msp']:+.1f}% ({m['verdict']})[/{color}]")
        console.print(t)

        console.print("\n[bold]Top markets by price:[/bold]")
        for market, price in analysis["top_markets"].items():
            console.print(f"  {market:<30} ₹{price:,}/qtl")
    else:
        print(f"\n── {analysis['commodity']} Market Prices ──")
        print(f"  Records   : {analysis['records_count']}")
        print(f"  Modal avg : ₹{analysis['modal_avg']:,}/qtl")
        print(f"  Max       : ₹{analysis['modal_max']:,}/qtl")
        print(f"  Min       : ₹{analysis['modal_min']:,}/qtl")
        if analysis.get("msp_comparison"):
            m = analysis["msp_comparison"]
            print(f"  MSP       : ₹{m['msp']:,}/qtl  ({m['pct_above_msp']:+.1f}% — {m['verdict']})")
        print("\n  Top markets:")
        for market, price in analysis["top_markets"].items():
            print(f"    {market:<28} ₹{price:,}/qtl")


# ── Scheduler (daily auto-refresh) ────────────────────────────────────────────

def run_scheduler(commodities: list[str], states: list[str]):
    try:
        import schedule
    except ImportError:
        log.error("Install 'schedule': pip install schedule")
        return

    def daily_refresh():
        log.info("Daily price refresh starting...")
        for commodity in commodities:
            for state in states:
                fetch_agmarknet(commodity=commodity, state=state, use_cache=False)
                time.sleep(1)
        log.info("Daily price refresh complete.")

    schedule.every().day.at("06:00").do(daily_refresh)
    daily_refresh()  # run immediately on start

    log.info("Scheduler started. Running daily at 06:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch Indian agricultural market prices")
    parser.add_argument("--commodity", default=None, help="Crop name (e.g. Wheat, Tomato)")
    parser.add_argument("--state",     default=None, help="State name (e.g. Punjab)")
    parser.add_argument("--market",    default=None, help="Mandi name")
    parser.add_argument("--days",      type=int, default=1, help="Fetch last N days of data")
    parser.add_argument("--no-cache",  action="store_true", help="Bypass cache")
    parser.add_argument("--serve",     action="store_true", help="Run as daily scheduler")
    args = parser.parse_args()

    if args.serve:
        run_scheduler(
            commodities=["Wheat", "Tomato", "Onion", "Potato", "Paddy", "Tur", "Gram"],
            states=["Uttar Pradesh", "Punjab", "Maharashtra", "Madhya Pradesh"],
        )
        return

    if args.days > 1 and args.commodity:
        records = fetch_price_history(args.commodity, args.state, args.days)
    else:
        records = fetch_agmarknet(
            commodity=args.commodity,
            state=args.state,
            market=args.market,
            use_cache=not args.no_cache,
        )

    if not records:
        print("No records returned. Check your API key and filters.")
        return

    analysis = analyse_prices(records, args.commodity or "All commodities")
    display_summary(analysis)

    if args.commodity:
        best = get_best_market(records)
        if best:
            print(f"\n  Best market : {best['market']} ({best['state']})")
            print(f"  Price       : ₹{best['modal_price']:,}/qtl on {best['date']}")


if __name__ == "__main__":
    main()
