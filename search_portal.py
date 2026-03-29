"""
search_portal.py
────────────────────────────────────────────────────────────────────────────
Interactive CLI search portal for government farmer schemes.

Implements the recommended 3-step flow:
  Step 1 → Filter by STATE
  Step 2 → Filter by CATEGORY (need type)
  Step 3 → Free-text semantic query → ChromaDB similarity search

Requirements:
    pip install chromadb sentence-transformers rich

Usage:
    python search_portal.py
    python search_portal.py --top 10        # return top-10 results (default 5)
    python search_portal.py --db ./chroma_db
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
from textwrap import fill, indent

import chromadb
from chromadb.utils import embedding_functions

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

# ── Config ────────────────────────────────────────────────────────────────

COLLECTION_NAME = "government_farmer_schemes"
DB_PATH         = "./chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Data: States & Categories ─────────────────────────────────────────────

STATES = {
    "0": "All India",          # shows Central + all states
    "1": "All India",          # Central schemes only
    "2": "Uttar Pradesh",
    "3": "Madhya Pradesh",
    "4": "Punjab",
    "5": "Maharashtra",
    "6": "West Bengal",
}

STATE_MENU = [
    ("0", "🇮🇳  All States (Central + State schemes)"),
    ("1", "🏛️   Central Government only"),
    ("2", "🌾  Uttar Pradesh"),
    ("3", "🌿  Madhya Pradesh"),
    ("4", "🚜  Punjab"),
    ("5", "🌊  Maharashtra"),
    ("6", "🍃  West Bengal"),
]

CATEGORIES = {
    "0": None,                              # all categories
    "1": "Income Support & Credit",
    "2": "Crop Insurance & Risk Protection",
    "3": "Irrigation & Water",
    "4": "Soil Health & Farming Practices",
    "5": "Market Access & Selling",
    "6": "Infrastructure & Storage",
    "7": "Technology & Mechanisation",
    "8": "Specialised Crop Missions",
}

CATEGORY_MENU = [
    ("0", "📋  All Needs / Any Category"),
    ("1", "💰  Income Support & Credit (loans, cash transfers, MSP)"),
    ("2", "🛡️   Crop Insurance & Risk Protection"),
    ("3", "💧  Irrigation & Water"),
    ("4", "🌱  Soil Health & Farming Practices"),
    ("5", "🛒  Market Access & Selling"),
    ("6", "🏗️   Infrastructure & Storage (warehouse, cold storage)"),
    ("7", "🚁  Technology & Mechanisation (drones, machinery)"),
    ("8", "🌾  Specialised Crop Missions (horticulture, bamboo, honey)"),
]

CATEGORY_HINTS = {
    "1": "Try: 'loan interest', 'cash transfer', 'kisan credit', 'income support'",
    "2": "Try: 'crop loss', 'weather insurance', 'natural calamity', 'fasal bima'",
    "3": "Try: 'drip irrigation', 'solar pump', 'water shortage', 'sprinkler'",
    "4": "Try: 'organic farming', 'soil testing', 'zero budget', 'natural farming'",
    "5": "Try: 'sell produce', 'mandi', 'market price', 'export', 'middlemen'",
    "6": "Try: 'godown', 'cold storage', 'warehouse', 'post harvest', 'processing unit'",
    "7": "Try: 'drone', 'tractor subsidy', 'machinery', 'digital farming', 'AI'",
    "8": "Try: 'horticulture', 'bamboo', 'beekeeping', 'oil palm', 'maize'",
    "0": "Try: 'accident benefit', 'solar pump', 'women farmer', 'SC ST', 'loan'",
}

# ── Console helper ────────────────────────────────────────────────────────

console = Console() if RICH else None

def print_header():
    if RICH:
        console.print()
        console.print(Panel.fit(
            "[bold green]🌾 Kisan Yojana Search Portal[/bold green]\n"
            "[dim]Find government schemes matching your needs[/dim]",
            border_style="green",
            padding=(1, 4),
        ))
    else:
        print("\n" + "═"*60)
        print("   🌾  KISAN YOJANA SEARCH PORTAL")
        print("   Find government schemes matching your needs")
        print("═"*60 + "\n")

def print_section(title: str):
    if RICH:
        console.print(f"\n[bold yellow]  {title}[/bold yellow]")
        console.print("  " + "─"*50)
    else:
        print(f"\n  {title}")
        print("  " + "─"*50)

def print_menu(items: list[tuple[str, str]]):
    for key, label in items:
        if RICH:
            console.print(f"   [cyan]{key}[/cyan]  {label}")
        else:
            print(f"   {key}  {label}")

def get_input(prompt: str) -> str:
    if RICH:
        return Prompt.ask(f"\n  [bold]{prompt}[/bold]").strip()
    else:
        return input(f"\n  {prompt}: ").strip()

def print_hint(text: str):
    if RICH:
        console.print(f"  [dim italic]{text}[/dim italic]")
    else:
        print(f"  {text}")

def print_error(text: str):
    if RICH:
        console.print(f"  [red]⚠  {text}[/red]")
    else:
        print(f"  ⚠  {text}")

def print_success(text: str):
    if RICH:
        console.print(f"  [green]✓  {text}[/green]")
    else:
        print(f"  ✓  {text}")


# ── Search logic ──────────────────────────────────────────────────────────

def build_where_filter(state_key: str, category_key: str) -> dict | None:
    """
    Build ChromaDB $and / $or where clause based on user selection.
    """
    conditions = []

    # State filter
    if state_key == "1":
        # Central only
        conditions.append({"level": {"$eq": "Central"}})
    elif state_key != "0":
        state_name = STATES[state_key]
        # Include Central schemes PLUS the selected state
        conditions.append({
            "$or": [
                {"level": {"$eq": "Central"}},
                {"state": {"$eq": state_name}},
            ]
        })

    # Category filter
    cat = CATEGORIES.get(category_key)
    if cat:
        conditions.append({"category": {"$eq": cat}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def run_search(collection, query: str, where: dict | None, top_k: int) -> list[dict]:
    kwargs = {
        "query_texts": [query],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
        # cosine distance → similarity score (0-100)
        similarity = round((1 - dist) * 100, 1)
        output.append({
            "rank": i + 1,
            "id": doc_id,
            "similarity": similarity,
            "scheme_name": meta.get("scheme_name", "Unknown"),
            "state": meta.get("state", "—"),
            "level": meta.get("level", "—"),
            "category": meta.get("category", "—"),
            "benefit_type": meta.get("benefit_type", "—"),
            "benefit_amount": meta.get("benefit_amount", "—"),
            "eligibility": meta.get("eligibility", "—"),
            "apply_url": meta.get("apply_url", "—"),
            "document": doc,
        })

    return output


# ── Result display ────────────────────────────────────────────────────────

def score_bar(score: float) -> str:
    filled = int(score / 10)
    return "█" * filled + "░" * (10 - filled)

def display_results(results: list[dict], query: str):
    if not results:
        print_error("No matching schemes found. Try a broader query.")
        return

    if RICH:
        console.print(f"\n  [bold green]Found {len(results)} schemes for:[/bold green] [italic]\"{query}\"[/italic]\n")

        for r in results:
            score_color = "green" if r["similarity"] >= 60 else "yellow" if r["similarity"] >= 40 else "red"
            bar = score_bar(r["similarity"])

            header = (
                f"[bold]#{r['rank']}  {r['scheme_name']}[/bold]   "
                f"[{score_color}]{bar} {r['similarity']}% match[/{score_color}]"
            )
            body_lines = [
                f"[dim]Category:[/dim]    {r['category']}",
                f"[dim]Level/State:[/dim] {r['level']} — {r['state']}",
                f"[dim]Benefit:[/dim]     {r['benefit_type']}",
            ]
            if r["benefit_amount"] and r["benefit_amount"] != "—":
                body_lines.append(f"[dim]Amount:[/dim]      [bold yellow]{r['benefit_amount']}[/bold yellow]")
            if r["eligibility"] and r["eligibility"] != "—":
                body_lines.append(f"[dim]Eligibility:[/dim] {r['eligibility']}")
            if r["apply_url"] and r["apply_url"] != "—":
                body_lines.append(f"[dim]Apply at:[/dim]    [link]{r['apply_url']}[/link]")

            body = "\n".join(body_lines)
            console.print(Panel(
                f"{header}\n\n{body}",
                border_style="green" if r["similarity"] >= 60 else "yellow",
                padding=(0, 2),
            ))
    else:
        print(f"\n  Found {len(results)} schemes for: \"{query}\"\n")
        print("  " + "═"*56)
        for r in results:
            bar = score_bar(r["similarity"])
            print(f"\n  #{r['rank']}  {r['scheme_name']}")
            print(f"       Match: {bar} {r['similarity']}%")
            print(f"       Category   : {r['category']}")
            print(f"       Level/State: {r['level']} — {r['state']}")
            print(f"       Benefit    : {r['benefit_type']}")
            if r["benefit_amount"] != "—":
                print(f"       Amount     : {r['benefit_amount']}")
            if r["eligibility"] != "—":
                print(f"       Eligibility: {r['eligibility']}")
            if r["apply_url"] != "—":
                print(f"       Apply at   : {r['apply_url']}")
        print("\n  " + "═"*56)


# ── Main portal loop ──────────────────────────────────────────────────────

def portal_loop(collection, top_k: int):
    print_header()

    while True:
        # ── STEP 1: State selection ────────────────────────────────────
        print_section("STEP 1 — Select your State")
        print_menu(STATE_MENU)

        while True:
            state_key = get_input("Enter number (0–6)")
            if state_key in STATES:
                break
            print_error("Invalid choice. Enter a number between 0 and 6.")

        state_label = STATE_MENU[int(state_key)][1]
        print_success(f"State: {state_label.strip()}")

        # ── STEP 2: Category selection ─────────────────────────────────
        print_section("STEP 2 — What do you need help with?")
        print_menu(CATEGORY_MENU)

        while True:
            cat_key = get_input("Enter number (0–8)")
            if cat_key in CATEGORIES:
                break
            print_error("Invalid choice. Enter a number between 0 and 8.")

        cat_label = CATEGORY_MENU[int(cat_key)][1]
        print_success(f"Category: {cat_label.strip()}")

        # ── STEP 3: Free-text semantic search ─────────────────────────
        print_section("STEP 3 — Describe your need")
        hint = CATEGORY_HINTS.get(cat_key, "")
        if hint:
            print_hint(f"💡 {hint}")

        query = get_input("Search (press Enter for broad search)").strip()
        if not query:
            query = CATEGORIES.get(cat_key) or "farmer government scheme"

        # Build filter and search
        where = build_where_filter(state_key, cat_key)

        if RICH:
            with console.status("[bold green]Searching schemes...[/bold green]", spinner="dots"):
                results = run_search(collection, query, where, top_k)
        else:
            print("\n  Searching...")
            results = run_search(collection, query, where, top_k)

        display_results(results, query)

        # ── Loop or exit ───────────────────────────────────────────────
        again = get_input("Search again? (y/n)").lower()
        if again not in ("y", "yes"):
            if RICH:
                console.print("\n[bold green]  Jai Kisan! 🌾[/bold green]\n")
            else:
                print("\n  Jai Kisan! 🌾\n")
            break


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kisan Yojana Search Portal")
    parser.add_argument("--top",  type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--db",   default=DB_PATH,   help="ChromaDB persistence directory")
    args = parser.parse_args()

    # Connect
    try:
        client = chromadb.PersistentClient(path=args.db)
    except Exception as e:
        print_error(f"Could not connect to ChromaDB at '{args.db}': {e}")
        print_error("Run seed_chromadb.py first.")
        sys.exit(1)

    # Embedding function (must match seeder)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        print_error(f"Collection '{COLLECTION_NAME}' not found.")
        print_error("Run: python seed_chromadb.py")
        sys.exit(1)

    count = collection.count()
    if count == 0:
        print_error("Collection is empty. Run: python seed_chromadb.py")
        sys.exit(1)

    if RICH:
        console.print(f"\n  [dim]Connected — {count} schemes loaded[/dim]")
    else:
        print(f"\n  Connected — {count} schemes loaded")

    portal_loop(collection, top_k=args.top)


if __name__ == "__main__":
    main()
