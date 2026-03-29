"""
seed_chromadb.py
────────────────────────────────────────────────────────────────────────────
Seeds government farmer schemes into a local ChromaDB collection.

Requirements:
    pip install chromadb sentence-transformers

Usage:
    python seed_chromadb.py
    python seed_chromadb.py --json path/to/schemes_chromadb.json
    python seed_chromadb.py --reset   # drops and re-creates the collection
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# ── Config ────────────────────────────────────────────────────────────────

COLLECTION_NAME = "government_farmer_schemes"
DB_PATH = "./chroma_db"                       # persistent local storage
JSON_PATH = "schemes_chromadb.json"           # default data file

# Using a multilingual model — handles Hindi/regional search terms well
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ── Helpers ───────────────────────────────────────────────────────────────

def load_schemes(json_path: str) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        print("[ERROR] No documents found in JSON.")
        sys.exit(1)

    print(f"[✓] Loaded {len(documents)} schemes from {json_path}")
    return documents


def flatten_metadata(raw: dict) -> dict:
    """
    ChromaDB metadata values must be str | int | float | bool.
    Lists (keywords, tags) are joined into comma-separated strings.
    None values are dropped.
    """
    flat = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, list):
            flat[k] = ", ".join(str(i) for i in v)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat


def build_batch(schemes: list[dict]) -> tuple[list, list, list]:
    ids, documents, metadatas = [], [], []
    for scheme in schemes:
        ids.append(scheme["id"])
        documents.append(scheme["document"])
        metadatas.append(flatten_metadata(scheme["metadata"]))
    return ids, documents, metadatas


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed farmer schemes into ChromaDB")
    parser.add_argument("--json", default=JSON_PATH, help="Path to schemes JSON file")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--db", default=DB_PATH, help="ChromaDB persistence directory")
    args = parser.parse_args()

    # 1. Load data
    schemes = load_schemes(args.json)

    # 2. Connect to ChromaDB (persistent)
    print(f"[→] Connecting to ChromaDB at '{args.db}' ...")
    client = chromadb.PersistentClient(path=args.db)

    # 3. Embedding function (multilingual sentence-transformers)
    print(f"[→] Loading embedding model: {EMBEDDING_MODEL}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 4. Create / reset collection
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[✓] Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},    # cosine similarity for semantic search
    )

    existing_count = collection.count()
    if existing_count > 0 and not args.reset:
        print(f"[!] Collection already has {existing_count} documents.")
        print("    Run with --reset to drop and re-seed. Exiting.")
        sys.exit(0)

    # 5. Build and insert in batches (ChromaDB recommends ≤ 500 per upsert)
    ids, documents, metadatas = build_batch(schemes)

    BATCH_SIZE = 50
    total = len(ids)
    inserted = 0

    print(f"\n[→] Inserting {total} schemes in batches of {BATCH_SIZE} ...")
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        inserted += len(batch_ids)
        print(f"    ✓ {inserted}/{total} schemes seeded")

    elapsed = time.time() - start

    # 6. Verify
    final_count = collection.count()
    print(f"\n{'═'*55}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Documents  : {final_count}")
    print(f"  Time taken : {elapsed:.1f}s")
    print(f"  DB path    : {Path(args.db).resolve()}")
    print(f"{'═'*55}")
    print("\n[✓] Seeding complete. Ready for search.\n")


if __name__ == "__main__":
    main()
