"""
FasalSetu — Government Scheme Document Ingestor
Populates ChromaDB from PDF files in data/schemes/.

Usage:
  python scripts/ingest_schemes.py --pdf-dir data/schemes/
  python scripts/ingest_schemes.py --pdf-dir data/schemes/ --verify
  python scripts/ingest_schemes.py --pdf-dir data/schemes/ --reset
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Seed data so the app works even without any PDFs
SEED_SCHEMES = [
    {
        "scheme_name": "PM-KISAN",
        "state": "central",
        "benefit": "₹6,000/year in 3 installments of ₹2,000",
        "eligibility": "All landholding farmer families with cultivable land",
        "apply_url": "https://pmkisan.gov.in",
        "text": (
            "PM Kisan Samman Nidhi provides income support of Rs 6000 per year to all landholding "
            "farmer families. The amount is transferred directly to bank accounts in three installments "
            "of Rs 2000 each. Small and marginal farmers with less than 2 hectares land are primary "
            "beneficiaries. Apply online at pmkisan.gov.in or through Common Service Centres."
        ),
    },
    {
        "scheme_name": "PMFBY - Pradhan Mantri Fasal Bima Yojana",
        "state": "central",
        "benefit": "Crop insurance with premium as low as 2% for kharif, 1.5% for rabi",
        "eligibility": "All farmers growing notified crops. Mandatory for loanee farmers.",
        "apply_url": "https://pmfby.gov.in",
        "text": (
            "PMFBY provides financial support to farmers suffering crop loss due to unforeseen events "
            "like natural calamities, pests and diseases. Premium rates are 2% for kharif crops, "
            "1.5% for rabi crops, and 5% for annual commercial/horticultural crops. Remaining premium "
            "is shared equally by central and state governments. Register through bank or CSC."
        ),
    },
    {
        "scheme_name": "Kisan Credit Card (KCC)",
        "state": "central",
        "benefit": "Short-term credit up to ₹3 lakh at 7% interest (4% with subsidy)",
        "eligibility": "All farmers, tenant farmers, oral lessees, sharecroppers, SHG members",
        "apply_url": "https://www.nabard.org/content1.aspx?id=596",
        "text": (
            "Kisan Credit Card provides flexible credit for agricultural and allied activities. "
            "Credit limit is based on land holding and crop cultivation costs. Interest subvention "
            "of 3% available for prompt repayment. Covers crop cultivation, post-harvest expenses, "
            "farm maintenance, allied activities like fishery and animal husbandry."
        ),
    },
    {
        "scheme_name": "PM Kusum Yojana",
        "state": "central",
        "benefit": "Subsidy up to 60% on solar pumps; sell surplus power to grid",
        "eligibility": "Individual farmers, cooperatives, panchayats, Water User Associations",
        "apply_url": "https://mnre.gov.in/solar/schemes",
        "text": (
            "PM-KUSUM (Pradhan Mantri Kisan Urja Suraksha evem Utthan Mahabhiyan) aims to provide "
            "energy and water security to farmers. Component A: 10,000 MW decentralized solar plants. "
            "Component B: standalone solar agriculture pumps. Component C: solarization of grid-connected "
            "pumps. Farmers can earn extra income by selling surplus power."
        ),
    },
    {
        "scheme_name": "Soil Health Card Scheme",
        "state": "central",
        "benefit": "Free soil testing and nutrient recommendations for each farm plot",
        "eligibility": "All farmers. Test cycle: once in 2 years",
        "apply_url": "https://soilhealth.dac.gov.in",
        "text": (
            "Soil Health Card scheme provides farmers with information on soil nutrient status and "
            "fertilizer recommendations to improve productivity. Cards issued once in two years. "
            "Contains information on 12 parameters: NPK macro, secondary nutrients (S), and micro "
            "nutrients (B, Fe, Mn, Zn, Cu, Mo). Apply at nearest Krishi Vigyan Kendra or agriculture office."
        ),
    },
    {
        "scheme_name": "eNAM - National Agriculture Market",
        "state": "central",
        "benefit": "Online trading platform to get best price across mandis",
        "eligibility": "Farmers registered with local APMC mandi",
        "apply_url": "https://enam.gov.in",
        "text": (
            "eNAM is a pan-India electronic trading portal that networks existing APMC mandis to create "
            "a unified national market. Farmers can upload produce quality data and get price discovery "
            "from buyers across India. Over 1000 mandis in 18 states integrated. Register through local "
            "mandi office with land documents and bank account."
        ),
    },
    {
        "scheme_name": "NABARD Agricultural Loans",
        "state": "central",
        "benefit": "Refinance support to banks for agricultural credit at subsidized rates",
        "eligibility": "Via cooperative banks, RRBs, commercial banks",
        "apply_url": "https://www.nabard.org",
        "text": (
            "NABARD provides refinance support to banks and financial institutions for agricultural "
            "and rural development. Farmers can access credit through cooperative banks, regional rural "
            "banks and commercial banks. Schemes cover crop loans, term loans for equipment, land "
            "development, plantation and horticulture, storage and warehousing."
        ),
    },
    {
        "scheme_name": "Pradhan Mantri Krishi Sinchai Yojana (PMKSY)",
        "state": "central",
        "benefit": "Subsidy for micro-irrigation (drip/sprinkler): up to 55% for small farmers",
        "eligibility": "All farmers; priority to water-scarce districts",
        "apply_url": "https://pmksy.gov.in",
        "text": (
            "PMKSY aims to ensure access to irrigation for every farm and improve water use efficiency. "
            "Per Drop More Crop component provides subsidy on drip and sprinkler irrigation systems. "
            "Small and marginal farmers get 55% subsidy, others get 45%. Watershed development and "
            "Har Khet Ko Pani components address water source creation."
        ),
    },
    {
        "scheme_name": "Paramparagat Krishi Vikas Yojana (PKVY)",
        "state": "central",
        "benefit": "₹50,000/hectare for 3 years to convert to organic farming",
        "eligibility": "Farmer groups of minimum 50 farmers covering 50 acres",
        "apply_url": "https://pgsindia-ncof.gov.in",
        "text": (
            "PKVY promotes organic farming through cluster approach. Farmer groups of 50 farmers "
            "form clusters covering 50 acres. Financial assistance of Rs 50,000 per hectare for "
            "3 years covers inputs, certification, and marketing. PGS-India certification provided "
            "free. Organic produce gets premium pricing access through eNAM and organic markets."
        ),
    },
    {
        "scheme_name": "Rashtriya Krishi Vikas Yojana (RKVY)",
        "state": "central",
        "benefit": "Grants for agricultural infrastructure, innovation, agri-startups",
        "eligibility": "State governments, FPOs, agri-startups via RAFTAAR programme",
        "apply_url": "https://rkvy.nic.in",
        "text": (
            "RKVY provides flexibility to states to plan and execute agricultural programmes based "
            "on local needs. RAFTAAR sub-scheme supports agri-entrepreneurship with seed funding "
            "of Rs 25 lakh for agri-startups. Supports infrastructure like cold chains, warehouses, "
            "soil testing labs, and precision farming technology adoption."
        ),
    },
]


def get_chroma_collection():
    try:
        import chromadb
        from pathlib import Path as P
        db_path = str(P(__file__).parent.parent / "data" / "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection("govt_schemes")
        return collection
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)


def get_embedder():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        print("WARNING: langchain-community or sentence-transformers not installed.")
        print("  Run: pip install langchain-community sentence-transformers")
        print("  Falling back to simple hash-based IDs (no semantic search).")
        return None


def ingest_seed_data(collection, embedder, reset: bool = False):
    print("\n[1] Ingesting seed scheme data...")

    if reset:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  Cleared {len(existing['ids'])} existing documents.")

    docs, ids, metadatas, embeddings_list = [], [], [], []

    for scheme in SEED_SCHEMES:
        doc_id = f"seed_{scheme['scheme_name'].replace(' ', '_').replace('(', '').replace(')', '')[:40]}"
        if doc_id in (collection.get(ids=[doc_id])["ids"] if not reset else []):
            print(f"  Skipping (already exists): {scheme['scheme_name']}")
            continue

        docs.append(scheme["text"])
        ids.append(doc_id)
        metadatas.append({
            "scheme_name": scheme["scheme_name"],
            "state": scheme["state"],
            "benefit": scheme["benefit"],
            "eligibility": scheme["eligibility"],
            "apply_url": scheme["apply_url"],
            "source": "seed_data",
        })

        if embedder:
            embeddings_list.append(embedder.embed_query(scheme["text"]))

    if docs:
        if embedder:
            collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embeddings_list)
        else:
            collection.add(documents=docs, ids=ids, metadatas=metadatas)
        print(f"  Ingested {len(docs)} seed schemes.")
    else:
        print("  All seed schemes already present.")


def ingest_pdfs(collection, embedder, pdf_dir: str):
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"\n[2] PDF directory not found: {pdf_dir} — skipping PDF ingestion.")
        return

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[2] No PDF files found in {pdf_dir} — skipping.")
        return

    try:
        import fitz  # pymupdf
    except ImportError:
        print("\n[2] pymupdf not installed. Run: pip install pymupdf")
        print("  Skipping PDF ingestion.")
        return

    print(f"\n[2] Ingesting {len(pdf_files)} PDF files from {pdf_dir}...")

    for pdf_file in pdf_files:
        print(f"  Processing: {pdf_file.name}")
        try:
            doc = fitz.open(str(pdf_file))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()

            if len(text.strip()) < 100:
                print(f"    WARNING: Very little text extracted — may be scanned PDF.")

            # Chunk with 100-char overlap
            chunk_size = 500
            overlap = 100
            chunks = []
            i = 0
            while i < len(text):
                chunks.append(text[i:i + chunk_size])
                i += chunk_size - overlap

            state = "central"
            # Infer state from filename e.g. "maharashtra_schemes.pdf"
            for state_name in ["maharashtra", "punjab", "uttar_pradesh", "rajasthan",
                                "madhya_pradesh", "karnataka", "gujarat", "bihar"]:
                if state_name in pdf_file.stem.lower():
                    state = state_name.replace("_", " ").title()
                    break

            docs = chunks
            ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "scheme_name": pdf_file.stem.replace("_", " ").title(),
                "state": state,
                "benefit": "See document",
                "eligibility": "See document",
                "apply_url": "",
                "source": pdf_file.name,
            } for _ in chunks]

            if embedder:
                embs = [embedder.embed_query(c) for c in chunks]
                collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embs)
            else:
                collection.add(documents=docs, ids=ids, metadatas=metadatas)

            print(f"    Ingested {len(chunks)} chunks.")

        except Exception as e:
            print(f"    ERROR processing {pdf_file.name}: {e}")


def verify(collection):
    print("\n[3] Verification...")
    total = collection.count()
    print(f"  Total documents in ChromaDB: {total}")

    sample = collection.get(limit=3)
    print(f"  Sample IDs: {sample['ids']}")
    for meta in sample["metadatas"]:
        print(f"    → {meta.get('scheme_name')} [{meta.get('state')}]")

    # Test query
    test_queries = ["solar pump subsidy", "crop insurance", "organic farming"]
    print("\n  Test queries:")
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=1)
        if results["documents"][0]:
            match = results["metadatas"][0][0].get("scheme_name", "?")
            print(f"    '{q}' → {match}")
        else:
            print(f"    '{q}' → no results")

    print("\n  ✓ ChromaDB ready for use.")


def main():
    parser = argparse.ArgumentParser(description="Ingest government scheme documents into ChromaDB")
    parser.add_argument("--pdf-dir", default="data/schemes", help="Directory containing scheme PDFs")
    parser.add_argument("--verify", action="store_true", help="Run verification after ingestion")
    parser.add_argument("--reset", action="store_true", help="Clear existing data before ingesting")
    args = parser.parse_args()

    print("=" * 55)
    print("  FASALSETU — SCHEME DOCUMENT INGESTOR")
    print("=" * 55)

    collection = get_chroma_collection()
    embedder = get_embedder()

    ingest_seed_data(collection, embedder, reset=args.reset)
    ingest_pdfs(collection, embedder, args.pdf_dir)

    if args.verify:
        verify(collection)

    print("\n  Done. ChromaDB is ready.")
    print(f"  Location: data/chroma_db/")


if __name__ == "__main__":
    main()
