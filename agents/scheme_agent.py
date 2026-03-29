import chromadb
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path

_chroma = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "data" / "chroma_db"))
_collection = _chroma.get_or_create_collection("govt_schemes")
_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@tool
def find_govt_schemes(query: str, state: str = "general") -> dict:
    """
    Search government agricultural schemes relevant to farmer's situation.
    Args: query (farmer's need), state (Indian state name for state-specific schemes)
    Returns: matching schemes with eligibility, benefit amount, how to apply.
    """
    search_query = f"{query} {state} farmer scheme subsidy"
    embedding = _embedder.embed_query(search_query)

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=3,
        where={"state": {"$in": [state, "central"]}}  # central = national schemes
    )

    schemes = []
    for i, doc in enumerate(results["documents"][0]):
        schemes.append({
            "scheme": results["metadatas"][0][i].get("scheme_name", "Unknown"),
            "benefit": results["metadatas"][0][i].get("benefit", "See document"),
            "eligibility": results["metadatas"][0][i].get("eligibility", ""),
            "apply_at": results["metadatas"][0][i].get("apply_url", ""),
            "excerpt": doc[:300]
        })

    return {"schemes": schemes, "query": query, "state": state}


def ingest_scheme_documents(pdf_dir: str):
    """Run once to populate ChromaDB from PDF files."""
    import fitz  # pymupdf
    import os

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(pdf_dir, fname))
        text = "\n".join(page.get_text() for page in doc)
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # overlap 100 chars

        _collection.add(
            documents=chunks,
            ids=[f"{fname}_{i}" for i in range(len(chunks))],
            metadatas=[{
                "source": fname,
                "state": "central",  # override per file as needed
                "scheme_name": fname.replace(".pdf", "").replace("_", " ")
            } for _ in chunks]
        )
    print(f"Ingested {pdf_dir} into ChromaDB")