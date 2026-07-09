"""Hybrid retrieval: BM25 (keyword) + vector similarity ensemble, plus exact-metadata date filtering."""

import re

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from settings import ITALIAN_STOPWORDS, RETRIEVER_K, ENSEMBLE_WEIGHTS, RIFLESSIONI_FOLDER


def preprocess_italian(text: str) -> list[str]:
    """Tokenizes text and strips Italian stopwords, for the BM25 retriever."""
    text = text.lower()
    tokens = re.findall(r"\b[a-zàèéìòù]+\b", text)
    return [t for t in tokens if t not in ITALIAN_STOPWORDS]


def create_bm25_retriever(chunks):
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=preprocess_italian)
    bm25_retriever.k = RETRIEVER_K
    return bm25_retriever


def create_vector_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


def create_ensemble_retriever(chunks, vectorstore):
    bm25_retriever = create_bm25_retriever(chunks)
    vector_retriever = create_vector_retriever(vectorstore)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )


def recupera_per_data(vectorstore, data_inizio: str, data_fine: str):
    """Exact metadata filter for generic TEMPORAL queries (not semantic search).
    Also filters to the Riflessioni folder, since untargeted temporal queries
    are typically journal-like rather than academic. Dates are ISO
    "YYYY-MM-DD", converted to int YYYYMMDD for Chroma's $gte/$lte operators."""
    inizio_int = int(data_inizio.replace("-", ""))
    fine_int = int(data_fine.replace("-", ""))

    return vectorstore.get(
        where={
            "$and": [
                {"creation_date": {"$gte": inizio_int}},
                {"creation_date": {"$lte": fine_int}},
                {"cartella": {"$eq": RIFLESSIONI_FOLDER}},
            ]
        }
    )

if __name__ == "__main__":
    from settings import VAULT_PATH
    from ingestion.loader import document_loader
    from ingestion.metadata import estrai_metadati_documenti
    from ingestion.splitter import split_documents
    from retrieval.embeddings import embedding_ollama
    from retrieval.vectorstore import load_vectorstore
    from pathlib import Path

    documents = document_loader(VAULT_PATH)
    documents = estrai_metadati_documenti(documents)
    chunks = split_documents(documents)

    embedding_model = embedding_ollama()
    vectorstore = load_vectorstore(embedding_model)
    import pickle
    from settings import CHUNKS_CACHE_PATH

    with open(CHUNKS_CACHE_PATH, "rb") as f:
        chunks_cache = pickle.load(f)

    print(f"Chunk in cache (pickle): {len(chunks_cache)}")
    print(f"Chunk generati ora da zero: {len(chunks)}")
    print(f"Documenti totali nel vectorstore persistito: {vectorstore._collection.count()}")

    query = "Cos'è il clustering?"

    # --- 1. Solo BM25, isolato ---
    print("=== Solo BM25 ===")
    bm25_retriever = create_bm25_retriever(chunks)
    for doc in bm25_retriever.invoke(query):
        nome = Path(doc.metadata["source"]).stem
        print(f"  {nome}")

    # --- 2. Solo vector search, isolato, CON punteggio di distanza ---
    print("\n=== Solo vector search (con distanza) ===")
    risultati_con_score = vectorstore.similarity_search_with_score(query, k=RETRIEVER_K)
    for doc, score in risultati_con_score:
        nome = Path(doc.metadata["source"]).stem
        print(f"  {nome}: distanza={score:.4f}")

    # --- 3. Ensemble completo, per confronto ---
    print("\n=== Ensemble (BM25 + vector combinati) ===")
    ensemble_retriever = create_ensemble_retriever(chunks, vectorstore)
    for doc in ensemble_retriever.invoke(query):
        nome = Path(doc.metadata["source"]).stem
        print(f"  {nome}")