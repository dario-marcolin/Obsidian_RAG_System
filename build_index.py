"""Builds the vector index from scratch: load vault -> metadata -> chunks -> embeddings -> ChromaDB."""

from settings import VAULT_PATH, CHROMA_PERSIST_DIR, CHUNKS_CACHE_PATH

from ingestion.loader import document_loader
from ingestion.metadata import estrai_metadati_documenti
from ingestion.splitter import split_documents
from retrieval.embeddings import embedding_ollama
from retrieval.vectorstore import build_vectorstore
import pickle


def build_index():
    documents = document_loader(VAULT_PATH)
    print(f"Documenti caricati: {len(documents)}")

    documents = estrai_metadati_documenti(documents)
    print("Metadati estratti dai documenti.")

    chunks = split_documents(documents)
    print(f"Documenti suddivisi in {len(chunks)} chunk.")

    # Cache chunks so pipeline.py doesn't redo loader+metadata+splitter
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunk salvati in cache: {CHUNKS_CACHE_PATH}")

    chunk_senza_tags = [c for c in chunks if "tags" not in c.metadata]
    chunk_senza_cartella = [c for c in chunks if "cartella" not in c.metadata]
    print(f"Chunk senza tags: {len(chunk_senza_tags)}")
    print(f"Chunk senza cartella: {len(chunk_senza_cartella)}")

    embedding_model = embedding_ollama()
    print("Modello di embedding creato.")

    vectorstore = build_vectorstore(chunks, embedding_model)
    print(f"Indice vettoriale costruito e salvato in: {CHROMA_PERSIST_DIR}")

    return vectorstore


if __name__ == "__main__":
    vectorstore = build_index()
    risultati = vectorstore.similarity_search("test", k=2)
    print(risultati)
