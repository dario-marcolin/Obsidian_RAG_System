"""Builds and loads the ChromaDB vector store."""

from langchain_chroma import Chroma
from settings import CHROMA_PERSIST_DIR
import shutil
from pathlib import Path


def build_vectorstore(chunks, embedding_model, batch_size=32):
    """Builds the index from scratch (used by build_index.py), in batches
    to avoid overloading Ollama with one huge payload on large vaults."""
    if Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)

    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model,
    )

    totale_batch = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        numero_batch = i // batch_size + 1

        try:
            vectorstore.add_documents(batch)
            print(f"Batch {numero_batch}/{totale_batch} indicizzato ({len(batch)} chunk)")
        except Exception as e:
            print(f"ERRORE nel batch {numero_batch}/{totale_batch} (chunk {i}-{i+len(batch)}): {e}")
            raise

    return vectorstore


def load_vectorstore(embedding_model):
    """Loads an existing index from disk (used by pipeline.py)."""
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model,
    )

if __name__ == "__main__":
    from ingestion.loader import document_loader
    from ingestion.metadata import estrai_metadati_documenti
    from ingestion.splitter import split_documents
    from retrieval.embeddings import embedding_ollama
    from settings import VAULT_PATH

    documents = document_loader(VAULT_PATH)
    documents = estrai_metadati_documenti(documents)
    chunks = split_documents(documents)

    embedding_model = embedding_ollama()
    vectorstore = build_vectorstore(chunks, embedding_model)

    print(f"Vectorstore creato con {len(chunks)} chunk")