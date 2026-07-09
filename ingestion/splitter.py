"""Splits documents into chunks for embedding/indexing."""

from settings import CHUNK_SIZE, CHUNK_OVERLAP, VAULT_PATH
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loader import document_loader
from ingestion.metadata import estrai_metadati_documenti


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # ChromaDB metadata values can't be lists, so join tags into a string
    for c in chunks:
        tags_list = c.metadata.get("tags", [])
        c.metadata["tags"] = ", ".join(tags_list) if tags_list else "nessun_tag"

    return chunks


if __name__ == "__main__":
    documents = document_loader(VAULT_PATH)
    documents = estrai_metadati_documenti(documents)
    chunks = split_documents(documents)

    chunk_senza_tags = [c for c in chunks if "tags" not in c.metadata]
    chunk_senza_cartella = [c for c in chunks if "cartella" not in c.metadata]

    print(f"Chunk senza tags: {len(chunk_senza_tags)}")
    print(f"Chunk senza cartella: {len(chunk_senza_cartella)}")