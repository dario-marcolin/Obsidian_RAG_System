"""Ollama embedding model factory."""

from langchain_ollama import OllamaEmbeddings
from settings import EMBEDDING_MODEL


def embedding_ollama():
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )
    return embedding


if __name__ == "__main__":
    model = embedding_ollama()
    test_vector = model.embed_query("test di embedding")
    print(f"Dimensione vettore: {len(test_vector)}")
