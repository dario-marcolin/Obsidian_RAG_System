"""Loads all markdown files from the vault directory."""

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from settings import VAULT_PATH


def document_loader(file):
    loader = DirectoryLoader(file, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, recursive=True)
    loaded_document = loader.load()
    return loaded_document

if __name__ == "__main__":
    # run with: python -m ingestion.loader
    documents = document_loader(VAULT_PATH)
    print(documents[0])