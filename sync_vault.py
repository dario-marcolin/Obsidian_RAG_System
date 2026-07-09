"""
Incremental sync of the Obsidian vault: processes only files that are
new or modified since the last run, and removes deleted files from the
index, instead of rebuilding everything via build_index.py.

Updates:
    1. ChromaDB (vectorstore)  -> via delete(where=...) + add_documents()
    2. chunks_cache.pkl        -> used by BM25 and the wikilink graph,
                                   which don't read from ChromaDB
    3. sync_state.json         -> {path: {hash, mtime}} from the last sync

Modified files use delete+reinsert instead of upsert-by-id: if a file's
length changes, the number of generated chunks changes too, which would
leave orphaned chunks behind under a per-id upsert.

Known limit: does not hot-reload an already-running server process —
only updates what's persisted to disk. A running server must be
restarted (or call /sync) to pick up fresh data.

Usage:
    python sync_vault.py
"""

import hashlib
import json
import pickle
from pathlib import Path

from langchain_community.document_loaders import UnstructuredMarkdownLoader

from settings import VAULT_PATH, CHUNKS_CACHE_PATH, SYNC_STATE_PATH
from ingestion.metadata import estrai_metadati_documenti
from ingestion.splitter import split_documents
from retrieval.embeddings import embedding_ollama
from retrieval.vectorstore import load_vectorstore


# ---------------------------------------------------------------------
# Sync state (sync_state.json) and chunk cache (chunks_cache.pkl)
# ---------------------------------------------------------------------

def carica_stato() -> dict:
    """Returns {} if no state file exists yet (first run: every file is treated as new)."""
    if not SYNC_STATE_PATH.exists():
        return {}
    with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def salva_stato(stato: dict) -> None:
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(stato, f, indent=2, ensure_ascii=False)


def carica_cache() -> list:
    """Returns [] if no cache file exists yet (before any build_index.py run)."""
    if not CHUNKS_CACHE_PATH.exists():
        return []
    with open(CHUNKS_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def salva_cache(chunks: list) -> None:
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)


# ---------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------

def calcola_hash(path: Path) -> str:
    """MD5 of the raw file content — only used to detect changes, not for security."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def scansiona_vault(vault_path: Path) -> dict:
    """Returns {absolute_path_str: mtime} for every .md file currently in the vault."""
    return {str(p): p.stat().st_mtime for p in vault_path.rglob("*.md")}


def aggiungi_a_chroma_a_lotti(vectorstore, chunks: list, batch_size: int = 32) -> None:
    """Inserts chunks into ChromaDB in explicit batches to avoid one oversized request to Ollama."""
    totale_batch = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        numero_batch = i // batch_size + 1

        try:
            vectorstore.add_documents(batch)
            print(f"[sync]   batch {numero_batch}/{totale_batch} indicizzato ({len(batch)} chunk)")
        except Exception as e:
            print(f"[sync] ERRORE nel batch {numero_batch}/{totale_batch} (chunk {i}-{i+len(batch)}): {e}")
            raise


def processa_file(path_str: str) -> list:
    """Loader + metadata + splitter for a single file (same chain as build_index.py)."""
    documenti = UnstructuredMarkdownLoader(path_str).load()
    documenti = estrai_metadati_documenti(documenti)
    chunks = split_documents(documenti)
    return chunks


# ---------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------

def sync_vault() -> dict:
    """Runs the incremental sync and returns a summary dict (used by the /sync endpoint)."""
    stato = carica_stato()
    cache = carica_cache()

    embedding_model = embedding_ollama()
    vectorstore = load_vectorstore(embedding_model)

    file_correnti = scansiona_vault(VAULT_PATH)

    path_precedenti = set(stato.keys())
    path_attuali = set(file_correnti.keys())

    nuovi = path_attuali - path_precedenti
    eliminati = path_precedenti - path_attuali
    da_verificare = path_attuali & path_precedenti

    # Among already-seen files, isolate those actually modified: cheap
    # mtime pre-filter first, hash comparison only if mtime changed.
    modificati = []
    for path_str in da_verificare:
        mtime_attuale = file_correnti[path_str]
        mtime_salvato = stato[path_str].get("mtime")

        if mtime_attuale == mtime_salvato:
            continue

        hash_attuale = calcola_hash(Path(path_str))
        if hash_attuale != stato[path_str].get("hash"):
            modificati.append(path_str)
        else:
            # mtime changed but content is identical: just refresh mtime
            stato[path_str]["mtime"] = mtime_attuale

    # Index for quickly removing stale chunks from the cache
    cache_per_source = {}
    for i, chunk in enumerate(cache):
        cache_per_source.setdefault(chunk.metadata["source"], []).append(i)

    indici_da_rimuovere = set()
    chunk_da_aggiungere = []

    # --- Deleted files ---
    for path_str in eliminati:
        vectorstore.delete(where={"source": path_str})
        indici_da_rimuovere.update(cache_per_source.get(path_str, []))
        del stato[path_str]
        print(f"[sync] ELIMINATO:  {path_str}")

    # --- Modified files (delete + reinsert) ---
    for path_str in modificati:
        vectorstore.delete(where={"source": path_str})
        indici_da_rimuovere.update(cache_per_source.get(path_str, []))

        nuovi_chunk = processa_file(path_str)
        chunk_da_aggiungere.extend(nuovi_chunk)

        stato[path_str] = {
            "hash": calcola_hash(Path(path_str)),
            "mtime": file_correnti[path_str],
        }
        print(f"[sync] MODIFICATO: {path_str} ({len(nuovi_chunk)} chunk)")

    # --- New files ---
    for path_str in nuovi:
        nuovi_chunk = processa_file(path_str)
        chunk_da_aggiungere.extend(nuovi_chunk)

        stato[path_str] = {
            "hash": calcola_hash(Path(path_str)),
            "mtime": file_correnti[path_str],
        }
        print(f"[sync] NUOVO:      {path_str} ({len(nuovi_chunk)} chunk)")

    cache_aggiornata = [c for i, c in enumerate(cache) if i not in indici_da_rimuovere]
    cache_aggiornata.extend(chunk_da_aggiungere)

    if chunk_da_aggiungere:
        aggiungi_a_chroma_a_lotti(vectorstore, chunk_da_aggiungere)

    salva_cache(cache_aggiornata)
    salva_stato(stato)

    invariati = len(da_verificare) - len(modificati)
    print(
        f"\n[sync] Completato — nuovi: {len(nuovi)} | "
        f"modificati: {len(modificati)} | eliminati: {len(eliminati)} | "
        f"invariati: {invariati}"
    )
    print(f"[sync] Chunk totali in cache dopo il sync: {len(cache_aggiornata)}")

    return {
        "nuovi": len(nuovi),
        "modificati": len(modificati),
        "eliminati": len(eliminati),
        "invariati": invariati,
        "chunk_totali": len(cache_aggiornata),
    }


if __name__ == "__main__":
    sync_vault()
