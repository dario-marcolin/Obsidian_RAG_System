"""
Central configuration for the Obsidian RAG System.
Tuning parameters (paths, models, hyperparameters) live here.
Secrets (API keys) are never hardcoded — they are read from env vars.
"""

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Incremental sync state: {absolute_path: {hash, mtime}} from the last
# successful run. Used by sync_vault.py to detect new/modified/deleted files.
SYNC_STATE_PATH = BASE_DIR / "sync_state.json"

# Obsidian vault folder to index.
VAULT_PATH = Path("/Users/dariomarcolin/Desktop/Dario's obsidian")

# Vault subfolder for personal journal entries, used to auto-filter
# generic temporal queries (e.g. "what did I write in June?").
RIFLESSIONI_FOLDER = "💭Riflessioni"

# --- Protected mode (demo mode) -------------------------------------------
# Folders considered "private": when a request sets modalita_protetta=True,
# chunks from these folders are dropped from context before generation,
# on both retrieval branches (temporal and content-based).
CARTELLE_PROTETTE = {RIFLESSIONI_FOLDER}

# Where ChromaDB persists the vector index on disk.
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Cache of already-processed chunks (avoids re-running loader+metadata+splitter
# on every pipeline.py start). Must be regenerated together with chroma_db.
CHUNKS_CACHE_PATH = BASE_DIR / "chunks_cache.pkl"


# --- Ingestion / Text Splitting -------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Tag extraction pattern (#tag, #tag/subtag)
TAG_PATTERN = r"#[\w/]+"
# Excludes false positives like hex colors (#FFF3A3A6)
HEX_COLOR_PATTERN = r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$"


# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL = "nomic-embed-text"  # served locally via Ollama


# --- Retrieval ---------------------------------------------------------------
RETRIEVER_K = 4  # number of chunks returned by each retriever

# EnsembleRetriever weights: [BM25, vector similarity]
ENSEMBLE_WEIGHTS = [0.4, 0.6]

ITALIAN_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "è", "e", "o", "ma", "che", "chi", "cui", "cosa", "come",
    "quando", "dove", "perché", "se", "non", "più", "anche",
    "questo", "questa", "quello", "quella", "si", "ci", "ne",
    "del", "della", "dei", "delle", "al", "alla", "dai", "dalle",
    "sono", "essere", "ha", "hanno", "cos",
}


# --- Query Classification -----------------------------------------------
CLASSIFIER_MODEL = "llama3.2:3b"  # via Ollama, forced JSON output


# --- Generation --------------------------------------------------------------
# Local model (fallback / testing), via Ollama.
LOCAL_GENERATION_MODEL = "llama3.2:3b"
LOCAL_GENERATION_TEMPERATURE = 0.1
LOCAL_GENERATION_MAX_TOKENS = 300  # maps to num_predict for Ollama

# Anthropic API model (primary generation).
ANTHROPIC_GENERATION_MODEL = "claude-sonnet-5"
ANTHROPIC_MAX_TOKENS = 1500

# API key is never hardcoded; set it before running any script, e.g.:
#   export ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if ANTHROPIC_API_KEY is None:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found in environment variables. "
        "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
    )

# --- Graph-aware retrieval --------------------------------------------------
GRAPH_HOP_LIMIT = 1  # wikilink graph expansion depth

# --- LangGraph parameters ---------------------------------------------------
MAX_RETRY = 2  # max rewriting attempts on insufficient grading

# --- Graph expansion for TEMPORAL queries -----------------------------------
SOGLIA_FREQUENZA_TEMPORALE = 2

# --- Context size control ----------------------------------------------------
MAX_CHUNK_CONTESTO = 20      # cap on chunks (after rerank) passed to grading/generation
GRADING_EXTRACT_CHARS = 150  # excerpt length per chunk in the grading context

# --- Default time window per mentioned time unit -----------------------------
# When data_inizio is missing, the applied window depends on the time unit
# actually mentioned in the query (days/week/month/year).
FINESTRA_GIORNI_PER_UNITA_GIORNO = 3
FINESTRA_GIORNI_PER_UNITA_SETTIMANA = 7
FINESTRA_GIORNI_PER_UNITA_MESE = 30
FINESTRA_GIORNI_PER_UNITA_ANNO = 365

# Conservative fallback for TEMPORAL queries that don't mention any of the
# units above explicitly (e.g. "recently").
FINESTRA_TEMPORALE_DEFAULT_GIORNI = 14

# --- Debug -------------------------------------------------------------------
DEBUG_GRAFO = True  # prints state after every node when running pipeline.py

# --- Conversation memory persistence -----------------------------------------
# SQLite file where the LangGraph checkpointer stores conversation checkpoints
# (chat_history per thread_id). Lives alongside chroma_db and chunks_cache.pkl
# as system state, not code. Deleting it wipes all past conversations (the
# vault index in chroma_db is unaffected).
CHECKPOINTS_DB_PATH = BASE_DIR / "checkpoints.sqlite"
