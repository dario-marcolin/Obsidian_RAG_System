# Obsidian RAG System

A Retrieval-Augmented Generation system for querying an Obsidian vault in natural language. It combines hybrid retrieval (BM25 + vector search), wikilink-graph expansion, and query classification (topic-based vs. date-based) to answer questions about your notes, orchestrated as a LangGraph pipeline with persistent conversation memory.

## Features

- **Hybrid retrieval**: BM25 keyword search + ChromaDB vector similarity, combined via an ensemble retriever.
- **Wikilink graph expansion**: retrieved notes are expanded through their `[[wikilinks]]` to pull in related context.
- **Query classification**: distinguishes topic questions ("What is clustering?") from date-range questions ("What did I write last week?"), each handled by a dedicated retrieval branch.
- **Conversational memory**: implicit references ("and that note from yesterday?") are resolved using chat history; conversations persist across server restarts via a SQLite-backed LangGraph checkpointer.
- **Incremental sync**: `sync_vault.py` re-indexes only new/modified/deleted files instead of rebuilding the whole index.
- **Two frontends**: a Gradio debug UI and a FastAPI backend (for the Obsidian plugin / external clients).
- **Protected mode**: an optional per-request flag excludes private folders (e.g. a personal journal) from the retrieved context.

## Architecture

```
ingestion/      load markdown files, extract metadata (tags, links, dates), split into chunks
retrieval/      embeddings, ChromaDB vectorstore, hybrid (BM25+vector) retriever, wikilink graph
query/          classifies each query as topic-based or date-based
orchestration/  LangGraph state, nodes, and graph wiring the pipeline together
generation/     prompt construction and the Anthropic API client
pipeline.py     builds the pipeline and exposes rispondi() to run a query
build_index.py  builds the vector index from scratch
sync_vault.py   incremental re-index of changed files
app_gradio.py   local debug UI
app_fastapi.py  HTTP backend (used by the Obsidian plugin)
```

Query flow: contextualize (resolve references from chat history) → classify (topic vs. date) → retrieve (hybrid search or date filter) → expand via wikilink graph → rerank/dedupe → format context → generate answer (Claude) → update memory.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with these models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings
  ollama pull llama3.2:3b        # query classification / contextualization
  ```
- An Anthropic API key (used for final answer generation)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
```

Set `VAULT_PATH` in [settings.py](settings.py) to point to your Obsidian vault.

## Usage

Build the index (first run, or full rebuild):
```bash
python build_index.py
```

Incrementally re-index after editing the vault:
```bash
python sync_vault.py
```

Run the debug UI:
```bash
python app_gradio.py
```

Run the HTTP backend:
```bash
python app_fastapi.py
# or: uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
```

Query from the command line:
```bash
python pipeline.py
```

## Notes

- `Test_Md_Docs/` (the test vault used during development) is not included in this repository — see `.gitignore`.
- Generated artifacts (`chroma_db/`, `chunks_cache.pkl`, `checkpoints.sqlite`, `sync_state.json`) are local state, rebuilt by `build_index.py` / `sync_vault.py`, and are not versioned.
