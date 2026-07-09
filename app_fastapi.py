"""
FastAPI backend for the Obsidian RAG System.

Exposes the compiled LangGraph pipeline over HTTP for external clients
(the Obsidian plugin, curl/Postman for testing).

- On startup: inizializza_sistema() runs once (in the lifespan handler)
  and the compiled graph is kept in app.state for the process lifetime.
- Per request: /chat reuses pipeline.rispondi() with the cached graph and
  the request's thread_id.

thread_id contract: if the client sends one, it's reused as-is (the
checkpointer restores that conversation's memory). If not (first message
of a new conversation), the server generates one and returns it; the
client must store it and resend it on later turns.

Run: uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
     (or: python app_fastapi.py)

New deps: pip install fastapi "uvicorn[standard]"
"""

import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from settings import VAULT_PATH
from pipeline import inizializza_sistema, rispondi
from sync_vault import sync_vault


# Serializes sync runs: sync_vault() mutates ChromaDB and rewrites
# chunks_cache.pkl, which is unsafe if two syncs overlap. acquire(blocking
# =False) below rejects a concurrent sync with 409 instead of queuing it.
_sync_lock = threading.Lock()


# ---------------------------------------------------------------------
# Lifespan: startup/shutdown
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once before the server accepts requests; stores the compiled graph in app.state."""
    print("Inizializzazione pipeline (carico indice + retriever + grafo LangGraph)...")
    app.state.grafo_compilato = inizializza_sistema(VAULT_PATH)
    print("Pipeline pronta. Il server ora accetta richieste.")

    yield

    print("Server in spegnimento.")


app = FastAPI(
    title="Obsidian RAG System — Backend",
    description="HTTP API around the LangGraph pipeline.",
    lifespan=lifespan,
)

# CORS: the Obsidian plugin (Electron) sends requests from "app://obsidian.md".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["app://obsidian.md", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None  # None on the first message of a new conversation
    # Demo mode: when True, excludes private folders (CARTELLE_PROTETTE) from context.
    modalita_protetta: bool = False


class ChatResponse(BaseModel):
    risposta: str
    stop_reason: str | None
    thread_id: str  # client must resend this on the next turn


class SyncResponse(BaseModel):
    nuovi: int
    modificati: int
    eliminati: int
    invariati: int
    chunk_totali: int


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    """Quick check that the server is up and the graph is loaded."""
    return {"status": "ok", "grafo_caricato": hasattr(app.state, "grafo_compilato")}


# 'def' not 'async def': rispondi() is blocking (calls Ollama/Anthropic
# synchronously), so FastAPI runs it in a threadpool and keeps the event
# loop free for /health and other requests.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="La query è vuota.")

    thread_id = req.thread_id or str(uuid.uuid4())

    try:
        risposta, stop_reason = rispondi(
            query, app.state.grafo_compilato, thread_id, req.modalita_protetta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella pipeline: {e}")

    if risposta is None:
        raise HTTPException(status_code=500, detail="Il grafo non ha prodotto una risposta.")

    return ChatResponse(risposta=risposta, stop_reason=stop_reason, thread_id=thread_id)


# 'def' not 'async def', same reason as /chat: sync_vault() and
# inizializza_sistema() are blocking.
@app.post("/sync", response_model=SyncResponse)
def sync():
    """Re-syncs the vault (ChromaDB + chunks cache) and reloads the graph, without restarting the server."""
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Un aggiornamento è già in corso. Attendi che finisca.",
        )
    try:
        riepilogo = sync_vault()
        app.state.grafo_compilato = inizializza_sistema(VAULT_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il sync: {e}")
    finally:
        _sync_lock.release()

    return SyncResponse(**riepilogo)


if __name__ == "__main__":
    # 127.0.0.1: local only, matches the plugin running on the same machine.
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
