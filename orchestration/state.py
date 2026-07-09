"""
Shared LangGraph state. Each node receives a RAGState and returns a
partial dict with only the keys it wants to update; LangGraph merges it.
NotRequired marks fields that don't exist until a specific node has run.
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langchain_core.documents import Document


class RAGState(TypedDict):
    # --- Query & contextualization ---
    query_originale: str
    # Query after the contextualization node rewrites it using chat_history
    # (e.g. "and that note from yesterday?" -> "What did I write yesterday
    # about the RAG project?"). Equals query_originale if no rewrite is needed.
    query_effettiva: NotRequired[str]

    # --- Conversation memory ---
    # operator.add: each node appends to the list instead of overwriting it.
    # Element format: {"ruolo": "utente" | "assistente", "contenuto": str}
    chat_history: Annotated[list[dict], operator.add]

    # --- Protected mode (demo mode) ---
    # True when the request comes from the plugin with protection enabled:
    # the rerank node drops chunks from private folders (CARTELLE_PROTETTE)
    # before generation. Set once per invocation in stato_iniziale(); no
    # node rewrites it. Absent (-> treated as False) for callers that don't
    # pass it, e.g. the CLI loop in pipeline.py.
    modalita_protetta: NotRequired[bool]

    # --- Classification (query.classifier) ---
    tipo: NotRequired[str]  # "CONTENUTISTICA" or "TEMPORALE"
    data_inizio: NotRequired[str | None]
    data_fine: NotRequired[str | None]

    # --- Retrieval ---
    # Raw retrieval output (hybrid ensemble for CONTENUTISTICA, vectorstore.get()
    # for TEMPORALE), normalized to list[Document] by the node regardless of source.
    chunk_recuperati: NotRequired[list[Document]]

    # Chunks added by graph expansion (espandi_vicini for CONTENUTISTICA,
    # the frequency-based variant for TEMPORALE).
    chunk_espansi: NotRequired[list[Document]]

    # Deduplicated, ordered union of chunk_recuperati + chunk_espansi;
    # what actually reaches formatta_contesto().
    chunk_rerankati: NotRequired[list[Document]]

    # --- CRAG cycle (grading + query rewriting) ---
    # Dormant: no active node writes these fields anymore (see the
    # "IN PAUSA" note in nodes.py). Kept in the schema in case the
    # pattern is reintroduced with a more reliable grading model.
    grading_esito: NotRequired[str]  # "sufficiente" | "insufficiente"
    retry_count: int  # starts at 0, no longer incremented by any active node
    contesto_incompleto: NotRequired[bool]

    # --- Generation ---
    contesto_formattato: NotRequired[str]
    risposta: NotRequired[str]
    stop_reason: NotRequired[str]


def stato_iniziale(query: str, modalita_protetta: bool = False) -> RAGState:
    """Builds the starting state for a new graph invocation. chat_history
    isn't force-reset here: if the graph runs with a checkpointer and the
    same thread_id, LangGraph restores it from the prior turn — this
    default only applies to a thread's first turn.

    modalita_protetta is per-invocation, not per-thread (it reflects the
    plugin toggle at the moment the question is asked), so it's set fresh
    every turn and isn't persisted like chat_history.
    """
    return {
        "query_originale": query,
        "chat_history": [],
        "retry_count": 0,
        "modalita_protetta": modalita_protetta,
    }
