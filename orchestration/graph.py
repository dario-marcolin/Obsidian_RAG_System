"""
Builds the LangGraph pipeline: node topology and branching only, no
business logic (that lives in nodes.py). Conversation memory is
persisted via SqliteSaver in checkpoints.sqlite, surviving server restarts.

New dependency: pip install langgraph-checkpoint-sqlite
"""

import sqlite3
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from settings import CHECKPOINTS_DB_PATH
from .state import RAGState
from .nodes import (
    crea_nodo_contestualizzazione,
    crea_nodo_classificazione,
    crea_nodo_retrieve_ibrido,
    crea_nodo_recupera_temporale,
    crea_nodo_rerank,
    crea_nodo_formattazione,
    crea_nodo_generazione,
    crea_nodo_aggiorna_memoria,
)


# ---------------------------------------------------------------------
# Router — branch after classification
# ---------------------------------------------------------------------

def route_dopo_classificazione(state: RAGState) -> Literal["retrieve_ibrido", "recupera_temporale"]:
    """Reads state["tipo"] (set by the classification node) to pick the retrieval branch."""
    if state["tipo"] == "TEMPORALE" and state["data_inizio"] and state["data_fine"]:
        return "recupera_temporale"
    return "retrieve_ibrido"


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def costruisci_grafo(ensemble_retriever, vectorstore, grafo_note, mappa_source):
    """Assembles the StateGraph. No CRAG cycle (grading + query rewriting)
    here — removed after tests showed it cost more than it helped with the
    local model available (see the "IN PAUSA" note in nodes.py)."""
    builder = StateGraph(RAGState)

    builder.add_node("contestualizza_query", crea_nodo_contestualizzazione())
    builder.add_node("classifica_query", crea_nodo_classificazione())
    builder.add_node(
        "retrieve_ibrido",
        crea_nodo_retrieve_ibrido(ensemble_retriever, vectorstore, grafo_note, mappa_source),
    )
    builder.add_node(
        "recupera_temporale",
        crea_nodo_recupera_temporale(vectorstore, grafo_note, mappa_source),
    )
    builder.add_node("rerank_chunk", crea_nodo_rerank())
    builder.add_node("formatta_contesto", crea_nodo_formattazione())
    builder.add_node("genera_risposta", crea_nodo_generazione())
    builder.add_node("aggiorna_memoria", crea_nodo_aggiorna_memoria())

    builder.add_edge(START, "contestualizza_query")
    builder.add_edge("contestualizza_query", "classifica_query")

    # Both retrieval branches converge on the same rerank node
    builder.add_edge("retrieve_ibrido", "rerank_chunk")
    builder.add_edge("recupera_temporale", "rerank_chunk")

    builder.add_edge("rerank_chunk", "formatta_contesto")
    builder.add_edge("formatta_contesto", "genera_risposta")
    builder.add_edge("genera_risposta", "aggiorna_memoria")
    builder.add_edge("aggiorna_memoria", END)

    builder.add_conditional_edges(
        "classifica_query",
        route_dopo_classificazione,
        {
            "retrieve_ibrido": "retrieve_ibrido",
            "recupera_temporale": "recupera_temporale",
        },
    )

    # check_same_thread=False: FastAPI runs 'def' endpoints in a threadpool,
    # so this connection is shared across threads. SqliteSaver serializes
    # access internally with a lock, so disabling the check is safe here.
    conn = sqlite3.connect(str(CHECKPOINTS_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
