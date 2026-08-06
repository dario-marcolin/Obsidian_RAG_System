"""Builds the RAG index/retriever/graph and exposes rispondi() to run a query through the compiled LangGraph."""

from settings import VAULT_PATH, CHUNKS_CACHE_PATH, DEBUG_GRAFO
import pickle
import uuid
from pathlib import Path
from retrieval.embeddings import embedding_ollama
from retrieval.vectorstore import load_vectorstore
from retrieval.hybrid_retriever import create_ensemble_retriever
from retrieval.graph_retriever import costruisci_grafo, mappa_note_a_source
from orchestration.state import stato_iniziale
from orchestration.graph import costruisci_grafo as costruisci_grafo_langgraph


def initialize_pipeline(vault_path):
    """Loads the vector index, cached chunks, and builds the wikilink graph."""
    with open(CHUNKS_CACHE_PATH, "rb") as f:
        chunked_documents = pickle.load(f)

    embedding_model = embedding_ollama()
    vectorstore = load_vectorstore(embedding_model)
    ensemble_retriever = create_ensemble_retriever(chunked_documents, vectorstore)

    grafo_note = costruisci_grafo(chunked_documents)
    mappa_source = mappa_note_a_source(chunked_documents)

    return ensemble_retriever, vectorstore, grafo_note, mappa_source


def inizializza_sistema(vault_path):
    """Wraps initialize_pipeline() and compiles the LangGraph orchestration graph (with checkpointer)."""
    ensemble_retriever, vectorstore, grafo_note, mappa_source = initialize_pipeline(vault_path)
    grafo_compilato = costruisci_grafo_langgraph(ensemble_retriever, vectorstore, grafo_note, mappa_source)
    return grafo_compilato


def _stampa_debug_nodo(nome_nodo, valori):
    """Prints only the relevant fields produced by each node."""
    print(f"\n[DEBUG] --- nodo: {nome_nodo} ---")

    if "query_effettiva" in valori:
        print(f"[DEBUG] query_effettiva: {valori['query_effettiva']!r}")

    if "testo_nota_corrente" in valori:
        if valori["testo_nota_corrente"]:
            print(
                f"[DEBUG] nota_corrente: {valori['nome_nota_corrente']!r} | "
                f"{len(valori['testo_nota_corrente'])} caratteri (fonte primaria)"
            )
        else:
            print("[DEBUG] nota_corrente: nessuna (toggle spento, nota vuota o cartella protetta)")

    if "tipo" in valori:
        print(
            f"[DEBUG] tipo: {valori['tipo']} | "
            f"data_inizio: {valori.get('data_inizio')} | "
            f"data_fine: {valori.get('data_fine')}"
        )

    if "chunk_recuperati" in valori:
        note = {Path(c.metadata["source"]).stem for c in valori["chunk_recuperati"]}
        print(f"[DEBUG] chunk_recuperati: {len(valori['chunk_recuperati'])} | note: {note}")

    if "chunk_espansi" in valori:
        note = {Path(c.metadata["source"]).stem for c in valori["chunk_espansi"]}
        print(f"[DEBUG] chunk_espansi (via grafo): {len(valori['chunk_espansi'])} | note: {note}")

    if "chunk_rerankati" in valori:
        print(f"[DEBUG] chunk_rerankati (dopo dedup + tetto MAX_CHUNK_CONTESTO): {len(valori['chunk_rerankati'])}")

    if "contesto_formattato" in valori:
        print(f"[DEBUG] contesto_formattato: {len(valori['contesto_formattato'])} caratteri")

    if "stop_reason" in valori:
        print(f"[DEBUG] stop_reason: {valori['stop_reason']}")

    if "chat_history" in valori:
        print(f"[DEBUG] chat_history: +{len(valori['chat_history'])} messaggi accodati")


def rispondi(query, grafo_compilato, thread_id, modalita_protetta=False, nota_corrente=None, debug=None):
    """Runs the compiled graph for one query and returns
    (risposta, stop_reason, nota_corrente_usata).

    nota_corrente_usata reports whether the note the caller sent actually
    reached the context. It can be False even when nota_corrente was sent
    (protected folder, empty note), and only the graph knows which — the
    plugin can't tell, since CARTELLE_PROTETTE is backend config. Without
    this the rejection would be a silent no-op for the user.

    modalita_protetta (demo mode): when True, the rerank node drops chunks
    from private folders (CARTELLE_PROTETTE). Per-invocation flag set by
    the caller (the /chat endpoint reads it from the plugin request body).

    nota_corrente: the note open in Obsidian when the question was asked,
    as {"nome", "path", "contenuto"}, or None. When present it becomes the
    primary context section, ahead of anything retrieval finds. Also
    per-invocation, and also read from the plugin request body — the CLI
    loop below never sends one (there's no open editor to speak of).

    debug=None uses settings.DEBUG_GRAFO; pass True/False to override it
    for this call.
    """
    if debug is None:
        debug = DEBUG_GRAFO

    config = {"configurable": {"thread_id": thread_id}}

    risposta = None
    stop_reason = None
    nota_corrente_usata = False

    stato = stato_iniziale(query, modalita_protetta, nota_corrente)
    for aggiornamento in grafo_compilato.stream(stato, config, stream_mode="updates"):
        # aggiornamento is {"node_name": {fields_written_by_that_node}}
        for nome_nodo, valori in aggiornamento.items():
            if debug:
                _stampa_debug_nodo(nome_nodo, valori)
            if "risposta" in valori:
                risposta = valori["risposta"]
            if "stop_reason" in valori:
                stop_reason = valori["stop_reason"]
            if "testo_nota_corrente" in valori:
                nota_corrente_usata = bool(valori["testo_nota_corrente"])

    return risposta, stop_reason, nota_corrente_usata


if __name__ == "__main__":
    print("Inizializzazione pipeline (carico indice + retriever + grafo LangGraph)...")
    grafo_compilato = inizializza_sistema(VAULT_PATH)
    print("Pipeline pronta. Scrivi una domanda (o 'exit' per uscire).\n")

    # One thread_id per CLI session: all questions share the same chat_history
    thread_id = str(uuid.uuid4())

    while True:
        domanda = input("DOMANDA: ").strip()
        if domanda.lower() in ("exit", "quit", ""):
            break
        # The CLI has no open editor, so no nota_corrente is ever sent and the
        # flag is always False here — unpacked and ignored.
        risposta, stop_reason, _ = rispondi(domanda, grafo_compilato, thread_id)
        print(f"\nRISPOSTA:\n{risposta}\n")
        print("-" * 60)
