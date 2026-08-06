"""
LangGraph node factories. Each "crea_nodo_*" takes heavy dependencies
(vectorstore, retriever, graph, etc.) once and returns the actual node
function, (state: RAGState) -> dict. Keeps non-serializable objects like
the vectorstore out of the state itself.

Usage in graph.py:
    nodo_classifica = crea_nodo_classificazione()
    graph.add_node("classifica", nodo_classifica)
"""

import ollama
import re
from datetime import datetime, timedelta
from pathlib import Path
from langchain_core.documents import Document

from query.classifier import classifica_e_estrai
from retrieval.hybrid_retriever import recupera_per_data
from retrieval.graph_retriever import espandi_vicini, espandi_vicini_per_frequenza, recupera_chunks_per_note
from generation.prompts import formatta_contesto, formatta_contesto_sintetico, formatta_contesto_con_nota_corrente
from generation.llm_client import genera_risposta_claude
from settings import GRAPH_HOP_LIMIT, CLASSIFIER_MODEL, MAX_RETRY, SOGLIA_FREQUENZA_TEMPORALE, MAX_CHUNK_CONTESTO, GRADING_EXTRACT_CHARS, FINESTRA_TEMPORALE_DEFAULT_GIORNI, FINESTRA_GIORNI_PER_UNITA_GIORNO, FINESTRA_GIORNI_PER_UNITA_SETTIMANA, FINESTRA_GIORNI_PER_UNITA_MESE, FINESTRA_GIORNI_PER_UNITA_ANNO, CARTELLE_PROTETTE, MAX_CARATTERI_NOTA_CORRENTE


# ---------------------------------------------------------------------
# Deterministic guardrail for contestualizza_query
# ---------------------------------------------------------------------

# Italian implicit-reference markers (demonstratives, "the same", explicit
# backreferences). Plain regex, no model call.
_PATTERN_RIFERIMENTI = re.compile(
    r"\b("
    r"quest[oaie]|quell[oaie]|tal[ei]|ciò|"
    r"ess[oaie]|lo stesso|la stessa|gli stessi|le stesse|"
    r"succitat[oaie]|sopracitat[oaie]|precedente|precedenti|"
    r"come (detto|accennato|menzionato)|"
    r"di cui (sopra|prima)"
    r")\b",
    re.IGNORECASE,
)


def necessita_contestualizzazione(query: str) -> bool:
    """True if the query has an implicit reference that needs chat_history
    to resolve (e.g. "which of THESE...", "and THAT note from before?").
    Guards against a bug where contestualizza_query, on open-ended
    non-referential questions, pulled unrelated topics from recent
    chat_history into the rewritten query and skewed retrieval."""
    return bool(_PATTERN_RIFERIMENTI.search(query))


# ---------------------------------------------------------------------
# Query contextualization (conversation memory)
# ---------------------------------------------------------------------

def crea_nodo_contestualizzazione():
    """Rewrites the query using chat_history to resolve implicit references.
    Leaves the query untouched if there's no history, or if
    necessita_contestualizzazione() finds nothing to resolve (avoids
    invoking the local model on open questions where it previously
    injected unrequested topics from history instead of just resolving
    a genuine reference)."""
    def nodo(state):
        chat_history = state.get("chat_history", [])
        query_originale = state["query_originale"]

        if not chat_history or not necessita_contestualizzazione(query_originale):
            return {"query_effettiva": query_originale}

        cronologia_testo = "\n".join(
            f"{m['ruolo']}: {m['contenuto']}" for m in chat_history[-6:]  # last 3 turns
        )

        prompt = f"""Data la cronologia della conversazione e l'ultima domanda dell'utente,
riformula la domanda in modo che sia comprensibile DA SOLA, senza bisogno della cronologia.
Se la domanda è già autonoma, ripetila identica.
Rispondi SOLO con la domanda riformulata, nessun altro testo.

CRONOLOGIA:
{cronologia_testo}

ULTIMA DOMANDA: {query_originale}

DOMANDA RIFORMULATA:"""

        response = ollama.chat(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        query_riformulata = response["message"]["content"].strip()

        if not query_riformulata.rstrip().endswith("?"):
            # Model answered instead of rewriting: discard and fall back to the original
            query_riformulata = query_originale

        return {"query_effettiva": query_riformulata}

    return nodo


# ---------------------------------------------------------------------
# Current note (open page as primary context)
# ---------------------------------------------------------------------

def crea_nodo_nota_corrente():
    """Validates the note the plugin says is open and turns it into the text
    that formatta_contesto uses as primary source.

    No dependencies (no vectorstore, no retriever): the plugin already sent
    the note's text, so there's nothing to retrieve. The factory shape is kept
    only for consistency with every other node here.

    Three things happen, in this order:

    1. Protected mode wins over the open note. The folder is derived from the
       vault-relative path with Path(...).parent.name, the same formula
       ingestion/metadata.py uses to compute the 'cartella' metadata that the
       rerank filter matches on — so both filters agree on what counts as a
       protected folder. Without this, a stray click on a note in a private
       folder during a demo would put it right back into the answer, which is
       exactly what the shield toggle exists to prevent.
    2. An empty (or whitespace-only) note is dropped: a brand-new empty page
       is not context, and announcing it to the model just invites remarks
       about the emptiness.
    3. The text is capped at MAX_CARATTERI_NOTA_CORRENTE so one pathological
       note can't eat the whole prompt. The truncation is announced inline
       rather than silent, so the model can say the note was cut instead of
       treating the fragment as the complete note.

    Returns testo_nota_corrente="" in every rejected case; the formatting node
    keys on that empty string, so there's a single "no current note" signal
    regardless of the reason.
    """
    def nodo(state):
        nota = state.get("nota_corrente")
        if not nota:
            return {"testo_nota_corrente": "", "nome_nota_corrente": ""}

        if state.get("modalita_protetta"):
            cartella = Path(nota.get("path", "")).parent.name
            if cartella in CARTELLE_PROTETTE:
                return {"testo_nota_corrente": "", "nome_nota_corrente": ""}

        testo = (nota.get("contenuto") or "").strip()
        if not testo:
            return {"testo_nota_corrente": "", "nome_nota_corrente": ""}

        if len(testo) > MAX_CARATTERI_NOTA_CORRENTE:
            testo = testo[:MAX_CARATTERI_NOTA_CORRENTE] + "\n\n[...] (nota troncata: troppo lunga)"

        return {
            "testo_nota_corrente": testo,
            "nome_nota_corrente": nota.get("nome", "") or Path(nota.get("path", "")).stem,
        }

    return nodo


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Default temporal window estimation (when data_inizio is missing)
# ---------------------------------------------------------------------

# Each tuple is (pattern matching the time unit, corresponding days).
_FINESTRE_PER_UNITA = [
    (re.compile(r"\bgiorn[oi]\b", re.IGNORECASE), FINESTRA_GIORNI_PER_UNITA_GIORNO),
    (re.compile(r"\bsettiman[ae]\b", re.IGNORECASE), FINESTRA_GIORNI_PER_UNITA_SETTIMANA),
    (re.compile(r"\bmes[ei]\b", re.IGNORECASE), FINESTRA_GIORNI_PER_UNITA_MESE),
    (re.compile(r"\bann[oi]\b", re.IGNORECASE), FINESTRA_GIORNI_PER_UNITA_ANNO),
]


def stima_finestra_giorni(query: str) -> int:
    """Estimates how many days back to look based on the time unit
    explicitly mentioned in the query (day/week/month/year) rather than a
    single fixed default — a flat 30-day window was producing 237 chunks
    for "negli ultimi giorni", far too broad for the intent. Falls back
    to FINESTRA_TEMPORALE_DEFAULT_GIORNI if no unit is recognized."""
    for pattern, giorni in _FINESTRE_PER_UNITA:
        if pattern.search(query):
            return giorni
    return FINESTRA_TEMPORALE_DEFAULT_GIORNI


def crea_nodo_classificazione():
    """Wraps classifica_e_estrai with a deterministic default: if the query
    is TEMPORALE but the model didn't extract a data_inizio (vague queries
    like "ultimi giorni"), applies a backward window from the known end
    date (or today) sized by stima_finestra_giorni(). Done in Python
    rather than the classification prompt because date arithmetic isn't a
    language-understanding task and llama3.2:3b has proven unreliable at
    it (see grading/rewriting, removed for the same reason).

    Without this default, a TEMPORAL query missing data_inizio wouldn't
    even take the recupera_temporale branch (route_dopo_classificazione
    requires both dates) and would silently fall through to generic
    semantic retrieval with an effectively unbounded range.
    """
    def nodo(state):
        classificazione = classifica_e_estrai(state["query_effettiva"])
        tipo = classificazione["tipo"]
        data_inizio = classificazione["data_inizio"]
        data_fine = classificazione["data_fine"]

        if tipo == "TEMPORALE" and data_inizio is None:
            riferimento = (
                datetime.strptime(data_fine, "%Y-%m-%d") if data_fine else datetime.now()
            )
            giorni_finestra = stima_finestra_giorni(state["query_effettiva"])
            data_inizio = (riferimento - timedelta(days=giorni_finestra)).strftime("%Y-%m-%d")
            if data_fine is None:
                data_fine = riferimento.strftime("%Y-%m-%d")

        return {
            "tipo": tipo,
            "data_inizio": data_inizio,
            "data_fine": data_fine,
        }

    return nodo


# ---------------------------------------------------------------------
# Retrieval — CONTENUTISTICA branch
# ---------------------------------------------------------------------

def crea_nodo_retrieve_ibrido(ensemble_retriever, vectorstore, grafo_note, mappa_source):
    """Hybrid retrieval (BM25 + vector) followed by graph expansion.
    chunk_recuperati/chunk_espansi are kept separate; rerank_chunk merges them."""
    def nodo(state):
        risultati = ensemble_retriever.invoke(state["query_effettiva"])

        note_seed = {Path(doc.metadata["source"]).stem for doc in risultati}
        vicini = espandi_vicini(note_seed, grafo_note, hops=GRAPH_HOP_LIMIT)
        chunk_espansi = recupera_chunks_per_note(vicini, vectorstore, mappa_source)

        return {
            "chunk_recuperati": risultati,
            "chunk_espansi": chunk_espansi,
        }

    return nodo


# ---------------------------------------------------------------------
# Retrieval — TEMPORALE branch
# ---------------------------------------------------------------------

def crea_nodo_recupera_temporale(vectorstore, grafo_note, mappa_source):
    """Exact date filter, followed by frequency-based graph expansion: a
    note linked by at least SOGLIA_FREQUENZA_TEMPORALE distinct daily
    notes in the period is considered relevant to the summary."""
    def nodo(state):
        risultati_raw = recupera_per_data(vectorstore, state["data_inizio"], state["data_fine"])

        # Normalize vectorstore.get()'s dict format to list[Document]
        chunk_recuperati = [
            Document(page_content=testo, metadata=meta)
            for testo, meta in zip(risultati_raw["documents"], risultati_raw["metadatas"])
        ]

        note_seed = {Path(c.metadata["source"]).stem for c in chunk_recuperati}
        vicini = espandi_vicini_per_frequenza(note_seed, grafo_note, soglia=SOGLIA_FREQUENZA_TEMPORALE)
        chunk_espansi = recupera_chunks_per_note(vicini, vectorstore, mappa_source)

        return {
            "chunk_recuperati": chunk_recuperati,
            "chunk_espansi": chunk_espansi,
        }

    return nodo


# ---------------------------------------------------------------------
# Re-ranking / deduplication
# ---------------------------------------------------------------------

def crea_nodo_rerank():
    """Merges chunk_recuperati and chunk_espansi, dedupes by (source,
    page_content) — not just source, since a note can produce multiple
    distinct chunks (e.g. different sections split by CHUNK_SIZE) that a
    source-only dedup would collapse into one, silently dropping real
    content — then caps the total at MAX_CHUNK_CONTESTO.

    The cap is positional (first N in arrival order: chunk_recuperati
    first, since it's already relevance-ordered by the hybrid ensemble,
    then chunk_espansi) rather than score-based; a real re-rank is a
    possible future improvement if this proves insufficient.

    Protected mode (demo mode): this node is the single point where both
    retrieval branches converge, making it the right chokepoint for
    filtering private folders — every chunk, whether retrieved directly
    or pulled in via graph expansion, passes through here, and any future
    retrieval branch inherits the protection automatically. The filter
    runs before the MAX_CHUNK_CONTESTO cap so the cap applies only to
    chunks that are actually visible.

    Current note: its chunks are dropped here for the same "single
    chokepoint" reason. The note already enters the context whole, as the
    primary section, straight from the editor — leaving its indexed chunks
    in the support section would repeat it, and repeat it in the STALE
    version ChromaDB happens to hold. Matching is by note name (stem), like
    everywhere else in this codebase.
    """
    def nodo(state):
        tutti_i_chunk = state.get("chunk_recuperati", []) + state.get("chunk_espansi", [])

        if state.get("modalita_protetta"):
            tutti_i_chunk = [
                c for c in tutti_i_chunk
                if c.metadata.get("cartella") not in CARTELLE_PROTETTE
            ]

        nome_nota_corrente = state.get("nome_nota_corrente")
        if nome_nota_corrente:
            tutti_i_chunk = [
                c for c in tutti_i_chunk
                if Path(c.metadata.get("source", "")).stem != nome_nota_corrente
            ]

        visti = set()
        chunk_unici = []
        for chunk in tutti_i_chunk:
            chiave = (chunk.metadata.get("source"), chunk.page_content)
            if chiave in visti:
                continue
            visti.add(chiave)
            chunk_unici.append(chunk)

        chunk_limitati = chunk_unici[:MAX_CHUNK_CONTESTO]

        return {"chunk_rerankati": chunk_limitati}

    return nodo


def crea_nodo_formattazione():
    """Explicit formatting step, needed after the CRAG grading cycle (which
    used to compute this as a side effect) was removed from the graph.

    When there's a current note, the context becomes two labelled sections
    (open note first, retrieved chunks as support) instead of one flat list;
    testo_nota_corrente being empty is the single signal for "no current
    note", whatever the reason it was rejected upstream."""
    def nodo(state):
        contesto_recuperato = formatta_contesto(state["chunk_rerankati"], is_get_result=False)

        testo_nota = state.get("testo_nota_corrente", "")
        if not testo_nota:
            return {"contesto_formattato": contesto_recuperato}

        contesto = formatta_contesto_con_nota_corrente(
            state["nome_nota_corrente"], testo_nota, contesto_recuperato
        )
        return {"contesto_formattato": contesto}

    return nodo


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def crea_nodo_generazione():
    """Thin wrapper around genera_risposta_claude.

    ha_nota_corrente adds the source-precedence rule to the system prompt. It's
    derived from testo_nota_corrente (what actually reached the context) and
    not from nota_corrente (what the plugin sent): a note rejected by the
    protected-mode filter or because it was empty must not make the prompt
    describe a primary section that isn't in the context."""
    def nodo(state):
        risposta, stop_reason = genera_risposta_claude(
            state["query_effettiva"],
            state["contesto_formattato"],
            state["tipo"],
            ha_nota_corrente=bool(state.get("testo_nota_corrente")),
        )
        return {"risposta": risposta, "stop_reason": stop_reason}

    return nodo


# ---------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------

def crea_nodo_aggiorna_memoria():
    """Appends the current turn to chat_history. The operator.add reducer
    on RAGState.chat_history means returning a list here appends rather
    than overwrites."""
    def nodo(state):
        nuovi_messaggi = [
            {"ruolo": "utente", "contenuto": state["query_originale"]},
            {"ruolo": "assistente", "contenuto": state["risposta"]},
        ]
        return {"chat_history": nuovi_messaggi}

    return nodo


# =======================================================================
# DORMANT (not wired into the graph, see graph.py)
# =======================================================================
#
# Grading and query rewriting (CRAG pattern) were removed from the active
# graph after tests showed llama3.2:3b as judge/rewriter did more harm
# than good — inconsistent grading on identical context, and one case
# where rewriting returned meta-conversational text instead of a query,
# corrupting the downstream chain.
#
# Left here unwired rather than deleted: to reintroduce this pattern
# (e.g. with a stronger grading model), re-insert "grading" and
# "riformula_query" between rerank_chunk and formatta_contesto in
# graph.py, with a route_dopo_grading conditional edge.
# ---------------------------------------------------------------------

def crea_nodo_grading():
    """Aggregate (not per-chunk) judgment of whether the formatted context
    is enough to answer the query, or needs a retrieval retry."""
    def nodo(state):
        # Compact context (titles + excerpts) is easier for llama3.2:3b to grade reliably
        contesto_sintetico = formatta_contesto_sintetico(
            state["chunk_rerankati"], is_get_result=False, n_caratteri=GRADING_EXTRACT_CHARS
        )

        prompt = f"""Valuta se il CONTESTO fornito contiene informazioni sufficienti
per rispondere alla DOMANDA. Rispondi SOLO con "sufficiente" o "insufficiente".

CONTESTO:
{contesto_sintetico}

DOMANDA: {state['query_effettiva']}

GIUDIZIO:"""

        response = ollama.chat(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        esito = response["message"]["content"].strip().lower()

        # Normalize imprecise model output to one of the two expected values
        esito = "sufficiente" if "sufficiente" in esito and "insufficiente" not in esito else "insufficiente"

        contesto_incompleto = (
            esito == "insufficiente" and state.get("retry_count", 0) >= MAX_RETRY
        )

        # Full context computed only now that it's actually needed, for the generation node
        contesto_completo = formatta_contesto(state["chunk_rerankati"], is_get_result=False)

        return {
            "grading_esito": esito,
            "contesto_formattato": contesto_completo,
            "contesto_incompleto": contesto_incompleto,
        }

    return nodo


def crea_nodo_riformulazione():
    """Rewrites the query when grading judges the context insufficient,
    and increments retry_count."""
    def nodo(state):
        prompt = f"""La seguente domanda non ha prodotto un contesto sufficiente
da un archivio di note personali. Riformulala in modo diverso, più specifico
o con termini alternativi, per migliorare il recupero di informazioni.
Rispondi SOLO con la domanda riformulata.

DOMANDA ORIGINALE: {state['query_effettiva']}

DOMANDA RIFORMULATA:"""

        response = ollama.chat(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        nuova_query = response["message"]["content"].strip()

        return {
            "query_effettiva": nuova_query,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return nodo
