"""Classifies a query as CONTENUTISTICA (topic-based) or TEMPORALE (date-range based), extracting dates when relevant."""

import json
import re
from datetime import datetime, timedelta
import ollama

from settings import CLASSIFIER_MODEL


# ---------------------------------------------------------------------
# Deterministic guardrail: absolute single-day temporal anchors
# ---------------------------------------------------------------------
# Resolving "oggi"/"ieri" to a date is lookup + arithmetic, not language
# understanding, so it's handled in Python instead of trusting the LLM
# (llama3.2:3b) — same rationale as stima_finestra_giorni in nodes.py.
# This also fixes a bug where "Cos'ho in programma di fare oggi?" was
# misclassified as CONTENUTISTICA and never matched today's daily note.
#
# Order matters: more specific anchors go first ("l'altro ieri" must win
# over "ieri", or it resolves to -1 instead of -2).
_ANCORE_GIORNO = [
    (re.compile(r"\bl['\s]?altro\s?ieri\b", re.IGNORECASE), -2),
    (re.compile(r"\bieri\b", re.IGNORECASE), -1),
    (re.compile(r"\boggi\b", re.IGNORECASE),  0),
    # (re.compile(r"\bdomani\b", re.IGNORECASE), +1),
    # ^ forward-looking: tomorrow's daily note likely doesn't exist yet,
    #   so a date filter would return nothing. Enable only if useful.
]


def risolvi_ancora_temporale(query: str) -> dict | None:
    """If the query contains a single-day temporal anchor (oggi/ieri/l'altro
    ieri), resolves type + dates in Python without calling the model.
    Returns None if no anchor is found, so the caller falls back to
    llama3.2:3b for references that need real understanding ("last week")."""
    for pattern, offset in _ANCORE_GIORNO:
        if pattern.search(query):
            giorno = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
            return {"tipo": "TEMPORALE", "data_inizio": giorno, "data_fine": giorno}
    return None


def classifica_e_estrai(query: str) -> dict:
    ancora = risolvi_ancora_temporale(query)
    if ancora is not None:
        return ancora

    oggi = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""Sei un classificatore per un sistema RAG che interroga note personali Obsidian.
Data odierna: {oggi}

Classifica la domanda dell'utente e, se necessario, estrai il range temporale.

Rispondi SOLO con un JSON valido in questo formato esatto, senza testo aggiuntivo:
{{
  "tipo": "CONTENUTISTICA" oppure "TEMPORALE",
  "data_inizio": "YYYY-MM-DD" oppure null,
  "data_fine": "YYYY-MM-DD" oppure null
}}

Regole:
- CONTENUTISTICA: la domanda verte su un ARGOMENTO/CONCETTO specifico (es. Python, clustering, GRU),
  anche se contiene verbi come "riassumi" o "spiega". Il segnale chiave è la presenza di un topic nominato.
  ATTENZIONE: parole generiche come "programma", "impegni", "task", "cose da fare", "appuntamenti",
  "attività" NON sono topic — sono contenitori generici, non concetti. Una domanda su "cosa ho in
  programma" NON è CONTENUTISTICA solo perché contiene la parola "programma".
- TEMPORALE: la domanda verte su un PERIODO o MOMENTO nel tempo (es. "giugno", "ieri", "oggi",
  "ultimo mese", "questa settimana"), senza un argomento specifico. Include sia "cosa ho scritto/pensato
  in quel periodo" sia "cosa ho in programma / cosa devo fare in quel giorno o periodo" (agenda, task).

Esempi:
- "Riassumi le mie note su Python" → CONTENUTISTICA (topic: Python, nessun periodo menzionato)
- "Riassumi giugno" → TEMPORALE (periodo: giugno, nessun topic specifico)
- "Cosa ho scritto su Python a giugno?" → TEMPORALE (ha entrambi, ma per ora trattala come TEMPORALE e filtra su giugno; il topic va gestito nel retrieval a valle)
- "Cos'ho in programma questa settimana?" → TEMPORALE (periodo: questa settimana; "programma" è una parola-contenitore, non un topic)
- "Spiegami il clustering" → CONTENUTISTICA
- "Cosa ho pensato ieri?" → TEMPORALE


Domanda: "{query}"
JSON:"""

    response = ollama.chat(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )

    contenuto = response["message"]["content"]

    try:
        return json.loads(contenuto)
    except json.JSONDecodeError:
        print(f"[classifier] Risposta non in formato JSON valido: {contenuto!r}")
        # Conservative fallback: treat as CONTENUTISTICA to avoid blocking the pipeline
        return {"tipo": "CONTENUTISTICA", "data_inizio": None, "data_fine": None}


if __name__ == "__main__":
    print("=== Ancore deterministiche (short-circuit, nessun modello) ===")
    ancore = [
        "Cos'ho in programma di fare oggi?",
        "Cosa ho pensato ieri?",
        "Cosa ho fatto l'altro ieri?",
        "Come si vive oggigiorno?",   # must NOT trigger "oggi" -> goes to the model
    ]
    for q in ancore:
        print(q, "→", risolvi_ancora_temporale(q))

    # Uncomment to test the llama3.2:3b path (requires Ollama running).
    # print("\n=== Percorso modello ===")
    # test_queries = [
    #     "Cos'è il clustering?",
    #     "Cos'ho in programma questa settimana?",
    #     "Fammi un riassunto delle mie riflessioni dell'ultimo mese",
    #     "Qual è la differenza tra GRU e LSTM?",
    #     "Cosa ho scritto la settimana scorsa?",
    # ]
    # for q in test_queries:
    #     print(q, "→", classifica_e_estrai(q))
