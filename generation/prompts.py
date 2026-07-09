"""Builds the system/user prompts and formats retrieved chunks into context text for generation."""

from pathlib import Path


def formatta_contesto(chunks_o_risultati, is_get_result=False):
    """Converts retrieved chunks into prompt text with wikilink citations.
    is_get_result=True for vectorstore.get() output (dict format, different from .invoke())."""
    contesto = []

    if is_get_result:
        docs_content = chunks_o_risultati["documents"]
        docs_metadata = chunks_o_risultati["metadatas"]
        iterator = zip(docs_content, docs_metadata)
    else:
        iterator = [(doc.page_content, doc.metadata) for doc in chunks_o_risultati]

    for content, metadata in iterator:
        nome_nota = Path(metadata["source"]).stem
        contesto.append(f"[[{nome_nota}]]\n{content}\n")

    return "\n---\n".join(contesto)


def formatta_contesto_sintetico(chunks_o_risultati, is_get_result=False, n_caratteri=150):
    """Compact version of formatta_contesto() for the grading node: note title
    + short excerpt instead of full text, since a small judge model (llama3.2:3b)
    grades a compact context more reliably than a blob of 50+ full chunks."""
    righe = []

    if is_get_result:
        docs_content = chunks_o_risultati["documents"]
        docs_metadata = chunks_o_risultati["metadatas"]
        iterator = zip(docs_content, docs_metadata)
    else:
        iterator = [(doc.page_content, doc.metadata) for doc in chunks_o_risultati]

    for content, metadata in iterator:
        nome_nota = Path(metadata["source"]).stem
        estratto = content[:n_caratteri].replace("\n", " ").strip()
        if len(content) > n_caratteri:
            estratto += "..."
        righe.append(f"[[{nome_nota}]]: {estratto}")

    return "\n".join(righe)


def costruisci_system_prompt(tipo: str) -> str:
    """System instructions (persona + rules), kept separate from the content
    (context + question) via the API's 'system' field so the language
    instruction isn't overridden by the dominant language of the content."""
    if tipo == "CONTENUTISTICA":
        return """Sei un assistente che risponde a domande usando ESCLUSIVAMENTE le note fornite come contesto.
Le note possono essere scritte in italiano, inglese, o entrambi.
Rispondi SEMPRE nella stessa lingua in cui è formulata la domanda dell'utente, indipendentemente dalla lingua del contesto —
riformulando liberamente i concetti con parole tue, NON traducendo letteralmente frase per frase, spiegando il concetto come lo spiegheresti a voce.
Se il contesto non contiene informazioni sufficienti, dillo esplicitamente invece di inventare.
Cita sempre le note di origine usando la sintassi [[nome nota]] già presente nel contesto."""

    else:  # TEMPORALE
        return """Sei un assistente che crea riassunti a partire da note personali datate.
Il contesto contiene tutte le note del periodo richiesto. Fai una sintesi organizzata (per temi o cronologicamente),
citando le note di origine con [[nome nota]].
Rispondi SEMPRE nella stessa lingua in cui è formulata la richiesta dell'utente, indipendentemente dalla lingua del contesto."""


def costruisci_prompt(query: str, contesto: str, tipo: str) -> str:
    """Returns the user-message content (context + question); system
    instructions live in costruisci_system_prompt(). The language reminder
    is repeated here, right before the question, as positional reinforcement."""
    etichetta_domanda = "DOMANDA" if tipo == "CONTENUTISTICA" else "RICHIESTA"
    etichetta_risposta = "RISPOSTA" if tipo == "CONTENUTISTICA" else "RIASSUNTO"

    return f"""CONTESTO:
{contesto}

{etichetta_domanda}: {query}

(Ricorda: rispondi nella stessa lingua di questa {etichetta_domanda.lower()}, non nella lingua del contesto sopra.)

{etichetta_risposta}:"""


if __name__ == "__main__":
    from settings import VAULT_PATH
    from ingestion.loader import document_loader
    from ingestion.metadata import estrai_metadati_documenti
    from ingestion.splitter import split_documents
    from retrieval.embeddings import embedding_ollama
    from retrieval.vectorstore import load_vectorstore  # loads, does NOT rebuild the index
    from retrieval.hybrid_retriever import create_ensemble_retriever, recupera_per_data

    # Recompute raw chunks only because BM25 needs them for its own index
    # (no embeddings); the vectorstore itself is loaded from disk, not rebuilt.
    documents = document_loader(VAULT_PATH)
    documents = estrai_metadati_documenti(documents)
    chunks = split_documents(documents)

    embedding_model = embedding_ollama()
    vectorstore = load_vectorstore(embedding_model)

    print(f"Chunk nel vectorstore (persistiti su disco): {vectorstore._collection.count()}")
    print(f"Chunk generati ora da split_documents: {len(chunks)}")
    print()

    def anteprima(testo, n=300):
        return testo[:n] + ("..." if len(testo) > n else "")

    # --- CONTENUTISTICA case (from .invoke()) ---
    ensemble_retriever = create_ensemble_retriever(chunks, vectorstore)
    risultati_contenuto = ensemble_retriever.invoke("Cos'è il clustering?")
    contesto_contenuto = formatta_contesto(risultati_contenuto, is_get_result=False)
    prompt_contenuto = costruisci_prompt("Cos'è il clustering?", contesto_contenuto, "CONTENUTISTICA")

    print("=== CONTENUTISTICA ===")
    print(f"Chunk nel contesto: {len(risultati_contenuto)}")
    print(f"Contiene wikilink '[[': {'[[' in contesto_contenuto}")
    print(f"Lunghezza prompt totale: {len(prompt_contenuto)} caratteri")
    print(f"Anteprima contesto:\n{anteprima(contesto_contenuto)}\n")

    # --- TEMPORALE case (from vectorstore.get()) ---
    risultati_temporale = recupera_per_data(vectorstore, "2026-06-01", "2026-06-10")
    contesto_temporale = formatta_contesto(risultati_temporale, is_get_result=True)
    prompt_temporale = costruisci_prompt(
        "Cosa ho scritto tra il 1 e il 10 giugno?", contesto_temporale, "TEMPORALE"
    )

    print("=== TEMPORALE ===")
    print(f"Chunk nel contesto: {len(risultati_temporale['documents'])}")
    print(f"Contiene wikilink '[[': {'[[' in contesto_temporale}")
    print(f"Lunghezza prompt totale: {len(prompt_temporale)} caratteri")
    print(f"Anteprima contesto:\n{anteprima(contesto_temporale)}")
