"""Builds a wikilink graph over the vault notes and expands retrieval results via graph traversal."""

from pathlib import Path
from collections import deque
from settings import GRAPH_HOP_LIMIT
from langchain_core.documents import Document

ESTENSIONI_ALLEGATI = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg", ".mp3", ".mp4"}


def normalizza_nome(nome: str) -> str:
    """Lowercases, trims, and reduces a path-qualified link ('Folder/Note') to just the note name."""
    ultimo_segmento = nome.strip().split("/")[-1]
    return " ".join(ultimo_segmento.lower().split())


def e_allegato(nome: str) -> bool:
    """True if the link points to a non-note file (image, pdf, etc.)."""
    return Path(nome).suffix.lower() in ESTENSIONI_ALLEGATI


def costruisci_grafo(chunks) -> dict:
    note_viste = {}

    for chunk in chunks:
        nome_nota = Path(chunk.metadata["source"]).stem
        if nome_nota in note_viste:
            continue
        links_raw = chunk.metadata.get("outgoing_links", "nessun_link")
        links = [] if links_raw == "nessun_link" else [l.strip() for l in links_raw.split(",")]
        note_viste[nome_nota] = links

    normalizzato_a_reale = {normalizza_nome(nome): nome for nome in note_viste}

    grafo = {nome_nota: {"out": [], "in": []} for nome_nota in note_viste}

    for nome_nota, links in note_viste.items():
        nome_nota_norm = normalizza_nome(nome_nota)

        for target_raw in links:
            if e_allegato(target_raw):
                continue

            target_norm = normalizza_nome(target_raw)
            if target_norm == nome_nota_norm:
                continue  # self-link

            # Resolve to the canonical filename, keeping only the last path segment otherwise
            target = normalizzato_a_reale.get(target_norm, target_raw.split("/")[-1])

            grafo[nome_nota]["out"].append(target)
            if target not in grafo:
                grafo[target] = {"out": [], "in": []}
            grafo[target]["in"].append(nome_nota)

    return grafo

def espandi_vicini(note_seed: list[str], grafo: dict, hops: int = GRAPH_HOP_LIMIT) -> set[str]:
    """BFS over the wikilink graph from seed notes, up to 'hops' levels,
    following both outgoing and incoming links. Returns discovered note
    names (excluding the seeds); ghost nodes (linked but with no actual
    chunk) are included and left for the caller to filter out."""
    seed_set = set(note_seed)
    visitate = set(seed_set)
    risultato = set()

    coda = deque((nota, 0) for nota in seed_set)

    while coda:
        nota_corrente, hop_corrente = coda.popleft()

        if hop_corrente >= hops:
            continue

        vicini = grafo.get(nota_corrente)
        if vicini is None:
            continue

        tutti_i_vicini = vicini["out"] + vicini["in"]

        for vicino in tutti_i_vicini:
            if vicino in visitate:
                continue

            visitate.add(vicino)
            risultato.add(vicino)
            coda.append((vicino, hop_corrente + 1))

    return risultato

def espandi_vicini_per_frequenza(note_seed: set[str], grafo: dict, soglia: int) -> set[str]:
    """Variant of espandi_vicini() for large seed sets (e.g. a week of daily
    notes in the TEMPORAL branch), where full BFS expansion would add too
    much noise. Keeps only neighbors linked by at least `soglia` distinct
    seeds. Deliberately different from espandi_vicini(): outgoing links
    only, fixed 1-hop depth, and an absolute (not percentage) threshold so
    behavior doesn't shift with the number of seed notes.

    note_seed: starting note names (e.g. daily notes in the period)
    grafo: adjacency dict from costruisci_grafo()
    soglia: minimum number of distinct seeds that must link a neighbor
    """
    conteggio = {}

    for nota in note_seed:
        vicini = grafo.get(nota)
        if vicini is None:
            continue

        # set() so a link repeated multiple times in one note counts once
        for vicino in set(vicini["out"]):
            if vicino in note_seed:
                continue  # don't count references between seeds themselves

            conteggio[vicino] = conteggio.get(vicino, 0) + 1

    return {nota for nota, volte in conteggio.items() if volte >= soglia}


def mappa_note_a_source(chunks) -> dict:
    """Maps note name -> full file path, needed to turn espandi_vicini()
    output into valid ChromaDB filters (indexed by 'source' = full path)."""
    mappa = {}
    for chunk in chunks:
        nome_nota = Path(chunk.metadata["source"]).stem
        if nome_nota not in mappa:
            mappa[nome_nota] = chunk.metadata["source"]
    return mappa


def recupera_chunks_per_note(nomi_note: set[str], vectorstore, mappa_source: dict) -> list[Document]:
    """Fetches the actual chunks from ChromaDB for a set of note names
    (typically espandi_vicini() output). Names missing from the map
    (ghost nodes, already-filtered attachments) are silently skipped."""
    sources_validi = [mappa_source[nome] for nome in nomi_note if nome in mappa_source]

    if not sources_validi:
        return []

    risultati_raw = vectorstore.get(
        where={"source": {"$in": sources_validi}}
    )

    # vectorstore.get() returns a dict; convert to Document objects to match
    # the rest of the pipeline (formatta_contesto, ensemble_retriever)
    return [
        Document(page_content=testo, metadata=meta)
        for testo, meta in zip(risultati_raw["documents"], risultati_raw["metadatas"])
    ]

if __name__ == "__main__":
    import pickle
    from collections import Counter
    from settings import CHUNKS_CACHE_PATH

    with open(CHUNKS_CACHE_PATH, "rb") as f:
        chunks = pickle.load(f)

    grafo = costruisci_grafo(chunks)

    n_note = len(grafo)
    n_link_totali = sum(len(v["out"]) for v in grafo.values())
    n_isolate = sum(1 for v in grafo.values() if not v["out"] and not v["in"])

    print(f"Note nel grafo: {n_note}")
    print(f"Link totali (archi): {n_link_totali}")
    print(f"Media link per nota: {n_link_totali / n_note:.2f}")
    print(f"Note completamente isolate: {n_isolate} ({n_isolate / n_note * 100:.0f}%)")

    nota_piu_connessa = max(grafo, key=lambda n: len(grafo[n]["out"]) + len(grafo[n]["in"]))
    grado_massimo = len(grafo[nota_piu_connessa]["out"]) + len(grafo[nota_piu_connessa]["in"])
    print(f"\nNota più connessa: {nota_piu_connessa} (grado {grado_massimo})")
    print(grafo[nota_piu_connessa])

    gradi = [len(v["out"]) + len(v["in"]) for v in grafo.values()]
    dist = Counter(gradi)

    print("\nDistribuzione grado (in+out):")
    for grado in sorted(dist):
        print(f"  {grado} collegamenti -> {dist[grado]} note")

    note_reali = {Path(c.metadata["source"]).stem for c in chunks}
    note_fantasma = set(grafo.keys()) - note_reali
    print(f"\nNote fantasma residue: {len(note_fantasma)}")
