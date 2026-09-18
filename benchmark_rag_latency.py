"""
Benchmark di latenza per l'Obsidian RAG System.

Misura la latenza end-to-end e, soprattutto, la scompone per nodo del grafo
LangGraph: così si distingue il tempo che dipende dal codice del progetto
(classificazione, retrieval ibrido, espansione via grafo, rerank) da quello
che dipende dall'API di generazione (Claude), che è rumore esterno.

È questa scomposizione a rendere i numeri difendibili in colloquio: un
"3.2s median end-to-end" da solo dice poco, "180ms di retrieval su 4.8k
chunk + 2.9s di generazione" dice cosa hai costruito tu.

USO
---
    # modalità diretta (default): importa la pipeline in-process, nessun server
    python benchmark_rag_latency.py

    # più ripetizioni per query, per stabilizzare le statistiche
    python benchmark_rag_latency.py --repeat 3

    # query da file esterno (una per riga, '#' per i commenti)
    python benchmark_rag_latency.py --queries mie_query.txt

    # modalità HTTP: misura anche l'overhead FastAPI (server già avviato)
    python app_fastapi.py                      # in un altro terminale
    python benchmark_rag_latency.py --mode http

PREREQUISITI
------------
- Ollama attivo (nomic-embed-text, llama3.2:3b)
- export ANTHROPIC_API_KEY="sk-ant-..."
- indice già costruito (build_index.py / sync_vault.py)

NOTA SULLA MODALITÀ HTTP
------------------------
Non serve avviare FastAPI per ottenere i numeri: la modalità diretta misura
esattamente la stessa pipeline (app_fastapi.py chiama pipeline.rispondi(), che
è ciò che viene misurato qui) senza il rumore di serializzazione HTTP, ed è
l'unica che può dare il breakdown per nodo. Avvia FastAPI solo se vuoi
dichiarare un numero "via API" — l'overhead su localhost è ~ms, quindi la
differenza tra le due modalità è un dato interessante ma non decisivo.
"""

import argparse
import csv
import json
import math
import pickle
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------
# QUERIES
# ---------------------------------------------------------------------
# Sostituisci con domande VERE sul tuo vault: il numero finale è difendibile
# solo se le query sono quelle che il sistema riceve davvero. Tieni entrambe
# le classi (CONTENUTISTICA e TEMPORALE), perché seguono rami di retrieval
# diversi e hanno latenze diverse — mostrarlo è un punto a favore.
QUERIES = [
    # --- CONTENUTISTICHE (ramo retrieve_ibrido: BM25 + vettoriale + grafo) ---
    "Quali sono i vantaggi di BM25 rispetto alla ricerca puramente vettoriale?",
    "Cosa ho scritto sulle differenze tra RAG e fine-tuning?",
    "Riassumi le mie note sul teorema di Bayes",
    "Come funziona il chunking con overlap e perché serve?",
    "Che cosa sono gli embedding e come si misura la loro similarità?",
    "Quali note ho sul clustering e quali algoritmi ho annotato?",
    "Cosa ho scritto a proposito del clustering?",
    "Quali sono i limiti che ho annotato sui modelli locali rispetto alle API?",
    "Spiegami cosa ho raccolto sul concetto di overfitting",
    "Quali riferimenti ho sui database vettoriali e sulle loro differenze?",
    "Cosa dicono le mie note sulla valutazione di un sistema RAG?",
    "Che collegamenti ho tracciato tra statistica bayesiana e machine learning?",

    # --- TEMPORALI (ramo recupera_temporale: filtro data + espansione) ---
    "Cosa ho scritto la settimana scorsa?",
    "Riassumi le mie note dell'ultimo mese",
    "Su cosa ho lavorato negli ultimi tre giorni?",
    "Quali argomenti ho toccato a giugno?",
    "Cosa ho annotato di recente sul progetto RAG?",
    "Quali sono stati i temi ricorrenti nelle mie note di quest'anno?",
]

# Nodi del grafo che rappresentano il lavoro "di retrieval" (codice del
# progetto) contro quello di generazione (API esterna). Usato per il
# breakdown; vedi orchestration/graph.py per la topologia.
NODO_GENERAZIONE = "genera_risposta"


# ---------------------------------------------------------------------
# Runner: modalità diretta (in-process)
# ---------------------------------------------------------------------

def crea_runner_diretto(debug=False):
    """Inizializza la pipeline una volta e restituisce (run_query, info_corpus).

    run_query(text) -> (latenza_totale, {nodo: secondi})

    Nota: replica il ciclo di stream di pipeline.rispondi() invece di
    chiamarlo, perché rispondi() restituisce solo la risposta finale e qui
    serve il tempo speso da ogni singolo nodo.
    """
    from settings import VAULT_PATH
    from pipeline import inizializza_sistema
    from orchestration.state import stato_iniziale

    print("Inizializzazione pipeline (indice + retriever + grafo LangGraph)...")
    t0 = time.perf_counter()
    grafo = inizializza_sistema(VAULT_PATH)
    t_init = time.perf_counter() - t0
    print(f"Pipeline pronta in {t_init:.1f}s.\n")

    def run_query(text, thread_id):
        stato = stato_iniziale(text, modalita_protetta=False, nota_corrente=None)
        config = {"configurable": {"thread_id": thread_id}}

        per_nodo = {}
        risposta = None
        start = time.perf_counter()
        ultimo = start

        for aggiornamento in grafo.stream(stato, config, stream_mode="updates"):
            ora = time.perf_counter()
            for nome_nodo, valori in aggiornamento.items():
                # Il tempo del nodo è l'intervallo tra la fine del nodo
                # precedente e l'emissione di questo aggiornamento.
                per_nodo[nome_nodo] = per_nodo.get(nome_nodo, 0.0) + (ora - ultimo)
                if debug:
                    print(f"    [{nome_nodo}] {ora - ultimo:.3f}s")
                if "risposta" in valori:
                    risposta = valori["risposta"]
            ultimo = ora

        totale = time.perf_counter() - start
        if risposta is None:
            raise RuntimeError("Il grafo non ha prodotto una risposta.")
        return totale, per_nodo

    return run_query, _info_corpus(), t_init


# ---------------------------------------------------------------------
# Runner: modalità HTTP (FastAPI)
# ---------------------------------------------------------------------

def crea_runner_http(base_url):
    """Runner contro il backend FastAPI già avviato. Nessun breakdown per
    nodo: dall'esterno si vede solo la latenza end-to-end (che include
    serializzazione HTTP e threadpool di FastAPI)."""
    import requests

    endpoint = f"{base_url.rstrip('/')}/chat"

    try:
        health = requests.get(f"{base_url.rstrip('/')}/health", timeout=5).json()
    except Exception as e:
        sys.exit(
            f"Impossibile raggiungere {base_url}/health ({e}).\n"
            "Avvia il backend con: python app_fastapi.py"
        )
    if not health.get("grafo_caricato"):
        sys.exit("Il server risponde ma il grafo non è caricato.")
    print(f"Backend raggiungibile su {base_url} (grafo caricato).\n")

    def run_query(text, thread_id):
        start = time.perf_counter()
        r = requests.post(
            endpoint,
            json={"query": text, "thread_id": thread_id, "modalita_protetta": False},
            timeout=180,
        )
        r.raise_for_status()
        payload = r.json()
        totale = time.perf_counter() - start
        if not payload.get("risposta"):
            raise RuntimeError("Risposta vuota dal backend.")
        return totale, {}

    return run_query, _info_corpus(), None


# ---------------------------------------------------------------------
# Dimensione del corpus (rende concreto il "across an N-note vault")
# ---------------------------------------------------------------------

def _info_corpus():
    """Legge chunks_cache.pkl per contare note e chunk indicizzati."""
    try:
        from settings import CHUNKS_CACHE_PATH
        with open(CHUNKS_CACHE_PATH, "rb") as f:
            chunks = pickle.load(f)
        note = {c.metadata.get("source") for c in chunks}
        caratteri = sum(len(c.page_content) for c in chunks)
        return {
            "n_note": len(note),
            "n_chunk": len(chunks),
            "caratteri_totali": caratteri,
        }
    except Exception as e:
        print(f"[avviso] impossibile leggere il corpus: {e}")
        return {"n_note": None, "n_chunk": None, "caratteri_totali": None}


# ---------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------

def percentile(valori_ordinati, p):
    """Percentile con metodo nearest-rank: corretto anche su campioni piccoli,
    dove l'interpolazione darebbe numeri che non corrispondono a nessuna
    misurazione reale."""
    if not valori_ordinati:
        return float("nan")
    rango = max(1, math.ceil(p * len(valori_ordinati)))
    return valori_ordinati[rango - 1]


def riassumi(latenze):
    ordinate = sorted(latenze)
    return {
        "n": len(latenze),
        "mean_s": statistics.mean(latenze),
        "p50_s": percentile(ordinate, 0.50),
        "p90_s": percentile(ordinate, 0.90),
        "p95_s": percentile(ordinate, 0.95),
        "min_s": ordinate[0],
        "max_s": ordinate[-1],
        "stdev_s": statistics.stdev(latenze) if len(latenze) > 1 else 0.0,
    }


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def benchmark(run_query, queries, repeat, warmup, conversational):
    """Esegue le query e raccoglie una riga per esecuzione.

    thread_id: di default uno nuovo per ogni query. Riusare lo stesso
    accumulerebbe chat_history nel checkpointer, attivando il nodo di
    contestualizzazione e facendo crescere la latenza con l'ordine delle
    query — cioè misurerebbe l'ordine, non il sistema. --conversational
    forza il comportamento opposto, se vuoi proprio misurare quel caso.
    """
    thread_condiviso = str(uuid.uuid4()) if conversational else None

    if warmup and queries:
        print(f"Warmup: {warmup} chiamata/e non misurata/e (cache, connessioni, modelli)...")
        for _ in range(warmup):
            try:
                run_query(queries[0], str(uuid.uuid4()))
            except Exception as e:
                sys.exit(f"Il warmup è fallito: {e}\nControlla Ollama, l'API key e l'indice.")
        print("Warmup completato.\n")

    risultati = []
    totale_esecuzioni = len(queries) * repeat
    i = 0

    for giro in range(1, repeat + 1):
        for q in queries:
            i += 1
            thread_id = thread_condiviso or str(uuid.uuid4())
            try:
                totale, per_nodo = run_query(q, thread_id)
            except Exception as e:
                print(f"[{i}/{totale_esecuzioni}]  ERRORE  -  {q[:55]}  ({e})")
                risultati.append({"query": q, "giro": giro, "errore": str(e)})
                continue

            gen = per_nodo.get(NODO_GENERAZIONE, 0.0)
            pipeline_s = totale - gen if per_nodo else None
            suffisso = f" (pipeline {pipeline_s:.2f}s + gen {gen:.2f}s)" if per_nodo else ""
            print(f"[{i}/{totale_esecuzioni}]  {totale:5.2f}s{suffisso}  -  {q[:55]}")

            risultati.append({
                "query": q,
                "giro": giro,
                "totale_s": totale,
                "generazione_s": gen if per_nodo else None,
                "pipeline_s": pipeline_s,
                "per_nodo": per_nodo,
                "errore": None,
            })

    return risultati


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def stampa_report(risultati, corpus, stats_file_prefix, mode, t_init):
    ok = [r for r in risultati if not r.get("errore")]
    falliti = [r for r in risultati if r.get("errore")]

    if not ok:
        sys.exit("Nessuna esecuzione riuscita: niente da riportare.")

    end_to_end = riassumi([r["totale_s"] for r in ok])
    ha_breakdown = any(r.get("per_nodo") for r in ok)

    print("\n" + "=" * 62)
    print("RISULTATI")
    print("=" * 62)
    if corpus["n_note"]:
        print(
            f"Corpus: {corpus['n_note']} note, {corpus['n_chunk']} chunk "
            f"(~{corpus['caratteri_totali'] // 1000}k caratteri indicizzati)"
        )
    print(f"Modalità: {mode}   |   esecuzioni riuscite: {len(ok)}   |   fallite: {len(falliti)}")
    if t_init:
        print(f"Cold start della pipeline (una tantum, non nelle misure): {t_init:.1f}s")

    print("\nLatenza end-to-end (s):")
    _stampa_riga_stat(end_to_end)

    if ha_breakdown:
        pipeline = riassumi([r["pipeline_s"] for r in ok])
        generazione = riassumi([r["generazione_s"] for r in ok])
        print("\n  di cui pipeline RAG (classificazione + retrieval + grafo + rerank):")
        _stampa_riga_stat(pipeline)
        print("\n  di cui generazione (API Anthropic, esterna al progetto):")
        _stampa_riga_stat(generazione)
        quota = 100 * pipeline["p50_s"] / end_to_end["p50_s"]
        print(f"\n  → la pipeline è il {quota:.1f}% della latenza mediana; il resto è l'LLM.")

        print("\nBreakdown per nodo del grafo (mediana, s):")
        nodi = {}
        for r in ok:
            for nodo, t in r["per_nodo"].items():
                nodi.setdefault(nodo, []).append(t)
        for nodo, tempi in sorted(nodi.items(), key=lambda kv: -statistics.median(kv[1])):
            mediana = statistics.median(tempi)
            barra = "█" * max(1, int(mediana / max(0.001, end_to_end["p50_s"]) * 40))
            print(f"  {nodo:<24} {mediana:6.3f}  {barra}")

    if falliti:
        print(f"\n{len(falliti)} esecuzioni fallite:")
        for r in falliti[:5]:
            print(f"  - {r['query'][:50]}: {r['errore'][:80]}")

    # --- file di output ---
    csv_path = f"{stats_file_prefix}.csv"
    json_path = f"{stats_file_prefix}.json"

    nodi_colonne = sorted({n for r in ok for n in r.get("per_nodo", {})})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "giro", "totale_s", "pipeline_s", "generazione_s"] + nodi_colonne)
        for r in ok:
            w.writerow(
                [
                    r["query"],
                    r["giro"],
                    f"{r['totale_s']:.3f}",
                    f"{r['pipeline_s']:.3f}" if r["pipeline_s"] is not None else "",
                    f"{r['generazione_s']:.3f}" if r["generazione_s"] is not None else "",
                ]
                + [f"{r.get('per_nodo', {}).get(n, 0):.3f}" for n in nodi_colonne]
            )

    riepilogo = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modalita": mode,
        "corpus": corpus,
        "cold_start_s": t_init,
        "n_query_distinte": len({r["query"] for r in ok}),
        "end_to_end": end_to_end,
    }
    if ha_breakdown:
        riepilogo["pipeline_rag"] = riassumi([r["pipeline_s"] for r in ok])
        riepilogo["generazione_llm"] = riassumi([r["generazione_s"] for r in ok])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(riepilogo, f, indent=2, ensure_ascii=False)

    print(f"\nDettaglio per esecuzione: {csv_path}")
    print(f"Riepilogo aggregato:      {json_path}")

    # --- bullet pronti ---
    print("\n" + "-" * 62)
    print("BULLET UTILIZZABILI (numeri reali di questa esecuzione)")
    print("-" * 62)
    n_note = corpus["n_note"] or "N"
    n_chunk = corpus["n_chunk"] or "M"
    print(
        f'  "{end_to_end["p50_s"]:.1f}s median / {end_to_end["p95_s"]:.1f}s p95 end-to-end query '
        f'latency over a {n_note}-note ({n_chunk}-chunk) Obsidian vault"'
    )
    if ha_breakdown:
        pipeline = riassumi([r["pipeline_s"] for r in ok])
        print(
            f'  "hybrid BM25+vector retrieval with wikilink-graph expansion in '
            f'{pipeline["p50_s"]*1000:.0f}ms median ({pipeline["p95_s"]*1000:.0f}ms p95); '
            f'remaining latency is LLM generation"'
        )
    print()


def _stampa_riga_stat(s):
    print(
        f"  p50 {s['p50_s']:.2f}   p90 {s['p90_s']:.2f}   p95 {s['p95_s']:.2f}   "
        f"mean {s['mean_s']:.2f} ± {s['stdev_s']:.2f}   min {s['min_s']:.2f}   max {s['max_s']:.2f}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def carica_query(path):
    righe = Path(path).read_text(encoding="utf-8").splitlines()
    query = [r.strip() for r in righe if r.strip() and not r.strip().startswith("#")]
    if not query:
        sys.exit(f"{path} non contiene query.")
    return query


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["direct", "http"], default="direct",
                   help="direct: pipeline in-process con breakdown per nodo (default). "
                        "http: contro FastAPI già avviato, solo end-to-end.")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="base URL del backend in modalità http")
    p.add_argument("--queries", help="file con una query per riga ('#' per i commenti)")
    p.add_argument("--repeat", type=int, default=1, help="ripetizioni dell'intero set (default: 1)")
    p.add_argument("--warmup", type=int, default=1, help="chiamate iniziali non misurate (default: 1)")
    p.add_argument("--conversational", action="store_true",
                   help="riusa un solo thread_id: misura il caso conversazionale (chat_history crescente)")
    p.add_argument("--out", default="rag_benchmark", help="prefisso dei file di output (default: rag_benchmark)")
    p.add_argument("--debug", action="store_true", help="stampa il tempo di ogni nodo durante l'esecuzione")
    args = p.parse_args()

    queries = carica_query(args.queries) if args.queries else QUERIES

    print(f"Benchmark RAG - {datetime.now().isoformat(timespec='seconds')}")
    print(f"{len(queries)} query x {args.repeat} giro/i = {len(queries) * args.repeat} esecuzioni "
          f"(modalità: {args.mode})\n")

    if args.mode == "direct":
        run_query, corpus, t_init = crea_runner_diretto(debug=args.debug)
    else:
        run_query, corpus, t_init = crea_runner_http(args.url)

    risultati = benchmark(run_query, queries, args.repeat, args.warmup, args.conversational)
    stampa_report(risultati, corpus, args.out, args.mode, t_init)


if __name__ == "__main__":
    main()
