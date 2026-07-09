"""
Diagnostic script for a single-day TEMPORAL query not retrieving the
expected daily note. Queries ChromaDB directly (bypasses the pipeline)
to isolate the problem between:
  (A) note not indexed,
  (B) creation_date missing/wrong/wrong type,
  (C) type/format mismatch in the filter (str "2026-07-09" vs int 20260709).

Usage:
    python diagnostica_daily_note.py            # uses today
    python diagnostica_daily_note.py 2026-07-09 # explicit date
"""

import sys
from datetime import datetime
from pathlib import Path

import chromadb

# Same path as CHROMA_PERSIST_DIR in settings.py (BASE_DIR / "chroma_db").
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
# Substring identifying the daily note in the 'source' field. Adjust if needed.
FRAMMENTO_SOURCE = "09-07-26"


def data_target() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return datetime.now().strftime("%Y-%m-%d")


data_iso = data_target()                       # "2026-07-09"
data_int = int(data_iso.replace("-", ""))      # 20260709
data_int_str = str(data_int)                   # "20260709" (if stored as string)

print(f"Data target: {data_iso}")
print(f"  -> come intero: {data_int}")
print(f"  -> come stringa compatta: {data_int_str!r}")
print(f"Chroma dir: {CHROMA_DIR}\n")

if not CHROMA_DIR.exists():
    print(f"!!! La cartella {CHROMA_DIR} non esiste. Path sbagliato?")
    sys.exit(1)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# list_collections shape differs across versions: handle both Collection
# objects (with .name) and plain strings.
collezioni = client.list_collections()
nomi = [c.name if hasattr(c, "name") else c for c in collezioni]
print("Collezioni trovate:", nomi, "\n")

for nome in nomi:
    col = client.get_collection(nome)
    tot = col.count()
    print(f"=== Collezione '{nome}' — {tot} record totali ===")

    dati = col.get(include=["metadatas"])
    ids = dati["ids"]
    metas = dati["metadatas"] or []

    # (A) Is the note indexed at all?
    corrispondenti = [
        (i, m) for i, m in zip(ids, metas)
        if FRAMMENTO_SOURCE in str(m.get("source", ""))
    ]
    print(f"[A] chunk con '{FRAMMENTO_SOURCE}' nel 'source': {len(corrispondenti)}")
    for i, m in corrispondenti[:5]:
        cd = m.get("creation_date")
        print(f"    - id={i}")
        print(f"      source={m.get('source')!r}")
        print(f"      creation_date={cd!r}  (tipo Python: {type(cd).__name__})")
    if len(corrispondenti) > 5:
        print(f"    ... e altri {len(corrispondenti) - 5} chunk")

    # (B/C) How many records match the target date, across formats?
    n_int = sum(1 for m in metas if m.get("creation_date") == data_int)
    n_str = sum(1 for m in metas if m.get("creation_date") == data_iso)
    n_intstr = sum(1 for m in metas if m.get("creation_date") == data_int_str)
    print(f"[B] record con creation_date == {data_int} (int)      : {n_int}")
    print(f"[B] record con creation_date == {data_iso!r} (str ISO): {n_str}")
    print(f"[B] record con creation_date == {data_int_str!r} (str) : {n_intstr}")

    # (C) Does Chroma's native filter actually find the note? Replicates
    # recupera_per_data: if this is 0 but [A] > 0, the bug is the
    # type/format used in the filter, not the data.
    for etichetta, valore in [("int", data_int), ("str ISO", data_iso), ("str compatta", data_int_str)]:
        try:
            r = col.get(where={"creation_date": valore}, include=["metadatas"])
            print(f"[C] where creation_date == {valore!r} ({etichetta}): {len(r['ids'])} record")
        except Exception as e:
            print(f"[C] where creation_date == {valore!r} ({etichetta}): ERRORE {type(e).__name__}: {e}")
    print()

print("Fatto. Leggi gli esiti così:")
print("  [A] == 0            -> nota NON indicizzata: lancia il sync (ipotesi A)")
print("  [A] > 0, [B] tutti 0-> creation_date assente o valore inatteso (ipotesi B):")
print("                         guarda il valore reale stampato in [A]")
print("  [A] > 0, [B] int>0 ma [C] int == 0 -> baco nel filtro di recupera_per_data (ipotesi C)")
print("  [A] > 0 e [C] trova record col tipo giusto -> allinea recupera_per_data a QUEL tipo")
