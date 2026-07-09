"""Gradio debug UI for the Obsidian RAG System pipeline."""

import uuid

import gradio as gr

from settings import VAULT_PATH
from pipeline import inizializza_sistema, rispondi

print("Inizializzazione pipeline (carico indice + retriever + grafo LangGraph)...")
grafo_compilato = inizializza_sistema(VAULT_PATH)
print("Pipeline pronta.")


def chiedi(domanda, thread_id):
    """thread_id lives in a gr.State (one instance per browser session);
    generated once on the first message and reused for later turns."""
    if not domanda.strip():
        return "Scrivi una domanda prima di inviare.", "", thread_id

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    risposta, stop_reason = rispondi(domanda, grafo_compilato, thread_id)

    avviso = "⚠️ Risposta troncata (max_tokens raggiunto)" if stop_reason == "max_tokens" else f"✅ Completata normalmente ({stop_reason})"

    return risposta, avviso, thread_id


demo = gr.Interface(
    fn=chiedi,
    inputs=[
        gr.Textbox(label="Question", placeholder="Es. What is clustering?", lines=2),
        gr.State(value=None),  # session thread_id, None until first generated
    ],
    outputs=[
        gr.Markdown(label="Answer"),
        gr.Textbox(label="Debug: generation state"),
        gr.State(),  # rewrites thread_id back into session state after each turn
    ],
    title="Obsidian's Vault RAG System — Debug",
    description="Test interface for the RAG pipeline on the Obsidian vault.",
)

if __name__ == "__main__":
    demo.launch()
