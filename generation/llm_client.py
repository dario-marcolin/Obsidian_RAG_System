"""Wraps the Anthropic API call for the final answer generation step."""

from generation.prompts import costruisci_prompt, costruisci_system_prompt
from settings import ANTHROPIC_API_KEY, ANTHROPIC_GENERATION_MODEL, ANTHROPIC_MAX_TOKENS
import anthropic

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def genera_risposta_claude(query: str, contesto: str, tipo: str) -> tuple[str, str]:
    system_prompt = costruisci_system_prompt(tipo)
    prompt = costruisci_prompt(query, contesto, tipo)
    response = client.messages.create(
        model=ANTHROPIC_GENERATION_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    testo_risposta = next(
        (blocco.text for blocco in response.content if blocco.type == "text"),
        ""
    )

    return testo_risposta, response.stop_reason


if __name__ == "__main__":
    contesto_di_prova = "[[Clustering]]\nIl clustering è una tecnica di apprendimento non supervisionato che raggruppa dati simili."
    risposta, stop_reason = genera_risposta_claude(
        query="Cos'è il clustering?",
        contesto=contesto_di_prova,
        tipo="CONTENUTISTICA",
    )
    print(risposta)
    print(f"\n[DEBUG] stop_reason: {stop_reason}")