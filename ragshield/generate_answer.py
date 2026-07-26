"""
Final answer generation: takes ONLY the sanitised context produced by
firewall.sanitised_context() and answers the user's query from it.

This module must never receive raw retrieved chunks directly — only
firewall-approved text. That boundary is enforced by app.py's wiring,
not by anything in this file, so keep that in mind when calling it.

Reuses llm_judge.get_client() for the Groq client rather than
constructing a second one — same fix in spirit as firewall.bind_embedder().
"""

from __future__ import annotations

from ragshield import config
from ragshield.llm_judge import get_client

_ANSWER_PROMPT_TEMPLATE = """Answer the user's question using ONLY the context below. If the
context doesn't contain enough information to answer, say so plainly
rather than guessing.

Context:
\"\"\"
{context}
\"\"\"

Question: {query}

Answer:
"""


def generate_answer(sanitised_context: str, user_query: str) -> str:
    """Generate the final answer from sanitised (firewall-approved) context only."""
    if not sanitised_context.strip():
        return (
            "All retrieved context was quarantined by RAGShield, so there's "
            "no safe content left to answer from."
        )

    prompt = _ANSWER_PROMPT_TEMPLATE.format(context=sanitised_context, query=user_query)

    response = get_client().chat.completions.create(
        model=config.GROQ_ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()