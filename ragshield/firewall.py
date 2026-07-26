"""
Firewall: orchestrates heuristics.py -> ml_classifier.py -> llm_judge.py
and produces the sanitised context that generate_answer.py is allowed to see.

This is also where the shared-embedder fix lives: rag_pipeline.py and
ml_classifier.py each load their own SentenceTransformer instance by
default (same model name, two copies in memory). bind_embedder() below
lets app.py wire them together so the model is only loaded once — call
it right after constructing your RagPipeline, before screening any chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ragshield import config, heuristics, llm_judge, ml_classifier
from ragshield.rag_pipeline import Chunk, RagPipeline

Status = Literal["SAFE", "BLOCKED", "REVIEW"]


@dataclass
class ChunkResult:
    chunk: Chunk
    status: Status  # UI badge: SAFE / BLOCKED / REVIEW (REVIEW = escalated to Layer 3)
    final_included: bool  # whether this chunk's text is passed to the LLM
    risk: float  # ML probability (or 1.0 for a critical-heuristic quarantine)
    reason: str


def bind_embedder(rag: RagPipeline) -> None:
    """
    Point ml_classifier at the RagPipeline's already-loaded embedder instead
    of letting it lazily load a second copy of the same SentenceTransformer
    model. Call this once, right after creating your RagPipeline, before
    screen_chunks()/evaluate_chunk() run.
    """
    ml_classifier._embedder = rag.embedder  # noqa: SLF001 - deliberate, see docstring


def evaluate_chunk(chunk: Chunk, user_query: str) -> ChunkResult:
    """Run one chunk through Layer 1 -> Layer 2 -> Layer 3 routing."""
    heuristic_result = heuristics.scan(chunk.text)

    if heuristic_result.critical_triggered:
        return ChunkResult(
            chunk=chunk,
            status="BLOCKED",
            final_included=False,
            risk=1.0,
            reason=(
                "Critical heuristic triggered: "
                f"{', '.join(heuristic_result.matched_rules)}"
            ),
        )

    ml_prob = float(ml_classifier.predict_proba([chunk.text])[0])

    if ml_prob < config.LOW_THRESHOLD:
        return ChunkResult(
            chunk=chunk,
            status="SAFE",
            final_included=True,
            risk=ml_prob,
            reason="Below ML risk threshold",
        )

    if ml_prob > config.HIGH_THRESHOLD:
        return ChunkResult(
            chunk=chunk,
            status="BLOCKED",
            final_included=False,
            risk=ml_prob,
            reason="Above ML risk threshold",
        )

    # Ambiguous band -> escalate to the contextual LLM judge.
    judge_result = llm_judge.judge(chunk.text, user_query)
    included = judge_result.verdict == "SAFE"

    return ChunkResult(
        chunk=chunk,
        status="REVIEW",  # kept as REVIEW even if the judge passed it, so the
        # UI/audit trail always shows this chunk was escalated, not
        # trivially approved by ML alone
        final_included=included,
        risk=ml_prob,
        reason=f"LLM judge verdict: {judge_result.verdict} — {judge_result.reasoning}",
    )


def screen_chunks(chunks: list[Chunk], user_query: str) -> list[ChunkResult]:
    """Run every retrieved chunk through the firewall."""
    return [evaluate_chunk(chunk, user_query) for chunk in chunks]


def sanitised_context(results: list[ChunkResult]) -> str:
    """Join the text of every chunk that passed (SAFE, or REVIEW+judge-approved)."""
    included = [r.chunk.text for r in results if r.final_included]
    return "\n\n".join(included)