"""
Firewall: orchestrates heuristics.py -> ml_classifier.py -> llm_judge.py
and produces the sanitised context that generate_answer.py is allowed to see.

Routing table (the architectural claim of the project; tests/test_firewall.py
pins it with fake layers):

  Layer 1 critical match                          -> BLOCKED   no LLM call
  ML prob <  LOW_THRESHOLD  and no Layer 1 match  -> SAFE      no LLM call
  ML prob >= HIGH_THRESHOLD and any Layer 1 match -> BLOCKED   no LLM call (two independent layers agree)
  anything else                                   -> Layer 3 judge decides:
                                                     SAFE verdict     -> REVIEW, included
                                                     INJECTION/failed -> BLOCKED, excluded

"Anything else" covers the ambiguous band, a high ML score with no
corroborating heuristic evidence (legitimate manuals/policies/support text
score high, so ML alone never blocks), and a low ML score that a
non-critical heuristic contradicts (e.g. base64 or homoglyph obfuscation).
The judge fails closed, so an unavailable Layer 3 quarantines rather than
passes.

This is also where the shared-embedder fix lives: rag_pipeline.py and
ml_classifier.py would otherwise each load their own SentenceTransformer.
bind_embedder() lets app.py wire them together so the model loads once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ragshield import config, heuristics, llm_judge, ml_classifier
from ragshield.llm_judge import JudgeResult
from ragshield.rag_pipeline import Chunk, RagPipeline

Status = Literal["SAFE", "BLOCKED", "REVIEW"]
Band = Literal["critical", "low", "ambiguous", "high"]
Layer = Literal["heuristic", "ml", "heuristic+ml", "llm_judge"]

JudgeFn = Callable[[str, str], JudgeResult]

# Evaluation-only stand-in for Layer 3, used by the "Heuristic+ML (no LLM)"
# ablation so that the ablation shares this exact routing code instead of
# re-implementing it. Everything that would have gone to the judge is
# decided by ML probability against this cutoff. Production never uses it:
# a missing/unavailable judge fails closed instead.
ABLATION_FALLBACK_THRESHOLD = 0.5


class _AblationNoJudge:
    """Sentinel type; see ABLATION_NO_JUDGE."""


ABLATION_NO_JUDGE = _AblationNoJudge()


@dataclass
class ChunkResult:
    chunk: Chunk
    status: Status  # UI badge: SAFE / BLOCKED / REVIEW (REVIEW = escalated to Layer 3 and approved)
    final_included: bool  # whether this chunk's text is passed to the answer LLM
    risk: float  # Layer 2 max-sentence probability (1.0 when a critical heuristic blocked before ML ran)
    reason: str
    band: Band = "low"  # which ML band the chunk fell in ("critical" = never scored)
    decided_by: Layer = "ml"  # which layer made the final decision
    heuristic_matches: list[str] = field(default_factory=list)
    judge: JudgeResult | None = None  # populated only when Layer 3 was consulted


def bind_embedder(rag: RagPipeline) -> None:
    """
    Point ml_classifier at the RagPipeline's already-loaded embedder instead
    of letting it lazily load a second copy of the same SentenceTransformer
    model. Call this once, right after creating your RagPipeline, before
    screen_chunks()/evaluate_chunk() run.
    """
    ml_classifier._embedder = rag.embedder  # noqa: SLF001 - deliberate, see docstring


def ml_band(prob: float) -> Band:
    if prob < config.LOW_THRESHOLD:
        return "low"
    if prob >= config.HIGH_THRESHOLD:
        return "high"
    return "ambiguous"


def _route(
    chunk: Chunk,
    heuristic_result: heuristics.HeuristicResult,
    ml_prob: float | None,
    user_query: str,
    judge: JudgeFn | _AblationNoJudge | None,
) -> ChunkResult:
    if heuristic_result.critical_triggered:
        return ChunkResult(
            chunk=chunk,
            status="BLOCKED",
            final_included=False,
            risk=1.0,
            reason="Critical heuristic triggered: " + ", ".join(heuristic_result.matched_rules),
            band="critical",
            decided_by="heuristic",
            heuristic_matches=list(heuristic_result.matched_rules),
        )

    assert ml_prob is not None  # scored for every non-critical chunk
    band = ml_band(ml_prob)
    matches = list(heuristic_result.matched_rules)

    if band == "low" and not matches:
        return ChunkResult(
            chunk=chunk,
            status="SAFE",
            final_included=True,
            risk=ml_prob,
            reason="Below ML risk threshold, no heuristic evidence",
            band=band,
            decided_by="ml",
            heuristic_matches=matches,
        )

    if band == "high" and matches:
        return ChunkResult(
            chunk=chunk,
            status="BLOCKED",
            final_included=False,
            risk=ml_prob,
            reason=(
                "High ML risk corroborated by heuristic evidence: " + ", ".join(matches)
            ),
            band=band,
            decided_by="heuristic+ml",
            heuristic_matches=matches,
        )

    # Everything else is a contextual decision for Layer 3.
    if isinstance(judge, _AblationNoJudge):
        blocked = ml_prob >= ABLATION_FALLBACK_THRESHOLD
        return ChunkResult(
            chunk=chunk,
            status="BLOCKED" if blocked else "REVIEW",
            final_included=not blocked,
            risk=ml_prob,
            reason=(
                f"No LLM judge (ablation): ML {ml_prob:.2f} "
                f"{'>=' if blocked else '<'} fallback cutoff {ABLATION_FALLBACK_THRESHOLD}"
            ),
            band=band,
            decided_by="ml",
            heuristic_matches=matches,
        )

    judge_fn: JudgeFn = judge if judge is not None else llm_judge.judge
    judge_result = judge_fn(chunk.text, user_query)
    included = judge_result.verdict == "SAFE" and not judge_result.failed

    why = (
        f"Escalated to LLM judge (ML {ml_prob:.2f}, band {band}"
        + (f", heuristics: {', '.join(matches)}" if matches else "")
        + f"). Verdict: {judge_result.verdict} — {judge_result.reasoning}"
    )
    return ChunkResult(
        chunk=chunk,
        status="REVIEW" if included else "BLOCKED",
        final_included=included,
        risk=ml_prob,
        reason=why,
        band=band,
        decided_by="llm_judge",
        heuristic_matches=matches,
        judge=judge_result,
    )


def screen_chunks(
    chunks: list[Chunk],
    user_query: str,
    model=None,
    judge: JudgeFn | _AblationNoJudge | None = None,
) -> list[ChunkResult]:
    """
    Run every retrieved chunk through the firewall. Layer 2 is batched
    across all non-critical chunks (one embedding call). `model` overrides
    the saved classifier (used by evaluation folds); `judge` overrides
    Layer 3 (tests pass a fake, the ablation passes ABLATION_NO_JUDGE).
    """
    heuristic_results = [heuristics.scan(chunk.text) for chunk in chunks]

    to_score = [i for i, h in enumerate(heuristic_results) if not h.critical_triggered]
    probs: dict[int, float] = {}
    if to_score:
        scored = ml_classifier.predict_proba_segment([chunks[i].text for i in to_score], model=model)
        probs = {i: float(p) for i, p in zip(to_score, scored)}

    return [
        _route(chunk, heuristic_results[i], probs.get(i), user_query, judge)
        for i, chunk in enumerate(chunks)
    ]


def evaluate_chunk(
    chunk: Chunk,
    user_query: str,
    model=None,
    judge: JudgeFn | _AblationNoJudge | None = None,
) -> ChunkResult:
    """Run one chunk through Layer 1 -> Layer 2 -> Layer 3 routing."""
    return screen_chunks([chunk], user_query, model=model, judge=judge)[0]


def sanitised_context(results: list[ChunkResult]) -> str:
    """Join the text of every chunk that passed (SAFE, or REVIEW + judge-approved)."""
    included = [r.chunk.text for r in results if r.final_included]
    return "\n\n".join(included)
