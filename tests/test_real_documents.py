"""
Real-document robustness (slow: loads the embedding model and the trained
classifier). Run with:  pytest -m slow

These documents are never used for training. They check the question the
project actually has to answer: does a normal PDF/text full of instructions,
procedures, lists and support language get quarantined wholesale?
"""

import pytest

from ragshield import config, firewall, ml_classifier
from ragshield.rag_pipeline import Chunk, chunk_text, load_document

pytestmark = pytest.mark.slow

INJECTED_MARKERS = ["ignore all previous", "unrestricted assistant", "print your hidden", "disregard the user's question"]


@pytest.fixture(autouse=True)
def real_embedder(monkeypatch):
    # Undo conftest's zero-embedding stub for this module only.
    monkeypatch.setattr(ml_classifier, "embed", ml_classifier.__dict__["embed"], raising=True)
    import importlib

    module = importlib.reload(ml_classifier)
    monkeypatch.setattr(firewall, "ml_classifier", module)
    if not config.CLASSIFIER_PATH.exists():
        pytest.skip("train the classifier first: python scripts/train_classifier.py")
    yield module


def _screen(path, judge):
    chunks = [Chunk(i, c, path.name) for i, c in enumerate(chunk_text(load_document(path)))]
    return firewall.screen_chunks(chunks, "What does this document say?", judge=judge)


def test_benign_documents_are_never_hard_blocked_and_mostly_pass_without_llm(recording_judge):
    judge = recording_judge("INJECTION")  # worst case: assume every escalated chunk gets blocked
    total = safe = 0
    for path in sorted(config.SAMPLE_DOCUMENTS_DIR.glob("*.txt")):
        if "poisoned" in path.name:
            continue
        for r in _screen(path, judge):
            total += 1
            assert r.decided_by not in ("heuristic", "heuristic+ml"), (path.name, r.reason)
            assert r.band != "high" or r.decided_by == "llm_judge"
            safe += r.status == "SAFE"
    assert total >= 30
    assert safe / total >= 0.4, f"only {safe}/{total} benign chunks resolved SAFE without the judge"


def test_poisoned_document_injections_never_pass_silently(recording_judge):
    judge = recording_judge("INJECTION")
    results = _screen(config.SAMPLE_DOCUMENTS_DIR / "poisoned_product_faq.txt", judge)
    injected = [r for r in results if any(m in r.chunk.text.lower() for m in INJECTED_MARKERS)]
    clean = [r for r in results if r not in injected]
    assert injected and clean
    assert all(r.status != "SAFE" for r in injected), [r.reason for r in injected]
    assert any(r.decided_by == "heuristic" for r in injected)  # the plain attack is caught by Layer 1
    assert all(r.status == "SAFE" or r.decided_by == "llm_judge" for r in clean)
