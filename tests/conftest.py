"""Shared fixtures. Unit tests never load the embedding model or call Groq."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "")  # unit tests must never hit the network

from ragshield import llm_judge, ml_classifier  # noqa: E402
from ragshield.rag_pipeline import Chunk  # noqa: E402


class FakeModel:
    """Stands in for LogisticRegression: returns a fixed P(injection) per row."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]] * len(X))


class RecordingJudge:
    def __init__(self, verdict="SAFE", error=None):
        self.verdict = verdict
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text, query):
        self.calls.append((text, query))
        return llm_judge.JudgeResult(self.verdict, 0.9, "fake", error=self.error)


@pytest.fixture(autouse=True)
def no_embedder(monkeypatch):
    """Replace embedding with zeros so Layer 2 never loads SentenceTransformers."""
    monkeypatch.setattr(ml_classifier, "embed", lambda texts: np.zeros((len(texts), 4), dtype="float32"))
    monkeypatch.setattr(llm_judge, "_client", None)


@pytest.fixture
def chunk():
    def make(text: str) -> Chunk:
        return Chunk(chunk_id=0, text=text, source="test")

    return make


@pytest.fixture
def fake_model():
    return FakeModel


@pytest.fixture
def recording_judge():
    return RecordingJudge
