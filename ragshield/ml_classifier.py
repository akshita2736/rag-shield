"""
Layer 2: semantic ML classifier.

SentenceTransformer embeddings -> Logistic Regression -> P(injection).

Deliberately a low-capacity linear model over pretrained embeddings: simple,
fast to retrain, and a clean ablation against the heuristic baseline. It is
not calibrated and makes no claim of coefficient-level interpretability.

Two design decisions matter here and are worth being able to defend:

1. Segment-level (max-sentence) scoring at inference. Each chunk is split
   into sentences, every sentence is scored, and the chunk's risk is the
   maximum. Whole-chunk embedding dilutes a single injected sentence inside
   otherwise benign text: in category-held-out testing, whole-chunk scoring
   detected 0% of unseen indirect injections at the 0.5 cutoff, max-sentence
   scoring detected 100%.

2. Sentence-level training units for benign records. Max-sentence scoring
   judges single sentences, so the classifier must have seen single benign
   sentences from realistic documents. training_units() therefore adds each
   sentence of every benign record as an extra label-0 example. This is
   label-safe (every sentence of a benign record is benign) and is NOT done
   for attack records, whose wrapper sentences in indirect_injection would
   be mislabelled. Before this fix, max-sentence scoring pushed 19 of 34
   legitimate real-document chunks above HIGH_THRESHOLD; after it, 0 did.

This module owns embedding + the classifier itself. scripts/train_classifier.py
calls train_classifier() and saves the result; firewall.py calls
predict_proba_segment() at inference time. Evaluation scripts use
fit_classifier() so fold models are trained exactly like production.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import platform

import joblib
import numpy as np
import sentence_transformers
import sklearn
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from ragshield import config

_embedder: SentenceTransformer | None = None
_classifier: LogisticRegression | None = None

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

SCORING_STRATEGY = "max_sentence"


class ClassifierNotTrainedError(RuntimeError):
    """Raised when predict_proba() is called before a classifier exists on disk."""


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, revision=config.EMBEDDING_MODEL_REVISION)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns shape (n_texts, embedding_dim)."""
    embeddings = _get_embedder().encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter: splits on '.', '!', '?' followed by
    whitespace. Not meant to handle abbreviations/decimals/ellipses
    perfectly; good enough for the prose this project's chunks contain.
    A chunk with no sentence-ending punctuation is returned as a single
    "sentence" (itself). Empty/whitespace-only text returns [].
    """
    stripped = text.strip()
    if not stripped:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    return sentences if sentences else [stripped]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def training_units(texts: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """
    Expand records into the units the classifier is trained on: every record
    as a whole, plus each individual sentence of every BENIGN record (see
    module docstring for why only benign records are expanded).
    """
    unit_texts: list[str] = []
    unit_labels: list[int] = []
    for text, label in zip(texts, labels):
        unit_texts.append(text)
        unit_labels.append(int(label))
        if int(label) == 0:
            sentences = split_sentences(text)
            if len(sentences) > 1:
                unit_texts.extend(sentences)
                unit_labels.extend([0] * len(sentences))
    return unit_texts, unit_labels


def fit_classifier(texts: list[str], labels: list[int]) -> LogisticRegression:
    """
    Fit the Layer 2 classifier in memory (never saved). Used both by
    train_classifier() and by evaluation folds, so a fold model is trained
    exactly the way the production model is.
    """
    unit_texts, unit_labels = training_units(texts, labels)
    embeddings = embed(unit_texts)
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    model.fit(embeddings, unit_labels)
    return model


def train_classifier(
    texts: list[str],
    labels: list[int],
    save_path: Path = config.CLASSIFIER_PATH,
    metadata: dict | None = None,
) -> LogisticRegression:
    """
    Fit Logistic Regression and save it to disk together with a provenance
    sidecar (<save_path stem>.meta.json) so evaluation can detect a stale
    model. labels: 1 = injection, 0 = benign.
    """
    model = fit_classifier(texts, labels)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    unit_texts, _ = training_units(texts, labels)
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
        "scoring_strategy": SCORING_STRATEGY,
        "python_version": platform.python_version(),
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "n_records": len(texts),
        "n_training_units": len(unit_texts),
        "n_positive_records": int(sum(int(l) for l in labels)),
        "sklearn_version": sklearn.__version__,
    }
    if metadata:
        meta.update(metadata)
    meta_path_for(save_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return model


def meta_path_for(model_path: Path = config.CLASSIFIER_PATH) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + ".meta.json")


def classifier_metadata(model_path: Path = config.CLASSIFIER_PATH) -> dict | None:
    """Provenance sidecar written by train_classifier(), or None if absent."""
    path = meta_path_for(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_classifier(path: Path = config.CLASSIFIER_PATH) -> LogisticRegression:
    """Load (and cache) a previously trained classifier from disk."""
    global _classifier
    if _classifier is None:
        path = Path(path)
        if not path.exists():
            raise ClassifierNotTrainedError(
                f"No trained classifier found at {path}. "
                "Run scripts/train_classifier.py first."
            )
        _classifier = joblib.load(path)
        _warn_if_environment_differs(path)
    return _classifier


def _warn_if_environment_differs(path: Path) -> None:
    """A pickled LogisticRegression over MiniLM embeddings is only reproducible
    in the environment that produced it; say so loudly instead of silently
    scoring with a mismatched stack."""
    meta = classifier_metadata(path)
    if not meta:
        return
    current = {
        "sklearn_version": sklearn.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
    }
    mismatches = {k: (meta.get(k), v) for k, v in current.items() if meta.get(k) not in (None, v)}
    if mismatches:
        print(
            "[ml_classifier] Warning: classifier was trained with a different environment "
            f"{mismatches}. Retrain with scripts/train_classifier.py to be safe."
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_proba(texts: list[str], model: LogisticRegression | None = None) -> np.ndarray:
    """
    Whole-chunk P(injection) for each text, shape (n_texts,).

    NOT the production path (firewall.py uses predict_proba_segment). Kept
    as the baseline that scripts/segmentation_experiment.py compares against.
    Pass `model` explicitly to score with an unsaved fold model.
    """
    clf = model if model is not None else load_classifier()
    if not texts:
        return np.zeros(0, dtype="float64")
    embeddings = embed(texts)
    # LogisticRegression.classes_ is [0, 1]; column 1 = P(class=1=injection)
    return clf.predict_proba(embeddings)[:, 1]


def predict_proba_segment(texts: list[str], model: LogisticRegression | None = None) -> np.ndarray:
    """
    Production scoring: split each text into sentences, score every sentence,
    return each text's maximum sentence-level injection probability. All
    sentences of all texts are embedded in one batch.

    A text with no sentences after splitting (empty/whitespace-only) gets
    risk 0.0 rather than an error.
    """
    clf = model if model is not None else load_classifier()

    all_sentences: list[str] = []
    owner_index: list[int] = []
    for i, text in enumerate(texts):
        for sentence in split_sentences(text):
            all_sentences.append(sentence)
            owner_index.append(i)

    max_probs = np.zeros(len(texts), dtype="float64")
    if not all_sentences:
        return max_probs

    sentence_probs = clf.predict_proba(embed(all_sentences))[:, 1]
    for owner, prob in zip(owner_index, sentence_probs):
        if prob > max_probs[owner]:
            max_probs[owner] = prob

    return max_probs


def sentence_scores(text: str, model: LogisticRegression | None = None) -> list[tuple[str, float]]:
    """Per-sentence probabilities for one text; used by the UI to explain a score."""
    clf = model if model is not None else load_classifier()
    sentences = split_sentences(text)
    if not sentences:
        return []
    probs = clf.predict_proba(embed(sentences))[:, 1]
    return list(zip(sentences, (float(p) for p in probs)))
