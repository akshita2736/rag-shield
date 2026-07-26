"""
Layer 2: semantic ML classifier.

SentenceTransformer embeddings -> Logistic Regression -> P(injection).

Deliberately a low-capacity linear model over pretrained embeddings, not
a fancier one (see build spec for the reasoning: simplicity + a clean
ablation against the heuristic baseline, not claims of coefficient-level
interpretability or automatic probability calibration).

This module owns embedding + the classifier itself. scripts/train_classifier.py
calls train_classifier() and saves the result; app.py/firewall.py call
predict_proba() at inference time. Kept separate from rag_pipeline.py's
embedder even though both load the same model name — different concerns,
and this module needs to work standalone (e.g. from scripts/, without
spinning up a FAISS index).
"""

from __future__ import annotations

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from ragshield import config

_embedder: SentenceTransformer | None = None
_classifier: LogisticRegression | None = None


class ClassifierNotTrainedError(RuntimeError):
    """Raised when predict_proba() is called before a classifier exists on disk."""


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns shape (n_texts, embedding_dim)."""
    embeddings = _get_embedder().encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")


def train_classifier(
    texts: list[str],
    labels: list[int],
    save_path=config.CLASSIFIER_PATH,
) -> LogisticRegression:
    """
    Fit Logistic Regression on embedded texts and save to disk.
    labels: 1 = injection, 0 = benign.
    """
    embeddings = embed(texts)
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    model.fit(embeddings, labels)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    return model


def load_classifier(path=config.CLASSIFIER_PATH) -> LogisticRegression:
    """Load a previously trained classifier from disk."""
    global _classifier
    if _classifier is None:
        if not path.exists():
            raise ClassifierNotTrainedError(
                f"No trained classifier found at {path}. "
                "Run scripts/train_classifier.py first."
            )
        _classifier = joblib.load(path)
    return _classifier


def predict_proba(texts: list[str], model: LogisticRegression | None = None) -> np.ndarray:
    """
    Return P(injection) for each text, shape (n_texts,).

    If `model` isn't passed explicitly, loads (and caches) the classifier
    saved at config.CLASSIFIER_PATH. Pass `model` explicitly during
    evaluation (e.g. category-held-out runs) where you're testing a
    freshly-fit model that hasn't been saved to disk.
    """
    clf = model if model is not None else load_classifier()
    embeddings = embed(texts)
    # LogisticRegression.classes_ is [0, 1]; column 1 = P(class=1=injection)
    return clf.predict_proba(embeddings)[:, 1]