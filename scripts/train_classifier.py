"""
scripts/train_classifier.py

Trains the Layer 2 semantic classifier (ragshield/ml_classifier.py) on
data/benchmark/benchmark.jsonl and saves it to config.CLASSIFIER_PATH
together with a provenance sidecar (benchmark hash, record counts,
timestamp) that scripts/run_evaluation.py and app.py use to detect a
stale model.

The saved classifier is trained on ALL benchmark records: it is the
production model. The random-split metrics printed at the end come from a
SEPARATE model fitted on the training portion only, purely as a smoke test
that training works and the classifier is not degenerate. Full evaluation
(group-held-out, category-held-out, ablation, error analysis) lives in
scripts/run_evaluation.py.

Run with: python scripts/train_classifier.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from ragshield import config, ml_classifier  # noqa: E402

# Decision threshold used ONLY for this script's own sanity-check metrics.
# It has nothing to do with config.LOW_THRESHOLD/HIGH_THRESHOLD, which are
# firewall routing bands, not a classification cutoff.
_SMOKE_TEST_DECISION_THRESHOLD = 0.5


def benchmark_sha256(path: Path = config.BENCHMARK_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_benchmark() -> tuple[list[str], list[int]]:
    if not config.BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"No benchmark found at {config.BENCHMARK_PATH}. "
            "Run scripts/build_benchmark.py first."
        )

    texts: list[str] = []
    labels: list[int] = []
    with open(config.BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(int(record["label"]))

    return texts, labels


def main():
    texts, labels = load_benchmark()
    print(f"[train_classifier] Loaded {len(texts)} records from {config.BENCHMARK_PATH}")

    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    print(f"[train_classifier] Label balance: {n_positive} injection / {n_negative} benign")

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Benchmark contains only one class — cannot train a classifier. "
            "Check scripts/build_benchmark.py output."
        )

    # --- Smoke test on a random split (separate, unsaved model) ---
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    print(
        f"[train_classifier] Smoke-test split: {len(train_texts)} train / "
        f"{len(test_texts)} test (stratified, seed={config.RANDOM_SEED})"
    )
    smoke_model = ml_classifier.fit_classifier(train_texts, train_labels)
    test_probs = ml_classifier.predict_proba_segment(test_texts, model=smoke_model)
    test_preds = (test_probs >= _SMOKE_TEST_DECISION_THRESHOLD).astype(int)

    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()

    print("\n[train_classifier] Random-split smoke-test metrics "
          f"(max-sentence scoring, decision threshold={_SMOKE_TEST_DECISION_THRESHOLD}):")
    print(f"  accuracy:  {accuracy:.3f}")
    print(f"  precision: {precision:.3f}")
    print(f"  recall:    {recall:.3f}")
    print(f"  f1:        {f1:.3f}")
    print(f"  confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")
    print(
        "  NOTE: the benchmark reuses seed sentences across categories, so a "
        "random split is optimistic. Group-held-out numbers are the honest ones."
    )

    # --- Production model on ALL records ---
    print("\n[train_classifier] Fitting production classifier on all records...")
    unit_texts, _ = ml_classifier.training_units(texts, labels)
    ml_classifier.train_classifier(
        texts,
        labels,
        metadata={
            "benchmark_path": str(config.BENCHMARK_PATH.relative_to(config.BASE_DIR)),
            "benchmark_sha256": benchmark_sha256(),
        },
    )
    print(
        f"[train_classifier] Saved classifier to {config.CLASSIFIER_PATH} "
        f"({len(texts)} records -> {len(unit_texts)} training units after benign "
        "sentence expansion) and provenance to "
        f"{ml_classifier.meta_path_for(config.CLASSIFIER_PATH)}"
    )
    print(
        "\n[train_classifier] Run scripts/run_evaluation.py for the ablation, "
        "category-held-out and error analysis, and "
        "scripts/run_evaluation.py --group-held-out for template-group cross-validation."
    )


if __name__ == "__main__":
    main()
