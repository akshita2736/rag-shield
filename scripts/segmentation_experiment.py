"""
scripts/segmentation_experiment.py

Diagnostic experiment behind two production decisions in ml_classifier.py:
(1) max-sentence scoring instead of whole-chunk scoring, and (2) expanding
benign records into sentence-level training units. It does not modify the
firewall; it reports the evidence.

For each combination of
    scoring strategy   whole | max_sentence | max_window2
    benign expansion   off | on
it reports
  * 5-fold template-group-held-out Heuristic+ML (no LLM) precision/recall/F1/FPR
  * category-held-out detection of indirect_injection at the 0.5 cutoff
  * how the chunks of the benign documents in data/sample_documents fall into
    the firewall's SAFE / judge / high bands, and the score of the injected
    vs clean chunks of poisoned_product_faq.txt

Writes results/segmentation_experiment.json.
Run with: python scripts/segmentation_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from ragshield import config, heuristics, ml_classifier  # noqa: E402
from ragshield.rag_pipeline import chunk_text, load_document  # noqa: E402

INJECTED_MARKERS = ["ignore all previous", "unrestricted assistant", "print your hidden", "disregard the user's question"]

_cache: dict[str, np.ndarray] = {}


def embed(texts: list[str]) -> np.ndarray:
    missing = list({t for t in texts if t not in _cache})
    if missing:
        for text, vec in zip(missing, ml_classifier.embed(missing)):
            _cache[text] = vec
    return np.stack([_cache[t] for t in texts])


def sentences(text: str) -> list[str]:
    return ml_classifier.split_sentences(text) or [text]


def windows(text: str, k: int) -> list[str]:
    s = sentences(text)
    return [" ".join(s[i : i + k]) for i in range(max(1, len(s) - k + 1))]


STRATEGIES = {
    "whole": lambda t: [t],
    "max_sentence": sentences,
    "max_window2": lambda t: windows(t, 2),
}


def fit(texts, labels, expand_benign: bool):
    if expand_benign:
        texts, labels = ml_classifier.training_units(texts, labels)
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    model.fit(embed(list(texts)), list(labels))
    return model


def score(texts, model, strategy) -> np.ndarray:
    segments, owners = [], []
    for i, text in enumerate(texts):
        for seg in STRATEGIES[strategy](text):
            segments.append(seg)
            owners.append(i)
    probs = model.predict_proba(embed(segments))[:, 1]
    out = np.zeros(len(texts))
    for owner, p in zip(owners, probs):
        out[owner] = max(out[owner], p)
    return out


def no_llm_predict(texts, probs) -> np.ndarray:
    preds = []
    for text, p in zip(texts, probs):
        h = heuristics.scan(text)
        if h.critical_triggered:
            preds.append(1)
        elif p < config.LOW_THRESHOLD and not h.any_match:
            preds.append(0)
        elif p >= config.HIGH_THRESHOLD and h.any_match:
            preds.append(1)
        else:
            preds.append(int(p >= 0.5))
    return np.array(preds)


def main():
    records = [json.loads(l) for l in config.BENCHMARK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in records]
    y = np.array([r["label"] for r in records])
    groups = [r["lineage_id"] for r in records]  # leakage-free grouping (see build_benchmark.py)

    benign_docs = sorted(p for p in config.SAMPLE_DOCUMENTS_DIR.glob("*.txt") if "poisoned" not in p.name)
    benign_chunks = [c for p in benign_docs for c in chunk_text(load_document(p))]
    poisoned = chunk_text(load_document(config.SAMPLE_DOCUMENTS_DIR / "poisoned_product_faq.txt"))
    poisoned_labels = [int(any(m in c.lower() for m in INJECTED_MARKERS)) for c in poisoned]

    rows = []
    for expand in (False, True):
        for strategy in STRATEGIES:
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            all_y, all_p = [], []
            for tr, te in skf.split(records, y, groups):
                model = fit([texts[i] for i in tr], [int(y[i]) for i in tr], expand)
                te_texts = [texts[i] for i in te]
                all_p.extend(no_llm_predict(te_texts, score(te_texts, model, strategy)))
                all_y.extend(y[te])
            ay, ap = np.array(all_y), np.array(all_p)
            tp = int(((ap == 1) & (ay == 1)).sum()); fp = int(((ap == 1) & (ay == 0)).sum())
            tn = int(((ap == 0) & (ay == 0)).sum()); fn = int(((ap == 0) & (ay == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            tr = [i for i, r in enumerate(records) if r["category"] != "indirect_injection"]
            ho = [i for i, r in enumerate(records) if r["category"] == "indirect_injection"]
            ind_probs = score([texts[i] for i in ho], fit([texts[i] for i in tr], [int(y[i]) for i in tr], expand), strategy)

            full = fit(texts, [int(v) for v in y], expand)
            doc_probs = score(benign_chunks, full, strategy)
            poison_probs = score(poisoned, full, strategy)

            row = {
                "benign_sentence_expansion": expand,
                "strategy": strategy,
                "group_cv_no_llm": {
                    "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                    "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                },
                "indirect_injection_held_out_detection_at_0.5": round(float((ind_probs >= 0.5).mean()), 4),
                "benign_sample_document_chunks": {
                    "n": len(benign_chunks),
                    "safe_band": int((doc_probs < config.LOW_THRESHOLD).sum()),
                    "judge_band": int(((doc_probs >= config.LOW_THRESHOLD) & (doc_probs < config.HIGH_THRESHOLD)).sum()),
                    "high_band": int((doc_probs >= config.HIGH_THRESHOLD).sum()),
                    "max_score": round(float(doc_probs.max()), 4),
                },
                "poisoned_document": {
                    "injected_chunk_scores": [round(float(p), 3) for p, l in zip(poison_probs, poisoned_labels) if l],
                    "clean_chunk_scores": [round(float(p), 3) for p, l in zip(poison_probs, poisoned_labels) if not l],
                },
            }
            rows.append(row)
            g = row["group_cv_no_llm"]; d = row["benign_sample_document_chunks"]
            print(
                f"expand={int(expand)} {strategy:13s} | group-CV P={g['precision']:.3f} R={g['recall']:.3f} "
                f"F1={g['f1']:.3f} FPR={g['false_positive_rate']:.3f} | indirect held-out="
                f"{row['indirect_injection_held_out_detection_at_0.5']:.2f} | docs safe/judge/high="
                f"{d['safe_band']}/{d['judge_band']}/{d['high_band']} | poisoned injected={row['poisoned_document']['injected_chunk_scores']}"
            )

    out = {
        "production_choice": {"strategy": ml_classifier.SCORING_STRATEGY, "benign_sentence_expansion": True},
        "benchmark_path": str(config.BENCHMARK_PATH.relative_to(config.BASE_DIR)),
        "n_records": len(records),
        "rows": rows,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "segmentation_experiment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {config.RESULTS_DIR / 'segmentation_experiment.json'}")


if __name__ == "__main__":
    main()
