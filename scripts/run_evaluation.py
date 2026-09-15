"""
scripts/run_evaluation.py

Full RAGShield evaluation. Every number reported here comes from a model
fitted inside this script on the training side of the relevant split; the
saved production classifier is never used for scoring, so a stale or
all-data model cannot leak into the metrics.

Default run (writes results/metrics.json and results/ablation_matrix.csv):

  1. Random stratified 80/20 split. A split-local classifier is fitted on the
     train side and the four ablation conditions are scored on the test side:
       Heuristic        any Layer 1 rule fired
       ML               Layer 2 max-sentence probability >= 0.5
       Heuristic+ML     the real firewall routing with Layer 3 replaced by the
                        ABLATION_FALLBACK_THRESHOLD cutoff (no LLM)
       Full Hybrid      the real firewall including live LLM judge calls
     Note: the benchmark reuses seed sentences across categories (obfuscated
     and indirect_injection records wrap the direct/role/extraction seeds), so
     a random split is optimistic by construction. It is kept because the
     ablation needs one common test set; the group-held-out run is the
     honest generalisation number.
  2. Category-held-out: for each attack family, fit on every OTHER family plus
     the benign records that are NOT in the main test split, then test on the
     held-out family plus the main split's benign records. (Earlier versions
     trained on all benign records, including the ones then used for FPR;
     that leak is closed here.) Reports ML-only detection at 0.5, the
     Heuristic+ML no-LLM routing, and a threshold sweep on the same
     probabilities.
  3. Error analysis: false positives / negatives of the Full Hybrid condition
     on the random test split with the firewall's reason for each.

--group-held-out (writes results/group_held_out_metrics.json):

  5-fold StratifiedGroupKFold. By default groups are the benchmark's
  `lineage_id`: every attack derived from (or paraphrasing) the same seed
  sentence, across ALL categories, stays on one side of the split, and benign
  records are grouped by document style. `--group-by template_group` runs the
  older category:template_group grouping for comparison; that grouping let a
  payload appear in training as a plain seed while its wrapped/obfuscated/
  paraphrased form was in test, which inflated recall. A fold-local
  classifier is fitted per fold and the real firewall is run on the held-out
  groups. Reports per fold: train/test size, held-out groups, TP/FP/TN/FN,
  precision, recall, F1, FPR, LLM calls and failures; plus the aggregate and
  per-category / per-template-group detection. This is the headline number.

  2. (cont.) Category-held-out additionally reports a lineage-clean variant:
  training also drops every record sharing a lineage_id with the held-out
  family. For families built from the shared seeds this removes most attack
  training data, which is itself the finding: with 21 seed lineages, "unseen
  family" and "unseen payload" are not the same thing. Detection is therefore
  also broken down by template_group so the 10 novel indirect payloads
  (indirect_novel, no lineage overlap with anything) are visible separately.

LLM judge accounting: every JudgeResult carries an `error` field. Calls that
failed (rate limit, network, unparseable output) are counted separately and
never silently reported as detections. If any failures occurred, the output
files carry a `judge_failure_warning` and the summary printed to the console
says so. With --skip-llm (or no GROQ_API_KEY) Layer 3 is replaced by the
ablation cutoff and `llm_judge_used` is recorded as false.

Benchmark records carry no user query, so a fixed PLACEHOLDER_QUERY is used
for every judge call during evaluation.

Run with:
  python scripts/run_evaluation.py [--skip-llm]
  python scripts/run_evaluation.py --group-held-out [--skip-llm] [--folds 5]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402

from ragshield import config, firewall, heuristics, ml_classifier  # noqa: E402
from ragshield.rag_pipeline import Chunk  # noqa: E402

# Decision threshold for the ML-only ablation column and the category-held-out
# detection rate. It is a measurement cutoff, not a firewall routing threshold
# (those are config.LOW_THRESHOLD / HIGH_THRESHOLD). Kept equal to the
# firewall's no-judge ablation cutoff so the two no-LLM views agree.
ML_DECISION_THRESHOLD = firewall.ABLATION_FALLBACK_THRESHOLD

PLACEHOLDER_QUERY = "What does this document say?"
N_ERROR_EXAMPLES = 5
THRESHOLD_SWEEP_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_GROUP_CV_SPLITS = 5


# ---------------------------------------------------------------------------
# Data loading / provenance
# ---------------------------------------------------------------------------


def load_benchmark_records() -> list[dict]:
    if not config.BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"No benchmark found at {config.BENCHMARK_PATH}. "
            "Run scripts/build_benchmark.py --from-baseline first."
        )
    records = []
    with open(config.BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def provenance(records: list[dict], llm_judge_used: bool) -> dict:
    counts = collections.Counter(r["category"] for r in records)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_path": str(config.BENCHMARK_PATH.relative_to(config.BASE_DIR)),
        "benchmark_sha256": hashlib.sha256(config.BENCHMARK_PATH.read_bytes()).hexdigest(),
        "n_records": len(records),
        "category_counts": dict(sorted(counts.items())),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "scoring_strategy": ml_classifier.SCORING_STRATEGY,
        "random_seed": config.RANDOM_SEED,
        "low_threshold": config.LOW_THRESHOLD,
        "high_threshold": config.HIGH_THRESHOLD,
        "ml_decision_threshold": ML_DECISION_THRESHOLD,
        "llm_judge_used": llm_judge_used,
        "judge_model": config.GROQ_JUDGE_MODEL if llm_judge_used else None,
        "models_fitted_inside_evaluation": True,
    }


def main_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = [r["label"] for r in records]
    return train_test_split(
        records, test_size=0.2, random_state=config.RANDOM_SEED, stratify=labels
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "n": len(y_true),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def judge_accounting(results: list[firewall.ChunkResult]) -> dict:
    calls = [r.judge for r in results if r.judge is not None]
    failures = collections.Counter(j.error for j in calls if j.failed)
    return {
        "llm_judge_calls": len(calls),
        "llm_judge_failures": int(sum(failures.values())),
        "llm_judge_failures_by_type": dict(failures),
        "decided_by": dict(collections.Counter(r.decided_by for r in results)),
    }


def decision_breakdown(records: list[dict], preds: list[int], results: list[firewall.ChunkResult]) -> dict:
    """
    Where did each error come from? Distinguishes, per false negative / false
    positive, whether the decision was made by Layers 1+2 alone, by the LLM
    judge (or a failed judge call), by a silent Layer 2 pass that never
    reached the judge, or by the no-LLM ablation cutoff standing in for the
    judge (i.e. an error the real judge would have had a chance to correct).
    """
    out: collections.Counter = collections.Counter()
    for record, pred, result in zip(records, preds, results):
        if pred == record["label"]:
            continue
        kind = "fn" if record["label"] == 1 else "fp"
        if result.judge is not None:
            where = "llm_judge_failed_call" if result.judge.failed else "llm_judge_verdict"
        elif result.decided_by in ("heuristic", "heuristic+ml"):
            where = "layers_1_2_block"
        elif result.status == "SAFE":
            where = "layer_2_silent_pass"
        else:
            where = "ablation_cutoff_would_reach_judge"
        out[f"{kind}_{where}"] += 1
    return dict(sorted(out.items()))


def failure_warning(accounting: dict) -> str | None:
    if accounting["llm_judge_failures"]:
        return (
            f"{accounting['llm_judge_failures']} of {accounting['llm_judge_calls']} LLM judge "
            f"calls FAILED ({accounting['llm_judge_failures_by_type']}). Failed calls fail "
            "closed (counted as INJECTION), so recall may be inflated and FPR overstated. "
            "Re-run once the API/rate limit recovers before quoting these numbers."
        )
    return None


# ---------------------------------------------------------------------------
# Per-condition predictors
# ---------------------------------------------------------------------------


def heuristic_predict(texts: list[str]) -> list[int]:
    """Crudest baseline: flag if ANY Layer 1 rule fired (critical or not)."""
    return [1 if heuristics.scan(t).any_match else 0 for t in texts]


def ml_predict(texts: list[str], model, threshold: float = ML_DECISION_THRESHOLD) -> list[int]:
    probs = ml_classifier.predict_proba_segment(texts, model=model)
    return [1 if p >= threshold else 0 for p in probs]


def firewall_predict(
    records: list[dict], model, use_llm: bool
) -> tuple[list[int], list[firewall.ChunkResult]]:
    """
    The real firewall (Layer 1 -> 2 -> 3 routing) on a list of records.
    With use_llm=False, Layer 3 is replaced by the ablation cutoff. Returns
    (predicted labels, chunk results); final_included=False -> predicted 1.
    """
    chunks = [Chunk(chunk_id=i, text=r["text"], source="benchmark") for i, r in enumerate(records)]
    results = firewall.screen_chunks(
        chunks,
        PLACEHOLDER_QUERY,
        model=model,
        judge=None if use_llm else firewall.ABLATION_NO_JUDGE,
    )
    return [0 if r.final_included else 1 for r in results], results


# ---------------------------------------------------------------------------
# Category-held-out (+ threshold sweep on the same probabilities)
# ---------------------------------------------------------------------------


def category_held_out_eval(
    records: list[dict], benign_test_records: list[dict]
) -> tuple[dict, dict, list[dict]]:
    benign_test_ids = {r["id"] for r in benign_test_records}
    results, category_errors, sweep_rows = {}, {}, []

    for category in config.ATTACK_CATEGORIES:
        held_out = [r for r in records if r["category"] == category]
        if not held_out:
            print(f"[run_evaluation] No examples for category '{category}' — skipping.")
            continue
        train_records = [
            r for r in records if r["category"] != category and r["id"] not in benign_test_ids
        ]
        model = ml_classifier.fit_classifier(
            [r["text"] for r in train_records], [r["label"] for r in train_records]
        )
        held_out_lineages = {r["lineage_id"] for r in held_out}
        clean_train = [r for r in train_records if r["lineage_id"] not in held_out_lineages]
        n_clean_attacks = sum(r["label"] for r in clean_train)
        clean_model = ml_classifier.fit_classifier(
            [r["text"] for r in clean_train], [r["label"] for r in clean_train]
        ) if n_clean_attacks >= 2 else None

        test_records = held_out + benign_test_records
        test_labels = [r["label"] for r in test_records]
        probs = ml_classifier.predict_proba_segment([r["text"] for r in test_records], model=model)
        n_held = len(held_out)

        preds = [1 if p >= ML_DECISION_THRESHOLD else 0 for p in probs]
        metrics = compute_metrics(test_labels, preds)
        metrics["unseen_family_detection_rate"] = round(sum(preds[:n_held]) / n_held, 4)

        hybrid_preds, _ = firewall_predict(test_records, model, use_llm=False)
        metrics["hybrid_no_llm_detection_rate"] = round(sum(hybrid_preds[:n_held]) / n_held, 4)
        metrics["hybrid_no_llm_false_positive_rate"] = compute_metrics(test_labels, hybrid_preds)[
            "false_positive_rate"
        ]
        metrics["n_train"] = len(train_records)
        metrics["n_train_attacks"] = int(sum(r["label"] for r in train_records))
        metrics["n_train_attacks_sharing_lineage_with_held_out"] = int(
            sum(1 for r in train_records if r["label"] == 1 and r["lineage_id"] in held_out_lineages)
        )
        by_group = {}
        for tg in sorted({r["template_group"] for r in held_out}):
            idx = [i for i, r in enumerate(held_out) if r["template_group"] == tg]
            by_group[tg] = {"n": len(idx), "detection_rate": round(sum(preds[i] for i in idx) / len(idx), 4)}
        metrics["detection_by_template_group"] = by_group
        if clean_model is not None:
            clean_probs = ml_classifier.predict_proba_segment([r["text"] for r in held_out], model=clean_model)
            metrics["lineage_clean"] = {
                "n_train_attacks": int(n_clean_attacks),
                "detection_rate": round(float((clean_probs >= ML_DECISION_THRESHOLD).mean()), 4),
                "note": "training additionally excludes every record sharing a lineage_id with the held-out family",
            }
        else:
            metrics["lineage_clean"] = {"n_train_attacks": int(n_clean_attacks), "detection_rate": None,
                                        "note": "not computable: fewer than 2 attack records remain after removing shared lineages"}
        results[category] = metrics

        category_errors[category] = [
            {
                "id": r["id"],
                "category": r["category"],
                "text": r["text"],
                "true_label": r["label"],
                "predicted_label": pred,
                "ml_risk_score": round(float(prob), 4),
                "threshold": ML_DECISION_THRESHOLD,
            }
            for r, prob, pred in zip(held_out, probs[:n_held], preds[:n_held])
            if pred != r["label"]
        ]

        for threshold in THRESHOLD_SWEEP_VALUES:
            t_preds = [1 if p >= threshold else 0 for p in probs]
            t_metrics = compute_metrics(test_labels, t_preds)
            sweep_rows.append(
                {
                    "category": category,
                    "threshold": threshold,
                    "precision": t_metrics["precision"],
                    "recall": t_metrics["recall"],
                    "f1": t_metrics["f1"],
                    "false_positive_rate": t_metrics["false_positive_rate"],
                    "unseen_family_detection_rate": round(sum(t_preds[:n_held]) / n_held, 4),
                }
            )

        lc = metrics["lineage_clean"]
        print(
            f"[run_evaluation] category-held-out '{category}': ML-only detection="
            f"{metrics['unseen_family_detection_rate']:.2f}, heuristic+ML detection="
            f"{metrics['hybrid_no_llm_detection_rate']:.2f}, FPR(ML-only)={metrics['false_positive_rate']:.2f} | "
            f"train attacks sharing lineage with held-out: {metrics['n_train_attacks_sharing_lineage_with_held_out']}/{metrics['n_train_attacks']} | "
            f"lineage-clean detection: {lc['detection_rate']} (train attacks {lc['n_train_attacks']}) | by group: "
            + ", ".join(f"{k}={v['detection_rate']:.2f}" for k, v in by_group.items())
        )

    return results, category_errors, sweep_rows


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(records, preds, results: list[firewall.ChunkResult]) -> dict:
    false_positives, false_negatives = [], []
    for record, pred, result in zip(records, preds, results):
        item = {
            "id": record["id"],
            "category": record["category"],
            "text": record["text"],
            "reason": result.reason,
            "band": result.band,
            "decided_by": result.decided_by,
            "ml_risk_score": round(float(result.risk), 4),
        }
        if pred == 1 and record["label"] == 0:
            false_positives.append(item)
        elif pred == 0 and record["label"] == 1:
            false_negatives.append(item)
    return {
        "false_positives": false_positives[:N_ERROR_EXAMPLES],
        "false_negatives": false_negatives[:N_ERROR_EXAMPLES],
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
    }


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------


def build_ablation_matrix(test_records: list[dict], split_model, run_llm: bool):
    texts = [r["text"] for r in test_records]
    labels = [r["label"] for r in test_records]

    heuristic_preds = heuristic_predict(texts)
    ml_preds = ml_predict(texts, split_model)
    hybrid_no_llm_preds, hybrid_no_llm_results = firewall_predict(test_records, split_model, use_llm=False)
    if run_llm:
        full_hybrid_preds, full_results = firewall_predict(test_records, split_model, use_llm=True)
    else:
        print("[run_evaluation] LLM judge skipped — Full Hybrid column reuses Heuristic+ML (no LLM calls made).")
        full_hybrid_preds, full_results = hybrid_no_llm_preds, hybrid_no_llm_results

    conditions = {
        "Heuristic": heuristic_preds,
        "ML": ml_preds,
        "Heuristic+ML": hybrid_no_llm_preds,
        "Full Hybrid": full_hybrid_preds,
    }

    rows = []
    for category in config.ATTACK_CATEGORIES + [config.BENIGN_CATEGORY]:
        idx = [i for i, r in enumerate(test_records) if r["category"] == category]
        if not idx:
            continue
        row = {"category": category, "n": len(idx)}
        for cond_name, preds in conditions.items():
            cat_preds = [preds[i] for i in idx]
            if category == config.BENIGN_CATEGORY:
                value = sum(1 for p in cat_preds if p == 0) / len(cat_preds)  # correctly passed
            else:
                value = sum(cat_preds) / len(cat_preds)  # detected
            row[cond_name] = round(value, 4)
        rows.append(row)

    overall = {"category": "Overall (F1)", "n": len(test_records)}
    for cond_name, preds in conditions.items():
        overall[cond_name] = compute_metrics(labels, preds)["f1"]
    rows.append(overall)

    per_condition = {name: compute_metrics(labels, preds) for name, preds in conditions.items()}
    return pd.DataFrame(rows), per_condition, full_hybrid_preds, full_results


# ---------------------------------------------------------------------------
# Group-held-out
# ---------------------------------------------------------------------------


def template_group_id(record: dict) -> str:
    return f"{record['category']}:{record['template_group']}"


def group_key(record: dict, group_by: str) -> str:
    if group_by == "lineage":
        return record["lineage_id"]
    if group_by == "template_group":
        return template_group_id(record)
    raise ValueError(f"unknown group_by {group_by!r}")


def group_held_out_eval(records: list[dict], run_llm: bool, n_splits: int, group_by: str = "lineage") -> dict:
    labels = [r["label"] for r in records]
    groups = [group_key(r, group_by) for r in records]
    if len(set(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct template groups, found {len(set(groups))}.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)

    fold_results, all_labels, all_preds, all_results, all_records = [], [], [], [], []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(records, labels, groups), start=1):
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        held_out_groups = sorted({groups[i] for i in test_idx})
        overlap = sorted({groups[i] for i in train_idx} & set(held_out_groups))
        if overlap:
            raise AssertionError(f"Group leakage in fold {fold_number}: {overlap}")
        # Lineage leakage check, whatever the grouping: report (and, when
        # grouping by lineage, forbid) any attack lineage on both sides.
        lineage_overlap = sorted(
            {records[i]["lineage_id"] for i in train_idx if records[i]["label"] == 1}
            & {records[i]["lineage_id"] for i in test_idx if records[i]["label"] == 1}
        )
        if lineage_overlap and group_by == "lineage":
            raise AssertionError(f"Lineage leakage in fold {fold_number}: {lineage_overlap}")

        fold_model = ml_classifier.fit_classifier(
            [r["text"] for r in train_records], [r["label"] for r in train_records]
        )
        preds, results = firewall_predict(test_records, fold_model, use_llm=run_llm)
        fold_labels = [r["label"] for r in test_records]

        metrics = compute_metrics(fold_labels, preds)
        fp_by_group = collections.Counter(
            groups[i] for i, p, r in zip(test_idx, preds, test_records) if p == 1 and r["label"] == 0
        )
        fn_by_category = collections.Counter(
            r["category"] for p, r in zip(preds, test_records) if p == 0 and r["label"] == 1
        )
        metrics.update(
            {
                "fold": fold_number,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "n_held_out_groups": len(held_out_groups),
                "held_out_groups": held_out_groups,
                "n_attack_lineages_leaking_into_train": len(lineage_overlap),
                "false_positives_by_group": dict(fp_by_group),
                "false_negatives_by_category": dict(fn_by_category),
                "error_breakdown": decision_breakdown(test_records, preds, results),
                **judge_accounting(results),
            }
        )
        fold_results.append(metrics)
        all_labels.extend(fold_labels)
        all_preds.extend(preds)
        all_results.extend(results)
        all_records.extend(test_records)
        print(
            f"[run_evaluation] fold {fold_number}: n_train={len(train_records)} n_test={len(test_records)} "
            f"groups={len(held_out_groups)} TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} "
            f"FN={metrics['fn']} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} FPR={metrics['false_positive_rate']:.3f} "
            f"judge_calls={metrics['llm_judge_calls']} judge_failures={metrics['llm_judge_failures']}"
        )

    per_template_group = {}
    for tg in sorted({r["template_group"] for r in all_records if r["label"] == 1}):
        idx = [i for i, r in enumerate(all_records) if r["label"] == 1 and r["template_group"] == tg]
        per_template_group[tg] = {"n": len(idx), "detection_rate": round(sum(all_preds[i] for i in idx) / len(idx), 4)}

    per_category = {}
    for category in config.ATTACK_CATEGORIES + [config.BENIGN_CATEGORY]:
        idx = [i for i, r in enumerate(all_records) if r["category"] == category]
        if not idx:
            continue
        if category == config.BENIGN_CATEGORY:
            per_category[category] = {
                "n": len(idx),
                "correctly_passed_rate": round(sum(1 for i in idx if all_preds[i] == 0) / len(idx), 4),
            }
        else:
            per_category[category] = {
                "n": len(idx),
                "detection_rate": round(sum(all_preds[i] for i in idx) / len(idx), 4),
            }

    accounting = judge_accounting(all_results)
    return {
        "evaluation_type": f"{group_by}-held-out cross-validation",
        "splitter": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "group_by": group_by,
        "group_id_definition": "lineage_id (attack seed lineage; benign:<template_group> for benign)" if group_by == "lineage" else "category + ':' + template_group",
        "n_groups": len(set(groups)),
        "n_attack_lineages": len({r["lineage_id"] for r in records if r["label"] == 1}),
        "any_train_test_group_overlap": False,
        "any_attack_lineage_leakage": any(f["n_attack_lineages_leaking_into_train"] for f in fold_results),
        "llm_judge_used": run_llm,
        "judge_failure_warning": failure_warning(accounting),
        "aggregate": {
            **compute_metrics(all_labels, all_preds),
            **accounting,
            "error_breakdown": decision_breakdown(all_records, all_preds, all_results),
        },
        "per_category": per_category,
        "per_template_group": per_template_group,
        "folds": fold_results,
        "provenance": provenance(records, run_llm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_run_llm(skip_llm: bool) -> bool:
    if skip_llm:
        return False
    if not config.GROQ_API_KEY:
        print("[run_evaluation] No GROQ_API_KEY set — LLM judge replaced by the no-LLM ablation cutoff.")
        return False
    return True


def run_group_held_out(args) -> None:
    run_llm = resolve_run_llm(args.skip_llm)
    records = load_benchmark_records()
    print(
        f"[run_evaluation] Group-held-out evaluation on {len(records)} records, "
        f"{args.folds} folds, grouped by {args.group_by}, LLM judge {'ON' if run_llm else 'OFF'}."
    )
    out = group_held_out_eval(records, run_llm, args.folds, group_by=args.group_by)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.GROUP_HELD_OUT_METRICS_PATH
    if args.group_by != "lineage":
        out_path = out_path.with_name(out_path.stem + f"_{args.group_by}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[run_evaluation] Wrote {out_path}")
    print(json.dumps(out["per_template_group"], indent=2))
    print(json.dumps({k: out["aggregate"][k] for k in ("precision", "recall", "f1", "false_positive_rate", "tp", "fp", "tn", "fn", "llm_judge_calls", "llm_judge_failures", "error_breakdown")}, indent=2))
    if out["judge_failure_warning"]:
        print("\n[run_evaluation] WARNING: " + out["judge_failure_warning"])


def run_main_evaluation(args) -> None:
    run_llm = resolve_run_llm(args.skip_llm)
    records = load_benchmark_records()
    print(f"[run_evaluation] Loaded {len(records)} records from {config.BENCHMARK_PATH}")

    train_records, test_records = main_split(records)
    print(f"[run_evaluation] Main split: {len(train_records)} train / {len(test_records)} test")

    # Split-local model. The saved production classifier is deliberately not
    # used: it is trained on all records, including this test split.
    split_model = ml_classifier.fit_classifier(
        [r["text"] for r in train_records], [r["label"] for r in train_records]
    )

    production_meta = ml_classifier.classifier_metadata()
    benchmark_hash = hashlib.sha256(config.BENCHMARK_PATH.read_bytes()).hexdigest()
    production_model_current = bool(production_meta) and production_meta.get("benchmark_sha256") == benchmark_hash
    if not production_model_current:
        print(
            "[run_evaluation] NOTE: the saved production classifier is missing or was trained on a "
            "different benchmark file. Evaluation is unaffected (it fits its own models), but run "
            "scripts/train_classifier.py before using the app."
        )

    print("\n[run_evaluation] Building ablation matrix"
          + (" (real LLM judge calls for escalated chunks)..." if run_llm else "..."))
    ablation_df, per_condition, full_preds, full_results = build_ablation_matrix(test_records, split_model, run_llm)
    summary = per_condition["Full Hybrid"]
    accounting = judge_accounting(full_results)
    print(f"[run_evaluation] Full-hybrid random-split metrics: {summary}")

    print("\n[run_evaluation] Running category-held-out evaluation (+ threshold sweep)...")
    benign_test_records = [r for r in test_records if r["category"] == config.BENIGN_CATEGORY]
    category_results, category_errors, sweep_rows = category_held_out_eval(records, benign_test_records)

    print("\n[run_evaluation] Running error analysis...")
    errors = error_analysis(test_records, full_preds, full_results)
    if not run_llm:
        errors["note"] = "Full Hybrid used the no-LLM ablation cutoff (LLM judge skipped or no API key)."
    print(
        f"[run_evaluation] {errors['total_false_positives']} FP, {errors['total_false_negatives']} FN "
        f"on the random test split"
    )

    metrics_out = {
        "summary": summary,
        "per_condition": per_condition,
        "judge_accounting": accounting,
        "error_breakdown": decision_breakdown(test_records, full_preds, full_results),
        "judge_failure_warning": failure_warning(accounting),
        "production_model_current": production_model_current,
        "category_held_out": category_results,
        "category_held_out_errors": category_errors,
        "threshold_sensitivity": sweep_rows,
        "error_analysis": errors,
        "config": provenance(records, run_llm),
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    ablation_df.to_csv(config.ABLATION_MATRIX_PATH, index=False)
    print(f"\n[run_evaluation] Wrote {config.METRICS_PATH} and {config.ABLATION_MATRIX_PATH}")
    print("\n" + ablation_df.to_string(index=False))
    if metrics_out["judge_failure_warning"]:
        print("\n[run_evaluation] WARNING: " + metrics_out["judge_failure_warning"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Replace the LLM judge with the no-LLM ablation cutoff (no API calls).")
    parser.add_argument("--group-held-out", action="store_true",
                        help="Run template-group-held-out cross-validation instead of the default evaluation.")
    parser.add_argument("--folds", type=int, default=DEFAULT_GROUP_CV_SPLITS,
                        help="Number of StratifiedGroupKFold splits for --group-held-out.")
    parser.add_argument("--group-by", choices=["lineage", "template_group"], default="lineage",
                        help="Grouping for --group-held-out: attack lineage (default, leakage-free) or the older "
                             "category:template_group. Output file gets a suffix for the non-default grouping.")
    args = parser.parse_args()

    if args.group_held_out:
        run_group_held_out(args)
    else:
        run_main_evaluation(args)


if __name__ == "__main__":
    main()
