"""Evaluation helpers: metric arithmetic and leakage-free splitting."""

import importlib.util
import json
from pathlib import Path

import pytest
from sklearn.model_selection import StratifiedGroupKFold

from ragshield import config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evaluation():
    spec = importlib.util.spec_from_file_location("run_evaluation", ROOT / "scripts" / "run_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def records():
    return [json.loads(l) for l in config.BENCHMARK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_compute_metrics_arithmetic(evaluation):
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = evaluation.compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 5, 1)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["false_positive_rate"] == round(1 / 6, 4)


def test_compute_metrics_handles_empty_classes(evaluation):
    m = evaluation.compute_metrics([0, 0], [0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["false_positive_rate"] == 0.0


def test_ml_decision_threshold_is_the_ablation_cutoff_not_a_routing_threshold(evaluation):
    from ragshield import firewall

    assert evaluation.ML_DECISION_THRESHOLD == firewall.ABLATION_FALLBACK_THRESHOLD == 0.5
    assert evaluation.ML_DECISION_THRESHOLD not in (config.LOW_THRESHOLD, config.HIGH_THRESHOLD)


def test_lineage_grouping_has_no_attack_lineage_on_both_sides(evaluation, records):
    labels = [r["label"] for r in records]
    groups = [evaluation.group_key(r, "lineage") for r in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    for train_idx, test_idx in splitter.split(records, labels, groups):
        train_lineages = {records[i]["lineage_id"] for i in train_idx if records[i]["label"] == 1}
        test_lineages = {records[i]["lineage_id"] for i in test_idx if records[i]["label"] == 1}
        assert not (train_lineages & test_lineages)
        assert any(records[i]["label"] == 0 for i in test_idx)  # stratified: benign in every fold


def test_template_group_grouping_does_leak_lineages(evaluation, records):
    """The reason the default changed: the older grouping puts a payload's seed in train and its variants in test."""
    labels = [r["label"] for r in records]
    groups = [evaluation.group_key(r, "template_group") for r in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    leaks = 0
    for train_idx, test_idx in splitter.split(records, labels, groups):
        train_lineages = {records[i]["lineage_id"] for i in train_idx if records[i]["label"] == 1}
        test_lineages = {records[i]["lineage_id"] for i in test_idx if records[i]["label"] == 1}
        leaks += len(train_lineages & test_lineages)
    assert leaks > 0


def test_main_split_is_deterministic_and_stratified(evaluation, records):
    a_train, a_test = evaluation.main_split(records)
    b_train, b_test = evaluation.main_split(records)
    assert [r["id"] for r in a_test] == [r["id"] for r in b_test]
    assert len(a_test) == round(0.2 * len(records))
    assert 0 < sum(r["label"] == 0 for r in a_test) < len(a_test)
