"""Benchmark integrity and reproducibility."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from ragshield import config, heuristics

ROOT = Path(__file__).resolve().parent.parent


def _load_build_benchmark():
    spec = importlib.util.spec_from_file_location("build_benchmark", ROOT / "scripts" / "build_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def records():
    return _records(config.BENCHMARK_PATH)


@pytest.fixture(scope="module")
def build_benchmark():
    return _load_build_benchmark()


def test_required_fields_and_labels(records):
    required = {"id", "text", "label", "category", "source_type", "generator", "template_group", "lineage_id"}
    for r in records:
        assert required <= set(r), r
        assert r["label"] in (0, 1)
        assert r["text"].strip()
        assert (r["label"] == 0) == (r["category"] == config.BENIGN_CATEGORY)


def test_category_counts_match_targets(records, build_benchmark):
    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    assert counts == build_benchmark.TARGET_COUNTS


def test_no_duplicate_ids_or_normalised_texts(records, build_benchmark):
    assert len({r["id"] for r in records}) == len(records)
    keys = [build_benchmark.normalized_text(r["text"]) for r in records]
    assert len(set(keys)) == len(keys)


def test_no_truncated_or_refusal_generated_text(records, build_benchmark):
    assert [build_benchmark.generated_quality_issue(r) for r in records] == [None] * len(records)


def test_no_critical_heuristic_fires_on_benign(records):
    assert [r["id"] for r in records if r["label"] == 0 and heuristics.scan(r["text"]).critical_triggered] == []


def test_previous_versions_are_subsets_of_the_current_benchmark(records):
    v1 = _records(config.BENCHMARK_V1_PATH)
    v2 = _records(config.BENCHMARK_DIR / "benchmark_v2_203.jsonl")
    assert len(v1) == 155 and len(v2) == 203 and len(records) == 213
    assert {r["text"] for r in v1} <= {r["text"] for r in v2} <= {r["text"] for r in records}


def test_every_attack_has_a_seed_or_novel_lineage(records, build_benchmark):
    seed_ids = set(build_benchmark.SEED_LINEAGE.values())
    for r in records:
        if r["label"] == 1:
            assert r["lineage_id"] in seed_ids or r["lineage_id"].startswith("novel_indirect_"), r["id"]
        else:
            assert r["lineage_id"] == f"benign:{r['template_group']}"


def test_seed_lineages_span_categories_which_is_why_lineage_grouping_exists(records):
    """Documents the leakage: most seed lineages appear in several attack categories."""
    spans = {}
    for r in records:
        if r["label"] == 1 and r["lineage_id"].startswith("seed_"):
            spans.setdefault(r["lineage_id"], set()).add(r["category"])
    assert len(spans) == 21
    assert sum(len(c) > 1 for c in spans.values()) >= 15


def test_novel_indirect_payloads_are_new_and_heuristic_free(records, build_benchmark):
    novel = [r for r in records if r["template_group"] == "indirect_novel"]
    assert len(novel) == len(build_benchmark.NOVEL_INDIRECT_PAYLOADS) == 10
    assert len({r["lineage_id"] for r in novel}) == 10  # each its own lineage
    assert all(r["lineage_id"].startswith("novel_indirect_") for r in novel)
    assert all(r["category"] == "indirect_injection" and r["label"] == 1 for r in novel)
    # payloads must not reuse seed wording and must never be hard-blocked by Layer 1
    # (the non-critical ai_addressed_instruction rule may add evidence)
    seeds = [t.lower() for t in build_benchmark.SEED_LINEAGE]
    for r in novel:
        assert not any(seed in r["text"].lower() for seed in seeds)
        assert not heuristics.scan(r["text"]).critical_triggered, r["id"]


def test_realistic_benign_domains_are_present_and_chunk_sized(records, build_benchmark):
    for domain, texts in build_benchmark.REALISTIC_BENIGN_SEEDS.items():
        group = f"benign_{domain}"
        rows = [r for r in records if r["template_group"] == group]
        assert len(rows) == len(texts) >= 5, group
        assert all(len(r["text"]) <= config.CHUNK_SIZE for r in rows)
        from ragshield.ml_classifier import split_sentences

        assert all(len(split_sentences(r["text"])) >= 2 for r in rows), "realistic chunks should be multi-sentence"


def test_enough_groups_for_group_held_out(records):
    groups = {f"{r['category']}:{r['template_group']}" for r in records}
    assert len(groups) >= 20
    assert len({r["lineage_id"] for r in records}) >= 40
    assert not any(g.endswith(("_01", "_02", "_03", "_04", "_05", "_06", "_07")) for g in groups), \
        "repair rows must share one group per category, not singleton groups"


def test_committed_benchmark_is_reproducible_from_baseline(build_benchmark):
    """`build_benchmark.py --from-baseline` must regenerate benchmark.jsonl byte for byte."""
    rebuilt, _ = build_benchmark.rebuild_from_baseline(config.BENCHMARK_BASELINE_PATH)
    build_benchmark.validate_records(rebuilt)
    random.Random(config.RANDOM_SEED).shuffle(rebuilt)
    expected = "".join(json.dumps(r) + "\n" for r in rebuilt)
    assert config.BENCHMARK_PATH.read_text(encoding="utf-8") == expected
