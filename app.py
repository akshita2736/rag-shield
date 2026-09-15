"""
RAGShield Streamlit app. Two tabs:

  1. RAGShield — upload/index documents, ask a question, see every retrieved
     chunk's firewall decision (status, band, risk, which layer decided and
     why), and get an answer generated from the sanitised context only.
  2. RAGShield Evaluation — reads results/*.json + results/ablation_matrix.csv
     written by scripts/run_evaluation.py: summary, ablation matrix,
     category-held-out, threshold sensitivity, error analysis, and the
     template-group-held-out cross-validation. Warns when results were
     produced from a different benchmark file than the one on disk, or when
     LLM judge calls failed during the run.

Run with: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ragshield import config, firewall, generate_answer, ml_classifier
from ragshield.rag_pipeline import RagPipeline

st.set_page_config(page_title="RAGShield", layout="wide")

STATUS_ICON = {"SAFE": "✅", "BLOCKED": "🚫", "REVIEW": "⚠️"}
STATUS_COLOR = {"SAFE": "green", "BLOCKED": "red", "REVIEW": "orange"}
DECIDED_BY_LABEL = {
    "heuristic": "Layer 1 (critical heuristic)",
    "ml": "Layer 2 (ML classifier)",
    "heuristic+ml": "Layers 1+2 (high ML risk corroborated by heuristics)",
    "llm_judge": "Layer 3 (LLM judge)",
}


# ---------------------------------------------------------------------------
# Session setup — the RagPipeline (and its SentenceTransformer model) is
# created once per session and reused across reruns. bind_embedder() must run
# right after creation so ml_classifier reuses the same model instance.
# ---------------------------------------------------------------------------

def _new_pipeline() -> RagPipeline:
    rag = RagPipeline()
    firewall.bind_embedder(rag)
    return rag


if "rag" not in st.session_state:
    st.session_state.rag = _new_pipeline()

rag: RagPipeline = st.session_state.rag

@st.cache_resource(show_spinner="No trained classifier found — training on the benchmark (about 20 s)...")
def _ensure_classifier() -> bool:
    """Fresh clone / cloud deploy safety net: train the Layer 2 model if it is missing."""
    if config.CLASSIFIER_PATH.exists():
        return True
    if not config.BENCHMARK_PATH.exists():
        return False
    import hashlib
    import json as _json

    records = [_json.loads(l) for l in config.BENCHMARK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ml_classifier.train_classifier(
        [r["text"] for r in records],
        [int(r["label"]) for r in records],
        metadata={
            "benchmark_path": str(config.BENCHMARK_PATH.relative_to(config.BASE_DIR)),
            "benchmark_sha256": hashlib.sha256(config.BENCHMARK_PATH.read_bytes()).hexdigest(),
            "trained_by": "app.py fallback",
        },
    )
    return True


classifier_ready = _ensure_classifier()
judge_ready = bool(config.GROQ_API_KEY)


def _benchmark_hash() -> str | None:
    if config.BENCHMARK_PATH.exists():
        return hashlib.sha256(config.BENCHMARK_PATH.read_bytes()).hexdigest()
    return None


def _index_paths(paths: list[Path]) -> None:
    added = rag.add_documents(paths)
    st.success(f"Indexed {added} chunks from {len(paths)} document(s).")


tab_main, tab_eval = st.tabs(["RAGShield", "RAGShield Evaluation"])


# ---------------------------------------------------------------------------
# Tab 1 — main demo
# ---------------------------------------------------------------------------

with tab_main:
    if not classifier_ready:
        st.error(
            "No trained classifier and no benchmark found. Run `python scripts/build_benchmark.py --from-baseline` "
            "and `python scripts/train_classifier.py`, then reload this page."
        )
    if not judge_ready:
        st.warning(
            "GROQ_API_KEY is not set. Layer 3 (LLM judge) is unavailable, so every chunk that needs "
            "contextual review is quarantined (fail closed), and answer generation is disabled. "
            "Add the key to a `.env` file to enable both."
        )

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    upload_col, sample_col, clear_col = st.columns([2, 2, 1])
    with upload_col:
        if st.button("Add to knowledge base", disabled=not uploaded_files):
            saved_paths = []
            for uploaded in uploaded_files:
                dest = config.DOCUMENTS_DIR / Path(uploaded.name).name  # basename only
                dest.write_bytes(uploaded.getvalue())
                saved_paths.append(dest)
            _index_paths(saved_paths)
    with sample_col:
        sample_paths = sorted(config.SAMPLE_DOCUMENTS_DIR.glob("*.txt")) if config.SAMPLE_DOCUMENTS_DIR.exists() else []
        if st.button("Load sample documents", disabled=not sample_paths,
                     help="Fictional support/manual/policy/API/lab/résumé documents plus one poisoned FAQ"):
            _index_paths(sample_paths)
    with clear_col:
        if st.button("Clear knowledge base", disabled=not rag.chunks):
            st.session_state.rag = _new_pipeline()
            st.rerun()

    if rag.chunks:
        sources = sorted({c.source for c in rag.chunks})
        st.caption(f"📚 {len(rag.chunks)} chunks indexed from {len(sources)} document(s): {', '.join(sources)}")

    st.divider()
    st.header("Ask a Question")
    user_query = st.text_input("Your question")
    ask_clicked = st.button("Ask", type="primary", disabled=not classifier_ready)

    if ask_clicked:
        if not rag.chunks:
            st.warning("Upload at least one document first.")
        elif not user_query.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving and screening context..."):
                retrieved = rag.retrieve(user_query)
                results = firewall.screen_chunks(retrieved, user_query)

            st.subheader("Retrieved Context")
            n_llm = sum(1 for r in results if r.judge is not None)
            st.caption(
                f"{len(results)} chunks retrieved · {sum(r.final_included for r in results)} passed · "
                f"{n_llm} sent to the LLM judge"
            )
            for i, result in enumerate(results, start=1):
                icon = STATUS_ICON[result.status]
                color = STATUS_COLOR[result.status]
                st.markdown(
                    f"**Chunk {i}** {icon} :{color}[{result.status}] · "
                    f"`{result.chunk.source}` · decided by {DECIDED_BY_LABEL[result.decided_by]}"
                )
                if result.band == "critical":
                    st.progress(1.0, text="Layer 1 critical match — blocked before ML scoring")
                else:
                    st.progress(
                        min(max(result.risk, 0.0), 1.0),
                        text=f"Layer 2 risk (max sentence): {result.risk:.2f} · band: {result.band}",
                    )
                if result.heuristic_matches:
                    st.caption("Layer 1 evidence: " + ", ".join(result.heuristic_matches))

                inclusion = "included in" if result.final_included else "excluded from"
                st.caption(f"{result.reason} — chunk {inclusion} the final answer.")
                if result.judge is not None and result.judge.failed:
                    st.error(f"Judge call failed ({result.judge.error}); quarantined by fail-closed policy.")
                elif result.judge is not None:
                    st.caption(f"Judge confidence: {result.judge.confidence:.2f}")

                with st.expander("View chunk text"):
                    st.write(result.chunk.text)
                    if result.band != "critical" and result.status != "SAFE":
                        top = sorted(ml_classifier.sentence_scores(result.chunk.text), key=lambda s: -s[1])[:3]
                        st.caption("Highest-scoring sentences:")
                        for sentence, prob in top:
                            st.caption(f"{prob:.2f} — {sentence}")
                st.divider()

            sanitised = firewall.sanitised_context(results)
            st.subheader("Final Answer")
            if not judge_ready:
                st.info(
                    "Answer generation needs GROQ_API_KEY. Sanitised context "
                    f"({sum(r.final_included for r in results)} of {len(results)} chunks) is shown below instead."
                )
                st.text(sanitised or "(empty — every chunk was quarantined)")
            else:
                try:
                    with st.spinner("Generating answer from sanitised context..."):
                        answer = generate_answer.generate_answer(sanitised, user_query)
                    st.write(answer)
                except Exception as exc:  # noqa: BLE001 - surface API problems to the user
                    st.error(f"Answer generation failed: {exc}")
            st.caption(
                f"Generated from {sum(r.final_included for r in results)} of "
                f"{len(results)} retrieved chunks (sanitised context only)."
            )


# ---------------------------------------------------------------------------
# Tab 2 — evaluation results
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _staleness_warning(section: dict | None, label: str) -> None:
    if not section:
        return
    current = _benchmark_hash()
    if current and section.get("benchmark_sha256") not in (None, current):
        st.warning(
            f"{label} were produced from a different benchmark file than the one on disk "
            "(hash mismatch). Re-run scripts/run_evaluation.py."
        )


def _judge_note(cfg: dict | None, warning: str | None) -> None:
    if not cfg:
        return
    if cfg.get("llm_judge_used"):
        st.caption(f"LLM judge: enabled ({cfg.get('judge_model')}) · seed {cfg.get('random_seed')} · "
                   f"routing bands [{cfg.get('low_threshold')}, {cfg.get('high_threshold')}] · "
                   f"ML measurement cutoff {cfg.get('ml_decision_threshold')}")
    else:
        st.caption("LLM judge: NOT used in this run — Layer 3 replaced by the no-LLM ablation cutoff "
                   f"({cfg.get('ml_decision_threshold')}). Numbers reflect Layers 1+2 only.")
    if warning:
        st.error(warning)


with tab_eval:
    st.header("RAGShield Evaluation")
    metrics = _load_json(config.METRICS_PATH)
    group_metrics = _load_json(config.GROUP_HELD_OUT_METRICS_PATH)

    if not metrics and not group_metrics:
        st.info(
            "No evaluation results yet. Run `python scripts/train_classifier.py`, then "
            "`python scripts/run_evaluation.py` and `python scripts/run_evaluation.py --group-held-out`."
        )

    # --- Group-held-out (headline) ---
    if group_metrics:
        st.subheader("Group-held-out cross-validation (headline)")
        st.caption(
            f"StratifiedGroupKFold grouped by **{group_metrics.get('group_by', 'template_group')}** "
            f"({group_metrics.get('group_id_definition', '')}); {group_metrics.get('n_groups', '?')} groups, "
            f"{group_metrics.get('n_attack_lineages', '?')} attack lineages. Every fold fits its own classifier and runs the "
            "real firewall on groups it never saw. Every attack in the benchmark descends from one of 21 seed sentences "
            "(obfuscated, indirect and paraphrased records are transforms of them), so grouping by lineage keeps all "
            "variants of a payload on one side of the split. Attack-lineage leakage into training: "
            f"{'none' if not group_metrics.get('any_attack_lineage_leakage') else 'PRESENT (older grouping)'}."
        )
        _staleness_warning(group_metrics.get("provenance"), "Group-held-out results")
        _judge_note(group_metrics.get("provenance"), group_metrics.get("judge_failure_warning"))
        agg = group_metrics["aggregate"]
        cols = st.columns(4)
        for col, key in zip(cols, ["precision", "recall", "f1", "false_positive_rate"]):
            col.metric(key.replace("_", " ").title(), f"{agg[key]:.3f}")
        st.caption(
            f"Aggregate over {agg['n']} held-out records: TP={agg['tp']} FP={agg['fp']} TN={agg['tn']} FN={agg['fn']} · "
            f"LLM judge calls {agg.get('llm_judge_calls', 0)} (failures {agg.get('llm_judge_failures', 0)})"
        )
        fold_rows = [
            {
                "fold": f["fold"], "train": f["n_train"], "test": f["n_test"], "groups": f["n_held_out_groups"],
                "TP": f["tp"], "FP": f["fp"], "TN": f["tn"], "FN": f["fn"],
                "precision": f["precision"], "recall": f["recall"], "f1": f["f1"], "FPR": f["false_positive_rate"],
                "judge calls": f.get("llm_judge_calls", 0), "judge failures": f.get("llm_judge_failures", 0),
            }
            for f in group_metrics["folds"]
        ]
        st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)
        per_cat = group_metrics.get("per_category")
        if per_cat:
            cat_rows = [
                {"category": c, "n": v["n"], "rate": v.get("detection_rate", v.get("correctly_passed_rate")),
                 "meaning": "detected" if "detection_rate" in v else "correctly passed"}
                for c, v in per_cat.items()
            ]
            st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)
        per_tg = group_metrics.get("per_template_group")
        if per_tg:
            st.caption(
                "Detection by attack template group. `indirect_novel` are ten handwritten indirect payloads that share no "
                "wording with the seeds; their detection rate is the honest measure of unseen-payload generalisation."
            )
            tg_rows = [{"template_group": k, "n": v["n"], "detection_rate": v["detection_rate"]} for k, v in per_tg.items()]
            st.dataframe(pd.DataFrame(tg_rows), width="stretch", hide_index=True)
        with st.expander("Per-fold held-out groups and error breakdown"):
            for f in group_metrics["folds"]:
                st.markdown(f"**Fold {f['fold']}** — held out: {', '.join(f.get('held_out_groups', f.get('held_out_template_groups', [])))}")
                if f.get("false_positives_by_group"):
                    st.caption(f"False positives by group: {f['false_positives_by_group']}")
                if f.get("false_negatives_by_category"):
                    st.caption(f"False negatives by category: {f['false_negatives_by_category']}")
        st.divider()

    if metrics:
        run_config = metrics.get("config", {})
        _staleness_warning(run_config, "Random-split results")

        # --- Summary ---
        st.subheader("Random-split summary (Full Hybrid)")
        summary = metrics.get("summary", {})
        cols = st.columns(4)
        for col, key in zip(cols, ["precision", "recall", "f1", "false_positive_rate"]):
            value = summary.get(key)
            col.metric(key.replace("_", " ").title(), f"{value:.3f}" if value is not None else "—")
        _judge_note(run_config, metrics.get("judge_failure_warning"))
        if metrics.get("production_model_current") is False:
            st.warning("The saved production classifier does not match the current benchmark. Run scripts/train_classifier.py.")
        st.divider()

        # --- Ablation matrix ---
        if config.ABLATION_MATRIX_PATH.exists():
            st.subheader("Ablation: what each layer contributes")
            st.caption(
                "Detection rate per attack category and correctly-passed rate for benign_hard_negative "
                "(every cell reads 'higher is better'). Overall (F1) summarises the whole test split. "
                "Heuristic+ML is the real routing with Layer 3 replaced by a 0.5 cutoff; Full Hybrid adds the judge."
            )
            ablation_df = pd.read_csv(config.ABLATION_MATRIX_PATH)
            st.dataframe(ablation_df, width="stretch", hide_index=True)
            chart_df = ablation_df[ablation_df["category"] != "Overall (F1)"].set_index("category")
            st.bar_chart(chart_df[[c for c in chart_df.columns if c not in ("category", "n")]])
        st.divider()

        # --- Category-held-out ---
        category_held_out = metrics.get("category_held_out")
        if category_held_out:
            st.subheader("Category-held-out: unseen attack families")
            st.caption(
                "For each family the classifier is trained on every OTHER family (and only benign records "
                "outside the shared benign test set), then tested on the held-out family plus that benign set. "
                "'ML-only' is Layer 2 at the 0.5 cutoff; 'Heuristic+ML' is the no-LLM firewall routing."
            )
            rows = [
                {
                    "category": cat,
                    "ML-only detection": m.get("unseen_family_detection_rate"),
                    "Heuristic+ML detection": m.get("hybrid_no_llm_detection_rate"),
                    "ML-only FPR": m.get("false_positive_rate"),
                    "train attacks sharing a lineage": f"{m.get('n_train_attacks_sharing_lineage_with_held_out', '?')}/{m.get('n_train_attacks', '?')}",
                    "lineage-clean detection": (m.get("lineage_clean") or {}).get("detection_rate"),
                }
                for cat, m in category_held_out.items()
            ]
            st.caption(
                "'train attacks sharing a lineage' shows how much of the training set descends from the same seeds as the "
                "held-out family; 'lineage-clean' removes those records before training. Low lineage-clean numbers mean the "
                "classifier recognises payloads it has seen in another form rather than injection intent in general."
            )
            cat_df = pd.DataFrame(rows)
            st.dataframe(cat_df, width="stretch", hide_index=True)
            st.bar_chart(cat_df.set_index("category")[["ML-only detection", "Heuristic+ML detection"]])
        st.divider()

        # --- Threshold sensitivity ---
        threshold_data = metrics.get("threshold_sensitivity")
        if threshold_data:
            st.subheader("Threshold sensitivity (diagnostic)")
            st.caption(
                "How each held-out family's detection would look at other ML cutoffs. Diagnostic only: "
                "routing thresholds are not tuned on held-out data."
            )
            threshold_df = pd.DataFrame(threshold_data)
            selected = st.selectbox("Category", sorted(threshold_df["category"].unique()), key="threshold_category")
            filtered = threshold_df[threshold_df["category"] == selected].set_index("threshold")
            st.line_chart(filtered[["unseen_family_detection_rate", "f1", "false_positive_rate"]])
        st.divider()

        # --- Error analysis ---
        error_analysis = metrics.get("error_analysis") or {}
        if error_analysis:
            st.subheader("Error analysis (random test split, Full Hybrid)")
            if error_analysis.get("note"):
                st.info(error_analysis["note"])
            fp_total = error_analysis.get("total_false_positives", 0)
            fn_total = error_analysis.get("total_false_negatives", 0)
            if fp_total == 0 and fn_total == 0:
                st.success("No misclassifications on this run's random test split.")
            else:
                st.caption(f"{fp_total} false positive(s), {fn_total} false negative(s).")
                fp_col, fn_col = st.columns(2)
                with fp_col:
                    st.markdown("**False positives** — benign content flagged")
                    for item in error_analysis.get("false_positives", []):
                        with st.expander(f"`{item['id']}` — {item['category']}"):
                            st.write(item["text"])
                            st.caption(item["reason"])
                with fn_col:
                    st.markdown("**False negatives** — injection that slipped through")
                    for item in error_analysis.get("false_negatives", []):
                        with st.expander(f"`{item['id']}` — {item['category']}"):
                            st.write(item["text"])
                            st.caption(item["reason"])
