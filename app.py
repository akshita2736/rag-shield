"""
RAGShield Streamlit app. Two tabs only, per the frozen spec:
  1. RAGShield  — upload documents, ask a question, see per-chunk firewall
     decisions, get an answer generated from sanitised context only.
  2. RAGShield Evaluation — reads results/metrics.json and
     results/ablation_matrix.csv (produced by scripts/run_evaluation.py)
     and displays them. Shows a friendly message if they don't exist yet.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ragshield import config, firewall, generate_answer
from ragshield.rag_pipeline import RagPipeline

st.set_page_config(page_title="RAGShield", layout="wide")

STATUS_COLOR = {"SAFE": "green", "BLOCKED": "red", "REVIEW": "orange"}


# ---------------------------------------------------------------------------
# Session setup — the RagPipeline (and its SentenceTransformer model) is
# created once per session and reused across reruns, not recreated on every
# button click. bind_embedder() must run right after creation, before any
# firewall screening happens, or ml_classifier loads its own second copy
# of the embedding model.
# ---------------------------------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = RagPipeline()
    firewall.bind_embedder(st.session_state.rag)

rag: RagPipeline = st.session_state.rag


tab_main, tab_eval = st.tabs(["RAGShield", "RAGShield Evaluation"])


# ---------------------------------------------------------------------------
# Tab 1 — main demo
# ---------------------------------------------------------------------------

with tab_main:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Add to knowledge base"):
        saved_paths = []
        for uploaded in uploaded_files:
            dest = config.DOCUMENTS_DIR / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            saved_paths.append(dest)

        added = rag.add_documents(saved_paths)
        st.success(f"Indexed {added} chunks from {len(saved_paths)} document(s).")

    st.divider()
    st.header("Ask Question")
    user_query = st.text_input("Your question")
    ask_clicked = st.button("Ask", type="primary")

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
            for i, result in enumerate(results, start=1):
                color = STATUS_COLOR[result.status]
                st.markdown(
                    f"**Chunk {i}** — :{color}[{result.status}] — risk `{result.risk:.2f}`"
                )
                with st.expander("View chunk text"):
                    st.write(result.chunk.text)
                if result.status != "SAFE":
                    st.caption(f"Why {result.status.lower()}? {result.reason}")

            st.divider()
            sanitised = firewall.sanitised_context(results)
            with st.spinner("Generating answer from sanitised context..."):
                answer = generate_answer.generate_answer(sanitised, user_query)

            st.subheader("Final Answer")
            st.write(answer)
            st.caption("Generated using sanitised context")


# ---------------------------------------------------------------------------
# Tab 2 — evaluation results
# ---------------------------------------------------------------------------

with tab_eval:
    st.header("RAGShield Evaluation")

    if not Path(config.METRICS_PATH).exists():
        st.info(
            "No evaluation results yet. Run `scripts/build_benchmark.py`, "
            "then `scripts/train_classifier.py`, then `scripts/run_evaluation.py`."
        )
    else:
        metrics = json.loads(Path(config.METRICS_PATH).read_text())

        st.subheader("Summary metrics")
        summary = metrics.get("summary", metrics)  # tolerate either shape
        cols = st.columns(4)
        for col, key in zip(cols, ["precision", "recall", "f1", "false_positive_rate"]):
            value = summary.get(key)
            col.metric(key.replace("_", " ").title(), f"{value:.2f}" if value is not None else "—")

        if Path(config.ABLATION_MATRIX_PATH).exists():
            st.subheader("Ablation matrix")
            df = pd.read_csv(config.ABLATION_MATRIX_PATH)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Ablation matrix not found yet — run scripts/run_evaluation.py.")