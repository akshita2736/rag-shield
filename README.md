# RAGShield

A hybrid prompt-injection detection and defense firewall for Retrieval-Augmented Generation (RAG) pipelines.

RAGShield analyzes retrieved document chunks before they enter an LLM's context window, detects potentially malicious instructions, and prevents high-risk content from influencing the final response.

## Overview

RAG systems retrieve external documents and pass relevant chunks to an LLM as context. However, retrieved documents are untrusted input and may contain instructions designed to manipulate the downstream model.

RAGShield introduces a security layer between retrieval and generation:

```text
Documents
    ↓
Chunking & Embedding
    ↓
FAISS Retrieval
    ↓
RAGShield Firewall
    ├── Heuristic Scanner
    ├── Semantic ML Classifier
    └── Contextual LLM Judge
    ↓
Sanitized Context
    ↓
LLM Response
```

The project focuses on an important distinction:

> **Instruction detection ≠ prompt-injection detection.**

Legitimate documents frequently contain instructional language. The goal is therefore not simply to detect instructions, but to distinguish legitimate document content from instructions attempting to redirect the behaviour of the RAG application's LLM.

## Detection Pipeline

### Layer 1 — Heuristic Scanner

Uses deterministic, inspectable rules to identify common injection patterns such as:

- instruction override attempts
- role manipulation
- system-prompt extraction
- suspicious delimiters
- basic obfuscation patterns

High-confidence critical rules can immediately quarantine a retrieved chunk.

### Layer 2 — Semantic ML Classifier

Uses SentenceTransformer embeddings with a Logistic Regression classifier to estimate the probability that a retrieved chunk contains a prompt injection.

The lightweight classifier enables semantic detection beyond exact keyword matching while keeping inference inexpensive.

### Layer 3 — Contextual LLM Judge

Ambiguous chunks are escalated to an LLM judge.

Rather than performing simple instruction classification, the judge considers the retrieved chunk together with the user's query to determine whether instructional language belongs to the document's legitimate content or attempts to manipulate the downstream AI system.

## Evaluation

RAGShield will be evaluated on a benchmark containing multiple prompt-injection families:

- Direct override
- Role manipulation
- Prompt extraction
- Indirect injection
- Obfuscated injection
- Paraphrased injection
- Benign instructional hard negatives

Evaluation includes:

- Precision
- Recall
- F1-score
- False Positive Rate
- Random train/test evaluation
- Attack-category-held-out evaluation
- Ablation comparison between heuristic, ML, and hybrid defenses

The category-held-out evaluation is designed to measure how well the detector generalizes to attack families that were not present during training.

## Tech Stack

- **Python**
- **Streamlit** — interface and application layer
- **FAISS** — vector similarity search
- **SentenceTransformers** — semantic embeddings
- **scikit-learn** — Logistic Regression classifier
- **LLM API** — contextual security judge and final generation

## Project Structure

```text
ragshield/
├── app.py
├── ragshield/
│   ├── config.py
│   ├── rag_pipeline.py
│   ├── heuristics.py
│   ├── ml_classifier.py
│   ├── llm_judge.py
│   ├── firewall.py
│   └── generate_answer.py
├── data/
│   ├── documents/
│   └── benchmark/
├── scripts/
│   ├── build_benchmark.py
│   ├── train_classifier.py
│   └── run_evaluation.py
├── models/
└── results/
```

## Status

🚧 **Currently under development**

Planned implementation order:

1. Vulnerable baseline RAG pipeline
2. Heuristic injection detection
3. Prompt-injection benchmark
4. Semantic ML classifier
5. Contextual LLM judge
6. Hybrid firewall and quarantine pipeline
7. Evaluation and ablation study
8. Streamlit demo

## Known Scope Limitations

The initial version focuses on English, chunk-level text prompt injections. Cross-chunk attacks, multilingual attacks, multimodal prompt injections, and production-scale adversarial testing are outside the scope of v1.

## Motivation

RAG applications treat retrieved documents as context, but retrieved content should not automatically be treated as trusted instructions.

RAGShield explores how lightweight deterministic rules, semantic machine learning, and contextual LLM reasoning can be combined to detect malicious retrieved content while minimizing false positives on legitimate instructional text.
