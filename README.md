# RAG-Shield: a 3-layer prompt-injection firewall for RAG pipelines

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://rag-shield.streamlit.app/)

RAG-Shield sits between document retrieval and the answer-generating LLM and screens every retrieved chunk before it reaches the model's context. Poisoned chunks are blocked, ambiguous ones get an LLM review, and only approved text is used to answer.


## Architecture

```text
Documents
   ↓
Chunking → MiniLM Embeddings → FAISS Retrieval
   ↓
RAGShield Firewall
   ├── Layer 1: Deterministic Heuristics
   ├── Layer 2: Embedding-Based Classifier
   └── Layer 3: Contextual LLM Judge (Groq)
   ↓
Sanitised Context
   ↓
Answer-Generating LLM
```

RAG-Shield focuses on an important distinction: instructional language is not automatically prompt injection. Real documents contain instructions intended for humans, while prompt injection attempts to manipulate the downstream AI.

## How It Works

- **Layer 1 — Heuristics:** Detects deterministic injection patterns, prompt extraction, role manipulation and obfuscation.
- **Layer 2 — ML Classifier:** Uses MiniLM embeddings + Logistic Regression for semantic risk scoring.
- **Layer 3 — LLM Judge:** Uses Groq to review ambiguous cases; failures are handled fail-closed.


## Routing

| Layer 1 | Layer 2 score | Decision | LLM call |
|---|---:|---|---|
| critical match | not scored | **BLOCKED** | no |
| no match | `< 0.30` | **SAFE** | no |
| any match | `≥ 0.70` | **BLOCKED** | no |
| anything else | anything else | **Layer 3 judge decides** (judge failure → **BLOCKED**) | yes |

A high ML score alone never blocks; Layer 1 evidence can trigger blocking or escalation depending on the rule.

---

### 5-Fold Lineage Group-Held-Out Evaluation:

- **213 records (125 attacks, 88 hard benign negatives):** direct override, role manipulation, prompt extraction, indirect, obfuscated and paraphrased attacks.
- Attack variants are grouped by **21 seed lineages** for 5-fold cross-validation, preventing related payloads from leaking across train and test folds.
- Each fold trains its own classifier and ensures that no attack lineage appears on both sides.


## Results


| Configuration (lineage-grouped CV) | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Layers 1+2 only (0.5 cutoff, no judge) | 0.955 | 0.848 | 0.898 | 0.057 |
| **Full hybrid, live Groq judge** | **0.992** | **0.976** | **0.984** | **0.011** |


- The full hybrid achieved **98.4% F1 and 1.1% FPR**, with **122 TP, 1 FP, 87 TN and 3 FN** across 5 folds.
- The evaluation used **118 live LLM judge calls with 0 failures**.
- The random 80/20 split achieved **1.000 F1**, but is treated as an optimistic secondary result rather than the main headline.


<details>
<summary>Screenshots</summary>

| Poisoned FAQ blocked | Benign banking question answered |
|---|---|
| ![poisoned](docs/screenshots/03_poisoned_query.png) | ![benign](docs/screenshots/04_benign_query.png) |

Evaluation tab: [`docs/screenshots/05_evaluation.png`](docs/screenshots/05_evaluation.png)
</details>

---

### Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ_API_KEY
python scripts/train_classifier.py
python scripts/run_evaluation.py --group-held-out      # headline cross-validation
streamlit run app.py
pytest                                                 # 87 unit tests (pytest -m slow for 2 real-document tests)
```

**Deploy (Streamlit Community Cloud):** main file `app.py`, Python 3.12, and `GROQ_API_KEY = "gsk_..."` under *Secrets*. 

## Repository structure
        
        ragshield/
        ├── heuristics.py
        ├── ml_classifier.py
        ├── llm_judge.py
        ├── firewall.py
        ├── rag_pipeline.py
        └── generate_answer.py
        
        scripts/
        ├── build_benchmark.py
        ├── train_classifier.py
        ├── run_evaluation.py
        └── segmentation_experiment.py
        
        data/          # Benchmark + sample documents
        results/       # Evaluation results
        tests/         # Pytest suite
        app.py         # Streamlit application
        docs/          # Technical report + screenshots


## Known Limitations

- **Novel-intent generalisation:** The classifier alone detects only 1/10 `indirect_novel` payloads; the full hybrid reaches 9/10 through Layer 1 escalation + LLM review.
- **Small benchmark size:** 213 records across 21 seed lineages plus 10 novel payloads, so results are not representative of all real-world attacks.
- **Pipeline limitations:** Fixed-size chunking can split attacks across boundaries, and some real-document chunks require LLM review.
- RAGShield reduces the attack surface but does not provide a formal guarantee against downstream LLM manipulation.

## Tech Stack
Python · FAISS · SentenceTransformers · Scikit-learn · Groq · Streamlit · Pytest
