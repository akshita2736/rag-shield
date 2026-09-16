# RAGShield — a 3-layer prompt-injection firewall for RAG pipelines

RAGShield sits between document retrieval and the answer-generating LLM, screening every retrieved chunk before it enters the model's context.

```
Documents → Chunking → MiniLM Embeddings → FAISS Retrieval
        → RAGShield Firewall
             Layer 1  Deterministic heuristics    (regex + obfuscation checks)
             Layer 2  Embedding classifier        (MiniLM → logistic regression, max-sentence scoring)
             Layer 3  Contextual LLM judge         (Groq, JSON verdict, fails closed)
        → Sanitised context → Answer-generating LLM
```

Instructional language alone isn't an injection — documents legitimately tell *humans* what to do. An injection tries to steer the downstream AI instead: override its rules, assign it a role, extract its system prompt, or redirect its output. RAGShield is built and evaluated around that distinction, and the same routing function (`firewall.screen_chunks`) runs in both the app and every evaluation condition, so production and evaluation can't silently diverge.

## Routing

| Layer 1 | Layer 2 score | Decision | LLM call |
|---|---|---|---|
| critical match | — | **BLOCKED** | no |
| no match | `< 0.30` | **SAFE** | no |
| any match | `≥ 0.70` | **BLOCKED** | no |
| anything else | anything else | **Layer 3 judge decides** | yes |

## Results

5-fold cross-validation, grouped by attack **lineage** (21 seed payloads; grouping by category alone lets variants leak across folds):

| Configuration | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Layers 1+2 only | 0.955 | 0.848 | 0.898 | 0.057 |
| **Full hybrid, live judge** | **0.992** | **0.976** | **0.984** | **0.011** |

Full hybrid on the random 80/20 split: precision, recall, and F1 all 1.000 on 43 held-out records (0 FP, 0 FN, 20 judge calls, 0 failures).

**Honest limitation:** `indirect_novel` — ten payloads sharing no wording with any training seed — is the true test of unseen-intent generalisation. The embedding classifier alone catches only 1/10. A Layer 1 rule for AI-directed language escalates these to the judge instead of passing them silently, recovering 9/10 in the full hybrid.

Full ablations, category-held-out breakdowns, real-document robustness tests, and reproducibility pins are in `docs/RAGShield_Report.pdf`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # add GROQ_API_KEY
python scripts/train_classifier.py
python scripts/run_evaluation.py --group-held-out  # headline cross-validation
streamlit run app.py
pytest                                             # 87 unit tests
```

Without a `GROQ_API_KEY`, evaluation falls back to a 0.5 cutoff instead of the judge, and the app quarantines anything needing review.

## Known limitations

* 213 records from 21 seed lineages plus 10 novel payloads — small per-fold counts, wide confidence intervals.
* ~40% of real-document chunks land in the review band and need an LLM call; recall on paraphrased and disguised attacks depends on the judge.
* Fixed 500-character chunking can split an attack across a boundary.
* Reduces attack surface — doesn't guarantee the answer model can't be manipulated; the prompt boundary (tagged untrusted context) is the last line of defence.