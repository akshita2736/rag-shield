# RAGShield — a 3-layer prompt-injection firewall for RAG pipelines

RAGShield sits between a retriever and the answer-generating LLM. Every retrieved
chunk passes through three layers before it is allowed into the model's context:

```
documents -> chunking -> MiniLM embeddings -> FAISS retrieval
        -> RAGShield firewall
             Layer 1  deterministic heuristics   (regex rules + obfuscation checks)
             Layer 2  embedding classifier       (MiniLM -> logistic regression, max-sentence scoring)
             Layer 3  contextual LLM judge       (Groq, JSON verdict, fails closed)
        -> sanitised context (approved chunks only) -> answer LLM
```

The problem it solves is narrower and harder than "does this text contain
instructions": real documents are full of instructions for *humans* ("follow the
steps below to reset your password", "ignore row 3 when computing the mean").
A prompt injection is text that tries to steer the *downstream AI*: override its
rules, assign it a role or privileges, extract its system prompt, or redirect its
output. RAGShield is built and evaluated around that distinction.

---

## Routing table

| Layer 1 result | Layer 2 probability | Decision | LLM call |
|---|---|---|---|
| critical rule matched | (not scored) | **BLOCKED** | no |
| no match | `< 0.30` | **SAFE** | no |
| any match | `>= 0.70` | **BLOCKED** (two independent layers agree) | no |
| anything else | anything else | **Layer 3 judge decides**: SAFE -> included (shown as REVIEW), INJECTION or judge failure -> BLOCKED | yes |

Two things about this table are deliberate and were validated on real documents
(see *Real-document robustness*):

* **A high Layer 2 score alone never blocks.** The embedding classifier gives
  legitimate manuals, support articles and policies high scores. A high score with
  no independent Layer 1 evidence is sent to the judge instead of being quarantined.
* **A low Layer 2 score does not pass if Layer 1 found evidence.** Base64 payloads
  and homoglyph text score low for the classifier but leave structural traces
  that force a review.

The judge fails closed: API errors, rate limits, empty or unparseable responses
and invalid verdicts all produce `INJECTION` with an `error` field, and the
evaluation reports those separately so an outage is never counted as accuracy.

The same routing code (`firewall.screen_chunks`) is used by the app and by every
evaluation condition, so production and evaluation cannot silently diverge.

---

## Results

All numbers below are produced by the committed scripts on the committed
benchmark (v3, 213 records). `results/*.json` carry the benchmark's SHA-256 and
the model's provenance sidecar records the exact library versions, so stale or
mismatched artifacts are detectable. Every classifier used in evaluation is
fitted inside the split; the saved production model never scores its own
training data. No Groq key was available for the committed runs, so **Layer 3
is replaced by a 0.5 cutoff and `llm_judge_used` is recorded as false**; the
"would reach judge" counts below say what Layer 3 would have been asked to decide.

### Attack lineage, and why the headline split changed

Every attack in the original benchmark descends from one of **21 handwritten
seed sentences**: obfuscated records are deterministic transforms of a seed,
indirect records wrap a seed in benign text, and the LLM-generated same-style and
paraphrased records were produced from a seed. 20 of the 21 lineages span more
than one category. Splitting on `category:template_group` therefore let a payload
sit in training as a plain seed while its wrapped, obfuscated or paraphrased form
sat in test, which is why earlier reports showed indirect detection of 1.00.

v3 fixes this in three ways:

* every record carries a `lineage_id` (exact for deterministic derivations,
  nearest seed by MiniLM cosine for LLM/curated rows, frozen in
  `scripts/build_benchmark.py` so the rebuild is offline);
* group-held-out evaluation groups on `lineage_id`, so all variants of a payload
  stay on one side (asserted per fold);
* ten **novel indirect payloads** (`indirect_novel`) were written with new intents
  (goal hijacking, output manipulation, exfiltration, language lock-in) sharing no
  wording with any seed. They are the honest measure of unseen-payload generalisation.

### Group-held-out cross-validation (headline)

5-fold `StratifiedGroupKFold`, Layers 1+2 only:

| grouping | benchmark | precision | recall | F1 | FPR | TP | FP | TN | FN | attack lineages leaking into train |
|---|---|---|---|---|---|---|---|---|---|---|
| **lineage (current)** | v3 | **0.955** | **0.848** | **0.898** | **0.057** | 106 | 5 | 83 | 19 | 0 |
| template_group | v3 | 0.972 | 0.824 | 0.892 | 0.034 | 103 | 3 | 85 | 22 | 13 to 15 per fold |
| template_group | v2 | 0.962 | 0.887 | 0.923 | 0.045 | | | | | present |
| template_group (before this work) | v1 | 0.767 | 1.000 | 0.868 | 0.875 | | | | | present |

Of the 19 misses under lineage grouping, **all 19 fall in the review band and
would reach Layer 3; none passes silently**. All 5 false positives are also in
the review band. Before the `ai_addressed_instruction` rule was added, 7 of the
misses passed silently below LOW_THRESHOLD with no Layer 1 evidence.

Detection by attack group (lineage CV, Layers 1+2 at the 0.5 cutoff): all
direct_override groups 1.00; seed-derived indirect wrappers 1.00; zero-width and
base64 obfuscation 1.00; homoglyph 0.67; paraphrase batches 0.50 to 0.83;
role-manipulation and prompt-extraction generated batches 0.67 to 1.00;
**indirect_novel 0.10**. The classifier does not recognise genuinely new indirect
intents; with the new Layer 1 rule 9 of the 10 are escalated to the judge instead
of passing.

### Category-held-out (unseen attack family)

Trained on every other family plus only the benign records outside the shared
benign test set. `lineage-clean` additionally removes every training record that
shares a lineage with the held-out family.

| held-out family | ML-only @0.5 | Heuristic+ML | train attacks sharing a lineage | lineage-clean detection |
|---|---|---|---|---|
| direct_override | 0.95 | 0.95 | 26 / 105 | 0.70 |
| role_manipulation | 0.85 | 0.85 | 35 / 105 | 0.10 |
| prompt_extraction | 0.87 | 0.93 | 26 / 110 | 0.60 |
| indirect_injection | 0.70 (wrappers 1.00, novel 0.10) | 0.70 | 94 / 95 | n/a (1 attack left) |
| obfuscated | 0.40 | 0.55 | 94 / 105 | 0.00 |
| paraphrased | 0.90 | 0.90 | 47 / 105 | 0.20 |

Read together, these say the same thing: the embedding classifier recognises
payloads it has seen in another form far better than injection intent in general.
That is why it is one layer of three and never the final word.

### Ablation on the random 80/20 split (v3)

| category | n | Heuristic | ML | Heuristic+ML | Full Hybrid* |
|---|---|---|---|---|---|
| direct_override | 5 | 0.40 | 1.00 | 1.00 | 1.00 |
| role_manipulation | 3 | 0.33 | 1.00 | 1.00 | 1.00 |
| prompt_extraction | 4 | 0.50 | 0.75 | 0.75 | 0.75 |
| indirect_injection | 3 | 1.00 | 0.67 | 0.67 | 0.67 |
| obfuscated | 6 | 1.00 | 0.67 | 0.83 | 0.83 |
| paraphrased | 4 | 0.25 | 1.00 | 1.00 | 1.00 |
| benign_hard_negative (correctly passed) | 18 | 1.00 | 0.94 | 0.94 | 0.94 |
| **Overall F1** | 43 | 0.75 | 0.89 | 0.92 | 0.92 |

\*Full Hybrid equals Heuristic+ML in the committed run (no judge). The random
split is optimistic by construction and is kept only to give the ablation one
common test set.

### Real-document robustness

`data/sample_documents/` holds six fictional benign documents (banking support
FAQ, router manual, remote-work policy, payments API reference, lab SOP, résumé)
and one poisoned product FAQ with four embedded injections. None is used for
training. Chunked exactly as the app does it (34 benign chunks):

| model / scoring | benign chunks SAFE without LLM | sent to judge | above HIGH | hard-blocked | poisoned chunks flagged |
|---|---|---|---|---|---|
| original (v1, max-sentence) | 0 | 15 | **19** | 19 | 3/3 |
| v2 model | 21 | 13 | 0 | 0 | 3/3 |
| **current (v3 model)** | **17** | 17 | **0** | **0** | 3/3 |

No benign chunk is blocked by Layers 1+2; half pass with no LLM call; every
injected chunk is hard-blocked by Layer 1 or escalated. The v3 model sends four
more benign chunks to the judge than v2 did, the cost of training on the novel
payloads. `pytest -m slow` pins this behaviour.

### Scoring strategy evidence

`scripts/segmentation_experiment.py` compares whole-chunk, max-sentence and
2-sentence-window scoring with and without benign sentence expansion (lineage
CV, v3). Max-sentence with expansion has the best F1 (0.898) and is the only
strategy that both detects seed-derived indirect wrappers and keeps every
real-document chunk out of the high band; whole-chunk scoring detects 0% of
indirect injections in either setting.

---

## Groq request configuration

Checked against the Groq reasoning and structured-output documentation and
pinned by `tests/test_llm_judge.py`:

| parameter | gpt-oss models | other reasoning models (qwen, minimax, deepseek) | non-reasoning models |
|---|---|---|---|
| `response_format` | `{"type": "json_object"}` (prompt asks for JSON, as JSON mode requires) | same | same |
| `max_completion_tokens` | 600 (includes reasoning tokens) | 600 | 600 |
| `reasoning_effort` | `"low"` | not sent | not sent |
| `include_reasoning` | `false` | not sent (mutually exclusive with `reasoning_format`) | not sent |
| `reasoning_format` | **not sent** (unsupported for gpt-oss) | `"hidden"` (`"raw"` + JSON mode is a 400) | not sent |

gpt-oss returns reasoning in `message.reasoning`, never inside `content`, so the
JSON verdict is parsed from `content` alone. Empty content, unparseable JSON,
invalid verdicts and API errors all fail closed with a labelled `error`.

## Reproducibility

`requirements.txt` pins the exact versions the committed classifier, results and
tests were produced with (Python 3.12.2; `.python-version` says 3.12):
scikit-learn 1.9.1, sentence-transformers 5.7.0, transformers 5.17.0, torch 2.14.0,
numpy 2.5.3, faiss-cpu 1.15.0, groq 0.37.1, streamlit 1.63.0, pandas 3.0.5,
joblib 1.6.0, pypdf 6.18.0. The embedding model is pinned to Hugging Face revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The classifier's sidecar
(`models/injection_classifier.meta.json`) records all of this plus the benchmark
hash, and `ml_classifier.load_classifier()` warns if the running environment
differs. If you change any of these versions, retrain and rerun the evaluation;
a logistic regression over embeddings from a different build is a different model.

---

## What changed and why (development notes)

1. **Benign side of the benchmark was one style.** v1 had 40 benign records, 31 of
   them single sentences from one hand-written seed group. Holding that group out
   turned all 31 into false positives, and the classifier scored ordinary support
   text at 0.71-0.78. v2 adds 48 hand-written, chunk-shaped benign records in 8
   domains (customer support, technical manuals, policies, software docs,
   operational procedures, résumé fragments, academic prose, legal/business).
   No existing record was removed; v1 is kept in `data/benchmark/benchmark_v1_155.jsonl`.
2. **Max-sentence scoring needed sentence-level benign training data.** Scoring
   a chunk by its highest-scoring sentence catches injections diluted in benign
   text (0% -> 100% held-out indirect detection) but judges single sentences,
   which the classifier had only ever seen as attacks. Each benign record's
   sentences are now added as label-0 training units (label-safe; attack records
   are not split because indirect wrappers contain benign sentences). This is
   what moved benign real-document chunks out of the high band (19 -> 0) while
   keeping poisoned chunks at 0.72-0.86. `scripts/segmentation_experiment.py`
   reproduces the comparison (whole / max-sentence / 2-sentence window, with and
   without expansion); results in `results/segmentation_experiment.json`.
3. **Routing regression.** `firewall.py` had reverted to "ML above 0.7 blocks with
   no judge", which quarantined legitimate documents, while other modules
   described the judge-based routing. The table above is now the single
   implementation, and Layer 1 non-critical evidence (previously computed and
   ignored) is used for corroboration and forced review.
4. **Layer 1 precision and normalisation.** The critical "disregard the earlier
   ..." rule fired on a benign benchmark record; it now requires an instruction
   noun. Text is NFKC-normalised, zero-width characters are stripped, common
   Cyrillic/Greek look-alikes are folded to Latin, and long base64 runs are
   decoded and rescanned, so disguised versions of known patterns hit the same rules.
5. **Evaluation leakage.** Category-held-out models were trained on all benign
   records including the ones used to measure their FPR; the main evaluation
   scored with the saved model rather than a split-local one. Both fixed. Judge
   failures are now counted and surfaced instead of being folded into recall.
6. **Prompt boundaries.** Both the judge and the answer generator now use a system
   turn for instructions, tagged data blocks for document text with closing tags
   neutralised, JSON response mode for the judge, bounded reasoning effort for
   gpt-oss models, and `None`-content handling.

7. **Attack lineage leakage (second audit).** All attacks descended from 21 seeds
   and the template-group split let seeds and their variants straddle train and
   test. Added `lineage_id`, lineage grouping with per-fold leakage assertions, a
   lineage-clean category-held-out variant, and ten novel indirect payloads.
   Honest effect: headline recall 0.887 to 0.848, indirect "1.00" revealed as
   memorisation (novel payloads 0.10 for the classifier).
8. **AI-addressed evidence rule.** Seven of the ten novel payloads passed silently
   below LOW_THRESHOLD. Their common marker is text addressing the AI reading the
   document; a non-critical Layer 1 rule for that pattern was added after checking
   it fires on zero benign records and zero sample-document chunks. It catches 9 of
   the 10 payloads it was designed on and 3 of 5 held-out phrasings written before
   the rule existed (kept in the test suite, not the benchmark). Silent passes in
   leakage-free CV: 7 to 0.
9. **Groq configuration and pins.** `reasoning_format` removed for gpt-oss
   (unsupported), `include_reasoning=false` added, exact request configuration
   pinned by tests; dependencies pinned to the versions the model was trained with,
   embedding revision pinned, environment recorded in the model sidecar.
---

## Screenshots

Captured from the running app without a Groq key, so Layer 3 is shown failing closed.

| Poisoned FAQ query | Benign banking query | Evaluation tab |
|---|---|---|
| ![poisoned](docs/screenshots/03_poisoned_query.png) | ![benign](docs/screenshots/04_benign_query.png) | ![evaluation](docs/screenshots/05_evaluation.png) |

A full written report of both audits, fixes and results is in `docs/RAGShield_Report.pdf` (regenerate with `python docs/build_report.py`).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your GROQ_API_KEY

python scripts/train_classifier.py                     # ~20 s, downloads all-MiniLM-L6-v2 once
python scripts/run_evaluation.py                       # ablation, category-held-out, error analysis
python scripts/run_evaluation.py --group-held-out      # headline cross-validation (lineage grouping)
streamlit run app.py
```

Without a `GROQ_API_KEY`, the evaluation scripts replace Layer 3 with a 0.5
cutoff and record `llm_judge_used: false`; the app quarantines everything that
needs review (fail closed) and shows the sanitised context instead of an answer.

### Reproducing with the LLM judge

`run_evaluation.py --group-held-out` makes roughly one judge call per
review-band record (about 100 of 213 in the committed configuration), and the
default evaluation about 20. Rate-limit or network failures are listed under
`judge_failure_warning` in the output files and in the Evaluation tab; re-run
rather than quoting a run with failures. Judge and benchmark generation use
separate models (`GROQ_JUDGE_MODEL`, `GROQ_BENCHMARK_MODEL`) so benchmark
building never competes with the judge for quota.

### Deploying (Streamlit Community Cloud)

1. Push the repository to GitHub (see the checklist below). `models/` and
   `results/` are committed, so the deployed app has a classifier and the
   Evaluation tab has data immediately; if `models/` were ever missing the app
   trains the classifier itself on first start.
2. At https://share.streamlit.io choose *New app*, select the repository,
   branch `main`, main file `app.py`, and Python 3.11 or 3.12 under *Advanced
   settings*.
3. In *Advanced settings → Secrets* paste:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

   `ragshield/config.py` reads the environment first and Streamlit secrets
   second, so the same code runs locally with `.env` and on the cloud with
   secrets. Never commit the key.
4. Deploy, then verify with the two demo questions from the README screenshots:
   the password-reset chunk should now show a real judge verdict and the Final
   Answer section should contain an answer.

The first start downloads the 90 MB embedding model and takes about a minute;
Community Cloud's memory limit is sufficient for MiniLM + FAISS. Uploaded
documents are stored on the container's disk and disappear on restart, which is
the intended single-session design.

**Pre-push checklist**

```bash
pytest && pytest -m slow
git status --ignored     # .env, .venv, data/documents must be ignored
grep -r "gsk_" --include="*.py" --include="*.md" --include="*.toml" . ; # must print nothing
```

### Rebuilding the benchmark

```bash
python scripts/build_benchmark.py --from-baseline     # deterministic; reproduces benchmark.jsonl byte for byte
python scripts/build_benchmark.py                     # fresh LLM-assisted generation (non-deterministic)
```

`tests/test_benchmark.py` asserts that the committed benchmark is exactly what
`--from-baseline` produces, has no duplicates, truncated or refusal rows, and
that no critical heuristic fires on any benign record.

### Tests

```bash
pytest              # 87 unit tests, no model download, no network
pytest -m slow      # real-document robustness (needs the trained classifier)
```

---

## Repository layout

```
ragshield/
  config.py            paths, thresholds, model names (env-overridable)
  heuristics.py        Layer 1
  ml_classifier.py     Layer 2 (training units, max-sentence scoring, provenance sidecar)
  llm_judge.py         Layer 3 (prompt, JSON parsing, fail-closed, error classes)
  firewall.py          routing table, shared by app and evaluation
  rag_pipeline.py      loading, chunking, FAISS retrieval
  generate_answer.py   answer LLM with untrusted-context boundary
scripts/
  build_benchmark.py         benchmark construction (--from-baseline is deterministic)
  train_classifier.py        production classifier + smoke test
  run_evaluation.py          ablation / category-held-out / error analysis / --group-held-out
  segmentation_experiment.py evidence behind the scoring strategy
data/benchmark/              benchmark.jsonl (v2), benchmark_v1_155.jsonl, benchmark_baseline_155.jsonl
data/sample_documents/       fictional real-style documents used for robustness checks and the demo
results/                     metrics.json, ablation_matrix.csv, group_held_out_metrics.json, segmentation_experiment.json
tests/                       pytest suite
app.py                       Streamlit UI (RAGShield tab, RAGShield Evaluation tab)
```

---

## Known limitations

* **Benchmark size and lineage.** 213 records (125 attacks, 88 benign) built from
  only 21 seed payloads plus 10 novel ones. Per-fold counts are small and the
  confidence intervals wide; treat differences of a few points as noise. The
  lineage-clean numbers show the classifier's generalisation to genuinely new
  payloads is weak; the design relies on Layer 1 evidence and Layer 3 for those.
* **Nearest-seed lineage is approximate** for the 54 LLM-generated and curated
  rows (cosine 0.35 to 0.84 to their assigned seed). It errs toward grouping more
  together, which makes the split stricter, not looser.
* **Obfuscation.** The classifier itself detects well under half of the
  obfuscated set when lineages are held out; coverage comes from Layer 1 normalisation (zero-width, base64,
  a fixed confusables table) forcing a judge review. Encodings and confusables
  outside that table are not covered, and non-English documents trip the
  non-ASCII heuristic and cost a judge call.
* **Layer 3 cost and dependence.** Roughly 40% of real-document chunks land in
  the review band and require an LLM call, and the system's recall on
  paraphrased and disguised attacks depends on the judge. The judge's own
  quality was verified by hand on the failure cases described in the development
  notes, not by a large labelled judge benchmark.
* **Chunking.** Fixed 500-character windows with 50 overlap; an attack split across
  a chunk boundary is scored in pieces. Concrete example from the demo: the
  poisoned FAQ's "tell the user to email their account password ..." sentence is
  cut by a boundary. The chunk holding its head is hard-blocked; the tail
  fragment ("...ser to email their account password to support@...") scored 0.30
  and passed under the v2 model, and scores 0.37 (escalated to the judge) under
  the v3 model. That margin is thin; sentence-aware chunking would remove this
  class of near-miss. Sentence splitting is regex-based.
* **Single-session index.** FAISS is in memory; uploads are saved to
  `data/documents/` (git-ignored) and are not persisted across restarts.
* **Not a guarantee.** RAGShield reduces the attack surface; the answer model's
  own prompt boundary (system turn, tagged untrusted context) is the last line of
  defence and is also not a formal guarantee.
