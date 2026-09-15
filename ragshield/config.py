"""
RAGShield configuration.

Kept as plain module-level constants (no pydantic-settings, no config
framework) on purpose: this is a small inspectable portfolio project, not
production infra. Everything here should be readable top to bottom in
under a minute.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; safe no-op if it isn't

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"  # user uploads from the Streamlit app (git-ignored)
SAMPLE_DOCUMENTS_DIR = DATA_DIR / "sample_documents"  # committed realistic demo/test documents
BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_PATH = BENCHMARK_DIR / "benchmark.jsonl"  # current benchmark (v2, 203 records)
BENCHMARK_V1_PATH = BENCHMARK_DIR / "benchmark_v1_155.jsonl"  # frozen v1 for comparison
BENCHMARK_BASELINE_PATH = BENCHMARK_DIR / "benchmark_baseline_155.jsonl"  # raw LLM output v1 was repaired from

MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "injection_classifier.joblib"
CLASSIFIER_META_PATH = MODELS_DIR / "injection_classifier.meta.json"  # provenance sidecar

RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
ABLATION_MATRIX_PATH = RESULTS_DIR / "ablation_matrix.csv"
GROUP_HELD_OUT_METRICS_PATH = RESULTS_DIR / "group_held_out_metrics.json"

# Make sure the directories that scripts/app.py will write into exist.
for _dir in (DOCUMENTS_DIR, BENCHMARK_DIR, MODELS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embeddings (Layer 2 input)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
# Pinned Hugging Face revision so every machine embeds identically. The
# classifier is a linear model over these embeddings: a different revision
# would silently be a different model. Recorded in the classifier's sidecar.
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # characters of overlap between adjacent chunks
RETRIEVAL_TOP_K = 4  # chunks returned per query

# ---------------------------------------------------------------------------
# Firewall routing thresholds (Layer 2 -> Layer 3 routing)
#
# These are ML-probability bands. They are never fused with the heuristic
# score into a single number; Layer 1 evidence is combined with the band
# by the routing table in firewall.evaluate_chunk():
#
#   critical heuristic match                     -> BLOCKED  (no LLM call)
#   prob <  LOW_THRESHOLD, no heuristic match     -> SAFE     (no LLM call)
#   prob >= HIGH_THRESHOLD, any heuristic match   -> BLOCKED  (no LLM call; two layers agree)
#   everything else                              -> Layer 3 LLM judge decides
#
# The high band alone does NOT block: validated on real documents, the
# embedding classifier gives legitimate instructional prose (support
# articles, manuals, policies) high scores, so a high score without
# independent Layer 1 evidence is sent to the contextual judge instead.
# ---------------------------------------------------------------------------

LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------

def _secret(name: str, default: str = "") -> str:
    """Environment variable first (.env / shell); falls back to Streamlit
    secrets so the same code works on Streamlit Community Cloud, where
    secrets are entered in the app's Settings panel rather than a .env file."""
    value = os.getenv(name)
    if value:
        return value
    try:  # only reached when the variable is missing, e.g. on Streamlit Cloud
        import streamlit as st  # noqa: WPS433 - optional dependency path

        return str(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001 - no secrets file / not running under Streamlit
        return default


GROQ_API_KEY = _secret("GROQ_API_KEY")
GROQ_JUDGE_MODEL = _secret("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")  # Layer 3 contextual judge
GROQ_ANSWER_MODEL = _secret("GROQ_ANSWER_MODEL", "openai/gpt-oss-20b")  # final answer generation

# Benchmark generation deliberately uses a separate model so that building
# the benchmark never competes with the production judge for quota.
GROQ_BENCHMARK_MODEL = _secret("GROQ_BENCHMARK_MODEL", "openai/gpt-oss-120b")

GROQ_MAX_RETRIES = 3  # SDK-level retries with backoff for 429/5xx before we give up
JUDGE_MAX_COMPLETION_TOKENS = 600  # includes reasoning tokens for reasoning models
JUDGE_REASONING_EFFORT = "low"  # only sent to models that support it (gpt-oss family)

if not GROQ_API_KEY:
    # Not raising here: some scripts (e.g. train_classifier.py) don't need
    # an LLM at all and shouldn't be blocked by a missing key. Code that
    # calls the LLM checks this itself and fails closed.
    print(
        "[config] Warning: GROQ_API_KEY is not set. Set it in a .env file "
        "or your environment before running anything that calls the LLM."
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

ATTACK_CATEGORIES = [
    "direct_override",
    "role_manipulation",
    "prompt_extraction",
    "indirect_injection",
    "obfuscated",
    "paraphrased",
]
BENIGN_CATEGORY = "benign_hard_negative"

RANDOM_SEED = 42  # used for train/test splitting, reproducible eval runs
