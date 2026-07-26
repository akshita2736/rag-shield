"""
RAGShield configuration.

Kept as plain module-level constants (no pydantic-settings, no config
framework) on purpose — this is a small inspectable portfolio project,
not production infra. Everything here should be readable top to bottom
in under a minute.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; safe no-op if it isn't

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (ragshield/)

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_PATH = BENCHMARK_DIR / "benchmark.jsonl"

MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "injection_classifier.joblib"

RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
ABLATION_MATRIX_PATH = RESULTS_DIR / "ablation_matrix.csv"

# Make sure the directories that scripts/app.py will write into exist.
for _dir in (DOCUMENTS_DIR, BENCHMARK_DIR, MODELS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embeddings (Layer 2 input)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformers, 384-dim

# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # characters of overlap between adjacent chunks
RETRIEVAL_TOP_K = 4  # chunks returned per query

# ---------------------------------------------------------------------------
# Firewall routing thresholds (Layer 2 -> Layer 3 routing)
#
# These are ML-probability-only thresholds. They are NOT combined with the
# heuristic score into a single fused number — see heuristics.py / firewall.py
# for why (critical heuristic matches short-circuit straight to quarantine
# instead, bypassing these thresholds entirely).
# ---------------------------------------------------------------------------

LOW_THRESHOLD = 0.3   # ML probability below this -> SAFE, no LLM call
HIGH_THRESHOLD = 0.7  # ML probability above this -> QUARANTINE, no LLM call
# Between LOW_THRESHOLD and HIGH_THRESHOLD -> escalate to Layer 3 LLM judge

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_JUDGE_MODEL = "llama-3.1-8b-instant"   # Layer 3 contextual judge
GROQ_ANSWER_MODEL = "llama-3.1-8b-instant"  # final answer generation

if not GROQ_API_KEY:
    # Not raising here — some scripts (e.g. train_classifier.py) don't need
    # an LLM at all and shouldn't be blocked by a missing key. app.py and
    # llm_judge.py should check this themselves before making a call.
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