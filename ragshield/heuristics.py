"""
Layer 1: deterministic heuristic scanner.

Each rule is named, weighted, and optionally marked `critical`. Critical
rules are high-precision patterns that, on match, short-circuit straight
to QUARANTINE in firewall.py — bypassing ML/LLM entirely. Non-critical
rules only contribute to the returned score/feature set; they do NOT get
fused with the ML probability into a single number (see config.py /
firewall.py for why).

Keep rules few and high-precision. The goal of this layer is to catch
obvious, cheap-to-detect attacks fast — not to be a complete detector.
That's what Layers 2/3 are for.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    weight: float
    critical: bool = False


@dataclass
class HeuristicResult:
    score: float  # sum of matched rule weights, NOT normalized to 0-1
    critical_triggered: bool
    matched_rules: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule definitions
#
# `critical=True` is reserved for patterns specific enough that they are
# very unlikely to appear in legitimate document content (see the benign
# hard-negative set in the benchmark — these rules should score 0 against
# all of it). If a critical rule ever fires on a benign example during
# evaluation, that's a signal to either reword the pattern or demote it
# to non-critical.
# ---------------------------------------------------------------------------

_RULES: list[Rule] = [
    # --- Direct override (critical: rarely appears outside an attack) ---
    Rule(
        name="direct_override_ignore_instructions",
        pattern=re.compile(
            r"\bignore (all|any|the) (previous|prior|above|earlier) instructions?\b",
            re.IGNORECASE,
        ),
        weight=1.0,
        critical=True,
    ),
    Rule(
        name="direct_override_disregard",
        pattern=re.compile(
            r"\bdisregard (all|any|the) (previous|prior|above|earlier)\b",
            re.IGNORECASE,
        ),
        weight=1.0,
        critical=True,
    ),
    Rule(
        name="direct_override_new_instructions",
        pattern=re.compile(r"\bnew instructions?:\s", re.IGNORECASE),
        weight=0.6,
        critical=False,  # could appear legitimately in a doc about instructions
    ),

    # --- Role manipulation ---
    Rule(
        name="role_manipulation_you_are_now",
        pattern=re.compile(r"\byou are now\b", re.IGNORECASE),
        weight=0.5,
        critical=False,
    ),
    Rule(
        name="role_manipulation_act_as",
        pattern=re.compile(r"\bact as (a|an|if you were)\b", re.IGNORECASE),
        weight=0.4,
        critical=False,
    ),
    Rule(
        name="role_manipulation_from_now_on",
        pattern=re.compile(r"\bfrom now on,? you\b", re.IGNORECASE),
        weight=0.5,
        critical=False,
    ),

    # --- Prompt extraction (critical: specific phrasing, rare in legit content) ---
    Rule(
        name="prompt_extraction_repeat_system",
        pattern=re.compile(
            r"\b(repeat|reveal|print|show) (your |the )?system prompt\b",
            re.IGNORECASE,
        ),
        weight=1.0,
        critical=True,
    ),
    Rule(
        name="prompt_extraction_what_were_you_told",
        pattern=re.compile(
            r"\bwhat (were|was) you (told|instructed)\b", re.IGNORECASE
        ),
        weight=0.6,
        critical=False,
    ),

    # --- Delimiter / formatting abuse ---
    Rule(
        name="fake_system_delimiter",
        pattern=re.compile(r"###\s*system\s*###", re.IGNORECASE),
        weight=0.8,
        critical=False,
    ),
    Rule(
        name="fake_inst_token",
        pattern=re.compile(r"\[/?INST\]", re.IGNORECASE),
        weight=0.6,
        critical=False,
    ),
]


def _obfuscation_score(text: str) -> tuple[float, list[str]]:
    """
    Cheap structural checks for obfuscated/encoded instructions.
    Returns (score_contribution, matched_rule_names).
    """
    matched = []
    score = 0.0

    # Zero-width / invisible unicode characters — a common disguise trick.
    zero_width_chars = ("\u200b", "\u200c", "\u200d", "\ufeff")
    if any(ch in text for ch in zero_width_chars):
        matched.append("obfuscation_zero_width_chars")
        score += 0.7

    # Long base64-looking blocks (heuristic: long run of base64 alphabet chars).
    if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", text):
        matched.append("obfuscation_base64_block")
        score += 0.5

    # High ratio of non-ASCII characters can indicate homoglyph substitution
    # (e.g. Cyrillic look-alikes replacing Latin letters mid-word).
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        non_ascii_ratio = sum(1 for ch in letters if ord(ch) > 127) / len(letters)
        if non_ascii_ratio > 0.15:
            matched.append("obfuscation_non_ascii_ratio")
            score += 0.4

    return score, matched


def scan(text: str) -> HeuristicResult:
    """Run all Layer 1 rules against a chunk of text."""
    normalized = unicodedata.normalize("NFKC", text)

    score = 0.0
    critical_triggered = False
    matched_rules: list[str] = []

    for rule in _RULES:
        if rule.pattern.search(normalized):
            matched_rules.append(rule.name)
            score += rule.weight
            if rule.critical:
                critical_triggered = True

    obf_score, obf_matched = _obfuscation_score(normalized)
    score += obf_score
    matched_rules.extend(obf_matched)

    return HeuristicResult(
        score=score,
        critical_triggered=critical_triggered,
        matched_rules=matched_rules,
    )


def rule_names() -> list[str]:
    """All non-obfuscation rule names, useful for evaluation/ablation reporting."""
    return [rule.name for rule in _RULES]