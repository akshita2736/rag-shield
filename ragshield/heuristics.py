"""
Layer 1: deterministic heuristic scanner.

Each rule is named, weighted, and optionally marked `critical`.

  * Critical rules are high-precision patterns that, on match, short-circuit
    straight to BLOCKED in firewall.py, bypassing Layers 2 and 3 entirely.
  * Non-critical rules never block on their own. They act as *evidence*
    that firewall.py combines with the Layer 2 probability: any non-critical
    match forces at least a Layer 3 review (so an obfuscated attack that the
    embedding classifier scores low is still looked at), and a non-critical
    match that corroborates a high Layer 2 score lets the firewall block
    without spending an LLM call. See firewall.evaluate_chunk() for the
    exact routing table.

Before any rule runs, the text is normalised so cheap disguises do not
defeat the patterns: NFKC normalisation, removal of zero-width characters,
and folding of common Cyrillic/Greek look-alike letters onto their Latin
counterparts. Long base64 runs are decoded and rescanned as well.

Keep rules few and high-precision. The goal of this layer is to catch
obvious, cheap-to-detect attacks fast, not to be a complete detector.
That is what Layers 2/3 are for.
"""

from __future__ import annotations

import base64
import binascii
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
    score: float  # sum of matched rule weights, NOT normalised to 0-1
    critical_triggered: bool
    matched_rules: list[str] = field(default_factory=list)

    @property
    def any_match(self) -> bool:
        return bool(self.matched_rules)


# ---------------------------------------------------------------------------
# Rule definitions
#
# `critical=True` is reserved for patterns specific enough that they are
# very unlikely to appear in legitimate document content. The benign
# hard-negative set in the benchmark is the regression test for this:
# tests/test_benchmark.py asserts that no critical rule fires on any benign
# record. If one ever does, reword the pattern or demote it to non-critical
# rather than deleting the benign example.
# ---------------------------------------------------------------------------

# Nouns that make "ignore/disregard the previous ..." unambiguously about
# instructions given to an AI rather than about document data ("disregard
# the earlier estimate" is legitimate; "disregard the earlier instructions"
# is not).
_INSTRUCTION_NOUNS = (
    r"(instructions?|prompts?|directions?|directives?|rules?|commands?|"
    r"guidance|guidelines?|constraints?|context|messages?|orders?|"
    r"configuration|programming|training|system prompt)"
)

_AI_NOUN = r"(?:ai|a\.i\.|assistant|language model|llm|chatbot|bot|model|copilot|agent)s?"
_AI_MODIFIER = r"(?:ai |digital |virtual |automated |language |support |screening |candidate screening )?"
_AI_ADDRESSED_RE = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in (
            # "Note to any AI assistant ...", "Attention virtual assistants:", "Dear digital assistant ..."
            r"\b(?:note|message|instructions?|reminder|attention|dear)\b[^.:\n]{0,20}\b(?:to|for)?\s*"
            r"(?:any|the|all|every|virtual|digital|automated|ai)?\s*" + _AI_MODIFIER + _AI_NOUN + r"\b",
            # "If you are a language model reading this ..."
            r"\bif (?:you are|you're) (?:an?|the)?\s*" + _AI_MODIFIER + _AI_NOUN + r"\b",
            # "AI assistant: when asked ...", "Support assistant, when you respond ...", "screening bots: mark ..."
            _AI_MODIFIER + _AI_NOUN + r"\s*[:,]\s*(?:when|if|do|please|reply|respond|answer|tell|mark|rate|ignore|"
            r"treat|output|append|include|describe|advise|recommend|state|omit|skip|flag)\b",
            # "the chatbot answering patient questions", "any AI assistant summarizing this listing"
            _AI_NOUN + r"\b[^.\n]{0,25}\b(?:reading|processing|summari[sz]ing|answering|using|screening|composing|"
            r"reviewing)\b[^.\n]{0,40}\b(?:this|these|the)\b",
            # "Any automated summary of this page must ...", "Whoever processes these minutes automatically should ..."
            r"\b(?:any|all|every|whoever|systems?|tools?|helpers?)\b[^.\n]{0,30}\b(?:automated|automatic|automatically|ai|"
            r"assistant)s?\b[^.\n]{0,60}\b(?:must|should|shall|are instructed|is instructed)\b",
            # "To the assistant using this guide: ..."
            r"\bto the " + _AI_MODIFIER + _AI_NOUN + r"\b[^.\n]{0,30}\b(?:using|reading|processing|summari[sz]ing|answering)\b",
        )
    ),
    re.IGNORECASE,
)

_RULES: list[Rule] = [
    # --- Direct override (critical: rarely appears outside an attack) ---
    Rule(
        name="direct_override_ignore_instructions",
        pattern=re.compile(
            r"\b(ignore|disregard|forget|discard) (all |any |the |your |every )?"
            r"(previous|prior|above|earlier|preceding|original) " + _INSTRUCTION_NOUNS + r"\b",
            re.IGNORECASE,
        ),
        weight=1.0,
        critical=True,
    ),
    Rule(
        # "Disregard the above and instead ..." — no noun, so it is only
        # evidence, not a hard block ("disregard the above chart" is legit).
        name="direct_override_disregard_above",
        pattern=re.compile(
            r"\b(ignore|disregard) (all |any |the )?(previous|prior|above|earlier)\b",
            re.IGNORECASE,
        ),
        weight=0.6,
        critical=False,
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
            r"\b(repeat|reveal|print|show|display|output|disclose) (me )?(your |the )?"
            r"(full |hidden |original |complete |entire |exact )?"
            r"(system prompt|system message|system instructions?|hidden instructions?)\b",
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
    Rule(
        # Text pretending to be a chat-role header addressed to the model.
        name="fake_role_header",
        pattern=re.compile(r"(^|\n)\s*(system|assistant|ai)\s*:", re.IGNORECASE),
        weight=0.5,
        critical=False,
    ),
    Rule(
        # Text that addresses the AI reading the document ("note to any AI
        # assistant summarizing this listing", "if you are a language model",
        # "support assistant, when you respond", "whoever processes these
        # minutes automatically should"). This vocative pattern is the
        # linguistic signature of indirect injection whose payload uses none of
        # the classic override/extraction vocabulary; the embedding classifier
        # scored 7 of 10 such payloads below LOW_THRESHOLD in leakage-free
        # cross-validation. Non-critical on purpose: it only guarantees a
        # Layer 3 review. Validated against every benign benchmark record and
        # all sample-document chunks (zero matches) before being added.
        name="ai_addressed_instruction",
        pattern=_AI_ADDRESSED_RE,
        weight=0.6,
        critical=False,
    ),
]

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff")  # ZWSP, ZWNJ, ZWJ, word joiner, BOM

# Common look-alike letters (Cyrillic / Greek) folded onto Latin so that a
# homoglyph-substituted attack matches the same regexes as the plain one.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "і": "i", "р": "p", "с": "c", "у": "y",
    "х": "x", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ɡ": "g",
    "А": "A", "Е": "E", "О": "O", "І": "I", "Р": "P", "С": "C", "Т": "T",
    "Н": "H", "К": "K", "М": "M", "В": "B", "Х": "X", "Ѕ": "S", "Ј": "J",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "ι": "i", "ε": "e",
})

_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_MIXED_SCRIPT_WORD_RE = re.compile(r"[A-Za-z]+[Ѐ-ӿͰ-Ͽ]|[Ѐ-ӿͰ-Ͽ]+[A-Za-z]")


def normalize(text: str) -> str:
    """NFKC-normalise, drop zero-width characters, fold confusable letters."""
    normalized = unicodedata.normalize("NFKC", text)
    for ch in _ZERO_WIDTH_CHARS:
        normalized = normalized.replace(ch, "")
    return normalized.translate(_CONFUSABLES)


def _decode_base64_runs(text: str) -> list[str]:
    """Return decoded, printable payloads for every long base64 run in text."""
    payloads = []
    for match in _BASE64_RUN_RE.finditer(text):
        blob = match.group(0)
        blob += "=" * (-len(blob) % 4)  # repair missing padding
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded and sum(ch.isprintable() for ch in decoded) / len(decoded) > 0.9:
            payloads.append(decoded)
    return payloads


def _match_rules(text: str, suffix: str = "") -> tuple[float, bool, list[str]]:
    score = 0.0
    critical = False
    matched: list[str] = []
    for rule in _RULES:
        if rule.pattern.search(text):
            matched.append(rule.name + suffix)
            score += rule.weight
            if rule.critical:
                critical = True
    return score, critical, matched


def _obfuscation_score(raw_text: str) -> tuple[float, list[str]]:
    """
    Cheap structural checks for obfuscated/encoded instructions. Runs on the
    RAW text (before normalisation) because normalisation deliberately
    removes the very artefacts these checks look for.
    """
    matched = []
    score = 0.0

    if any(ch in raw_text for ch in _ZERO_WIDTH_CHARS):
        matched.append("obfuscation_zero_width_chars")
        score += 0.7

    if _BASE64_RUN_RE.search(raw_text):
        matched.append("obfuscation_base64_block")
        score += 0.5

    # A single word mixing Latin with Cyrillic/Greek letters is a strong
    # homoglyph signal (real multilingual text switches script between
    # words, not inside them).
    if _MIXED_SCRIPT_WORD_RE.search(raw_text):
        matched.append("obfuscation_mixed_script_word")
        score += 0.7

    # High ratio of non-ASCII letters can also indicate substitution. This
    # fires on genuinely non-English documents too, which is why it is
    # non-critical: it costs a Layer 3 review, never a block.
    letters = [ch for ch in raw_text if ch.isalpha()]
    if letters:
        non_ascii_ratio = sum(1 for ch in letters if ord(ch) > 127) / len(letters)
        if non_ascii_ratio > 0.15:
            matched.append("obfuscation_non_ascii_ratio")
            score += 0.4

    return score, matched


def scan(text: str) -> HeuristicResult:
    """Run all Layer 1 rules against a chunk of text."""
    if not text:
        return HeuristicResult(score=0.0, critical_triggered=False, matched_rules=[])

    normalized = normalize(text)
    score, critical_triggered, matched_rules = _match_rules(normalized)

    # Decode long base64 runs and rescan the payload with the same rules.
    for payload in _decode_base64_runs(normalized):
        p_score, p_critical, p_matched = _match_rules(normalize(payload), suffix="[base64]")
        score += p_score
        critical_triggered = critical_triggered or p_critical
        matched_rules.extend(p_matched)

    obf_score, obf_matched = _obfuscation_score(text)
    score += obf_score
    matched_rules.extend(obf_matched)

    return HeuristicResult(
        score=score,
        critical_triggered=critical_triggered,
        matched_rules=matched_rules,
    )


def rule_names() -> list[str]:
    """All pattern-rule names (excluding structural obfuscation checks)."""
    return [rule.name for rule in _RULES]
