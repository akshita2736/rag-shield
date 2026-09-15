"""
docs/build_report.py

Builds docs/RAGShield_Report.pdf from the committed results files and the
screenshots in docs/screenshots/. Numbers in the tables are read from
results/*.json so the report cannot drift from the evaluation outputs;
narrative text states the values at the time of writing and is reviewed by hand.

Optional tooling, not a runtime dependency: pip install reportlab pillow
Run:  python docs/build_report.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs/screenshots"
OUT = ROOT / "docs/RAGShield_Report.pdf"
SLICES = ROOT / "docs/_slices"

metrics = json.loads((ROOT / "results/metrics.json").read_text())
group = json.loads((ROOT / "results/group_held_out_metrics.json").read_text())
tg = json.loads((ROOT / "results/group_held_out_metrics_template_group.json").read_text())["aggregate"]
seg = json.loads((ROOT / "results/segmentation_experiment.json").read_text())
ablation = (ROOT / "results/ablation_matrix.csv").read_text().strip().splitlines()
agg = group["aggregate"]
eb = agg["error_breakdown"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1f2937"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1f2937"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=24, leading=30, spaceAfter=14)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11, alignment=TA_CENTER, textColor=colors.HexColor("#4b5563"))
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#4b5563"), spaceBefore=2, spaceAfter=10)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8.5, leading=11, backColor=colors.HexColor("#f3f4f6"), leftIndent=6, borderPadding=4, spaceAfter=8)
W = letter[0] - 1.2 * inch


def P(t, st=BODY):
    return Paragraph(t, st)


def table(rows, widths=None, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    widths = widths or [W / len(rows[0])] * len(rows[0])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")))
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9fafb")))
    t.setStyle(TableStyle(style))
    return t


def slice_image(path, max_ratio=1.25):
    """Split a tall screenshot into page-sized slices, cutting on near-blank rows."""
    SLICES.mkdir(exist_ok=True)
    im = PILImage.open(path).convert("RGB")
    w, h = im.size
    target = int(w * max_ratio)
    arr = np.asarray(im)
    blank = (np.abs(arr.astype(int) - arr[5, 5]).sum(axis=2) < 30).mean(axis=1) > 0.995
    pieces, y = [], 0
    while y < h:
        end = min(y + target, h)
        if end < h:
            cands = [r for r in range(end, max(y + target // 2, end - 400), -1) if blank[r]]
            if cands:
                end = cands[0]
        pieces.append(im.crop((0, y, w, end)))
        y = end
    out = []
    for i, piece in enumerate(pieces):
        p = SLICES / f"{path.stem}_{i}.png"
        piece.save(p)
        out.append((p, piece.size))
    return out


def figure(path, caption, max_ratio=1.25):
    flow = []
    for i, (p, (w, h)) in enumerate(slice_image(path, max_ratio)):
        flow.append(Image(str(p), width=W, height=W * h / w))
        flow.append(P(caption if i == 0 else f"(continued) {caption.split('.')[0]}.", CAP))
    return flow


story = []
# ------------------------------------------------------------------ title
story += [Spacer(1, 1.4 * inch), P("RAGShield", TITLE), P("Reliability Framework for RAG Systems", SUB),
          P("Debugging, audit and finalisation report (two audit passes)", SUB), Spacer(1, 0.3 * inch),
          P(f"Prepared {datetime.date.today():%d %B %Y} · project directory <font face='Courier'>~/Claude/rag-shield</font>", SUB),
          Spacer(1, 0.5 * inch)]
story.append(table([["Headline (group-held-out, Layers 1+2)", "Before (v1, template-group split)", "After (v3, lineage split)"],
                    ["Precision", "0.767", f"{agg['precision']:.3f}"], ["Recall", "1.000", f"{agg['recall']:.3f}"],
                    ["F1", "0.868", f"{agg['f1']:.3f}"], ["False positive rate", "0.875", f"{agg['false_positive_rate']:.3f}"],
                    ["Attack lineages leaking into training", "present", "0"],
                    ["Attacks passing silently (no Layer 3 review) in CV", "n/a", f"{eb.get('fn_layer_2_silent_pass', 0)}"],
                    ["Legitimate real-document chunks quarantined without review", "19 / 34 (56%)", "0 / 34"]],
                   widths=[W * 0.5, W * 0.25, W * 0.25]))
story.append(P("'Before' is the shipped repository evaluated with the same script and no LLM judge; 'After' is the committed state after two audits, "
               "evaluated with attack-lineage grouping (stricter than the original split). Every number in this document was produced by running the code. "
               "Section 13 covers the second audit and the GO/NO-GO verdict.", CAP))
story.append(PageBreak())

# ------------------------------------------------------------------ 1
story += [P("1. Executive summary", H1),
P("RAGShield is a three-layer prompt-injection firewall that sits between a FAISS retriever and the answer-generating LLM. Layer 1 is a deterministic heuristic scanner, "
  "Layer 2 an embedding classifier (all-MiniLM-L6-v2 → logistic regression), Layer 3 a contextual LLM judge on Groq. The central design idea is that "
  "<b>instructional language is not prompt injection</b>: manuals, policies and support articles are full of instructions for humans, whereas an injection is text that tries to steer the downstream AI."),
P("The repository as shipped had three interacting problems. The classifier had been trained on 40 single-sentence benign examples against 115 attacks, so it scored ordinary "
  "document prose at 0.7 or higher. The firewall had regressed to blocking anything above 0.7 without consulting the judge. And the group-held-out evaluation collapsed to 31 false "
  "positives out of 31 when the single benign style was held out. Together these meant that a normal PDF full of procedures would have had more than half of its text quarantined silently."),
P("First audit fixes: the benign side of the benchmark was extended with 48 realistic multi-sentence document chunks in eight domains; benign records are additionally trained at sentence "
  "level so max-sentence scoring judges units the model has seen; the routing table was unified so a high ML score alone never blocks and Layer 1 evidence forces a judge review; Layer 1 gained "
  "Unicode normalisation and base64 rescanning; two evaluation leaks were closed; judge failures are counted separately; both LLM prompts were hardened. Thresholds (0.30 / 0.70) were not changed."),
P("A second audit (section 13) then traced attack lineage through the benchmark: all attacks descend from 21 seed sentences, and the original group split let a seed sit in training while its "
  "wrapped or paraphrased form sat in test. The benchmark now carries an explicit lineage_id, evaluation groups on it, ten genuinely novel indirect payloads were added, and the honest numbers are "
  "lower than the first report's: recall 0.848 instead of 0.887, and the classifier detects only 1 of 10 novel indirect payloads. A Layer 1 rule for text that addresses the AI now escalates "
  "9 of those 10 to the judge, so no attack in leakage-free cross-validation passes without Layer 3 review. The Groq request configuration was corrected against current documentation and "
  "all dependencies were pinned to the versions the model was trained with."),
P("Verification: 87 unit tests and 2 real-document tests pass; every evaluation script was rerun on the final code and model; the Streamlit app was driven end to end in a headless browser. "
  "Live Groq calls could not be exercised (no API key available), so judge accuracy remains to be confirmed by the owner with the two commands in section 10."),
]

# ------------------------------------------------------------------ 2
story += [P("2. Architecture and routing", H1),
P("Documents → chunking (500 chars, 50 overlap) → MiniLM embeddings → FAISS retrieval → <b>RAGShield firewall</b> → sanitised context → answer LLM. "
  "The same routing function (<font face='Courier'>firewall.screen_chunks</font>) is used by the Streamlit app and by every evaluation condition."),
table([["Layer 1 result", "Layer 2 probability", "Decision", "LLM call"],
       ["critical rule matched", "not scored", "BLOCKED", "no"],
       ["no match", "&lt; 0.30", "SAFE", "no"],
       ["any match", "≥ 0.70", "BLOCKED (two independent layers agree)", "no"],
       ["anything else", "anything else", "Layer 3 judge decides: SAFE → included (shown as REVIEW); INJECTION or any judge failure → BLOCKED", "yes"]],
      widths=[W * 0.2, W * 0.18, W * 0.5, W * 0.12]),
P("A <b>high Layer 2 score alone never blocks</b>, because legitimate manuals and policies score high. A <b>low Layer 2 score does not pass if Layer 1 found evidence</b>: base64 payloads, "
  "homoglyph text and text that addresses the AI score low for the classifier but leave detectable traces. Layer 3 fails closed with a labelled error class."),
]

# ------------------------------------------------------------------ 3 audit (first pass)
story += [PageBreak(), P("3. First-audit findings", H1)]
bugs = [["Issue", "Severity", "File(s)", "Evidence", "Fix applied"],
["Routing regression: ML ≥ 0.7 hard-blocked with no judge", "Critical", "firewall.py, config.py", "19 of 34 real-document chunks scored ≥ 0.7 → 56% of legitimate text quarantined without review", "Unified routing table (section 2)"],
["Classifier scored legitimate prose 0.71–0.78", "Critical", "ml_classifier.py, benchmark", "Reproduced (password-reset sentence = 0.767); 40 single-sentence benign vs 115 attacks", "Benchmark v2 (+48 realistic benign chunks); sentence-level benign training units"],
["Group-held-out collapse", "Critical", "benchmark", "benign_seed fold 31 FP / 0 TN; no-LLM aggregate FPR 0.875", "Same as above; FPR now 0.057 under the stricter lineage split"],
["Critical heuristic fired on a benign record", "High", "heuristics.py", "'Disregard the earlier estimate…' hard-blocked", "Rule requires an instruction noun"],
["Obfuscated attacks passed silently below LOW", "High", "firewall.py, heuristics.py", "6 base64/homoglyph records scored &lt; 0.30; Layer 1 evidence unused", "Any Layer 1 match forces a judge review; zero-width stripping, confusable folding, base64 rescan"],
["Category-held-out trained on its own benign test rows", "High", "run_evaluation.py", "train set excluded only the attack category", "Benign test records excluded from training"],
["Main evaluation scored with the saved production model", "Medium", "run_evaluation.py", "Loaded CLASSIFIER_PATH instead of a split-local fit", "Split-local models everywhere; provenance sidecar"],
["Judge failures indistinguishable from verdicts", "High", "llm_judge.py, run_evaluation.py", "Fail-closed INJECTION carried no marker", "JudgeResult.error; evaluation counts failures"],
["Prompt construction: no system turn, breakable framing, None content unhandled", "High", "llm_judge.py, generate_answer.py", "gpt-oss reasoning could exhaust max_tokens=300 → None → exception", "System turn, tagged blocks, JSON mode, bounded reasoning, None handling"],
["Benchmark at benchmark/ while code reads data/benchmark/; baseline git-ignored", "Medium", "layout, .gitignore", "rebuild path broken on a fresh clone", "Relocated; un-ignored; --from-baseline reproduces byte for byte"],
["Duplicate requirements; deprecated Streamlit argument; misleading risk label", "Low", "requirements.txt, app.py", "pip tolerated it; Streamlit warned", "Cleaned"]]
story.append(table(bugs, widths=[W * 0.25, W * 0.08, W * 0.14, W * 0.28, W * 0.25]))
story.append(P("Actual bugs found in the first audit. Architectural, evaluation, benchmark and documentation findings and accepted limitations are in the README's development notes and Known limitations.", CAP))

# ------------------------------------------------------------------ 4 root cause
story += [PageBreak(), P("4. Root-cause investigation (first audit)", H1),
table([["Observation", "Shipped state", "Meaning"],
       ["Benign chunks (34) by ML band, max-sentence scoring", "0 safe · 15 judge band · 19 above 0.70", "19 chunks blocked with no review under the shipped routing"],
       ["Reported false-positive sentence", "'Please follow the steps below in order to reset your password.' = 0.767", "Matches the 0.76–0.78 figure reported during development"],
       ["Worst-scoring real sentences", "'Managers must approve remote arrangements in writing.' = 0.79", "Model learned authority/rule vocabulary, not 'aimed at the AI'"],
       ["class_weight='balanced' alone", "33/34 chunks move to the judge band, 0 to safe", "Prior correction is not the fix; the model had never seen document prose"],
       ["Group-held-out (no LLM), benign_seed fold", "TP 23 · FP 31 · TN 0 · FN 0", "Reproduces the reported collapse"]],
      widths=[W * 0.3, W * 0.38, W * 0.32]),
P("Conclusion: a data-distribution problem, not a threshold problem. Moving HIGH_THRESHOLD would have hidden it.", CAP),
P("Segmentation experiment (rerun on v3 with lineage grouping)", H2)]
rows = [["Benign sentence expansion", "Strategy", "Group-CV P / R / F1 / FPR", "Held-out indirect detection", "Benign doc chunks safe / judge / high", "Poisoned injected chunk scores"]]
for r in seg["rows"]:
    g = r["group_cv_no_llm"]; d = r["benign_sample_document_chunks"]
    rows.append(["on" if r["benign_sentence_expansion"] else "off", r["strategy"],
                 f"{g['precision']:.3f} / {g['recall']:.3f} / {g['f1']:.3f} / {g['false_positive_rate']:.3f}",
                 f"{r['indirect_injection_held_out_detection_at_0.5']:.2f}", f"{d['safe_band']} / {d['judge_band']} / {d['high_band']}",
                 ", ".join(f"{x:.2f}" for x in r["poisoned_document"]["injected_chunk_scores"])])
story.append(table(rows, widths=[W * 0.13, W * 0.14, W * 0.25, W * 0.13, W * 0.17, W * 0.18]))
story.append(P("Production choice: max-sentence scoring with benign sentence expansion. Whole-chunk scoring detects 0% of held-out indirect injections; max-sentence without expansion pushes "
               "almost every real chunk into the judge band and three above 0.70. The held-out indirect column now includes the ten novel payloads, which is why it reads 0.70 rather than 1.00.", CAP))

# ------------------------------------------------------------------ 5 changes
story += [PageBreak(), P("5. What changed, file by file, and why", H1)]
changes = [["File", "Change", "Why"],
["scripts/build_benchmark.py", "REALISTIC_BENIGN_SEEDS (48 chunks); NOVEL_INDIRECT_PAYLOADS (10); lineage_id on every record with a frozen nearest-seed map; repair rows grouped per category; --from-baseline deterministic rebuild", "Benign diversity was the root cause; attack lineage must be explicit; the committed benchmark must be reproducible offline"],
["data/benchmark/", "benchmark.jsonl is v3 (213); v1 and v2 frozen; baseline un-ignored", "Preserve earlier versions for comparison; rebuild works on a fresh clone"],
["ragshield/ml_classifier.py", "training_units() adds each benign sentence as a label-0 unit; fit_classifier() shared by training and evaluation; provenance sidecar with library versions and embedding revision; environment-mismatch warning on load", "Align training units with max-sentence inference; fold models trained like production; reproducibility"],
["ragshield/heuristics.py", "normalize(): NFKC, zero-width removal, confusable folding; base64 rescan; 'disregard the earlier…' requires an instruction noun; mixed-script, fake-role-header and ai_addressed_instruction evidence rules", "Disguised patterns must hit the same rules; a critical rule fired on a benign record; novel indirect payloads passed silently"],
["ragshield/firewall.py", "Routing table of section 2; ChunkResult gains band, decided_by, heuristic_matches, judge; batched Layer 2; ABLATION_NO_JUDGE sentinel", "Fix the regression; make Layer 1 evidence useful; one routing implementation"],
["ragshield/llm_judge.py", "System turn + tagged block; JSON mode; gpt-oss: reasoning_effort low + include_reasoning false (no reasoning_format); qwen/minimax/deepseek: reasoning_format hidden; max_completion_tokens; None handling; JudgeResult.error", "Security boundary; correct Groq parameters per current docs; failures distinguishable from verdicts"],
["ragshield/generate_answer.py", "System turn declaring retrieved text UNTRUSTED; tagged blocks; same reasoning parameter rules", "Answer model must not treat document instructions as trusted"],
["ragshield/config.py", "Routing comments; env/Streamlit-secrets fallback for keys; pinned embedding revision; judge settings", "Documentation drift; cloud deploy; reproducibility"],
["scripts/run_evaluation.py", "Split/fold-local models only; benign leak closed; judge accounting; error-source breakdown; lineage grouping with per-fold leakage assertions; lineage-clean category-held-out; per-template-group detection; provenance", "Every number must come from a model that never saw its test data, including through a payload's other forms"],
["scripts/train_classifier.py", "Production model on all records; smoke test on a separate split; provenance", "Evaluation no longer depends on the saved model"],
["scripts/segmentation_experiment.py", "Replaces test_segment_detection.py; lineage grouping", "Evidence behind the scoring decision, reproducible"],
["app.py", "Fail-closed banner; sample-document loader; per-chunk band/layer/evidence/judge; auto-train if the model is missing; Evaluation tab shows grouping, lineage leakage status, per-group detection, lineage-clean column, staleness and judge-failure warnings", "Explainability; honest display; works on a fresh clone or cloud deploy"],
["tests/", "87 unit tests + 2 slow: routing, heuristics (incl. in-sample vs held-out AI-addressed phrasings), judge parsing/fail-closed and exact Groq kwargs vs SDK signature, prompt boundaries, benchmark integrity/reproducibility/lineage, metric arithmetic, split leakage", "Pin the routing table, security properties and evaluation validity"],
["requirements.txt, .python-version, README, .gitignore, docs/", "Exact version pins; README with both audits; models/ and results/ committed with hashes; report and screenshot scripts", "GitHub and deployment readiness"]]
story.append(table(changes, widths=[W * 0.2, W * 0.45, W * 0.35]))

# ------------------------------------------------------------------ 6 results
story += [PageBreak(), P("6. Results (all produced by the committed code on the committed benchmark v3)", H1),
P("No Groq key was available locally, so every committed run uses Layers 1+2 with the judge replaced by the 0.5 ablation cutoff and records llm_judge_used: false. "
  "The results files embed the benchmark's SHA-256; the app warns if it no longer matches."),
P(f"6.1 Group-held-out cross-validation, grouped by {group.get('group_by')} (headline)", H2),
P("Groups are attack lineages (all variants of one seed payload) plus benign document styles; a per-fold assertion guarantees no attack lineage is on both sides. Section 13.1 compares this with the older template-group split.", CAP)]
rows = [["Fold", "Train", "Test", "Groups held out", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "FPR"]]
for f in group["folds"]:
    rows.append([f["fold"], f["n_train"], f["n_test"], f["n_held_out_groups"], f["tp"], f["fp"], f["tn"], f["fn"],
                 f"{f['precision']:.3f}", f"{f['recall']:.3f}", f"{f['f1']:.3f}", f"{f['false_positive_rate']:.3f}"])
rows.append(["Aggregate", "", agg["n"], group["n_groups"], agg["tp"], agg["fp"], agg["tn"], agg["fn"], f"{agg['precision']:.3f}", f"{agg['recall']:.3f}", f"{agg['f1']:.3f}", f"{agg['false_positive_rate']:.3f}"])
story.append(table(rows))
story.append(P(f"Error sources: {eb.get('fn_ablation_cutoff_would_reach_judge', 0)} of the {agg['fn']} misses fell in the review band and would reach Layer 3 in production; "
               f"{eb.get('fn_layer_2_silent_pass', 0)} passed silently. All {agg['fp']} false positives were in the review band.", CAP))
pc = group["per_category"]
story.append(table([["Category", "n", "Rate", "Meaning"]] + [[c, v["n"], f"{v.get('detection_rate', v.get('correctly_passed_rate')):.3f}", "detected" if "detection_rate" in v else "correctly passed"] for c, v in pc.items()],
                   widths=[W * 0.35, W * 0.15, W * 0.2, W * 0.3]))
ptg = group.get("per_template_group", {})
story.append(table([["Attack template group", "n", "Detection (Layers 1+2 at 0.5)"]] + [[k, v["n"], f"{v['detection_rate']:.2f}"] for k, v in ptg.items()], widths=[W * 0.5, W * 0.2, W * 0.3]))
story.append(P("Detection by template group under lineage grouping. indirect_novel (ten handwritten payloads sharing no wording with any seed) is the honest unseen-payload number: "
               "the classifier detects 1 of 10; the ai_addressed_instruction rule escalates 9 of 10 to Layer 3.", CAP))

story.append(P("6.2 Ablation on the random 80/20 split", H2))
story.append(table([c.split(",") for c in ablation]))
story.append(P("Detection rate per attack category and correctly-passed rate for benign_hard_negative. Full Hybrid equals Heuristic+ML here because no judge was available. "
               "The random split is optimistic by construction and is kept only to give the ablation one common test set.", CAP))

story.append(P("6.3 Category-held-out (unseen attack family)", H2))
rows = [["Held-out family", "ML-only @0.5", "Heuristic+ML", "ML-only FPR", "Train attacks sharing a lineage", "Lineage-clean detection", "By template group"]]
for c, m in metrics["category_held_out"].items():
    lcd = (m.get("lineage_clean") or {}).get("detection_rate")
    rows.append([c, f"{m['unseen_family_detection_rate']:.2f}", f"{m['hybrid_no_llm_detection_rate']:.2f}", f"{m['false_positive_rate']:.2f}",
                 f"{m.get('n_train_attacks_sharing_lineage_with_held_out', '?')} / {m.get('n_train_attacks', '?')}",
                 "n/a" if lcd is None else f"{lcd:.2f}",
                 "; ".join(f"{k.replace(c + '_', '').replace('obfuscation_', '')}={v['detection_rate']:.2f}" for k, v in m.get("detection_by_template_group", {}).items())])
story.append(table(rows, widths=[W * 0.14, W * 0.08, W * 0.09, W * 0.08, W * 0.13, W * 0.11, W * 0.37]))
story.append(P("Trained on every other family plus only the benign records outside the shared benign test set. 'Lineage-clean' additionally removes every training record sharing a lineage with the "
               "held-out family; for indirect_injection only one attack record survives, which is itself the finding: with 21 seed lineages, 'unseen family' and 'unseen payload' were not the same thing.", CAP))

story.append(P("6.4 Real-document robustness", H2))
story.append(table([["Model / scoring", "Benign chunks SAFE without LLM", "Sent to judge", "Above 0.70", "Poisoned injected chunks flagged"],
                    ["Shipped (v1 model, max-sentence)", "0 / 34", "15", "19", "3 / 3"],
                    ["v2 model (max-sentence, benign sentence expansion)", "21 / 34", "13", "0", "3 / 3"],
                    ["Final (v3 model, trained with the novel payloads)", "17 / 34", "17", "0", "3 / 3"]],
                   widths=[W * 0.36, W * 0.18, W * 0.13, W * 0.13, W * 0.2]))
story.append(P("Six fictional benign documents never used in training. Under the final system no benign chunk is blocked by Layers 1+2, half pass with no LLM call, and every injected chunk of the poisoned FAQ "
               "is hard-blocked by Layer 1 or escalated. The v3 model sends four more benign chunks to the judge than v2 did. tests/test_real_documents.py pins this behaviour.", CAP))

# ------------------------------------------------------------------ 7 security
story += [PageBreak(), P("7. Security boundary audit", H1)]
sec = [["#", "Requirement", "Status", "How it is enforced / verified"],
["1–3", "Retrieved text cannot override the system prompt, developer instructions, or impersonate trusted instructions", "Done", "System turn states retrieved text is untrusted data; &lt;retrieved_context&gt; block with neutralised closing tags; test_generate_answer.py"],
["4–5", "Sanitised context contains only approved chunks; blocked chunks never reach generation", "Done", "sanitised_context() joins final_included only; test asserts the attack string is absent from the messages sent"],
["6–8", "Judge failures, malformed responses and API errors fail closed", "Done", "INJECTION with error class for API exceptions, None content, unparseable JSON, invalid verdict; real SDK connection failure exercised offline"],
["9", "Critical heuristic matches hard-block", "Done", "Routing row 1; zero critical fires on 88 benign records"],
["10–13", "Direct, indirect, role-manipulation and prompt-extraction attacks detected", "Verified without judge; honest gaps stated", "Lineage CV detection: direct 1.00, seed-derived indirect 1.00, role 0.95, extraction 0.93; novel indirect payloads 0.10 by ML, 9/10 escalated by Layer 1"],
["14–15", "Obfuscation and paraphrase tested", "Verified, limitation noted", "All 20 obfuscated records leave Layer 1 evidence; classifier alone 0.40–0.55; paraphrased 0.90"],
["16", "Legitimate instructional content not automatically blocked", "Done", "High ML score alone never blocks; 0 of 34 real benign chunks hard-blocked"],
["17", "User query and document content clearly separated", "Done", "Separate tagged blocks in both prompts"],
["18", "Answer generation cannot treat document instructions as trusted", "Done (best effort)", "Prompt boundary; documented as defence in depth, not a formal guarantee"]]
story.append(table(sec, widths=[W * 0.06, W * 0.34, W * 0.16, W * 0.44]))

# ------------------------------------------------------------------ 8 streamlit
story += [PageBreak(), P("8. Streamlit application walkthrough", H1),
P("Captured from the running app in a headless browser after all fixes (docs/capture_screenshots.py). They show exactly what a user sees without the Groq key: Layer 3 is unavailable, so the app says so and fails closed.")]
story += figure(SHOTS / "01_home.png", "Figure 1. Tab 1 on launch. The banner states that Layer 3 is unavailable and what that means (fail closed). Upload accepts .txt/.pdf; a sample-document loader is provided for demos.", 1.0)
story += figure(SHOTS / "02_loaded.png", "Figure 2. After 'Load sample documents': 39 chunks from 7 fictional documents indexed in FAISS (six benign, one poisoned FAQ).", 1.0)
story += figure(SHOTS / "03_poisoned_query.png", "Figure 3. Query that retrieves the poisoned FAQ. Two chunks are blocked by Layer 1 (critical rules), one by Layer 2 corroborated by a Layer 1 rule with no LLM call; each card shows band, deciding layer, evidence and highest-scoring sentences. The tail fragment of an injected sentence cut at a chunk boundary scores 0.37 and is escalated (it passed at 0.30 under the v2 model): the documented chunking limitation, now a near-miss rather than a miss.")
story += figure(SHOTS / "04_benign_query.png", "Figure 4. Benign banking query. Support-article chunks pass with no LLM call; the password-reset chunk (ambiguous band) is escalated and, with no key, fails closed with an explicit message; the poisoned chunk also retrieved is hard-blocked.")
story += figure(SHOTS / "05_evaluation.png", "Figure 5. Tab 2, RAGShield Evaluation: lineage-grouped cross-validation with leakage status, per-fold TP/FP/TN/FN, per-category and per-template-group detection (indirect_novel visible), random-split summary, ablation, category-held-out with the lineage-clean column, threshold sensitivity and error analysis.")

# ------------------------------------------------------------------ 9-12
story += [PageBreak(), P("9. Repository layout and how to run", H1),
P("<font face='Courier'>ragshield/</font> config.py · heuristics.py (Layer 1) · ml_classifier.py (Layer 2) · llm_judge.py (Layer 3) · firewall.py (routing) · rag_pipeline.py · generate_answer.py<br/>"
  "<font face='Courier'>scripts/</font> build_benchmark.py · train_classifier.py · run_evaluation.py · segmentation_experiment.py<br/>"
  "<font face='Courier'>data/benchmark/</font> benchmark.jsonl (v3) · benchmark_v2_203.jsonl · benchmark_v1_155.jsonl · benchmark_baseline_155.jsonl &nbsp; <font face='Courier'>data/sample_documents/</font> 7 fictional documents<br/>"
  "<font face='Courier'>results/</font> metrics.json · ablation_matrix.csv · group_held_out_metrics.json (+ _template_group.json) · segmentation_experiment.json &nbsp; <font face='Courier'>tests/</font> pytest suite &nbsp; "
  "<font face='Courier'>docs/</font> this report, build_report.py, capture_screenshots.py, screenshots/ &nbsp; <font face='Courier'>app.py</font> · README.md"),
P("Quick start", H2),
P("python -m venv .venv &amp;&amp; source .venv/bin/activate &nbsp;&nbsp;(Windows: .venv\\Scripts\\activate)<br/>pip install -r requirements.txt<br/>cp .env.example .env &nbsp;&nbsp;# add GROQ_API_KEY<br/>"
  "python scripts/train_classifier.py<br/>python scripts/run_evaluation.py<br/>python scripts/run_evaluation.py --group-held-out<br/>pytest &nbsp;&nbsp;&amp;&amp;&nbsp;&nbsp; pytest -m slow<br/>streamlit run app.py", CODE),
P("10. What still needs the API key", H1),
P("Live Groq behaviour was not exercised. The SDK was verified to accept the exact parameters used and a real connection failure was shown to produce a classified fail-closed verdict, "
  "but judge accuracy on the escalated chunks is unconfirmed. After adding the key, run the two evaluation commands above (about 20 and 100 judge calls). Quote only runs whose output shows "
  "<font face='Courier'>judge_failures: 0</font>; otherwise re-run after the rate limit recovers. Add the full-hybrid row to the README's group-held-out table from those outputs."),
P("11. Resume defensibility", H1),
P("<b>3-layer firewall</b>: intact and consistent across code, evaluation and documentation, with a routing table that can be explained in one sentence per row. "
  "<b>Adversarial benchmark</b>: direct, obfuscated (zero-width, base64, homoglyph), paraphrased, indirect (seed-derived and novel) and extraction attacks plus 88 hard benign negatives across nine styles; "
  "explicit attack lineage; deterministic rebuild; integrity tests. <b>Rule-based vs ML vs hybrid comparison</b>: four-condition ablation with precision, recall, F1 and FPR per attack type, "
  "category-held-out (naive and lineage-clean) and lineage-grouped cross-validation, with leakage closed and judge failures accounted for. The honest headline is the lineage-grouped table; "
  "the honest caveat is that the classifier does not generalise to genuinely new payloads on its own, and the design relies on Layer 1 escalation and Layer 3 for those."),
P("12. Next steps: adding the key, pushing to GitHub, deploying", H1),
P("12.1 Add the key locally and confirm Layer 3 works", H2),
P("cp .env.example .env &nbsp;&nbsp;# then edit: GROQ_API_KEY=gsk_...<br/>streamlit run app.py", CODE),
P("Load the sample documents and ask <i>How do I reset my online banking password?</i> The password-reset chunk that previously said 'Judge unavailable' must now show <i>Verdict: SAFE</i> with a confidence, "
  "and the Final Answer section must contain a generated answer. Then ask <i>How do I reset the thermostat to factory settings?</i> and confirm the poisoned chunks are still blocked. "
  "'Judge call failed (rate_limited)' means quota, not a bug; wait and retry."),
P("12.2 Run the LLM-backed evaluation and update the README", H2),
P("python scripts/run_evaluation.py<br/>python scripts/run_evaluation.py --group-held-out", CODE),
P("12.3 Pre-push checklist", H2),
table([["Check", "Command", "Expected"],
       ["Tests pass", "pytest &amp;&amp; pytest -m slow", "87 passed; 2 passed"],
       ["No secrets in the tree", "grep -r \"gsk_\" --include=*.py --include=*.md --include=*.toml .", "no output"],
       ["Ignored files really ignored", "git status --ignored", ".env, .venv/, data/documents/ listed under ignored"],
       ["Results match benchmark", "open Evaluation tab", "no 'hash mismatch' warning"]],
      widths=[W * 0.25, W * 0.45, W * 0.3]),
P("12.4 Push to GitHub", H2),
P("cd rag-shield<br/>git init -b main<br/>git add .<br/>git commit -m \"RAGShield: 3-layer RAG prompt-injection firewall with reproducible evaluation\"<br/>"
  "git remote add origin https://github.com/&lt;your-user&gt;/rag-shield.git<br/>git push -u origin main", CODE),
P("models/ (about 5 KB) and results/ are committed deliberately so a fresh clone and the cloud deploy work immediately; both carry the benchmark hash. .env, .venv/ and uploaded documents are ignored."),
P("12.5 Deploy on Streamlit Community Cloud", H2),
table([["Step", "What to do"],
       ["1", "Sign in at share.streamlit.io with GitHub and choose <b>New app</b>."],
       ["2", "Repository: your rag-shield repo · Branch: main · Main file path: app.py."],
       ["3", "<b>Advanced settings</b> → Python version 3.12 (matches .python-version and the pinned wheels)."],
       ["4", "<b>Advanced settings → Secrets</b>: paste <font face='Courier'>GROQ_API_KEY = \"gsk_...\"</font> (TOML). config.py reads the environment first and Streamlit secrets second."],
       ["5", "Deploy. First start takes about a minute (downloads the pinned MiniLM revision). If models/ were missing the app trains the classifier itself on first start (verified)."],
       ["6", "Verify with the two demo questions from 12.1, then share the URL (Settings → Sharing)."]],
      widths=[W * 0.08, W * 0.92]),
P("Operational notes: uploaded documents live on the container disk and vanish on restart (intended single-session design); the free tier sleeps after inactivity; Groq free-tier rate limits are per minute, "
  "so bursts can produce fail-closed 'rate_limited' cards, which the UI labels explicitly.", BODY),
]

# ------------------------------------------------------------------ 13 second audit
story += [PageBreak(), P("13. Second audit: lineage leakage, Groq configuration, reproducibility, GO/NO-GO", H1),
P("13.1 Attack lineage leakage (the most important finding)", H2),
P("The benchmark builder was traced end to end. Every attack in v1/v2 descends from one of 21 handwritten seed sentences: obfuscated rows are deterministic transforms of a seed (zero-width insertion, "
  "base64 wrapping, homoglyph substitution), indirect rows embed a seed verbatim in one of four benign wrappers, and the LLM-generated same-style and paraphrased rows were produced from a seed. "
  "Recovering lineage: 61 rows invert exactly (21 seeds, 20 wrappers, 20 de-obfuscations); the 54 LLM-generated and curated rows were mapped to their nearest seed by MiniLM cosine similarity "
  "(0.35 to 0.84) and the mapping frozen in the builder. 20 of the 21 lineages span more than one category."),
P("Consequence: the previous category:template_group grouping let a payload sit in training as a plain seed while its wrapped form sat in test. Under that grouping 13 to 15 attack lineages leaked "
  "into training in every fold, and category-held-out for indirect_injection trained on 94 of 95 attack records sharing a lineage with the held-out set. The reported indirect detection of 1.00 "
  "was payload memorisation, exactly as suspected."),
P("Changes: (1) <font face='Courier'>lineage_id</font> on every record (benign rows use benign:&lt;template_group&gt;); (2) group-held-out groups on lineage by default, with a per-fold assertion that no attack lineage "
  "appears on both sides, and <font face='Courier'>--group-by template_group</font> retained for comparison; (3) a lineage-clean variant of category-held-out; (4) ten novel indirect payloads with new intents "
  "(goal hijacking, output manipulation, exfiltration, language lock-in) that share no wording with any seed, each its own lineage. Nothing was removed; v2 is frozen as benchmark_v2_203.jsonl."),
table([["Group-held-out (Layers 1+2)", "precision", "recall", "F1", "FPR", "TP / FP / TN / FN", "lineage leakage"],
       ["lineage grouping, v3 (current headline)", f"{agg['precision']:.3f}", f"{agg['recall']:.3f}", f"{agg['f1']:.3f}", f"{agg['false_positive_rate']:.3f}", f"{agg['tp']} / {agg['fp']} / {agg['tn']} / {agg['fn']}", "none (asserted)"],
       ["template_group grouping, v3", f"{tg['precision']:.3f}", f"{tg['recall']:.3f}", f"{tg['f1']:.3f}", f"{tg['false_positive_rate']:.3f}", f"{tg['tp']} / {tg['fp']} / {tg['tn']} / {tg['fn']}", "13 to 15 lineages per fold"],
       ["template_group grouping, v2 (first report)", "0.962", "0.887", "0.923", "0.045", "102 / 4 / 84 / 13", "present"]],
      widths=[W * 0.3, W * 0.09, W * 0.09, W * 0.09, W * 0.09, W * 0.16, W * 0.18]),
P("Honest reading: the leakage-free recall is 0.848, not 0.887; the novel indirect payloads are detected by the classifier at 0.10; lineage-clean category-held-out ranges from 0.00 (obfuscated) "
  "to 0.70 (direct_override). The classifier recognises payloads it has seen in another form far better than injection intent in general, which is why it is one layer of three.", CAP),
P("13.2 Closing the gap the novel payloads exposed", H2),
P("In leakage-free CV, 7 of the 10 novel payloads scored below LOW_THRESHOLD with no Layer 1 evidence: they passed silently and Layer 3 never saw them. Their shared marker is text that addresses "
  "the AI reading the document ('note to any AI assistant summarizing this listing', 'if you are a language model', 'support assistant, when you respond', 'whoever processes these minutes "
  "automatically should'). A non-critical Layer 1 rule, ai_addressed_instruction, was added for that pattern. Method, in order: five extra payloads were written before the rule was designed; the "
  "rule was designed on the ten benchmark payloads only; it was checked against all 88 benign records and 34 sample-document chunks (zero matches) and against every other attack (zero matches). "
  "Result: 9 of 10 in-sample payloads and 3 of 5 held-out phrasings are escalated; the held-out five live in the test suite, not the benchmark. Silent passes in leakage-free CV went from 7 to 0; "
  "aggregate precision/recall/F1 are unchanged because the ablation's 0.5 cutoff still counts a review-band record as a miss. The rule never blocks on its own."),
P("13.3 Groq request configuration", H2),
P("Checked against console.groq.com/docs/reasoning and /docs/structured-outputs and against the installed SDK signature (groq 0.37.1). For gpt-oss models Groq accepts reasoning_effort in "
  "{low, medium, high} and include_reasoning, and does NOT support reasoning_format; reasoning is returned in message.reasoning, never inside content. JSON object mode is available on all models "
  "and requires the prompt to ask for JSON, which the system prompt does. The judge sends temperature 0, max_completion_tokens 600 (not the older max_tokens), response_format json_object, "
  "reasoning_effort low and include_reasoning false for gpt-oss; reasoning_format hidden for qwen/minimax/deepseek families (raw + JSON mode is a 400); nothing extra for non-reasoning models. "
  "include_reasoning and reasoning_format are never sent together. tests/test_llm_judge.py pins the exact kwargs, checks every kwarg against the installed SDK signature, and exercises the full "
  "request/response path with a mocked client. Live calls were not made."),
P("13.4 Reproducibility", H2),
P("requirements.txt pins the exact versions the committed classifier and results were produced with: Python 3.12.2 (.python-version), scikit-learn 1.9.1, sentence-transformers 5.7.0, transformers 5.17.0, "
  "torch 2.14.0, numpy 2.5.3, faiss-cpu 1.15.0, groq 0.37.1, streamlit 1.63.0, pandas 3.0.5, joblib 1.6.0, pypdf 6.18.0. The embedding model is pinned to Hugging Face revision "
  "1110a243fdf4706b3f48f1d95db1a4f5529b4d41 in config.py and used by both the classifier and the retriever. The classifier sidecar records Python, scikit-learn, sentence-transformers, torch and numpy "
  "versions, the embedding revision, training-unit counts and the benchmark SHA-256; load_classifier() warns when the running environment differs. The model was retrained and every evaluation rerun "
  "after the pins and the benchmark change."),
P("13.5 Verification checklist", H2),
table([["Item", "How verified", "Result"],
       ["Benchmark counts, duplicates, empty records", "tests/test_benchmark.py; --from-baseline byte-for-byte rebuild", "213 records, counts match targets, 0 duplicates, 0 empty"],
       ["Train/test leakage (benign)", "category-held-out excludes shared benign test rows; split-local models only", "closed"],
       ["Attack lineage leakage", "lineage_id on every record; per-fold assertion; tests/test_evaluation.py", "0 lineages leak under lineage grouping; 13–15 under old grouping (documented)"],
       ["Group / category-held-out splits", "StratifiedGroupKFold(5, shuffle, seed 42); every fold has benign rows", "verified"],
       ["Threshold selection", "0.30 / 0.70 unchanged since v1; 0.5 is a measurement cutoff only", "no tuning on held-out data"],
       ["Metric arithmetic", "test_compute_metrics_arithmetic (hand-computed confusion matrix)", "correct"],
       ["Production vs evaluation routing", "both call firewall.screen_chunks; ABLATION_NO_JUDGE sentinel for no-LLM", "single implementation"],
       ["Layer 1/2/3 behaviour", "routing tests with fakes; heuristics tests incl. obfuscation and AI-addressed rule", "87 unit tests pass"],
       ["Judge failure handling", "mocked rate limit / empty / malformed / invalid verdict; real SDK connection failure offline", "fails closed with labelled error"],
       ["Prompt boundary / context sanitisation", "system turn, tagged blocks, neutralised closing tags; blocked text absent from answer messages", "tests pass"],
       ["Streamlit behaviour", "driven headlessly: load samples, poisoned query, benign query, evaluation tab; auto-train when model missing", "no errors"],
       ["Provenance / benchmark hash", "model sidecar and results carry SHA-256; app warns on mismatch", "all match current benchmark"],
       ["Groq request + JSON parsing", "kwargs pinned and checked against SDK signature; parser tests", "pass; live call not possible"],
       ["Dependency compatibility", "exact pins; embedding revision pin; environment recorded and checked on load", "pinned; model retrained under the pinned stack"]],
      widths=[W * 0.24, W * 0.44, W * 0.32]),
P("13.6 Verdict: GO, with two stated conditions", H2),
P("<b>GO</b> for publishing the repository and deploying the app. The architecture survived the audit: the problems found were in the benchmark's construction and the evaluation's split, not in "
  "the three-layer design, and they are now measured honestly. Two conditions remain and are not hidden: (1) Layer 3 has still not been exercised live; add the key, run the two evaluation commands, "
  "and quote only runs with judge_failures = 0. (2) The classifier's generalisation to genuinely new payloads is weak (1 of 10 novel indirect payloads; lineage-clean 0.00 to 0.70); the design covers "
  "this with Layer 1 escalation and Layer 3 review, and the README says so in plain terms. Do not describe the classifier as detecting unseen attacks on its own."),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"RAGShield report · page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                        title="RAGShield – debugging, audit and finalisation report", author="Claude (with Sparsh Sharma)")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
import shutil  # noqa: E402

shutil.rmtree(SLICES, ignore_errors=True)
print(OUT, OUT.stat().st_size // 1024, "KB")
