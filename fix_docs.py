"""Run once from the project root:  python fix_docs.py
Patches docs/build_report.py (Figure 3 caption, security-table row) and
docs/capture_screenshots.py (stuck tooltip). Safe to re-run."""
from pathlib import Path

def patch(path, edits):
    p = Path(path)
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    for label, old, new in edits:
        if new in s:
            print(f"[skip] {label}: already applied")
        elif old in s:
            s = s.replace(old, new, 1)
            print(f"[done] {label}")
        else:
            print(f"[MISS] {label}: text not found, send this line to Claude")
    if crlf:
        s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8"))

patch("docs/capture_screenshots.py", [(
    "tooltip fix",
    '    """Streamlit scrolls inside an inner container, so size the viewport to the content."""\n',
    '    """Streamlit scrolls inside an inner container, so size the viewport to the content."""\n'
    '    page.evaluate("() => document.activeElement && document.activeElement.blur()")\n'
    '    page.mouse.move(1, 1)\n'
    '    page.keyboard.press("Escape")\n'
    '    time.sleep(0.8)\n',
)])

patch("docs/build_report.py", [
    ("Figure 3 caption",
     "The tail fragment of an injected sentence cut at a chunk boundary scores 0.37 and is escalated (it passed at 0.30 under the v2 model): the documented chunking limitation, now a near-miss rather than a miss.",
     "The tail fragment of an injected sentence cut at a chunk boundary scores 0.37 and is escalated to the live judge, which returns SAFE (confidence 0.99), so it stays in the context; the final answer is unaffected, but this is the documented chunking limitation in practice: a fragment of a split attack can pass Layer 3 on its own."),
    ("security table rows 10-13",
     "Lineage CV detection: direct 1.00, seed-derived indirect 1.00, role 0.95, extraction 0.93; novel indirect payloads 0.10 by ML, 9/10 escalated by Layer 1",
     "Lineage CV, full hybrid with live judge: direct 1.00, role 1.00, extraction 1.00, indirect 0.97; novel indirect payloads 0.10 by the classifier alone, 0.90 with Layer 1 escalation and the live judge"),
])