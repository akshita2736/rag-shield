"""
docs/capture_screenshots.py

Drives the running Streamlit app (streamlit run app.py, port 8501) headlessly
and saves full-page screenshots to docs/screenshots/. Optional tooling, not a
runtime dependency: pip install playwright && playwright install chromium
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"

JS_SCROLL_H = """() => {
  let best = document.documentElement.scrollHeight;
  for (const el of document.querySelectorAll('section, div')) {
    if (el.scrollHeight > el.clientHeight + 5 && el.scrollHeight > best) best = el.scrollHeight;
  }
  return best;
}"""


def wait_idle(page, secs=2.5):
    time.sleep(secs)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=60000)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)


def capture(page, name):
    """Streamlit scrolls inside an inner container, so size the viewport to the content."""
    page.set_viewport_size({"width": 1400, "height": 1000})
    time.sleep(0.8)
    h = int(min(max(page.evaluate(JS_SCROLL_H), 1000), 9000)) + 60
    page.set_viewport_size({"width": 1400, "height": h})
    time.sleep(1.5)
    page.screenshot(path=OUT / name)
    page.set_viewport_size({"width": 1400, "height": 1000})
    time.sleep(0.5)


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
        page.goto(URL)
        wait_idle(page, 6)
        capture(page, "01_home.png")

        page.get_by_role("button", name="Load sample documents").click()
        wait_idle(page, 5)
        capture(page, "02_loaded.png")

        def ask(question, name):
            box = page.get_by_label("Your question")
            box.fill(question)
            box.press("Enter")
            wait_idle(page, 1)
            page.get_by_role("button", name="Ask", exact=True).click()
            wait_idle(page, 8)
            for exp in page.get_by_text("View chunk text").all():
                try:
                    exp.click()
                    time.sleep(0.4)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1.5)
            capture(page, name)

        ask("How do I reset the thermostat to factory settings?", "03_poisoned_query.png")
        ask("How do I reset my online banking password?", "04_benign_query.png")

        page.get_by_role("tab", name="RAGShield Evaluation").click()
        wait_idle(page, 3)
        for exp in page.get_by_text("Per-fold held-out groups and error breakdown").all():
            try:
                exp.click()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(1.5)
        capture(page, "05_evaluation.png")
        browser.close()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
