"""Drive the running web app with Edge (Playwright), exercise the real UI and save
screenshots to docs/screenshots/. Requires the server on http://localhost:8420.

Usage: .venv/Scripts/python.exe scripts/20_ui_screenshots.py [--dark]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8420/"
dark = "--dark" in sys.argv


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        ctx = b.new_context(viewport={"width": 1360, "height": 900}, color_scheme="dark" if dark else "light")
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error",) else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(URL, wait_until="load", timeout=90000)
        page.wait_for_timeout(3500)
        tag = "dark_" if dark else ""
        page.screenshot(path=str(OUT / f"{tag}01_home.png"))

        # --- live demo: run a Hinglish query through the real UI
        page.locator("#textInput").fill("gehu mein peela rust aa gaya hai kya spray karun")
        page.locator("#submitBtn").click()
        page.wait_for_selector("#decisionCard:not([hidden])", timeout=60000)
        page.wait_for_timeout(800)
        page.locator("#demo").scroll_into_view_if_needed()
        page.locator("#demo").screenshot(path=str(OUT / f"{tag}02_demo_escalated.png"))

        # --- an auto-answered query
        page.locator("#textInput").fill("which tomato variety is good for me?")
        page.locator("#submitBtn").click()
        page.wait_for_timeout(2500)
        page.locator("#demo").screenshot(path=str(OUT / f"{tag}03_demo_answered.png"))

        # --- advisor console: select first pending item
        page.wait_for_timeout(500)
        first = page.locator("#queueList .queue-item").first
        if first.count():
            first.click()
            page.wait_for_timeout(500)
        page.locator("#advisor").screenshot(path=str(OUT / f"{tag}04_advisor.png"))

        for sec, name in [("operations", "05_operations"), ("evaluation", "06_evaluation"), ("system", "07_system"), ("about", "08_about")]:
            page.locator(f"#{sec}").scroll_into_view_if_needed()
            page.wait_for_timeout(900)
            page.locator(f"#{sec}").screenshot(path=str(OUT / f"{tag}{name}.png"))

        # --- mobile
        m = b.new_context(viewport={"width": 390, "height": 844}, color_scheme="dark" if dark else "light", is_mobile=True).new_page()
        m.goto(URL, wait_until="load", timeout=90000)
        m.wait_for_timeout(2500)
        m.screenshot(path=str(OUT / f"{tag}09_mobile_home.png"))
        overflow = m.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        print("mobile horizontal overflow (px):", overflow)
        b.close()
    print("JS errors:", errors if errors else "none")
    print("saved:", sorted(p.name for p in OUT.glob(f"{tag}*.png")))


if __name__ == "__main__":
    main()
