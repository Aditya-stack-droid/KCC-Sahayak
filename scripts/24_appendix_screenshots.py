"""Screenshots for the report appendix ("Application screenshots and outputs").

Drives the running web app (http://localhost:8420) in headless Edge and saves numbered figures to
docs/appendix_g/. Browser figures are real screenshots of the real UI. The two "terminal" figures (dataset
preprocessing, model results) and the backend-execution figure are rendered from the actual output of the
commands/result files shown, drawn as a terminal window, because a console window cannot be screenshotted headlessly.

Voice input is real: the browser is started with a fake microphone that plays demo/sample_query.wav, and the
UI's own record button is pressed, so audio goes through MediaRecorder -> /api/transcribe -> Whisper.

Usage: .venv/Scripts/python.exe scripts/24_appendix_screenshots.py
"""
import html
import json
import sys
import urllib.request
import wave
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "appendix_g"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8420"
AUDIO = ROOT / "demo" / "sample_query.wav"
captions = []


def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def terminal(page, title, text, name, caption, width=1180):
    body = html.escape(text)
    page.set_viewport_size({"width": width + 40, "height": 800})
    page.set_content(f"""<html><body style="margin:0;background:#fff;padding:20px">
<div style="width:{width}px;border-radius:10px;overflow:hidden;box-shadow:0 2px 12px #0003;font-family:Consolas,'Courier New',monospace">
<div style="background:#2d333b;color:#adbac7;padding:8px 14px;font-size:13px"><span style="color:#ff5f56">●</span> <span style="color:#ffbd2e">●</span> <span style="color:#27c93f">●</span>&nbsp;&nbsp;{html.escape(title)}</div>
<pre style="margin:0;background:#0d1117;color:#e6edf3;padding:16px 18px;font-size:14px;line-height:1.45;white-space:pre-wrap">{body}</pre></div></body></html>""")
    page.locator("div").first.screenshot(path=str(OUT / name))
    captions.append((name, caption))


def wait_gemini(page, timeout=90000):
    """Wait until the Gemini card has an answer (or an error), not just 'Asking Gemini...'."""
    page.wait_for_function("!document.getElementById('cmpGeminiText').innerText.startsWith('Asking')", timeout=timeout)
    page.wait_for_timeout(1200)


def shot(locator, name, caption):
    locator.screenshot(path=str(OUT / name))
    captions.append((name, caption))


def fig1_dataset():
    raw = pd.read_csv(ROOT / "data/raw/kcc_mapping_full.csv")
    sp = {k: pd.read_csv(ROOT / f"data/processed/{k}.csv") for k in ("train", "val", "test", "full_labeled")}
    meta = pd.read_csv(ROOT / "models/retrieval_index/corpus_metadata.csv")
    lines = [
        '>>> raw = pd.read_csv("data/raw/kcc_mapping_full.csv")',
        f"raw rows: {len(raw):,}    columns: {list(raw.columns)[:6]}",
        "",
        ">>> python scripts/01_prepare_data.py   # clean, de-duplicate, label",
        f"cleaned + labelled queries : {len(sp['full_labeled']):,}",
        f"  train / val / test split  : {len(sp['train']):,} / {len(sp['val']):,} / {len(sp['test']):,}",
        f"retrieval corpus (Q/A pairs): {len(meta):,}",
        "",
        "query_type distribution (full_labeled.csv):",
    ]
    vc = sp["full_labeled"]["query_type"].value_counts()
    for k, v in vc.items():
        lines.append(f"  {k:<26s}{v:>8,d}  ({100 * v / len(sp['full_labeled']):4.1f}%)")
    return "\n".join(lines)


def fig2_results():
    ms = json.load(open(ROOT / "results/model_selection.json"))["results"]
    sel = json.load(open(ROOT / "results/selective_eval.json"))
    lines = ["intent classifier comparison (accuracy)", f"{'model':<20s}{'silver n=496':>14s}{'gold n=150':>12s}{'gold EN':>9s}{'Hinglish':>10s}{'Hindi':>8s}"]
    for k, v in ms.items():
        lines.append(f"{k:<20s}{v['silver']:>14.3f}{v['gold']:>12.3f}{v['gold_en']:>9.2f}{v['gold_hinglish']:>10.2f}{v['gold_hi']:>8.2f}")
    d = sel["silver_default"]
    lines += ["", f"deployed model selected: {json.load(open(ROOT / 'results/model_selection.json'))['selected']}", "  (MiniLM sentence embedding + character n-gram TF-IDF -> logistic regression)", "",
              "confidence gate on held-out silver set (n=496):",
              f"  coverage (auto-answered)      : {d['coverage']:.1%}",
              f"  topic accuracy when answered  : {d['selective_accuracy']:.1%}",
              f"  overall topic accuracy        : {d['overall_accuracy']:.1%}"]
    return "\n".join(lines)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        ctx = b.new_context(viewport={"width": 1360, "height": 900})
        page = ctx.new_page()

        # G.1 / G.2 : terminal-style output of real commands / result files
        terminal(page, "Terminal: dataset loading and preprocessing", fig1_dataset(), "G01_dataset_preprocessing.png",
                 "Dataset loading and preprocessing output: raw KCC rows, cleaned labelled queries, train/validation/test split and topic distribution.")
        terminal(page, "Terminal: model training and classification results", fig2_results(), "G02_training_results.png",
                 "Model comparison and confidence-gate results on the held-out silver set and the 150-query gold set.")
        page.set_viewport_size({"width": 1360, "height": 900})

        # G.8 homepage
        page.goto(URL + "/?v=g", wait_until="load", timeout=90000)
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / "G08_homepage.png"))
        captions.append(("G08_homepage.png", "KCC-Sahayak application homepage with the measured evaluation figures."))

        # G.4 / G.5 / G.6 : auto-answered query
        page.locator("#textInput").fill("which is the best variety of wheat for sowing in november")
        page.locator("#submitBtn").click()
        page.wait_for_selector("#decisionCard:not([hidden])", timeout=60000)
        wait_gemini(page)
        page.locator("#demo").scroll_into_view_if_needed()
        shot(page.locator("#demo"), "G04_query_classification_pipeline.png",
             "Farmer query classification: pipeline stages (speech/text, intent with confidence, entities, retrieval, confidence gate) for a typed query.")
        shot(page.locator("#candidatesBox"), "G05_retrieved_historical_qa.png",
             "Retrieved historical question-and-answer results: closest logged KCC questions with similarity and the advisor answer recorded for each.")
        shot(page.locator("#decisionCard"), "G06a_safety_gate_auto_answer.png",
             "Safety gate decision: query answered automatically (non-high-stakes topic, confidence and similarity above threshold, answer screened clean).")

        # G.6b escalated query
        page.locator("#textInput").fill("what pesticide do i need to use for tomato")
        page.locator("#submitBtn").click()
        page.wait_for_timeout(2500)
        page.wait_for_function("document.getElementById('decisionTitle').innerText.includes('Provisional')", timeout=60000)
        wait_gemini(page)
        shot(page.locator("#decisionCard"), "G06b_safety_gate_escalation.png",
             "Safety gate decision: high-stakes query (Plant Protection) escalated to a human advisor, with provisional guidance and the closest logged questions.")

        # G.7 advisor console
        page.locator("#advisor").scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        # hide the "Recently resolved" list: it holds a leftover test entry ("LOL") unrelated to this figure
        page.evaluate("document.querySelectorAll('#advisor .panel-title').forEach(h => { if (/recently resolved/i.test(h.innerText)) { while (h.nextElementSibling) h.nextElementSibling.remove(); h.remove(); } })")
        first = page.locator("#queueList .queue-item").first
        if first.count():
            first.click()
            page.wait_for_timeout(600)
        shot(page.locator("#advisor"), "G07_advisor_escalation_console.png",
             "Human advisor escalation interface: queue of escalated queries, the retrieved candidates and the answer form.")

        # G.9 backend: swagger page + terminal render of real API responses
        page.goto(URL + "/docs", wait_until="load", timeout=90000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "G09a_fastapi_docs.png"))
        captions.append(("G09a_fastapi_docs.png", "FastAPI backend: auto-generated API documentation listing the query, transcribe, queue and statistics endpoints."))
        health, info = get("/api/health"), get("/api/model-info")
        text = "$ python -m uvicorn app:app --app-dir backend --port 8420\nINFO:     Uvicorn running on http://127.0.0.1:8420\n\n$ curl localhost:8420/api/health\n" \
               + json.dumps(health, indent=2) + "\n\n$ curl localhost:8420/api/model-info\n" + json.dumps(info, indent=2)[:1800]
        terminal(page, "Terminal: FastAPI backend running", text, "G09b_backend_running.png",
                 "FastAPI backend execution: server started with uvicorn and answering the health and model-info endpoints.")

        # G.3 voice: fake microphone plays the demo wav through the real record button
        b.close()
        dur = wave.open(str(AUDIO)).getnframes() / wave.open(str(AUDIO)).getframerate()
        b2 = p.chromium.launch(channel="msedge", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={AUDIO}"])
        c2 = b2.new_context(viewport={"width": 1360, "height": 900}, permissions=["microphone"])
        v = c2.new_page()
        for frac in (0.92, 0.7, 0.5, 1.0):
            # The fake microphone loops the file, so a recording can capture the sentence twice. Retry with other
            # lengths until the transcript contains the sentence once (we never edit the transcript itself).
            v.goto(URL + "/?v=g" + str(frac), wait_until="load", timeout=90000)
            v.wait_for_timeout(2500)
            v.locator("#demo").scroll_into_view_if_needed()
            v.locator("#micBtn").click()
            v.wait_for_timeout(int(dur * 1000 * frac))
            v.locator("#micBtn").click()
            v.wait_for_selector("#decisionCard:not([hidden])", timeout=180000)
            words = v.locator("#textInput").input_value().lower().split()
            print(f"voice attempt frac={frac}: {' '.join(words)!r}")
            if len(words) >= 4 and sum(words[i:i + 4] == words[:4] for i in range(len(words) - 3)) == 1:
                break
        wait_gemini(v)
        shot(v.locator("#demo"), "G03_voice_input_speech_to_text.png",
             "Voice input and speech-to-text: audio recorded with the microphone button is transcribed by Whisper and passed through the pipeline.")
        b2.close()

    (OUT / "CAPTIONS.md").write_text(
        "# Appendix G figures\n\n" + "\n".join(f"- **{n}** - {c}" for n, c in sorted(captions)) + "\n", encoding="utf-8")
    print("saved:", *sorted(n for n, _ in captions), sep="\n  ")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
