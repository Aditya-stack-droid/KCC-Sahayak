"""End-to-end *voice* evaluation: text -> speech (Windows SAPI) -> Whisper -> intent.

Measures the part of the "voice-first" claim that was never tested: how much
does speech recognition degrade the pipeline? Uses the 50 English gold queries.

Honest caveats (also written to the output JSON):
  * the speech is synthetic (clean, studio-quality TTS), so real phone-line farmer
    speech will be harder -- these numbers are an UPPER bound on voice performance;
  * only English voices are available on this machine, so Hindi/Hinglish speech
    could not be tested.

Outputs: results/voice_eval.json, results/voice_eval_rows.csv
Usage:   .venv/Scripts/python.exe scripts/15_voice_eval.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
AUDIO = ROOT / "demo" / "voice_eval"

FFMPEG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"


def find_ffmpeg_dir():
    for p in FFMPEG_DIR.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"):
        return str(p.parent)
    return None


def norm(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(1, len(r))


def synth(rows):
    AUDIO.mkdir(parents=True, exist_ok=True)
    items = [{"id": r.id, "text": r.text} for r in rows.itertuples()]
    spec = AUDIO / "spec.json"
    spec.write_text(json.dumps(items), encoding="utf-8")
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$items = Get-Content -Raw -Encoding UTF8 '{spec}' | ConvertFrom-Json
$voices = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }}
Write-Host ('voices: ' + ($voices -join ', '))
$i = 0
foreach ($it in $items) {{
    $s.SelectVoice($voices[$i % $voices.Count]); $i++
    $s.SetOutputToWaveFile((Join-Path '{AUDIO}' ($it.id + '.wav')))
    $s.Speak($it.text)
}}
$s.Dispose()
"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(out.stdout.strip())
    if out.returncode != 0:
        raise RuntimeError(out.stderr)


def main():
    ff = find_ffmpeg_dir()
    if ff:
        os.environ["PATH"] += os.pathsep + ff
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")
    en = gold[gold["style"] == "en"].reset_index(drop=True)
    if all((AUDIO / f"{i}.wav").exists() for i in en["id"]):
        print("reusing existing synthetic audio")
    else:
        print(f"synthesising {len(en)} English queries ...")
        synth(en)

    from asr import transcribe
    from pipeline import KCCSahayakPipeline

    pipe = KCCSahayakPipeline()
    rows = []
    for r in en.itertuples():
        wav = AUDIO / f"{r.id}.wav"
        base = transcribe(str(wav), use_domain_prompt=False)
        asr = transcribe(str(wav), use_domain_prompt=True)
        t_label, t_conf, _ = pipe.classify_intent(r.text)
        a_label, a_conf, _ = pipe.classify_intent(asr["text"])
        b_label, _, _ = pipe.classify_intent(base["text"])
        rows.append({"id": r.id, "label": r.label, "reference": r.text, "transcript": asr["text"],
                     "transcript_no_prompt": base["text"],
                     "wer": wer(r.text, asr["text"]), "wer_no_prompt": wer(r.text, base["text"]),
                     "pred_text": t_label, "pred_asr": a_label, "pred_asr_no_prompt": b_label,
                     "conf_text": t_conf, "conf_asr": a_conf})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "voice_eval_rows.csv", index=False, encoding="utf-8")
    out = {
        "n": int(len(df)),
        "mean_wer": round(float(df["wer"].mean()), 4),
        "median_wer": round(float(df["wer"].median()), 4),
        "intent_accuracy_typed_text": round(float((df["pred_text"] == df["label"]).mean()), 4),
        "intent_accuracy_after_asr": round(float((df["pred_asr"] == df["label"]).mean()), 4),
        "prediction_unchanged_by_asr": round(float((df["pred_text"] == df["pred_asr"]).mean()), 4),
        "baseline_no_vocab_hint": {
            "mean_wer": round(float(df["wer_no_prompt"].mean()), 4),
            "intent_accuracy_after_asr": round(float((df["pred_asr_no_prompt"] == df["label"]).mean()), 4),
        },
        "domain_terms_fixed": int(((df["wer_no_prompt"] > df["wer"])).sum()),
        "domain_terms_broken": int(((df["wer_no_prompt"] < df["wer"])).sum()),
        "asr_model": "openai-whisper small (with agricultural vocabulary hint)",
        "caveats": [
            "speech is synthetic Windows TTS (clean audio) -- an upper bound on real phone-line farmer speech",
            "English voices only; Hindi / Hinglish speech was not tested",
        ],
    }
    (RES / "voice_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
