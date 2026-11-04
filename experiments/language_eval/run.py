#!/usr/bin/env python
"""Language-specific capability report (English / Hinglish / Hindi).

Reuses the committed 150-query gold set (50 per register) and the shipped
components; no model is trained or re-tuned here. Reports, per language:

* intent accuracy on the gold queries (from ``results/research_eval.json``,
  measured elsewhere and quoted with its source);
* register-detection accuracy (``src/lang.py``);
* entity-coverage (share of queries where ``src/ner.py`` finds any crop/pest);
* dosage/risk detection recall vs the new screener on authored per-language
  dosage strings (``answer_screening``).

It is a **capability** report, not a fairness benchmark: the gold set is small
(n=50 per register) and LLM-authored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

from lang import detect_register  # noqa: E402
from ner import extract_entities  # noqa: E402

try:
    from confidence_gate import answer_risk as _v1
except Exception:  # pragma: no cover
    from src.confidence_gate import answer_risk as _v1  # type: ignore
from answer_screening.screening import risk_reason as _v2  # noqa: E402

GOLD = ROOT / "data" / "gold" / "gold_queries.csv"
RESEARCH = ROOT / "results" / "research_eval.json"
DEFAULT_OUT = ROOT / "results" / "language_eval"

# Per-language dosage strings (authored) for the screener comparison.
DOSAGE_PROBES = {
    "en": ["Spray 2 ml monocrotophos per litre.", "Apply 50 kg urea per acre."],
    "hinglish": ["2 ml rogor per litre pani me spray karein.", "Ek acre me 50 kg urea dalen."],
    "roman_hindi": ["dawa 2 mili per litre pani me dalen.", "Dawa ki 2 boond per litre pani me dalen."],
    "devanagari": ["कीटनाशक दवा 2 मिली प्रति लीटर पानी में डालें।", "खेत में २ किलो यूरिया डालें।"],
}


def _rate(num, den):
    return round(num / den, 4) if den else None


def build() -> dict:
    gold = pd.read_csv(GOLD)
    research = json.loads(RESEARCH.read_text(encoding="utf-8")) if RESEARCH.exists() else {}
    prod = (research.get("main") or {}).get("production (emb+char, aug)", {})

    registers = {}
    for style, frame in gold.groupby("style"):
        texts = frame["text"].astype(str).tolist()
        reg_ok = sum(1 for t in texts if detect_register(t) == style)
        entity_ok = sum(1 for t in texts if any(extract_entities(t)[k] for k in ("crop", "symptom")))
        registers[style] = {
            "n": len(texts),
            "register_detection_accuracy": _rate(reg_ok, len(texts)),
            "entity_coverage": _rate(entity_ok, len(texts)),
        }

    intent = {
        "en": (prod.get("gold_en") or {}).get("acc"),
        "hinglish": (prod.get("gold_hinglish") or {}).get("acc"),
        "hi": (prod.get("gold_hi") or {}).get("acc"),
        "source": "results/research_eval.json main['production (emb+char, aug)']",
        "roman_hindi": None,
        "roman_hindi_reason": "no separate Roman-Hindi gold queries were authored",
    }

    dosage = {}
    for lang, probes in DOSAGE_PROBES.items():
        dosage[lang] = {
            "n": len(probes),
            "v1_recall": _rate(sum(1 for t in probes if _v1(t) is not None), len(probes)),
            "v2_recall": _rate(sum(1 for t in probes if _v2(t) is not None), len(probes)),
        }

    return {
        "protocol": {
            "gold": "data/gold/gold_queries.csv (50 en / 50 hinglish / 50 hi, LLM-authored)",
            "no_training": "components reused read-only; nothing retrained",
        },
        "register_and_entities": registers,
        "intent_accuracy": intent,
        "dosage_detection_recall": dosage,
        "limitations": [
            "Gold set is small (n=50 per register) and LLM-authored, not human-verified.",
            "Entity coverage counts any crop/pest hit; there is no entity ground truth, so no precision/recall.",
            "Roman-Hindi has no separate gold queries; its intent accuracy is not measured here.",
            "Devanagari register detection is script-based and near-perfect; it says nothing about understanding.",
        ],
    }


def render(m: dict) -> str:
    lines = [
        "# Language capability report",
        "",
        "**Evidence class:** reuse of the committed gold set and shipped components; no training.",
        "Small, LLM-authored sample (n=50 per register) — a capability check, not a benchmark.",
        "",
        "## Intent accuracy (from `results/research_eval.json`)",
        "",
        "| Language | Accuracy | Source |",
        "|---|---:|---|",
    ]
    for lang in ("en", "hinglish", "hi", "roman_hindi"):
        v = m["intent_accuracy"].get(lang)
        lines.append(f"| {lang} | {v if v is not None else 'not measured'} | {m['intent_accuracy']['source'] if v is not None else m['intent_accuracy']['roman_hindi_reason']} |")
    lines += [
        "",
        "## Register detection and entity coverage (recomputed)",
        "",
        "| Language | n | register detection | entity coverage |",
        "|---|---:|---:|---:|",
    ]
    for style, b in m["register_and_entities"].items():
        lines.append(f"| {style} | {b['n']} | {b['register_detection_accuracy']} | {b['entity_coverage']} |")
    lines += [
        "",
        "## Dosage/risk detection recall on authored per-language probes",
        "",
        "| Language | n | v1 (shipped) | v2 (candidate) |",
        "|---|---:|---:|---:|",
    ]
    for lang, b in m["dosage_detection_recall"].items():
        lines.append(f"| {lang} | {b['n']} | {b['v1_recall']} | {b['v2_recall']} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {x}" for x in m["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "REPORT.md"):
        if (out / name).exists() and not args.overwrite:
            print(f"refusing to overwrite {out / name} (pass --overwrite)")
            return 1
    m = build()
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(render(m), encoding="utf-8")
    print(json.dumps({"register_and_entities": m["register_and_entities"],
                      "intent_accuracy": m["intent_accuracy"],
                      "dosage_detection_recall": m["dosage_detection_recall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
