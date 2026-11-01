"""Interactive/scripted demo of the full KCC-Sahayak pipeline, including a
real audio -> text -> triage run if demo/sample_query.wav exists.

Usage:
    .venv/Scripts/python.exe scripts/05_demo_pipeline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import KCCSahayakPipeline  # noqa: E402

DEMO_AUDIO = ROOT / "demo" / "sample_query.wav"

TEXT_QUERIES = [
    "what is the control measure for aphid infestation in mustard crop",
    "asking about fertilizer dose for wheat crop",
    "how to avail kisan credit card loan for wheat crop",
    "asking about weather forecast for sowing paddy",
    "asking about a good variety of tomato seed for this season",
    "asking about treatment of low milk production in cow",
]


def print_result(label, result):
    print("\n" + "=" * 90)
    print(f"[{label}] {result['input_text']}")
    if result.get("asr"):
        print(f"  (ASR detected_language={result['asr']['detected_language']}, "
              f"raw='{result['asr']['raw_text'].strip()}')")
    print(f"  Intent      : {result['query_type']}  (confidence={result['intent_confidence']:.2f})")
    print(f"  Entities    : {result['entities']}")
    top = result["retrieved_top3"][0]
    print(f"  Top match   : sim={top['similarity']:.2f}  '{top['matched_question']}'")
    print(f"  DECISION    : {result['decision']}")
    print(f"  Reason      : {result['decision_reason']}")
    if result["suggested_answer"]:
        print(f"  Answer      : {result['suggested_answer']}")


def main():
    pipeline = KCCSahayakPipeline()

    for q in TEXT_QUERIES:
        result = pipeline.run(q)
        print_result("TEXT", result)

    if DEMO_AUDIO.exists():
        result = pipeline.run(text=None, from_audio=str(DEMO_AUDIO))
        print_result("AUDIO", result)
    else:
        print(f"\n(No demo audio found at {DEMO_AUDIO} -- skipping voice-input demo. "
              f"Run scripts/00_generate_demo_audio.ps1 to synthesize one.)")


if __name__ == "__main__":
    main()
