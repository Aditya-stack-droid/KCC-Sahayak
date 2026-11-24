import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lang import detect_register  # noqa: E402


def test_assamese_script_is_detected():
    assert detect_register("মোৰ ধানৰ পথাৰত পোক লাগিছে") == "as"


def test_other_registers_unchanged():
    assert detect_register("what is the sowing time for mustard") == "en"
    assert detect_register("गेहूं की बुवाई कब करें") == "hi"
    assert detect_register("gehu mein peela rust aa gaya hai kya karna chahiye") == "hinglish"
