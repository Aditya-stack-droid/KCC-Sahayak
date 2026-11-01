import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lang import detect_register  # noqa: E402
from ner import extract_entities  # noqa: E402
from weak_labels import label_query  # noqa: E402


# ---- language register detection ----
def test_devanagari_is_hindi():
    assert detect_register("मेरी गेहूं की फसल में पीला रतुआ लगा है") == "hi"


def test_roman_hindi_is_hinglish():
    assert detect_register("gehu mein peela rust aa gaya hai kya spray karna chahiye") == "hinglish"


def test_plain_english_is_english():
    assert detect_register("what is the control measure for aphid infestation in mustard") == "en"


def test_empty_and_symbols_default_to_english():
    assert detect_register("") == "en"
    assert detect_register("1234 !!!") == "en"


def test_english_with_one_stray_hindi_word_stays_english():
    assert detect_register("please tell me the price of wheat in the mandi today and next week") == "en"


# ---- entity extraction ----
def test_extracts_crop_and_symptoms():
    e = extract_entities("aphid infestation in mustard crop")
    assert "mustard" in e["crop"]
    assert "aphid" in e["symptom"] and "infestation" in e["symptom"]


def test_no_entities_returns_empty_lists():
    e = extract_entities("what is the phone number of the office")
    assert e == {"crop": [], "symptom": []}


def test_entity_matching_is_case_insensitive_and_whole_word():
    assert extract_entities("WHEAT rust")["crop"] == ["wheat"]
    assert extract_entities("buckwheatish")["crop"] == []  # no substring matches


# ---- keyword labeller (kept as a baseline) ----
def test_keyword_labeller_basic_cases():
    assert label_query("control measure for aphid infestation") == "Plant Protection"
    assert label_query("kisan credit card loan") == "Government Scheme"
    assert label_query("hello") == "Other"
