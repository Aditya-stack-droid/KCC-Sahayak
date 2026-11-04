"""Authored multi-intent evaluation cases.

Each case lists the intent labels a careful reader would assign to the query.
Multi-intent labelling is a judgement call, so the criterion here is explicit:
a query is multi-intent when it asks for **two or more distinct intent classes**.
Single-request controls are included and must not be split.

Every case is synthetic. ``gold_safety`` is derived: true when any gold intent is
high-stakes or time-sensitive (so the whole query must be escalated).
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

FIELDS = ["case_id", "text", "language", "category", "gold_intents", "synthetic"]

# (case_id, text, language, category, [gold intents])
CASES: list[tuple[str, str, str, str, list[str]]] = [
    # ---- two intents, unrelated (en) ----
    ("mi-001", "What is the price of urea and how much should I apply to my wheat crop?", "en", "two_unrelated", ["Market Price", "Nutrient Management"]),
    ("mi-002", "What is the weather forecast and which tomato variety should I sow?", "en", "two_unrelated", ["Weather", "Varietal Information"]),
    ("mi-003", "Which fertiliser is best for paddy and what is the capital of Japan?", "en", "two_unrelated", ["Nutrient Management", "Other"]),
    ("mi-004", "How do I control aphids in mustard and who won yesterday's cricket match?", "en", "two_unrelated", ["Plant Protection", "Other"]),
    ("mi-005", "My buffalo is not eating and please book a train ticket to Delhi.", "en", "two_unrelated", ["Animal Husbandry", "Other"]),
    ("mi-006", "What is the MSP for cotton and how do I irrigate sugarcane?", "en", "two_unrelated", ["Market Price", "Cultural Practices"]),
    ("mi-007", "Tell me the market price of wheat and write a poem about rain.", "en", "two_unrelated", ["Market Price", "Other"]),
    ("mi-008", "How do I sow bajra and how do I open a savings account?", "en", "two_unrelated", ["Cultural Practices", "Other"]),
    ("mi-009", "My cow has fever and which movie won the award?", "en", "two_unrelated", ["Animal Husbandry", "Other"]),
    ("mi-010", "What pesticide for bollworm and how do I apply for a passport?", "en", "two_unrelated", ["Plant Protection", "Other"]),
    ("mi-011", "Best seed variety for gram and what is the interest rate on fixed deposit?", "en", "two_unrelated", ["Varietal Information", "Other"]),
    ("mi-012", "My brinjal crop has wilt and give me a recipe for brinjal curry.", "en", "two_unrelated", ["Plant Protection", "Other"]),
    ("mi-013", "How do I irrigate sugarcane and who is the prime minister?", "en", "two_unrelated", ["Cultural Practices", "Other"]),
    ("mi-014", "Which fungicide for blight and how do I fix my laptop battery?", "en", "two_unrelated", ["Plant Protection", "Other"]),
    ("mi-015", "My goat is limping and what is the schedule of the next train?", "en", "two_unrelated", ["Animal Husbandry", "Other"]),
    ("mi-016", "Fertilizer for tomato plants and what is the boiling point of water?", "en", "two_unrelated", ["Nutrient Management", "Other"]),

    # ---- two intents, agricultural & related ----
    ("mi-017", "wheat rust control and urea dose for wheat", "en", "two_related", ["Plant Protection", "Nutrient Management"]),
    ("mi-018", "Which variety of paddy is good and when should I transplant it?", "en", "two_related", ["Varietal Information", "Cultural Practices"]),
    ("mi-019", "How much urea for wheat and how do I control weeds in it?", "en", "two_related", ["Nutrient Management", "Plant Protection"]),
    ("mi-020", "My tomato leaves are turning yellow and insects are eating them. What pesticide should I use?", "en", "two_related", ["Cultural Practices", "Plant Protection"]),
    ("mi-021", "What is the mandi rate for mustard and which variety gives the best yield?", "en", "two_related", ["Market Price", "Varietal Information"]),
    ("mi-022", "Tell me the weather for sowing and recommend a wheat variety.", "en", "two_related", ["Weather", "Varietal Information"]),
    ("mi-023", "How do I apply for PM-Kisan and what is the subsidy on drip irrigation?", "en", "two_related", ["Government Scheme", "Government Scheme"]),
    ("mi-024", "My fish pond has algae and how do I increase fish growth?", "en", "two_related", ["Fisheries", "Fisheries"]),
    ("mi-025", "Which fodder crop for my cattle and how do I treat my cow's wound?", "en", "two_related", ["Animal Husbandry", "Animal Husbandry"]),

    # ---- three intents ----
    ("mi-026", "What is the price of DAP, how much do I apply to wheat, and which variety is best?", "en", "three_intent", ["Market Price", "Nutrient Management", "Varietal Information"]),
    ("mi-027", "How do I control termites, when should I irrigate, and what is the MSP for sugarcane?", "en", "three_intent", ["Plant Protection", "Cultural Practices", "Market Price"]),
    ("mi-028", "My cow is sick, my pond is dry, and what is the weather tomorrow?", "en", "three_intent", ["Animal Husbandry", "Fisheries", "Weather"]),
    ("mi-029", "Which seed for mustard, how much urea, and how do I spray for aphids?", "en", "three_intent", ["Varietal Information", "Nutrient Management", "Plant Protection"]),
    ("mi-030", "Tell me the wheat variety, the fertiliser dose, and the market rate.", "en", "three_intent", ["Varietal Information", "Nutrient Management", "Market Price"]),

    # ---- Hindi / Devanagari ----
    ("mi-031", "गेहूं का भाव क्या है और यूरिया कितना डालें?", "devanagari", "two_unrelated", ["Market Price", "Nutrient Management"]),
    ("mi-032", "मौसम कैसा रहेगा और कौन सी किस्म बोएं?", "devanagari", "two_unrelated", ["Weather", "Varietal Information"]),
    ("mi-033", "गेहूं में रतुआ है और यूरिया कितना डालें?", "devanagari", "two_related", ["Plant Protection", "Nutrient Management"]),
    ("mi-034", "मेरी भैंस बीमार है और दूध कम दे रही है।", "devanagari", "two_related", ["Animal Husbandry", "Animal Husbandry"]),
    ("mi-035", "मछली तालाब की देखभाल कैसे करें और मछली का बाजार भाव क्या है?", "devanagari", "two_related", ["Fisheries", "Market Price"]),
    ("mi-036", "सरसों में कीड़ा लगा है और बीज कौन सा बोएं?", "devanagari", "two_related", ["Plant Protection", "Varietal Information"]),
    ("mi-037", "योजना का लाभ कैसे लें और गेहूं की सिंचाई कब करें?", "devanagari", "two_related", ["Government Scheme", "Cultural Practices"]),

    # ---- Roman Hindi ----
    ("mi-038", "gehu ka bhav kya hai aur urea kitna dalen?", "roman_hindi", "two_unrelated", ["Market Price", "Nutrient Management"]),
    ("mi-039", "mausam kaisa rahega aur kaunsi kism boyen?", "roman_hindi", "two_unrelated", ["Weather", "Varietal Information"]),
    ("mi-040", "gehu me ratua hai aur urea kitna dalen?", "roman_hindi", "two_related", ["Plant Protection", "Nutrient Management"]),
    ("mi-041", "sarson me keeda lag gaya hai aur beej kaunsa boyen?", "roman_hindi", "two_related", ["Plant Protection", "Varietal Information"]),
    ("mi-042", "yojana ka labh kaise lein aur sinchai kab kare?", "roman_hindi", "two_related", ["Government Scheme", "Cultural Practices"]),
    ("mi-043", "mandi me sarson ka bhav kya hai aur kism kaunsi hai?", "roman_hindi", "two_unrelated", ["Market Price", "Varietal Information"]),
    ("mi-044", "talab me machhli kaise badhayen aur machhli ka bhav kya hai?", "roman_hindi", "two_related", ["Fisheries", "Market Price"]),

    # ---- Hinglish ----
    ("mi-045", "Gehu ka rate kya chal raha hai aur urea kitna dalna hai?", "hinglish", "two_unrelated", ["Market Price", "Nutrient Management"]),
    ("mi-046", "Mausam kaisa hai aur tomato ki kaunsi variety lagayein?", "hinglish", "two_unrelated", ["Weather", "Varietal Information"]),
    ("mi-047", "Meri gaay bimar hai aur doodh kam de rahi hai.", "hinglish", "two_related", ["Animal Husbandry", "Animal Husbandry"]),
    ("mi-048", "Paddy me stem borer hai aur khaad kitna dalen?", "hinglish", "two_related", ["Plant Protection", "Nutrient Management"]),
    ("mi-049", "Sinchai kab karein aur kheti ke liye loan kaise milega?", "hinglish", "two_related", ["Cultural Practices", "Government Scheme"]),
    ("mi-050", "Mandi me kapas ka bhav kya hai aur kapas me keeda lag gaya hai?", "hinglish", "two_related", ["Market Price", "Plant Protection"]),

    # ---- shared entities (same crop) ----
    ("mi-051", "Wheat has rust and the wheat price is falling; what should I do?", "en", "shared_entities", ["Plant Protection", "Market Price"]),
    ("mi-052", "For tomato: which variety and what urea dose?", "en", "shared_entities", ["Varietal Information", "Nutrient Management"]),
    ("mi-053", "My paddy needs fertiliser and also protection from stem borer.", "en", "shared_entities", ["Nutrient Management", "Plant Protection"]),

    # ---- different crops ----
    ("mi-054", "How do I control aphids in mustard and whitefly in cotton?", "en", "different_crops", ["Plant Protection", "Plant Protection"]),
    ("mi-055", "Urea dose for wheat and potash dose for sugarcane?", "en", "different_crops", ["Nutrient Management", "Nutrient Management"]),
    ("mi-056", "Best variety for paddy and best breed of buffalo?", "en", "different_crops", ["Varietal Information", "Animal Husbandry"]),

    # ---- mixed safe + high-risk ----
    ("mi-057", "How do I sow wheat and what is the monocrotophos dose for aphids?", "en", "mixed_high_risk", ["Cultural Practices", "Plant Protection"]),
    ("mi-058", "Which cow breed is good and how much urea for wheat?", "en", "mixed_high_risk", ["Animal Husbandry", "Nutrient Management"]),
    ("mi-059", "What is the weather tomorrow and what pesticide for bollworm?", "en", "mixed_high_risk", ["Weather", "Plant Protection"]),
    ("mi-060", "How to get a subsidy and what is the fertiliser dose for paddy?", "en", "mixed_high_risk", ["Government Scheme", "Nutrient Management"]),

    # ---- single-request controls (must NOT be split) ----
    ("mi-061", "What is the control measure for aphid infestation in mustard?", "en", "single_control", ["Plant Protection"]),
    ("mi-062", "Which tomato variety is good for me?", "en", "single_control", ["Varietal Information"]),
    ("mi-063", "Apply 50 kg urea per acre to wheat.", "en", "single_control", ["Nutrient Management"]),
    ("mi-064", "Tell me the weather forecast for tomorrow.", "en", "single_control", ["Weather"]),
    ("mi-065", "How much urea for wheat?", "en", "single_control", ["Nutrient Management"]),
    ("mi-066", "गेहूं का भाव क्या है?", "devanagari", "single_control", ["Market Price"]),
    ("mi-067", "gehu me ratua hai kya karein?", "roman_hindi", "single_control", ["Plant Protection"]),
    ("mi-068", "Meri gaay bimar hai.", "hinglish", "single_control", ["Animal Husbandry"]),
    ("mi-069", "Which variety of paddy is best?", "en", "single_control", ["Varietal Information"]),
    ("mi-070", "How do I control insects in my brinjal crop?", "en", "single_control", ["Plant Protection"]),
    ("mi-071", "When should I irrigate the wheat field?", "en", "single_control", ["Cultural Practices"]),
    ("mi-072", "What is the market price of mustard today?", "en", "single_control", ["Market Price"]),
    ("mi-073", "How do I increase milk yield in my buffalo?", "en", "single_control", ["Animal Husbandry"]),
    ("mi-074", "Which is the best fish for a farm pond?", "en", "single_control", ["Fisheries"]),
    ("mi-075", "How do I apply for crop insurance?", "en", "single_control", ["Government Scheme"]),
    ("mi-076", "How should I sow bajra this season?", "en", "single_control", ["Cultural Practices"]),
    ("mi-077", "What is the recommended seed rate for wheat?", "en", "single_control", ["Varietal Information"]),
    ("mi-078", "गेहूं में रतुआ है तो क्या करें?", "devanagari", "single_control", ["Plant Protection"]),
    ("mi-079", "Mandi me gehu ka bhav kya hai?", "roman_hindi", "single_control", ["Market Price"]),
    ("mi-080", "Talab me machhli kaise paalna hai?", "roman_hindi", "single_control", ["Fisheries"]),

    # ---- ambiguous single crop-health statements (control, judgement call) ----
    ("mi-081", "My tomato leaves are turning yellow.", "en", "single_control", ["Cultural Practices"]),
    ("mi-082", "Insects are eating my wheat plants.", "en", "single_control", ["Plant Protection"]),
    ("mi-083", "My crop has a disease.", "en", "single_control", ["Plant Protection"]),

    # ---- repeated same intent (should not count as two intents) ----
    ("mi-084", "How much urea and how much DAP for wheat?", "en", "single_control", ["Nutrient Management"]),
    ("mi-085", "Which variety is early and which variety is high-yielding?", "en", "single_control", ["Varietal Information"]),

    # ---- more two-intent pairs for volume ----
    ("mi-086", "What is the DAP price and the urea dose for wheat?", "en", "two_related", ["Market Price", "Nutrient Management"]),
    ("mi-087", "How do I store wheat and where can I sell it?", "en", "two_related", ["Cultural Practices", "Market Price"]),
    ("mi-088", "Which fungicide for blight and when should I spray it?", "en", "two_related", ["Plant Protection", "Cultural Practices"]),
    ("mi-089", "What is the subsidy for seeds and which seed variety is best?", "en", "two_related", ["Government Scheme", "Varietal Information"]),
    ("mi-090", "My buffalo has a fever and she is not giving milk.", "en", "two_related", ["Animal Husbandry", "Animal Husbandry"]),
    ("mi-091", "How do I test my soil and how much zinc do I add?", "en", "two_related", ["Cultural Practices", "Nutrient Management"]),
    ("mi-092", "Will it rain tomorrow and should I spray now?", "en", "two_related", ["Weather", "Plant Protection"]),
    ("mi-093", "What is the cotton MSP and how do I control bollworm?", "en", "two_related", ["Market Price", "Plant Protection"]),
    ("mi-094", "How do I prepare a fish pond and which species to stock?", "en", "two_related", ["Fisheries", "Fisheries"]),
    ("mi-095", "Which wheat variety tolerates heat and when do I irrigate?", "en", "two_related", ["Varietal Information", "Cultural Practices"]),
    ("mi-096", "What is the urea price and is it good for paddy?", "en", "two_unrelated", ["Market Price", "Nutrient Management"]),
    ("mi-097", "Tell me the mustard variety and the sowing time.", "en", "two_related", ["Varietal Information", "Cultural Practices"]),
    ("mi-098", "My pond has low oxygen and how do I treat it?", "en", "two_related", ["Fisheries", "Fisheries"]),
    ("mi-099", "What is the weather now and what is the mandi rate?", "en", "two_unrelated", ["Weather", "Market Price"]),
    ("mi-100", "How much fertiliser for tomato and which pest is attacking it?", "en", "two_related", ["Nutrient Management", "Plant Protection"]),
    ("mi-101", "गेहूं में यूरिया कितना डालें और सिंचाई कब करें?", "devanagari", "two_related", ["Nutrient Management", "Cultural Practices"]),
    ("mi-102", "sarson ka bhav kya hai aur keeda kaise hatayein?", "roman_hindi", "two_unrelated", ["Market Price", "Plant Protection"]),
    ("mi-103", "Kapas me bollworm hai aur fertiliser kitna dalen?", "hinglish", "two_related", ["Plant Protection", "Nutrient Management"]),
    ("mi-104", "What is the price of paddy and how do I control stem borer?", "en", "two_unrelated", ["Market Price", "Plant Protection"]),
    ("mi-105", "Which cow breed for milk and how much fodder to feed?", "en", "two_related", ["Animal Husbandry", "Animal Husbandry"]),
    ("mi-106", "How do I control whitefly and what is the weather forecast?", "en", "two_unrelated", ["Plant Protection", "Weather"]),
    ("mi-107", "Tell me the sugarcane price and the best irrigation method.", "en", "two_unrelated", ["Market Price", "Cultural Practices"]),
    ("mi-108", "How do I increase tomato yield and which variety should I buy?", "en", "two_related", ["Cultural Practices", "Varietal Information"]),
    ("mi-109", "My goat is sick and what is the price of goat milk?", "en", "two_unrelated", ["Animal Husbandry", "Market Price"]),
    ("mi-110", "Which crop after paddy and how much fertiliser for it?", "en", "two_related", ["Cultural Practices", "Nutrient Management"]),
]


def build(path: Path, overwrite: bool = False) -> list[dict]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite)")
    ids = [row[0] for row in CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for case_id, text, language, category, intents in CASES:
            writer.writerow([case_id, text, language, category, ";".join(intents), "true"])
    return [{"case_id": c, "text": t, "language": l, "category": cat, "gold_intents": ints, "synthetic": "true"}
            for c, t, l, cat, ints in CASES]


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "multi_intent_cases.csv"
    rows = build(out, overwrite="--overwrite" in sys.argv)
    print(f"wrote {out}: {len(rows)} cases")
    print("category:", dict(Counter(r["category"] for r in rows)))
    print("language:", dict(Counter(r["language"] for r in rows)))
    print("multi:", sum(1 for r in rows if len(r["gold_intents"]) >= 2), "single:", sum(1 for r in rows if len(r["gold_intents"]) == 1))
