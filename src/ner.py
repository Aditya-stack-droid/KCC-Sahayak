"""Rule-based entity extraction (crop, pest/symptom) via glossary matching.

No spaCy model download is required -- this is a plain longest-match
gazetteer matcher over the CROPS / PESTS_SYMPTOMS lists, mirroring the
low-labeling-cost approach the literature survey attributes to the KCC NER
paper (ResearchGate, 2021).
"""
import re

from glossary import CROPS, PESTS_SYMPTOMS

_crop_pattern = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(CROPS, key=len, reverse=True)) + r")\b"
)
_pest_pattern = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(PESTS_SYMPTOMS, key=len, reverse=True)) + r")\b"
)


def extract_entities(text: str) -> dict:
    text_l = text.lower()
    crops = sorted(set(_crop_pattern.findall(text_l)))
    symptoms = sorted(set(_pest_pattern.findall(text_l)))
    return {"crop": crops, "symptom": symptoms}
