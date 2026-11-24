"""Lightweight script/register detection for farmer queries.

Returns one of:
  "hi"       Devanagari Hindi
  "hinglish" Hindi written in Roman script
  "as"       Assamese (Bengali-Assamese script; Bengali shares the script and is treated the same)
  "en"       English / everything else

A heuristic for display and analytics, not something the pipeline depends on:
the multilingual encoder handles all three registers directly.

Roman-script Hindi is detected from two word lists:
  _FUNCTION  grammatical words / verb forms that essentially never occur in
             English text (one hit is weak evidence, two is strong);
  _NOUNS     Hindi farm vocabulary that also appears as loan words inside
             English sentences ("mandi", "bigha"), so it only counts in bulk.
Words that are also English ("me", "par", "ho", "hi", "to", "na") are excluded
on purpose.
"""
import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ASSAMESE = re.compile(r"[ঀ-৿]")

_FUNCTION = {
    "ke", "ka", "ki", "ko", "mein", "hai", "hain", "hoga", "hogi", "honge", "kya", "kaise",
    "kaisa", "kaisi", "kab", "kitna", "kitni", "kitne", "kaun", "kaunsi", "kaunsa", "kahan",
    "kyon", "nahi", "nahin", "aur", "se", "liye", "mera", "meri", "mere", "hamare", "hamara",
    "mujhe", "chahiye", "batao", "bataye", "bataiye", "karna", "karni", "karein", "kare",
    "karun", "dalun", "daalna", "daalein", "lagana", "lagwana", "milega", "milegi", "milenge",
    "hota", "hoti", "hote", "hua", "hui", "hue", "gaya", "gayi", "gaye", "rahe", "raha",
    "rahi", "rahega", "aayega", "lag", "lage", "lagi", "bhi", "toh", "tak", "saal", "agle",
    "abhi", "bahut", "zyada", "sabse", "wala", "wali", "wale", "kisi", "isko", "uska",
}
_NOUNS = {
    "bhav", "fasal", "gehu", "dhan", "sarson", "kapas", "khaad", "dawai", "bhains", "gaay",
    "bakri", "machhli", "talab", "mandi", "baarish", "mausam", "bigha", "kheti", "beej",
    "paani", "doodh", "mitti", "kism", "paidawar",
}


def detect_register(text: str) -> str:
    if not text:
        return "en"
    letters = [c for c in text if c.isalpha()]
    if letters:
        deva = sum(1 for c in letters if _DEVANAGARI.match(c))
        if deva / len(letters) > 0.3:
            return "hi"
        if sum(1 for c in letters if _ASSAMESE.match(c)) / len(letters) > 0.3:
            return "as"
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return "en"
    strong = sum(1 for t in tokens if t in _FUNCTION)
    nouns = sum(1 for t in tokens if t in _NOUNS)
    if strong >= 2 or (strong >= 1 and nouns >= 1) or nouns >= 3:
        return "hinglish"
    return "en"
