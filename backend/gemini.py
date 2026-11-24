"""Optional second opinion from Google's Gemini, shown beside the KCC-corpus answer.

Design rules (this is generated text, unlike the corpus, so it can be wrong):
  * It is NEVER used as the system's answer and never changes the confidence-gate decision.
  * The UI labels it "AI-generated, not verified".
  * The prompt forbids exact pesticide / fertiliser doses and tells the model to point the farmer
    to a local KVK or the Kisan Call Centre (1800-180-1551) for those.
  * No key, no call: without GEMINI_API_KEY the feature reports "not configured" and nothing is sent anywhere.

The API key is read from the environment or from a local `.env` file (git-ignored). It is never logged,
never returned to the browser, and never written to the database.
Optional: GEMINI_MODEL (default below).
"""
import json
import os
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-3.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_S = 25

SYSTEM_PROMPT = (
    "You are an agricultural extension assistant for Indian farmers, supporting the Kisan Call Centre. "
    "Reply in the same language and script as the farmer's question: a question written in English gets an English reply, "
    "Hinglish in Roman script gets Hinglish, Hindi in Devanagari gets Hindi. "
    "Be practical and concise: at most 90 words, plain sentences, no markdown, no headings. "
    "NEVER give exact pesticide, herbicide, fungicide or fertiliser doses, concentrations or mixing ratios, and never name a "
    "specific brand. For pest, disease or nutrient questions give only general integrated-management steps (scouting, "
    "cultural and biological options, the type of product) and tell the farmer to confirm the product and dose with the "
    "local Krishi Vigyan Kendra (KVK) or the Kisan Call Centre on 1800-180-1551. "
    "You have no live data and no regional price data: if the question is about current or upcoming weather, market "
    "prices, or other real-time conditions, say you cannot know them and point to the IMD / Meghdoot app, Agmarknet "
    "or e-NAM, or the Kisan Call Centre. "
    "If the question is not about agriculture, farming, livestock or fisheries, say you can only help with farming questions. "
    "If you are unsure, say so."
)

_cache: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 200


def _load_dotenv():
    """Minimal .env reader (KEY=VALUE per line); never overrides real environment variables."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def configured() -> bool:
    _load_dotenv()
    return bool(os.environ.get("GEMINI_API_KEY"))


def model_name() -> str:
    _load_dotenv()
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _generate(system: str, text: str, temperature: float = 0.3, tag: str = "") -> dict:
    """Return {"available": bool, "text": str|None, "model": str, "error": str|None}. Never raises."""
    model = model_name()
    if not configured():
        return {"available": False, "text": None, "model": model,
                "error": "Gemini is not configured. Add GEMINI_API_KEY to a .env file and restart the server."}
    user_text = text.strip()
    key_text = f"{tag}|{user_text}"
    if key_text in _cache:
        _cache.move_to_end(key_text)
        return _cache[key_text]
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    req = urllib.request.Request(
        ENDPOINT.format(model=model), data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        out = " ".join(p.get("text", "") for p in parts).strip()
        if not out:
            reason = cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason") or "empty response"
            return {"available": True, "text": None, "model": model, "error": f"Gemini returned no text ({reason})."}
        result = {"available": True, "text": out, "model": model, "error": None}
    except urllib.error.HTTPError as e:
        # 400/403 usually means a bad key or model name; 429 means quota. Do not echo the response body.
        hint = {400: "bad request or unknown model", 401: "invalid API key", 403: "API key not allowed",
                404: "model not found (set GEMINI_MODEL)", 429: "quota or rate limit reached"}.get(e.code, "request failed")
        return {"available": True, "text": None, "model": model, "error": f"Gemini error {e.code}: {hint}."}
    except Exception as e:  # network down, timeout, bad JSON
        return {"available": True, "text": None, "model": model, "error": f"Could not reach Gemini ({type(e).__name__})."}
    _cache[key_text] = result
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return result


def suggest(text: str) -> dict:
    """Second-opinion advice for a farmer's question."""
    return _generate(SYSTEM_PROMPT, text)


# Target-language instructions. The KCC corpus is English, so other languages are handled by translating the
# question to English before retrieval and the answer back afterwards. Machine translation: shown as unverified.
LANGUAGES = {
    "en": "English",
    "hi": "Hindi in Devanagari script",
    "hinglish": "Hinglish (Hindi written in Roman script)",
    "as": "Assamese in Assamese script",
}


def translate(text: str, lang: str) -> dict:
    """Translate `text` into the language code `lang` (en, hi, hinglish, as). Same return shape as suggest()."""
    target = LANGUAGES.get(lang)
    if target is None:
        return {"available": True, "text": None, "model": model_name(), "error": f"Unsupported language '{lang}'."}
    system = (f"You translate Indian farming text into {target}. Output only the translation: no notes, quotes or "
              "markdown. Keep crop and variety names, numbers, units, phone numbers and abbreviations (KVK, IMD) "
              "exactly as given. If the text is already in the target language, return it unchanged.")
    return _generate(system, text, temperature=0.1, tag=f"tr:{lang}")

