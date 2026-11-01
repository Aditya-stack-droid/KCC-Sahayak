"""Speech-to-text front end. Wraps openai-whisper so the pipeline can take a
raw farmer call recording (wav/mp3) as input instead of pre-typed text --
the gap the literature survey identifies in every prior KCC-NLP system
(KisanQRS, AgriLLM, KrushiMitra all start from already-transcribed text).

Whisper's output is treated as noisy: normalize_transcript() lightly cleans
common code-switched Hindi-English filler/punctuation artifacts before the
text is handed to the intent classifier / retrieval stages.
"""
import re

_WHISPER_MODEL = None

# Whisper decodes unfamiliar farming words as common English ones ("mandi" -> "monday",
# "bajra" -> "batch", "rohu" -> "rojo"). A vocabulary hint in the decoder prompt biases
# it towards the right spellings.
DOMAIN_PROMPT = (
    "Kisan Call Centre farmer query. Words: mandi, bajra, jowar, sarson, gehu, dhan, kapas, "
    "rohu, catla, PM-Kisan, KCC, urea, DAP, MOP, kharif, rabi, bigha, kanal, quintal, MSP, "
    "kisan credit card, fasal bima, sugarcane, mustard, paddy, wheat, cotton, aphid, whitefly."
)


def _get_model(model_size: str = "small"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper

        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


def transcribe(audio_path: str, model_size: str = "small", language: str | None = None,
               use_domain_prompt: bool = True) -> dict:
    """Transcribe an audio file. `language=None` lets Whisper auto-detect
    (useful for code-switched Hindi/English/Haryanvi calls)."""
    model = _get_model(model_size)
    prompt = DOMAIN_PROMPT if use_domain_prompt else None
    result = model.transcribe(audio_path, language=language, initial_prompt=prompt)
    # Whisper often labels short Hindi speech as Urdu and writes Arabic script, which the
    # rest of the pipeline (Devanagari / Roman Hindi) cannot use. Farmers calling an Indian
    # helpline are overwhelmingly speaking Hindi, so re-decode as Hindi.
    if language is None and result.get("language") == "ur":
        result = model.transcribe(audio_path, language="hi", initial_prompt=prompt)
    return {
        "text": normalize_transcript(result["text"]),
        "raw_text": result["text"],
        "detected_language": result.get("language"),
    }


def normalize_transcript(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,%-]", "", text)
    return text
