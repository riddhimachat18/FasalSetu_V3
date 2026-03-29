"""
agents/voice_agent.py — FasalSetu
====================================
Handles voice input in local Indian languages.
Converts speech → text → normalized English query for the orchestrator.

Pipeline:
  Audio file / microphone → Whisper (speech-to-text) → language detect
  → translate to English if needed → orchestrator → response
  → optional TTS back in source language

Supported languages (Whisper multilingual):
  Hindi (hi), Bengali (bn), Telugu (te), Marathi (mr),
  Tamil (ta), Gujarati (gu), Kannada (kn), Malayalam (ml),
  Punjabi (pa), Odia (or)

Dependencies:
  pip install openai-whisper deep-translator gTTS pydub
  brew install ffmpeg   (macOS)
  apt install ffmpeg    (Linux)

For demo without audio hardware:
  Use the text translation tools directly — pass Hindi text,
  get normalized English + Hindi response back.
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional
from langchain.tools import tool

logger = logging.getLogger(__name__)

# ── Language configuration ─────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",    "bn": "Bengali",  "te": "Telugu",
    "mr": "Marathi",  "ta": "Tamil",    "gu": "Gujarati",
    "kn": "Kannada",  "ml": "Malayalam","pa": "Punjabi",
    "or": "Odia",     "en": "English",
}

# Common farming terms: Hindi → English (fast lookup without API)
_HINDI_FARMING_GLOSSARY = {
    "मिट्टी":      "soil",
    "खाद":         "fertiliser",
    "पानी":        "water",
    "फसल":        "crop",
    "बीज":         "seeds",
    "कीड़े":       "pests",
    "रोग":         "disease",
    "गेहूं":       "wheat",
    "धान":         "paddy",
    "मक्का":       "maize",
    "दाल":         "pulses",
    "सोयाबीन":    "soybean",
    "कपास":        "cotton",
    "सरसों":       "mustard",
    "टमाटर":       "tomato",
    "प्याज":       "onion",
    "आलू":         "potato",
    "नाइट्रोजन":  "nitrogen",
    "फास्फोरस":   "phosphorus",
    "पोटेशियम":   "potassium",
    "यूरिया":      "urea",
    "सिंचाई":     "irrigation",
    "बारिश":      "rain",
    "मौसम":       "weather",
    "बाजार":      "market",
    "भाव":        "price",
    "MSP":        "MSP",
    "न्यूनतम समर्थन मूल्य": "minimum support price",
    "सरकारी योजना":   "government scheme",
    "कृषि":        "agriculture",
    "किसान":       "farmer",
    "KVK":         "KVK",
    "कीटनाशक":    "pesticide",
    "फफूंदनाशक":  "fungicide",
    "कमी":         "deficiency",
    "अधिकता":      "excess",
    "उपज":        "yield",
    "सीजन":       "season",
    "खरीफ":       "kharif",
    "रबी":        "rabi",
}


def _glossary_translate(text: str, source_lang: str = "hi") -> str:
    """Fast glossary-based translation for common farming terms."""
    if source_lang != "hi":
        return text
    result = text
    for hindi, english in sorted(_HINDI_FARMING_GLOSSARY.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        result = result.replace(hindi, english)
    return result


def _detect_language(text: str) -> str:
    """
    Simple heuristic language detection based on Unicode ranges.
    Returns ISO 639-1 code.
    """
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    bengali     = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    telugu      = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    tamil       = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    gujarati    = sum(1 for c in text if "\u0A80" <= c <= "\u0AFF")
    kannada     = sum(1 for c in text if "\u0C80" <= c <= "\u0CFF")
    malayalam   = sum(1 for c in text if "\u0D00" <= c <= "\u0D7F")
    gurmukhi    = sum(1 for c in text if "\u0A00" <= c <= "\u0A7F")

    scores = {
        "hi": devanagari, "mr": devanagari,  # Devanagari shared
        "bn": bengali,    "te": telugu,
        "ta": tamil,      "gu": gujarati,
        "kn": kannada,    "ml": malayalam,
        "pa": gurmukhi,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 2 else "en"


def _translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text to English.
    Uses deep-translator, falls back to glossary-only translation.
    """
    if source_lang == "en":
        return text

    # Glossary pass first (always fast)
    glossary_result = _glossary_translate(text, source_lang)

    # Try deep-translator (optional dependency)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        logger.info("Translated [%s→en]: '%s' → '%s'", source_lang, text[:50], translated[:50])
        return translated
    except ImportError:
        logger.debug("deep-translator not installed — using glossary translation only")
        return glossary_result
    except Exception as e:
        logger.warning("Translation failed: %s — using glossary", e)
        return glossary_result


def _transcribe_audio(audio_path: str, language: Optional[str] = None) -> dict:
    """
    Transcribe audio file using OpenAI Whisper.
    Returns {text, language, confidence}.
    """
    try:
        import whisper
        model = whisper.load_model("base")  # ~150MB, runs on CPU

        if language:
            result = model.transcribe(audio_path, language=language)
        else:
            result = model.transcribe(audio_path)

        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": len(result.get("segments", [])),
        }
    except ImportError:
        return {
            "error": "Whisper not installed. Run: pip install openai-whisper",
            "text":  "",
        }
    except Exception as e:
        return {"error": str(e), "text": ""}


def _text_to_speech(text: str, language: str = "hi", output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert text to speech using gTTS.
    Returns path to audio file, or None if gTTS not available.
    """
    try:
        from gtts import gTTS
        tts_lang = language if language in ["hi", "bn", "te", "mr", "ta", "gu", "en"] else "hi"
        tts = gTTS(text=text, lang=tts_lang, slow=False)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            output_path = tmp.name
            tmp.close()

        tts.save(output_path)
        return output_path
    except ImportError:
        logger.debug("gTTS not installed — skipping TTS output")
        return None
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return None


# ── LangChain Tools ────────────────────────────────────────────────────────

@tool
def process_voice_query(audio_file_path: str, hint_language: str = "") -> dict:
    """
    Process a voice/audio query from a farmer.
    Transcribes audio, detects language, translates to English,
    and returns a normalized query ready for the orchestrator.

    Args:
        audio_file_path: Path to audio file (WAV, MP3, M4A, OGG supported)
        hint_language:   Optional language hint (hi, bn, te, mr, ta, gu, kn, ml, pa)
                         Leave empty for automatic detection.

    Returns:
        original_text, detected_language, english_query, ready for orchestrator.
    """
    path = Path(audio_file_path)
    if not path.exists():
        return {"error": f"Audio file not found: {audio_file_path}"}

    # Transcribe
    lang_hint = hint_language if hint_language in SUPPORTED_LANGUAGES else None
    transcription = _transcribe_audio(str(path), language=lang_hint)

    if transcription.get("error"):
        return {"error": transcription["error"]}

    raw_text      = transcription["text"]
    detected_lang = transcription.get("language", "hi")

    # Translate
    english_query = _translate_to_english(raw_text, detected_lang)

    return {
        "original_text":    raw_text,
        "detected_language": detected_lang,
        "language_name":    SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "english_query":    english_query,
        "ready_for_agent":  True,
        "note": f"Transcribed {transcription.get('segments', '?')} segments via Whisper",
    }


@tool
def translate_farmer_query(text: str, source_language: str = "auto") -> dict:
    """
    Translate a farmer's text query from a local Indian language to English.
    Use when the farmer types in Hindi, Marathi, Telugu, etc.

    Args:
        text:            The farmer's query in their local language
        source_language: Language code (hi, bn, te, mr, ta, gu, kn, ml, pa, en)
                         Use 'auto' for automatic detection.

    Returns:
        original_text, detected_language, english_translation.

    Example:
        text="मेरी गेहूं की फसल में पीले धब्बे हैं क्या करूं?"
        → "My wheat crop has yellow spots, what should I do?"
    """
    if source_language == "auto" or source_language not in SUPPORTED_LANGUAGES:
        detected = _detect_language(text)
    else:
        detected = source_language

    english = _translate_to_english(text, detected)

    return {
        "original_text":     text,
        "detected_language": detected,
        "language_name":     SUPPORTED_LANGUAGES.get(detected, "Unknown"),
        "english_query":     english,
        "glossary_applied":  detected == "hi",
    }


@tool
def speak_response(response_text: str, language: str = "hi") -> dict:
    """
    Convert a text response to speech in the farmer's language.
    Returns path to the generated audio file.

    Args:
        response_text: The response to speak (should already be in target language)
        language:      Target language code (hi, bn, te, mr, ta, gu, en)

    Returns:
        audio_file_path if TTS available, else text_only flag.
    """
    audio_path = _text_to_speech(response_text, language)

    if audio_path:
        return {
            "audio_file":  audio_path,
            "language":    SUPPORTED_LANGUAGES.get(language, language),
            "text":        response_text,
            "tts_success": True,
        }
    return {
        "audio_file":  None,
        "tts_success": False,
        "text":        response_text,
        "note":        "Install gTTS for audio output: pip install gTTS",
    }


@tool
def get_language_support_info() -> dict:
    """
    List all supported languages and their capabilities.
    Returns language codes, names, and available features.
    """
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "speech_to_text": {
            "engine":   "OpenAI Whisper (base model)",
            "install":  "pip install openai-whisper",
            "note":     "Runs offline on CPU. ~150MB model download on first use.",
        },
        "text_to_speech": {
            "engine":  "Google TTS (gTTS)",
            "install": "pip install gTTS",
            "note":    "Requires internet connection for TTS generation.",
        },
        "translation": {
            "engine":   "deep-translator + built-in farming glossary",
            "install":  "pip install deep-translator",
            "offline":  "Built-in Hindi farming glossary works without internet",
            "glossary_terms": len(_HINDI_FARMING_GLOSSARY),
        },
        "offline_mode": {
            "available": True,
            "features":  [
                "Hindi farming term glossary (no internet needed)",
                "Language detection via Unicode ranges",
                "Whisper transcription (after initial model download)",
            ],
        },
    }
