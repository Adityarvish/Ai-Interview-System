import logging
import os
import time
from pathlib import Path


from groq import Groq, AuthenticationError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)


GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            # Fail early with a clear message rather than a cryptic 401 later.
            logger.error(
                "[STT] GROQ_API_KEY environment variable is not set. "
                "Set it in your .env file or shell before starting the server."
            )
           
        _groq_client = Groq(api_key=api_key or None)
        logger.info("[STT] Groq client initialised (model=%s)", GROQ_WHISPER_MODEL)
    return _groq_client


# ── Public transcription function ─────────────────────────────────────────────

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    
    t0 = time.perf_counter()
    path = Path(audio_path)

    
    if not path.exists():
        logger.error("[STT] Audio file not found: %s", audio_path)
        return ""

    file_size = path.stat().st_size
    if file_size == 0:
        logger.error("[STT] Audio file is empty (0 bytes): %s", audio_path)
        return ""

    logger.info("[STT] Transcribing %s (%d bytes) via Groq …", path.name, file_size)

   
    try:
        client = _get_groq_client()

        with open(audio_path, "rb") as audio_file:
            
            transcription = client.audio.transcriptions.create(
                file=(path.name, audio_file.read()),   
                model=GROQ_WHISPER_MODEL,
                response_format="json",               
                language=language,                     
            )
       

        transcript = (transcription.text or "").strip()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[STT] Done in %dms → %d chars: '%s'",
            elapsed_ms, len(transcript), transcript[:80],
        )
        return transcript

    # ── Structured error handling ─────────────────────────────────────────────

    except AuthenticationError as e:
        # 401 — API key is missing, revoked, or malformed.
        logger.error(
            "[STT] Groq authentication failed — check GROQ_API_KEY. "
            "Details: %s", e
        )
        return ""

    except APIConnectionError as e:
        # Network-level failure: DNS, TCP timeout, TLS error, etc.
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "[STT] Network error reaching Groq after %dms — "
            "check internet connectivity. Details: %s", elapsed_ms, e
        )
        return ""

    except APIStatusError as e:
        # 4xx / 5xx from Groq (rate limit, server error, bad request, …).
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "[STT] Groq API error %d after %dms — %s",
            e.status_code, elapsed_ms, e.message
        )
        return ""

    except FileNotFoundError:
        # Shouldn't reach here (caught above), but guard against TOCTOU races.
        logger.error("[STT] Audio file disappeared before upload: %s", audio_path)
        return ""

    except Exception as e:
        # Catch-all: unexpected SDK changes, OS errors, etc.
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(
            "[STT] Unexpected error after %dms transcribing %s: %s",
            elapsed_ms, path.name, e
        )
        return ""



class SpeechToTextService:
    """
    Thin wrapper around transcribe_audio() that preserves the class-based
    interface expected by socket_routes.py.

    The heavy lifting (Groq client, error handling, logging) lives in
    transcribe_audio() so it can also be called directly if needed.
    """

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        
        return transcribe_audio(audio_path, language=language)


# ── Module-level singleton (matches existing import in socket_routes.py) ───────
stt_service = SpeechToTextService()
