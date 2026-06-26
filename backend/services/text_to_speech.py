import logging
import os
import time
from pathlib import Path

import requests
import base64 as _b64

logger = logging.getLogger(__name__)



try:
    from groq import Groq, AuthenticationError, APIConnectionError, APIStatusError
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GROQ_SDK_AVAILABLE = False
   
    AuthenticationError = APIConnectionError = APIStatusError = Exception
    logger.critical(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  [TTS] FATAL — groq SDK not installed                       ║\n"
        "║  Run:  pip install groq>=0.9.0                              ║\n"
        "║  Then restart the server.                                   ║\n"
        "║  Until then ALL TTS calls will fail and the frontend        ║\n"
        "║  will fall back to the browser's built-in speechSynthesis.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

from config.settings import Config



GROQ_TTS_MODEL  = "canopylabs/orpheus-v1-english"

GROQ_TTS_VOICE  = os.getenv("GROQ_TTS_VOICE", "daniel").strip()

GROQ_TTS_MAX_CHARS = 1800

#
GROQ_TTS_FORMAT = "wav"

SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_TTS_URL     = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_MODEL   = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_SPEAKER     = os.getenv("SARVAM_TTS_SPEAKER", "anushka")  
SARVAM_LANG_CODES  = {"hi": "hi-IN", "mr": "mr-IN"}
SARVAM_TTS_MAX_CHARS = 1500  


def _sarvam_tts_chunk(text: str, language: str) -> bytes:
    """Call Sarvam AI TTS for a single chunk of text. Returns raw WAV bytes."""
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY environment variable is not set. "
            "Add it to backend/.env:  SARVAM_API_KEY=sk_... "
            "(get a free-tier key at https://www.sarvam.ai)"
        )

    target_lang = SARVAM_LANG_CODES.get(language, "hi-IN")
    headers = {
        "API-Subscription-Key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text],
        "target_language_code": target_lang,
        "speaker": SARVAM_SPEAKER,
        "model": SARVAM_TTS_MODEL,
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
    }
    resp = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Sarvam TTS HTTP {resp.status_code} — {resp.text[:300]}"
        )
    data = resp.json()
    audios = data.get("audios") or []
    if not audios:
        raise RuntimeError(f"Sarvam TTS returned no audio — {data}")
    return _b64.b64decode(audios[0])


def _sarvam_text_to_speech(text: str, output_file: str, language: str) -> "str | None":
    """Synthesize Hindi/Marathi speech via Sarvam AI and write a WAV file."""
    t0 = time.perf_counter()
    output_path = Config.AUDIO_FOLDER / output_file
    clean_text  = text.strip()
    chunks      = _chunk_text(clean_text, SARVAM_TTS_MAX_CHARS)

    logger.info(
        "[TTS] provider=SarvamAI  lang=%s  model=%s  speaker=%s  chunks=%d  chars=%d  output=%s",
        language, SARVAM_TTS_MODEL, SARVAM_SPEAKER, len(chunks), len(clean_text), output_file,
    )

    try:
        raw_audio_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            part_bytes = _sarvam_tts_chunk(chunk, language)
            if len(part_bytes) < 44:
                logger.error("[TTS] [Sarvam] Chunk %d/%d returned suspiciously small audio (%d bytes)",
                             i + 1, len(chunks), len(part_bytes))
                return None
            raw_audio_parts.append(part_bytes)

        final_audio = _concat_wav_chunks(raw_audio_parts)
        output_path.write_bytes(final_audio)

        if not output_path.exists() or output_path.stat().st_size < 200:
            logger.error("[TTS] [Sarvam] Output file missing or too small: %s", output_path)
            return None

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("[TTS] [Sarvam] ✓ done in %dms  size_kb=%d  file=%s",
                     elapsed_ms, output_path.stat().st_size // 1024, output_file)
        return str(output_path)

    except RuntimeError:
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception("[TTS] [Sarvam] ✗ Unexpected error after %dms: %s", elapsed_ms, e)
        raise RuntimeError(f"Sarvam TTS error: {e}") from e

# ── Groq client singleton ─────────────────────────────────────────────────────
_groq_client: "Groq | None" = None


def _get_groq_client() -> "Groq":
   
    global _groq_client
    if _groq_client is None:
        if not _GROQ_SDK_AVAILABLE:
            raise RuntimeError(
                "groq SDK is not installed. Run: pip install groq>=0.9.0"
            )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to backend/.env:  GROQ_API_KEY=gsk_..."
            )
        _groq_client = Groq(api_key=api_key)
        logger.info(
            "[TTS] ✓ Groq client ready  provider=GroqOrpheus  model=%s  voice=%s  fmt=%s",
            GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT
        )
    return _groq_client


# ── Debug / health report ─────────────────────────────────────────────────────

def get_tts_debug_report() -> dict:
    """
    Return a dict describing the active TTS configuration.
    Used by /api/tts/health and logged at startup.

    Returns:
        {
          "provider":       "GroqOrpheus",
          "sdk_installed":  True,
          "api_key_set":    True,
          "model":          "canopylabs/orpheus-v1-english",
          "voice":          "leah",
          "output_format":  "wav",
          "client_ready":   True,
          "available_voices": [...],
          "status":         "ok" | "degraded" | "error",
          "error":          null | "...",
        }
    """
    api_key_set  = bool(os.getenv("GROQ_API_KEY", "").strip())
    client_ready = _groq_client is not None

    status = "ok"
    error  = None
    if not _GROQ_SDK_AVAILABLE:
        status = "error"
        error  = "groq SDK not installed — run: pip install groq>=0.9.0"
    elif not api_key_set:
        status = "error"
        error  = "GROQ_API_KEY not set in environment / .env"
    elif not client_ready:
        status = "degraded"
        error  = "Client not yet initialised (will init on first TTS call)"

    return {
        "provider":         "GroqOrpheus",
        "sdk_installed":    _GROQ_SDK_AVAILABLE,
        "api_key_set":      api_key_set,
        "model":            GROQ_TTS_MODEL,
        "voice":            GROQ_TTS_VOICE,
        "output_format":    GROQ_TTS_FORMAT,
        "client_ready":     client_ready,
        "available_voices": ["autumn", "diana", "hannah", "austin", "daniel", "troy"],
        "status":           status,
        "error":            error,
        "multilingual": {
            "provider":      "SarvamAI",
            "languages":     ["hi", "mr"],
            "api_key_set":   bool(SARVAM_API_KEY),
            "model":         SARVAM_TTS_MODEL,
            "speaker":       SARVAM_SPEAKER,
            "status":        "ok" if SARVAM_API_KEY else "error",
            "error":         None if SARVAM_API_KEY else (
                "SARVAM_API_KEY not set — Hindi/Marathi TTS will fail. "
                "Get a free-tier key at https://www.sarvam.ai"
            ),
        },
    }


# ── Text chunking helpers ─────────────────────────────────────────────────────

import re
import struct


def _chunk_text(text: str, max_chars: int) -> list[str]:
    
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        
        window = remaining[:max_chars]
        # Find last sentence-ending punctuation in window
        match = None
        for pattern in [r'[.!?]\s', r'[,;]\s', r'\s']:
            matches = list(re.finditer(pattern, window))
            if matches:
                match = matches[-1]
                break

        if match:
            cut = match.start() + 1  # include the punctuation
        else:
            cut = max_chars 

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining.strip())

    return [c for c in chunks if c]


def _find_wav_data_chunk(wav_bytes: bytes) -> tuple[int, int]:
    
    if len(wav_bytes) < 12:
        raise ValueError("WAV too short to contain a RIFF header")
    if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file (magic={wav_bytes[0:4]!r})")

    # Walk RIFF chunks starting after the 12-byte RIFF header.
    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id   = wav_bytes[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        if chunk_id == b"data":
            return offset + 8, chunk_size          # found it
        # Chunks are padded to even byte boundaries
        offset += 8 + chunk_size + (chunk_size & 1)

    raise ValueError("WAV file contains no 'data' chunk")


def _concat_wav_chunks(wav_parts: list[bytes]) -> bytes:
    
    if len(wav_parts) == 1:
        return wav_parts[0]

    # Extract raw PCM samples from every chunk by parsing RIFF properly.
    pcm_parts: list[bytes] = []
    for i, part in enumerate(wav_parts):
        try:
            data_start, data_size = _find_wav_data_chunk(part)
        except ValueError as exc:
            logger.error("[TTS] _concat_wav_chunks: chunk %d/%d invalid WAV — %s",
                         i + 1, len(wav_parts), exc)
            raise
        pcm_parts.append(part[data_start: data_start + data_size])

    total_pcm      = b"".join(pcm_parts)
    total_pcm_size = len(total_pcm)

    first = wav_parts[0]

    # Locate the fmt chunk in the first part to copy its payload verbatim.
    fmt_payload = b""
    offset = 12
    while offset + 8 <= len(first):
        cid  = first[offset:offset + 4]
        csz  = struct.unpack_from("<I", first, offset + 4)[0]
        if cid == b"fmt ":
            fmt_payload = first[offset + 8: offset + 8 + csz]
            break
        offset += 8 + csz + (csz & 1)

    if not fmt_payload:
        raise ValueError("WAV first chunk has no 'fmt ' sub-chunk")

    # Assemble: RIFF header + fmt chunk + data chunk
    fmt_chunk  = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    data_chunk = b"data" + struct.pack("<I", total_pcm_size) + total_pcm
    riff_body  = b"WAVE" + fmt_chunk + data_chunk
    riff_header = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body

    return riff_header


# ── Public reusable function ──────────────────────────────────────────────────

def text_to_speech(text: str, output_file: str, language: str = "en") -> "str | None":
   
    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not text or not text.strip():
        logger.error("[TTS] Rejected empty text — returning None")
        return None

    # ── Route to Sarvam AI for Hindi / Marathi ────────────────────────────────
    if language in SARVAM_LANG_CODES:
        return _sarvam_text_to_speech(text, output_file, language)

    output_path = Config.AUDIO_FOLDER / output_file

   
    logger.info(
        "[TTS_PROVIDER] provider=GroqOrpheus  model=%s  voice=%s  fmt=%s  "
        "chars=%d  output=%s",
        GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT,
        len(text.strip()), output_file,
    )

    # ── API call ──────────────────────────────────────────────────────────────
    try:
        client = _get_groq_client()
        clean_text = text.strip()

        chunks = _chunk_text(clean_text, GROQ_TTS_MAX_CHARS)
        logger.info("[TTS] Synthesising %d chunk(s) for %d chars", len(chunks), len(clean_text))

        raw_audio_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            response = client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                input=chunk,
                response_format=GROQ_TTS_FORMAT,
            )
            part_bytes = response.read()
            if len(part_bytes) < 44:  # WAV header alone is 44 bytes
                logger.error("[TTS] Chunk %d/%d returned suspiciously small audio (%d bytes)",
                             i + 1, len(chunks), len(part_bytes))
                return None
            raw_audio_parts.append(part_bytes)
            logger.debug("[TTS] Chunk %d/%d done  bytes=%d", i + 1, len(chunks), len(part_bytes))

        # ── Concatenate WAV chunks ────────────────────────────────────────────
        
        final_audio = _concat_wav_chunks(raw_audio_parts)

        output_path.write_bytes(final_audio)

        # ── Post-write validation ─────────────────────────────────────────────
        if not output_path.exists():
            logger.error("[TTS] write_to_file() returned but file missing: %s", output_path)
            return None

        file_size  = output_path.stat().st_size
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if file_size < 200:
            # A real WAV with even 50ms of audio is ~4 KB. Under 200 bytes = corrupted.
            logger.error(
                "[TTS] File too small (%d bytes) — Groq returned bad audio. "
                "file=%s  elapsed=%dms", file_size, output_file, elapsed_ms
            )
            try:
                output_path.unlink()
            except OSError:
                pass
            return None

        # ── Debug report on every successful synthesis ────────────────────────
        logger.info(
            "[TTS_REPORT] ✓  provider=GroqOrpheus  voice=%s  model=%s  "
            "elapsed_ms=%d  size_kb=%d  format=%s  file=%s  "
            "timestamp=%s",
            GROQ_TTS_VOICE, GROQ_TTS_MODEL,
            elapsed_ms, file_size // 1024, GROQ_TTS_FORMAT,
            output_file,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return str(output_path)

    # ── Structured error handling ─────────────────────────────────────────────

    except RuntimeError as e:
        # Missing SDK or API key — raised by _get_groq_client() — re-raise so caller sees it
        logger.critical("[TTS] ✗ Configuration error — %s", e)
        raise

    except AuthenticationError as e:
        msg = f"401 Authentication failed — GROQ_API_KEY is invalid or revoked. Details: {e}"
        logger.error("[TTS] ✗ %s  key_prefix=%s...", msg, os.getenv("GROQ_API_KEY", "")[:8])
        raise RuntimeError(msg) from e

    except APIConnectionError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"Network error after {elapsed_ms}ms — cannot reach api.groq.com. Details: {e}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except APIStatusError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # e.body contains the full JSON response from Groq — most useful for debugging
        body = getattr(e, "body", None) or getattr(e, "message", str(e))
        msg = f"Groq API HTTP {e.status_code} after {elapsed_ms}ms — {body}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except OSError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"File write error after {elapsed_ms}ms — path={output_path} — {e}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except RuntimeError:
        raise  # already formatted above (e.g. missing SDK / API key)

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"Unexpected error after {elapsed_ms}ms — '{text[:40]}...' — {type(e).__name__}: {e}"
        logger.exception("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e




class TextToSpeechService:
    """Thin wrapper around text_to_speech() preserving the class interface."""

    def generate_speech(self, text: str, output_filename: str, language: str = "en") -> str:
        """
        Generate TTS audio. Routes to Sarvam AI for Hindi/Marathi, Groq Orpheus
        for English. Return the audio file path.
        Raises RuntimeError with the actual provider error message on failure.
        """
        return text_to_speech(text, output_filename, language=language)


# ── Module-level singleton ────────────────────────────────────────────────────
tts_service = TextToSpeechService()


# ── Startup self-test ─────────────────────────────────────────────────────────
def _startup_tts_check() -> None:
    """
    Run at module import time.  Logs the TTS configuration and catches
    common misconfigurations (missing SDK, missing API key) immediately
    rather than silently failing on the first interview question.
    """
    report = get_tts_debug_report()
    if report["status"] == "ok" or report["status"] == "degraded":
        logger.info(
            "[TTS_STARTUP] provider=%s  model=%s  voice=%s  fmt=%s  "
            "sdk=%s  key_set=%s  status=%s",
            report["provider"], report["model"], report["voice"],
            report["output_format"], report["sdk_installed"],
            report["api_key_set"], report["status"],
        )
        if report["status"] == "degraded":
            logger.warning("[TTS_STARTUP] %s", report["error"])
    else:
        logger.critical(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  [TTS_STARTUP] TTS IS NOT FUNCTIONAL                        ║\n"
            "║  Status: %-51s║\n"
            "║  Error:  %-51s║\n"
            "║  All interview questions will use browser speechSynthesis.  ║\n"
            "║  Fix the issue above and restart the server.                ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            report["status"], (report["error"] or "")[:51],
        )


_startup_tts_check()


if _GROQ_SDK_AVAILABLE and os.getenv("GROQ_API_KEY", "").strip():
    try:
        _get_groq_client()
        logger.info("[TTS] ✓ Groq client pre-initialized at startup")
    except Exception as _e:
        logger.critical("[TTS] ✗ Groq client pre-init failed at startup: %s", _e)
