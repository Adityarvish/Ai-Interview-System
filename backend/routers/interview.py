import asyncio
import base64
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.responses import JSONResponse

from config.settings import Config
from services.database_service import DatabaseService
from services.interview_engine import InterviewEngine
from services.per_question_db_service import generate_interview_summary
from services.text_to_speech import tts_service
from schemas.interview import (
    HealthResponse, StartInterviewResponse, FinalReportResponse,
    TTSResponse, TTSRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_RESUME_EXT = {'pdf', 'txt'}

# ── Module-level database instance ────────────────────────────────────────────
_db = DatabaseService()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delete_file_safe(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _ext_ok(filename: str, allowed: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def _safe_filename(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace('..', '_')


def _tts_b64(text: str, interview_id: str, language: str = "en"):
    
    from services.text_to_speech import GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT

    t0       = time.perf_counter()
    out_name = f"ai_{interview_id}_{uuid.uuid4().hex}.wav"

    logger.info(
        "[TTS] [%s] lang=%s  model=%s  voice=%s  fmt=%s  chars=%d",
        interview_id, language, GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT, len(text)
    )

    try:
        path = tts_service.generate_speech(text, out_name, language=language)

        if path is None:
            elapsed = int((time.perf_counter() - t0) * 1000)
            logger.error("[TTS_FAILED] [%s] after %dms — generate_speech() returned None.",
                         interview_id, elapsed)
            return None, None

        suffix = Path(path).suffix.lstrip('.')
        with open(path, 'rb') as f:
            audio_bytes = f.read()

        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        timestamp  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        b64        = base64.b64encode(audio_bytes).decode('ascii')
        size_bytes = len(audio_bytes)
        elapsed    = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "[TTS_FILE] [%s]  model=%s  voice=%s  file=%s  size=%d  ts=%s  hash=%s",
            interview_id, GROQ_TTS_MODEL, GROQ_TTS_VOICE,
            out_name, size_bytes, timestamp, audio_hash
        )
        logger.info(
            "[TTS_COMPLETED] [%s] elapsed_ms=%d  size_kb=%d  fmt=%s  voice=%s  hash=%s",
            interview_id, elapsed, size_bytes // 1024, suffix, GROQ_TTS_VOICE, audio_hash
        )
        _delete_file_safe(path)
        return b64, suffix

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.exception(
            "[TTS_FAILED] [%s] after %dms — %s\n"
            "             Frontend will fall back to browser speechSynthesis.",
            interview_id, elapsed, e
        )
        return None, None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/api/health", response_model=HealthResponse)
async def health():
    return {"success": True, "status": "healthy", "service": "AI Interview System v8.2"}


@router.post("/api/start-interview")
async def start_interview(
    request: Request,
    candidate_name: str = Form(...),
    job_description: str = Form(...),
    resume: UploadFile = File(...),
    language: str = Form("en"),
):
    
    t_req = time.perf_counter()
    try:
        logger.info("[START] /api/start-interview received")

        name = candidate_name.strip()
        jd   = job_description.strip()
        lang = (language or 'en').strip().lower()
        if lang not in ('en', 'hi', 'mr'):
            lang = 'en'

        if not name or not jd:
            raise HTTPException(status_code=400,
                                detail="candidate_name and job_description are required")
        if not _ext_ok(resume.filename, ALLOWED_RESUME_EXT):
            raise HTTPException(status_code=400, detail="Resume must be PDF or TXT")

        fname       = _safe_filename(resume.filename)
        resume_path = Config.RESUME_FOLDER / f"{uuid.uuid4().hex}_{fname}"

        contents = await resume.read()
        resume_path.write_bytes(contents)
        logger.info(
            f"[START] Resume saved: {resume_path.name} "
            f"({resume_path.stat().st_size} bytes) "
            f"in {int((time.perf_counter()-t_req)*1000)} ms"
        )

        loop = asyncio.get_event_loop()

        # ── Engine initialisation ─────────────────────────────────────────────
        def _init_engine():
            t_init       = time.perf_counter()
            interview_id = str(uuid.uuid4())
            engine       = InterviewEngine()
            r            = engine.initialize_interview(name, str(resume_path), jd, language=lang)
            if not r['success']:
                return None, None, r
            logger.info(f"[START] Engine init in {int((time.perf_counter()-t_init)*1000)} ms")
            return interview_id, engine, r

        interview_id, engine, r = await loop.run_in_executor(None, _init_engine)

        if engine is None:
            logger.error(f"[START] Engine init failed: {r}")
            raise HTTPException(status_code=500, detail=r.get('error', 'Engine init failed'))

        from sockets.handlers import ENGINES
        ENGINES[interview_id] = engine

        
        # ── Generate first question ───────────────────────────────────────────
        def _gen_first_q():
            t_q     = time.perf_counter()
            first_q = engine.generate_first_question()
            logger.info(
                f"[START] First question in {int((time.perf_counter()-t_q)*1000)} ms: "
                f"'{first_q[:80]}'"
            )
            return first_q

        first_q  = await loop.run_in_executor(None, _gen_first_q)
        total_ms = int((time.perf_counter() - t_req) * 1000)
        logger.info(f"[START] Response in {total_ms} ms — interview_id={interview_id}")

        return {
            "success":        True,
            "interview_id":   interview_id,
            "candidate_id":   interview_id,
            "first_question": first_q,
            "timing":         {"startup_ms": total_ms},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"start_interview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/final-report/{interview_id}")
async def final_report(interview_id: str):
    """Return the final evaluation for a completed interview."""
    loop = asyncio.get_event_loop()

    try:
        # Primary: return the already-persisted evaluation
        doc = await loop.run_in_executor(None, _db.get_final_evaluation, interview_id)

        if not doc:
            # Fail-safe: evaluation not yet written — rebuild from per-question aggregate
            logger.warning(f"[REPORT] No evaluation for {interview_id} — rebuilding")
            evaluation = await generate_interview_summary(
                interview_id     = interview_id,
                candidate_name   = "Candidate",
                interview_status = "completed",
            )
            doc = await loop.run_in_executor(
                None, _db.save_final_evaluation, interview_id, evaluation
            )
            logger.info(f"[REPORT] Fail-safe evaluation generated for {interview_id}")

        if doc is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return {"success": True, "evaluation": doc}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"final_report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/tts")
async def api_tts(body: TTSRequest):
    
    text         = (body.text or '').strip()
    interview_id = (body.interview_id or 'ondemand')

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Look up the candidate's chosen language from the active engine so the
    # frontend doesn't need to pass it on every request.
    language = (body.language or '').strip().lower()
    if not language:
        from sockets.handlers import ENGINES
        engine = ENGINES.get(interview_id)
        language = (engine.candidate_info.get('language', 'en') if engine else 'en')

    loop = asyncio.get_event_loop()
    audio_b64, audio_fmt = await loop.run_in_executor(None, _tts_b64, text, interview_id, language)

    if not audio_b64:
        raise HTTPException(status_code=500,
                            detail="TTS failed — check GROQ_API_KEY in backend/.env")

    return {
        "success":      True,
        "audio":        audio_b64,
        "audio_format": audio_fmt or 'wav',
    }


@router.get("/api/tts/health")
async def tts_health():
    
    from services.text_to_speech import get_tts_debug_report
    report      = get_tts_debug_report()
    status_code = 200 if report["status"] == "ok" else 503
    return JSONResponse(content=report, status_code=status_code)


@router.get("/debug-tts")
async def debug_tts(text: str = Query(default="This is a test sentence.")):
   
    from services.text_to_speech import (
        tts_service as _tts_svc,
        GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT,
        get_tts_debug_report,
    )

    out_name  = f"debug_{uuid.uuid4().hex}.wav"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info("[DEBUG_TTS] Generating audio for text=%r", text[:80])

    loop = asyncio.get_event_loop()

    def _gen():
        t0   = time.perf_counter()
        path = _tts_svc.generate_speech(text, out_name)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        with open(path, 'rb') as f:
            raw = f.read()
        return path, raw, elapsed_ms

    try:
        path, raw, elapsed_ms = await loop.run_in_executor(None, _gen)

        file_hash = hashlib.md5(raw).hexdigest()
        file_size = len(raw)

        logger.info(
            "[DEBUG_TTS] file=%s  size=%d  hash=%s  voice=%s  model=%s  elapsed_ms=%d",
            out_name, file_size, file_hash, GROQ_TTS_VOICE, GROQ_TTS_MODEL, elapsed_ms
        )

        return {
            "success":    True,
            "filename":   out_name,
            "file_size":  file_size,
            "hash":       file_hash,
            "provider":   "GroqOrpheus",
            "voice":      GROQ_TTS_VOICE,
            "model":      GROQ_TTS_MODEL,
            "format":     GROQ_TTS_FORMAT,
            "elapsed_ms": elapsed_ms,
            "timestamp":  timestamp,
            "text":       text,
            "file_path":  path,
        }
    except Exception as e:
        logger.exception("[DEBUG_TTS] Failed: %s", e)
        from services.text_to_speech import get_tts_debug_report
        report = get_tts_debug_report()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "tts_health": report}
        )