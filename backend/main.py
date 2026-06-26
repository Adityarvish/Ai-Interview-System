import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import socketio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles  

from config.settings import Config
from config.database import init_db
from config.per_question_models import create_per_question_tables  
from routers.interview import router as interview_router
from routers.per_question_router import router as pq_router         
from sockets.handlers import sio

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger(__name__)

BACKEND_DIR  = Path(__file__).parent.resolve()
FRONTEND_DIR = BACKEND_DIR.parent / 'frontend'



@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    t_boot = time.perf_counter()

    
    def _handle_task_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        logger.error(
            "[ASYNCIO] Unhandled background task exception: %s | %s",
            context.get("message"),
            exc,
            exc_info=exc if exc else False,
        )

    asyncio.get_event_loop().set_exception_handler(_handle_task_exception)

   
    init_db()
    logger.info(f"[BOOT] Database initialised in {int((time.perf_counter() - t_boot)*1000)} ms")

    create_per_question_tables()
    logger.info("[BOOT] per_question_evaluations table ensured")

    # Groq connectivity check
    if not Config.GROQ_API_KEY:
        logger.error(
            "[BOOT] GROQ_API_KEY is not set in backend/.env — "
            "the system will not be able to generate questions or evaluations. "
            "Get your key at https://console.groq.com/keys"
        )
    else:
        from services.llm_service import GroqService
        loop = asyncio.get_event_loop()

        def _check():
            return GroqService().check_connection()

        ok = await loop.run_in_executor(None, _check)
        if ok:
            logger.info(
                "[BOOT] Groq API reachable ✓  primary=%s  fallback=%s",
                Config.GROQ_PRIMARY_MODEL, Config.GROQ_FALLBACK_MODEL,
            )
        else:
            logger.warning(
                "[BOOT] Groq API probe failed — check GROQ_API_KEY in backend/.env "
                "and ensure https://api.groq.com is reachable from this host."
            )

    # Background warmup task
    async def _warmup_task():
        loop = asyncio.get_event_loop()

        def _warmup():
            from services.warm_cache import warmup_all, start_audio_cleanup_loop
            from services.text_to_speech import get_tts_debug_report
            warmup_all()
            start_audio_cleanup_loop(Config.AUDIO_FOLDER)
            report = get_tts_debug_report()
            if report["status"] != "ok":
                logger.critical(
                    "[BOOT] TTS health check FAILED — status=%s  error=%s\n"
                    "       Interviews will use browser speechSynthesis as fallback.\n"
                    "       Fix: %s",
                    report["status"], report["error"],
                    "pip install groq>=0.9.0  and  set GROQ_API_KEY in backend/.env",
                )
            else:
                logger.info(
                    "[BOOT] TTS health check OK — provider=%s  model=%s  voice=%s",
                    report["provider"], report["model"], report["voice"],
                )

        await loop.run_in_executor(None, _warmup)

    asyncio.create_task(_warmup_task())
    logger.info("[BOOT] Background model warm-up + cleanup loop spawned")
    logger.info(f"[BOOT] Frontend dir: {FRONTEND_DIR}")
    logger.info(f"[BOOT] API docs: http://{Config.HOST}:{Config.PORT}/docs")

    yield  # ── application runs here ─────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[SHUTDOWN] Application shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Voice Interview System",
    version="8.2.0",
    description="AI-powered voice interview platform using Groq LLM, STT, and TTS.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # FIX #3: lifespan passed here instead of @app.on_event
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(interview_router)
app.include_router(pq_router)        

# ── Static files ──────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "AI Interview System"})


@app.get("/{path:path}")
async def spa_fallback(path: str):
    if not path.startswith("api/") and not path.startswith("socket.io"):
        full = FRONTEND_DIR / path
        if full.is_file():
            return FileResponse(str(full))
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
    return JSONResponse({"success": False, "error": "Not found"}, status_code=404)


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if request.url.path.startswith('/api/'):
        return JSONResponse({"success": False, "error": "Not found"}, status_code=404)
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"success": False, "error": "Not found"}, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.exception(f"500: {exc}")
    return JSONResponse({"success": False, "error": "Internal server error"}, status_code=500)


# ── Mount Socket.IO ASGI app ──────────────────────────────────────────────────
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path="/socket.io",
)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:socket_app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level="info",
    )