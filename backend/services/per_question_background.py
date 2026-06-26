from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from evaluator.pipeline import PerQuestionPipeline
from services.per_question_db_service import PerQuestionDBService

logger = logging.getLogger(__name__)

# ── Per-interview concurrency limiter ─────────────────────────────────────────

_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}
_MAX_CONCURRENT = 3
_SEM_LOCK: Optional[asyncio.Lock] = None


async def _get_sem(interview_id: str) -> asyncio.Semaphore:
    global _SEM_LOCK
    if _SEM_LOCK is None:
        _SEM_LOCK = asyncio.Lock()
    async with _SEM_LOCK:
        if interview_id not in _SEMAPHORES:
            _SEMAPHORES[interview_id] = asyncio.Semaphore(_MAX_CONCURRENT)
        return _SEMAPHORES[interview_id]


# ── Main background coroutine ─────────────────────────────────────────────────

async def trigger_per_question_eval(
    interview_id:    str,
    question_id:     str,
    question:        str,
    answer:          str,
    job_description: str,
    topic:           str = "",
    difficulty:      str = "medium",
    resume_context:  str = "",
    question_type:   str = "",
) -> None:
    """
    Run the 16-phase pipeline in the background.
    Persists per-question result then updates the interview aggregate.
    Never raises — all exceptions are logged.

    question_type: pass the interview stage name (e.g. "behavioral", "technical",
                   "general"). When "behavioral", the STAR phase is activated.
    """
    t_start = time.perf_counter()
    sem     = await _get_sem(interview_id)

    try:
        async with sem:
            logger.info(
                f"[BG] Start: interview={interview_id} Q{question_id} "
                f"topic={topic or 'General'} diff={difficulty}"
            )

            pipeline = PerQuestionPipeline(
                question_id     = question_id,
                interview_id    = interview_id,
                question        = question,
                answer          = answer,
                job_description = job_description,
                topic           = topic,
                difficulty      = difficulty,
                resume_context  = resume_context,
                question_type   = question_type,
            )
            result = await pipeline.run()

            db = PerQuestionDBService()
            row_id = await db.save(result)
            agg    = await db.update_aggregate(interview_id)

            elapsed = int((time.perf_counter() - t_start) * 1000)
            logger.info(
                f"[BG] Done: interview={interview_id} Q{question_id} "
                f"score={result.question_score:.1f} weighted={result.weighted_score:.1f} "
                f"decision={agg.get('decision')} overall={agg.get('overall_score')} "
                f"elapsed={elapsed}ms row_id={row_id}"
            )

    except asyncio.CancelledError:
        logger.warning(f"[BG] Cancelled: interview={interview_id} Q{question_id}")

    except Exception as exc:
        elapsed = int((time.perf_counter() - t_start) * 1000)
        logger.error(
            f"[BG] Failed: interview={interview_id} Q{question_id} "
            f"after {elapsed}ms — {exc}",
            exc_info=True,
        )


# ── Cleanup ───────────────────────────────────────────────────────────────────

def release_semaphore(interview_id: str) -> None:
    """Call at interview end to free semaphore memory."""
    _SEMAPHORES.pop(interview_id, None)
    logger.debug(f"[BG] Semaphore released for {interview_id}")
