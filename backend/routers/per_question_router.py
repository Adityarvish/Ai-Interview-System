from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field, validator

from evaluator.pipeline import PerQuestionPipeline
from services.per_question_background import trigger_per_question_eval
from services.per_question_db_service import PerQuestionDBService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pq-eval", tags=["per-question-eval"])
_db    = PerQuestionDBService()


# ── Request / Response models ─────────────────────────────────────────────────

class SubmitAnswerRequest(BaseModel):
    interview_id:    str
    question_id:     str
    question:        str
    answer:          str
    job_description: str
    topic:           Optional[str] = ""
    difficulty:      Optional[str] = Field(default="medium", pattern="^(easy|medium|hard)$")
    resume_context:  Optional[str] = ""
    question_type:   Optional[str] = Field(default="general", pattern="^(technical|behavioral|general)$")


class SubmitAnswerResponse(BaseModel):
    success:      bool
    message:      str
    interview_id: str
    question_id:  str
    status:       str   # "queued"


class SyncEvalResponse(BaseModel):
    success:         bool
    interview_id:    str
    question_id:     str
    topic:           str
    difficulty:      str
    question_type:   str
    question_score:  float
    weighted_score:  float
    status:          str
    breakdown:       dict
    star:            dict   # score, components, missing — populated for behavioral questions only
    feedback:        dict
    ideal_answer:    str
    key_concepts:    List[str]
    elapsed_ms:      int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=SubmitAnswerResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_answer(body: SubmitAnswerRequest):
    
    asyncio.create_task(
        trigger_per_question_eval(
            interview_id    = body.interview_id,
            question_id     = body.question_id,
            question        = body.question,
            answer          = body.answer,
            job_description = body.job_description,
            topic           = body.topic or "",
            difficulty      = body.difficulty or "medium",
            resume_context  = body.resume_context or "",
            question_type   = body.question_type or "general",
        )
    )
    logger.info(
        f"[ROUTER] Queued evaluation: interview={body.interview_id} "
        f"Q{body.question_id} topic={body.topic} type={body.question_type}"
    )
    return SubmitAnswerResponse(
        success      = True,
        message      = "Answer received. Evaluation running in background.",
        interview_id = body.interview_id,
        question_id  = body.question_id,
        status       = "queued",
    )


@router.post("/evaluate-sync", response_model=SyncEvalResponse)
async def evaluate_sync(body: SubmitAnswerRequest):
    """
    Synchronous evaluation — waits for full result before responding.
    Use for testing, admin re-scoring, or low-latency single-answer flows.
    """
    t0 = time.perf_counter()

    pipeline = PerQuestionPipeline(
        question_id     = body.question_id,
        interview_id    = body.interview_id,
        question        = body.question,
        answer          = body.answer,
        job_description = body.job_description,
        topic           = body.topic or "",
        difficulty      = body.difficulty or "medium",
        resume_context  = body.resume_context or "",
    )

    try:
        result = await pipeline.run()
    except Exception as exc:
        logger.exception(f"[ROUTER] evaluate-sync failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist and update aggregate
    db = PerQuestionDBService()
    await db.save(result)
    agg = await db.update_aggregate(body.interview_id)

    elapsed = int((time.perf_counter() - t0) * 1000)

    return SyncEvalResponse(
        success        = True,
        interview_id   = result.interview_id,
        question_id    = str(result.question_id),
        topic          = result.topic,
        difficulty     = result.difficulty,
        question_type  = result.question_type,
        question_score = result.question_score,
        weighted_score = result.weighted_score,
        status         = result.status,
        breakdown = {
            "correctness": result.correctness_score,
            "relevance":   result.relevance_score,
            "depth":       result.depth_score,
            "coverage":    result.coverage_score,
            "clarity":     result.clarity_score,
            "anti_gaming_penalty": result.anti_gaming_penalty,
        },
        star = {
            "score":      result.star_score,
            "components": result.star_components,
            "missing":    result.star_missing,
        },
        feedback = {
            "strengths":               result.strengths,
            "weaknesses":              result.weaknesses,
            "improvement_suggestions": result.improvement_suggestions,
        },
        ideal_answer = result.ideal_answer,
        key_concepts = result.key_concepts,
        elapsed_ms   = elapsed,
    )


@router.get("/{interview_id}/questions")
async def get_questions(interview_id: str):
    """All per-question results for an interview."""
    questions = await _db.get_question_results(interview_id)
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No question evaluations found for interview {interview_id}. "
                   "Evaluation may still be running — retry in a moment.",
        )
    return {
        "success":      True,
        "interview_id": interview_id,
        "count":        len(questions),
        "questions":    questions,
    }


@router.get("/{interview_id}/question/{question_id}")
async def get_single_question(interview_id: str, question_id: str):
    """Full result for one question. Returns 202 if still processing."""
    questions = await _db.get_question_results(interview_id)
    match = next(
        (q for q in questions if str(q["question_id"]) == str(question_id)),
        None,
    )
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"Q{question_id} not found for interview {interview_id}. "
                   "Still processing or invalid ID.",
        )
    # Return 202 if the row exists but evaluation hasn't finished yet
    http_status = 202 if match.get("status") == "processing" else 200
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_status,
        content={"success": True, "evaluation": match},
    )


@router.get("/{interview_id}/aggregate")
async def get_aggregate(interview_id: str):
    """
    Returns Phases 13-15 output: overall_score, skill_scores, decision.
    Updates in near-real-time as question evaluations complete.
    """
    agg = await _db.get_aggregate(interview_id)
    if not agg:
        raise HTTPException(
            status_code=404,
            detail=f"No aggregate found for interview {interview_id}. "
                   "No questions evaluated yet.",
        )
    return {"success": True, "aggregate": agg}


@router.get("/{interview_id}/final-report")
async def get_final_report(interview_id: str):
    """
    Full combined report: aggregate + all per-question breakdowns.
    Use this endpoint for the results page.
    """
    agg, questions = await asyncio.gather(
        _db.get_aggregate(interview_id),
        _db.get_question_results(interview_id),
        return_exceptions=True,
    )

    if isinstance(agg, Exception) or not agg:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation data for interview {interview_id}.",
        )
    if isinstance(questions, Exception):
        questions = []

    return {
        "success":      True,
        "interview_id": interview_id,
        "aggregate": agg,
        "questions": questions,
        "summary": {
            "overall_score":        agg["overall_score"],
            "decision":             agg["decision"],
            "skill_scores":         agg["skill_scores"],
            "questions_evaluated":  agg["questions_evaluated"],
            "questions_filtered":   agg["questions_filtered"],
            "top_strengths":        (agg.get("all_strengths")  or [])[:5],
            "top_weaknesses":       (agg.get("all_weaknesses") or [])[:5],
            "top_suggestions":      (agg.get("all_suggestions") or [])[:5],
        },
    }
