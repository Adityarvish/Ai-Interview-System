"""
services/per_question_db_service.py — Async database layer for per-question evaluations.

Tables used:
  per_question_evaluations  — one row per Q&A pair

interview_aggregates has been removed. Aggregate data (overall_score,
skill_scores, decision) is computed on-the-fly from per_question_evaluations
by the PerQuestionDBService.update_aggregate() method, which now returns a
plain dict instead of persisting to a separate table.

generate_interview_summary() builds the evaluations row from per-question
data — this is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.database import SessionLocal
from config.per_question_models import PerQuestionEvaluation
from evaluator.pipeline import PerQuestionResult
from llm import client as llm_client
from scoring.aggregator import (
    compute_skill_scores,
    make_decision,
)

logger = logging.getLogger(__name__)


# ── Sync helpers (called via run_in_executor) ─────────────────────────────────

def _save_sync(result: PerQuestionResult) -> int:
    """Upsert a PerQuestionEvaluation row. Returns the row PK."""
    session = SessionLocal()
    try:
        row = (
            session.query(PerQuestionEvaluation)
            .filter_by(
                interview_id=result.interview_id,
                question_id=str(result.question_id),
            )
            .first()
        )
        if not row:
            row = PerQuestionEvaluation(
                interview_id=result.interview_id,
                question_id=str(result.question_id),
            )
            session.add(row)

        row.topic               = result.topic
        row.question_type       = result.question_type
        row.difficulty          = result.difficulty
        row.question_text       = result.question
        row.answer_text         = result.answer
        row.ideal_answer        = result.ideal_answer
        row.key_concepts        = result.key_concepts

        row.relevance_score     = result.relevance_score
        row.correctness_score   = result.correctness_score
        row.depth_score         = result.depth_score
        row.coverage_score      = result.coverage_score
        row.clarity_score       = result.clarity_score
        row.anti_gaming_penalty = result.anti_gaming_penalty

        row.star_score          = result.star_score
        row.star_components     = result.star_components
        row.star_missing        = result.star_missing

        row.question_score      = result.question_score
        row.weighted_score      = result.weighted_score

        row.depth_reason        = result.depth_reason
        row.clarity_reason      = result.clarity_reason

        row.strengths               = result.strengths
        row.weaknesses              = result.weaknesses
        row.improvement_suggestions = result.improvement_suggestions

        row.status              = result.status
        row.filter_reason       = result.filter_reason
        row.eval_ms             = result.total_elapsed_ms
        row.evaluated_at        = datetime.now(timezone.utc)

        session.commit()
        session.refresh(row)
        return row.id

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _compute_aggregate_sync(interview_id: str) -> Dict:
    """
    Compute aggregate from all PerQuestionEvaluation rows for this interview.
    Returns a plain dict — nothing is persisted to a separate table.

    Phases 13, 14, 15 run here.
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(PerQuestionEvaluation)
            .filter_by(interview_id=interview_id)
            .all()
        )

        processed = [r for r in rows if r.status == "processed"]
        filtered  = [r for r in rows if r.status == "filtered"]

        # Phase 13: aggregate (filtered questions count as 0 in denominator)
        total_questions = len(rows)
        if total_questions == 0:
            overall = 0.0
        else:
            score_sum = sum(r.weighted_score for r in processed)
            overall   = round(min(100.0, max(0.0, score_sum / total_questions)), 2)

        # Phase 14: skill map — exclude generic stage-name topics
        _STAGE_NAMES = {"greeting", "introduction", "closing",
                        "general", "technical", "behavioral"}
        q_dicts = [
            {"topic": r.topic or "General", "final_score": r.weighted_score}
            for r in processed
            if (r.topic or "General").lower() not in _STAGE_NAMES
        ]
        if not q_dicts:
            q_dicts = [
                {"topic": r.topic or "General", "final_score": r.weighted_score}
                for r in processed
            ]
        skill_map = compute_skill_scores(q_dicts)

        # Phase 15: decision
        decision = make_decision(overall)

        # Collected feedback
        all_strengths   = [s for r in processed for s in (r.strengths   or [])]
        all_weaknesses  = [w for r in processed for w in (r.weaknesses  or [])]
        all_suggestions = [s for r in processed for s in (r.improvement_suggestions or [])]

        return {
            "interview_id":        interview_id,
            "overall_score":       overall,
            "skill_scores":        skill_map,
            "decision":            decision,
            "questions_evaluated": len(processed),
            "questions_filtered":  len(filtered),
            "all_strengths":       all_strengths,
            "all_weaknesses":      all_weaknesses,
            "all_suggestions":     all_suggestions,
            "last_updated":        datetime.now(timezone.utc).isoformat(),
        }

    finally:
        session.close()


def _get_question_results_sync(interview_id: str) -> List[Dict]:
    session = SessionLocal()
    try:
        rows = (
            session.query(PerQuestionEvaluation)
            .filter_by(interview_id=interview_id)
            .order_by(PerQuestionEvaluation.evaluated_at.asc())
            .all()
        )
        results = []
        for r in rows:
            results.append({
                "id":                    r.id,
                "interview_id":          r.interview_id,
                "question_id":           r.question_id,
                "question_type":         r.question_type,
                "topic":                 r.topic,
                "difficulty":            r.difficulty,
                "question_text":         r.question_text,
                "answer_text":           r.answer_text,
                "ideal_answer":          r.ideal_answer,
                "key_concepts":          r.key_concepts,
                "relevance_score":       r.relevance_score,
                "correctness_score":     r.correctness_score,
                "depth_score":           r.depth_score,
                "coverage_score":        r.coverage_score,
                "clarity_score":         r.clarity_score,
                "anti_gaming_penalty":   r.anti_gaming_penalty,
                "star_score":            r.star_score,
                "star_components":       r.star_components,
                "star_missing":          r.star_missing,
                "question_score":        r.question_score,
                "weighted_score":        r.weighted_score,
                "depth_reason":          r.depth_reason,
                "clarity_reason":        r.clarity_reason,
                "strengths":             r.strengths,
                "weaknesses":            r.weaknesses,
                "improvement_suggestions": r.improvement_suggestions,
                "status":                r.status,
                "filter_reason":         r.filter_reason,
                "eval_ms":               r.eval_ms,
                "evaluated_at":          r.evaluated_at.isoformat() if r.evaluated_at else None,
            })
        return results
    finally:
        session.close()


# ── Async service class ───────────────────────────────────────────────────────

class PerQuestionDBService:

    async def save(self, result: PerQuestionResult) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save_sync, result)

    async def update_aggregate(self, interview_id: str) -> Dict:
        """Compute and return aggregate dict (no DB write — no separate table)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute_aggregate_sync, interview_id)

    async def get_aggregate(self, interview_id: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        agg = await loop.run_in_executor(None, _compute_aggregate_sync, interview_id)
        # Return None if no rows exist yet
        if agg["questions_evaluated"] == 0 and agg["questions_filtered"] == 0:
            # Check whether any rows at all exist
            def _has_rows():
                session = SessionLocal()
                try:
                    return session.query(PerQuestionEvaluation).filter_by(
                        interview_id=interview_id
                    ).count() > 0
                finally:
                    session.close()
            has = await loop.run_in_executor(None, _has_rows)
            if not has:
                return None
        return agg

    async def get_question_results(self, interview_id: str) -> List[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_question_results_sync, interview_id)


# ── generate_interview_summary ────────────────────────────────────────────────
# Builds the evaluations row from per-question aggregate data.
# Called from routers/interview.py and sockets/handlers.py.

async def generate_interview_summary(
    interview_id: str,
    candidate_name: str,
    interview_status: str = "completed",
) -> Dict:
    """
    Build the evaluations payload from per-question data.
    Falls back to LLM-generated summary if aggregate is missing.
    """
    loop = asyncio.get_event_loop()
    agg  = await loop.run_in_executor(None, _compute_aggregate_sync, interview_id)

    overall_score = agg.get("overall_score", 0.0)
    decision      = agg.get("decision", "HOLD")
    skill_scores  = agg.get("skill_scores", {})

    # Map decision → recommendation label used in evaluations table
    rec_map = {"SELECT": "Hire", "HOLD": "Hold", "REJECT": "Reject"}
    recommendation = rec_map.get(decision, "Hold")

    strengths         = (agg.get("all_strengths")   or [])[:10]
    improvement_areas = (agg.get("all_weaknesses")  or [])[:10]

    # Fetch per-question rows once — used both for the report sub-metrics
    # below and for the LLM summary prompt.
    q_results = await loop.run_in_executor(
        None, _get_question_results_sync, interview_id
    )
    # Include ALL questions (filtered ones have 0.0 sub-scores) so that
    # skipped/empty answers correctly reduce metric averages.  Only
    # processed rows supply meaningful sub-scores; filtered rows contribute 0.
    all_q = q_results
    processed_q = [r for r in q_results if r.get("status") == "processed"]

    total_q_count = len(all_q)

    # Sub-scores are stored on a 0-10 scale; the report is 0-100, so ×10.
    # Denominator = total questions asked (including filtered), matching
    # how _compute_aggregate_sync() calculates overall_score.
    def _avg_subscore(key: str) -> float:
        if total_q_count == 0:
            return 0.0
        vals = [r.get(key) or 0.0 for r in all_q]
        return round((sum(vals) / total_q_count) * 10, 2)

    # technical_knowledge_score = average correctness_score (Phase 5 —
    # answer-vs-ideal-answer semantic similarity) across all questions.
    avg_correctness = _avg_subscore("correctness_score")
    technical_knowledge_score = avg_correctness

    # The other metrics — average real per-question sub-scores instead of
    # hardcoded overall_score copies or dead topic-name lookups.
    avg_clarity   = _avg_subscore("clarity_score")     # Phase 8 — readability/structure/fluency
    avg_depth     = _avg_subscore("depth_score")       # Phase 6 — detail/explanation/examples
    avg_coverage  = _avg_subscore("coverage_score")    # Phase 7 — key-concept coverage ratio
    avg_relevance = _avg_subscore("relevance_score")   # Phase 3 — answer addresses the question asked

    communication_score = avg_clarity
    confidence_score     = round((avg_depth + avg_clarity) / 2, 2)
    clarity_score        = avg_clarity
    role_fitment_score   = round((avg_coverage + avg_relevance) / 2, 2)
    problem_solving_score = avg_depth

    # Build a short LLM summary from the aggregate
    try:
        summary_prompt = (
            f"Candidate: {candidate_name}\n"
            f"Overall score: {overall_score}/100  Decision: {decision}\n"
            f"Skill scores: {skill_scores}\n"
            f"Top strengths: {strengths[:5]}\n"
            f"Areas to improve: {improvement_areas[:5]}\n\n"
            "Write a concise 3-sentence professional interview summary."
        )
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm_client.completion(summary_prompt, max_tokens=200),
        )
        summary = resp.strip() if resp else ""
    except Exception as e:
        logger.warning(f"[SUMMARY] LLM summary failed for {interview_id}: {e}")
        summary = (
            f"{candidate_name} completed the interview with an overall score of "
            f"{overall_score:.1f}/100. Decision: {decision}."
        )

    return {
        "interview_id":              interview_id,
        "candidate_name":            candidate_name,
        "interview_status":          interview_status,
        "overall_score":             overall_score,
        "confidence_score":          confidence_score,
        "communication_score":       communication_score,
        "problem_solving_score":     problem_solving_score,
        "technical_knowledge_score": technical_knowledge_score,
        "role_fitment_score":        role_fitment_score,
        "clarity_score":             clarity_score,
        "recommendation":            recommendation,
        "summary":                   summary,
        "strengths":                 strengths,
        "improvement_areas":         improvement_areas,
        
    }