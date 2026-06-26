from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Float, Index, Integer, String, Text, DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB

from config.database import Base, engine


def _now():
    return datetime.now(timezone.utc)


class PerQuestionEvaluation(Base):
    
    __tablename__ = "per_question_evaluations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    interview_id    = Column(String(36), nullable=False, index=True)
    question_id     = Column(String(64), nullable=False)
    question_type   = Column(String(32),  default="general")
    topic           = Column(String(128), default="General")
    difficulty      = Column(String(16),  default="medium")

   
    question_text   = Column(Text, nullable=True)
    answer_text     = Column(Text, nullable=True)
    ideal_answer    = Column(Text, nullable=True)
    key_concepts    = Column(JSONB, default=list)

    
    relevance_score     = Column(Float, default=0.0)
    correctness_score   = Column(Float, default=0.0)
    depth_score         = Column(Float, default=0.0)
    coverage_score      = Column(Float, default=0.0)
    clarity_score       = Column(Float, default=0.0)
    anti_gaming_penalty = Column(Float, default=0.0)

    # Phase X — STAR evaluation (behavioral questions only; 0.0 when not run)
    star_score      = Column(Float, default=0.0)
    star_components = Column(JSONB, default=dict)
    star_missing    = Column(JSONB, default=list)

    # Aggregated (Phase 10 + 12)
    question_score  = Column(Float, default=0.0)
    weighted_score  = Column(Float, default=0.0)

    # LLM reasoning traces
    depth_reason    = Column(Text, nullable=True)
    clarity_reason  = Column(Text, nullable=True)

    # Qualitative feedback (Phase 16)
    strengths               = Column(JSONB, default=list)
    weaknesses              = Column(JSONB, default=list)
    improvement_suggestions = Column(JSONB, default=list)

    # Lifecycle
    status        = Column(String(20), default="processing")
    filter_reason = Column(String(64), nullable=True)
    eval_ms       = Column(Integer, default=0)
    evaluated_at  = Column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_pqeval_interview_qid", "interview_id", "question_id"),
    )


# ── DDL helper ────────────────────────────────────────────────────────────────

def create_per_question_tables():
    
    Base.metadata.create_all(
        bind=engine,
        tables=[PerQuestionEvaluation.__table__],
        checkfirst=True,
    )
    import logging
    logging.getLogger(__name__).info("[DB] per_question_evaluations ensured.")
