from sqlalchemy import Column, String, Float, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone
from config.database import Base


def _now():
    return datetime.now(timezone.utc)


class Evaluation(Base):
    __tablename__ = "evaluations"

    interview_id              = Column(String(36), primary_key=True)
    candidate_name            = Column(String(255), default="")
    interview_status          = Column(String(50),  default="completed")
    overall_score             = Column(Float,        default=0)
    confidence_score          = Column(Float,        default=0)
    communication_score       = Column(Float,        default=0)
    problem_solving_score     = Column(Float,        default=0)
    technical_knowledge_score = Column(Float,        default=0)
    role_fitment_score        = Column(Float,        default=0)
    clarity_score             = Column(Float,        default=0)
    recommendation            = Column(String(50),   default="Hold")
    summary                   = Column(Text,         default="")
    strengths                 = Column(JSONB,        default=list)
    improvement_areas         = Column(JSONB,        default=list)
    decision_score            = Column(Float,        default=0)
    created_at                = Column(DateTime(timezone=True), default=_now)
