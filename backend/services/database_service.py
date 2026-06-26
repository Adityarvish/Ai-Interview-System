from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session

from config.database import SessionLocal
from config.models import Evaluation

logger = logging.getLogger(__name__)


def _row_to_dict(row) -> dict | None:
    """Convert a SQLAlchemy model instance to a plain dict."""
    if row is None:
        return None
    d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


class DatabaseService:
    """SQLAlchemy-backed service — evaluations table only."""

    def __init__(self):
        self._session: Session = SessionLocal()

    def _commit(self):
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def close(self):
        self._session.close()

    # ── Evaluations ────────────────────────────────────────────────────────────

    def save_final_evaluation(self, interview_id: str, evaluation: dict) -> dict:
        """Idempotent upsert for the final evaluation record."""
        row = (
            self._session.query(Evaluation)
            .filter_by(interview_id=interview_id)
            .first()
        )
        doc = {
            "interview_id":              interview_id,
            "candidate_name":            evaluation.get("candidate_name", ""),
            "interview_status":          evaluation.get("interview_status", "completed"),
            "overall_score":             evaluation.get("overall_score", 0),
            "confidence_score":          evaluation.get("confidence_score", 0),
            "communication_score":       evaluation.get("communication_score", 0),
            "problem_solving_score":     evaluation.get("problem_solving_score", 0),
            "technical_knowledge_score": evaluation.get("technical_knowledge_score", 0),
            "role_fitment_score":        evaluation.get("role_fitment_score", 0),
            "clarity_score":             evaluation.get("clarity_score", 0),
            "recommendation":            evaluation.get("recommendation", "Hold"),
            "summary":                   evaluation.get("summary", ""),
            "strengths":                 evaluation.get("strengths", []),
            "improvement_areas":         evaluation.get("improvement_areas", []),
            "decision_score":            evaluation.get("decision_score", evaluation.get("overall_score", 0)),
            "created_at":                datetime.now(timezone.utc),
        }

        if row:
            for k, v in doc.items():
                setattr(row, k, v)
        else:
            row = Evaluation(**doc)
            self._session.add(row)

        self._commit()
        logger.info(
            f"[DB] Saved evaluation: {interview_id} | "
            f"overall={doc['overall_score']} | rec={doc['recommendation']}"
        )
        return {**doc, "created_at": doc["created_at"].isoformat()}

    def get_final_evaluation(self, interview_id: str) -> dict | None:
        row = (
            self._session.query(Evaluation)
            .filter_by(interview_id=interview_id)
            .first()
        )
        return _row_to_dict(row)
