from __future__ import annotations

from typing import Dict, List, Optional



TECHNICAL_WEIGHTS: Dict[str, float] = {
    "correctness": 0.60,
    "relevance":   0.13,
    "depth":       0.10,
    "coverage":    0.10,
    "clarity":     0.07,
}

BEHAVIORAL_WEIGHTS: Dict[str, float] = {
    "star_score":  0.30,
    "relevance":   0.20,
    "clarity":     0.20,
    "depth":       0.15,
    "correctness": 0.10,
    "coverage":    0.05,
}

GENERAL_WEIGHTS: Dict[str, float] = {
    "correctness": 0.25,
    "relevance":   0.25,
    "depth":       0.15,
    "coverage":    0.15,
    "clarity":     0.20,
}

_WEIGHT_MAPS: Dict[str, Dict[str, float]] = {
    "technical":  TECHNICAL_WEIGHTS,
    "behavioral": BEHAVIORAL_WEIGHTS,
    "general":    GENERAL_WEIGHTS,
}

# Sub-scores already on a 0-100 scale (everything else is assumed 0-10).
_HUNDRED_SCALE_KEYS = {"star_score"}


def compute_question_score(question_type: str, scores: Dict[str, float]) -> float:
   
    weights = _WEIGHT_MAPS.get((question_type or "").lower(), GENERAL_WEIGHTS)

    total = 0.0
    for key, weight in weights.items():
        raw = float(scores.get(key, 0.0) or 0.0)
        normalized = raw if key in _HUNDRED_SCALE_KEYS else raw * 10
        total += normalized * weight

    return round(min(100.0, max(0.0, total)), 2)


# ── Phase 12 — Difficulty multiplier ─────────────────────────────────────────

DIFFICULTY_MULTIPLIER: Dict[str, float] = {
    "easy":   1.0,
    "medium": 1.1,
    "hard":   1.2,
}


def apply_difficulty_weight(score: float, difficulty: str) -> float:
    """
    Phase 12: Multiply score by difficulty factor and clamp to 100.
    Higher-difficulty questions contribute proportionally more to the overall.
    """
    mult = DIFFICULTY_MULTIPLIER.get(difficulty.lower(), 1.0)
    return round(min(100.0, score * mult), 2)


# ── Phase 13 — Aggregate overall score ───────────────────────────────────────

def aggregate_overall_score(weighted_scores: List[float]) -> float:
    """
    Phase 13: Simple mean of difficulty-weighted per-question scores.
    Returns 0.0 if the list is empty.
    """
    if not weighted_scores:
        return 0.0
    return round(sum(weighted_scores) / len(weighted_scores), 2)


# ── Phase 14 — Topic/skill grouping ──────────────────────────────────────────

def compute_skill_scores(
    question_results: List[Dict],
) -> Dict[str, float]:
    """
    Phase 14: Group per-question scores by topic and compute per-topic average.

    Each item in question_results must have:
      topic (str)          — e.g. "Python", "System Design", "DSA"
      final_score (float)  — difficulty-weighted score 0-100

    Returns {"Python": 82.5, "DSA": 71.0, ...}
    """
    topic_scores: Dict[str, List[float]] = {}
    for q in question_results:
        topic = (q.get("topic") or "General").strip()
        score = float(q.get("final_score", 0.0))
        topic_scores.setdefault(topic, []).append(score)

    return {
        topic: round(sum(scores) / len(scores), 2)
        for topic, scores in topic_scores.items()
    }


# ── Phase 15 — Final decision ─────────────────────────────────────────────────

def make_decision(overall_score: float) -> str:
   
    if overall_score >= 75:
        return "SELECT"
    elif overall_score >= 40:
        return "HOLD"
    else:
        return "REJECT"
