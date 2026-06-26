from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from embeddings.engine import cosine_similarity, coverage_ratio
from llm import client as llm
from llm import prompts
from scoring.aggregator import (
    apply_difficulty_weight,
    compute_question_score,
    make_decision,
)

from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=1)
def _get_embedding_model():
    """Load sentence-transformer once, cached for the process lifetime."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("[PIPELINE] Embedding model loaded (all-MiniLM-L6-v2)")
    return model

logger = logging.getLogger(__name__)


# ── Phase 0 — Transcript Cleaning ────────────────────────────────────────────

class TranscriptCleaner:
    """
    Phase 0: Clean raw STT transcript before evaluation pipeline.

    Operations (all regex, no LLM — near-zero latency):
      • Remove filler words (um, uh, like, you know, basically…)
      • Strip disfluencies and repeated words (stutters)
      • Normalise whitespace and punctuation
      • Expand common technical contractions (I've → I have, etc.)
      • Preserve technical terms and acronyms (case-sensitive guard)

    Moved from services/semantic_evaluator.py — canonical copy lives here.
    """

    _FILLERS = re.compile(
        r'\b(um+|uh+|uhh+|er+|hmm+|like(?!\s+a\s+\w)|you know|you know what i mean|'
        r'basically|literally|actually|honestly|right\?|okay\?|so+\s|kind of|sort of|'
        r'i mean|i guess|i think that|let me think|well\s+so|right so)\b',
        re.IGNORECASE
    )
    _REPEAT_WORDS = re.compile(r'\b(\w+)(\s+\1){2,}\b', re.IGNORECASE)
    _MULTI_SPACE  = re.compile(r' {2,}')
    _CONTRACTIONS = {
        "i've": "I have", "i'm": "I am", "i'd": "I would", "i'll": "I will",
        "it's": "it is", "that's": "that is", "we've": "we have",
        "we're": "we are", "they're": "they are", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "can't": "cannot",
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
        "there's": "there is", "here's": "here is", "let's": "let us",
    }

    @classmethod
    def clean(cls, raw: str) -> str:
        if not raw:
            return ""
        text = raw.strip()

        # Expand contractions
        for contraction, expanded in cls._CONTRACTIONS.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)

        # Remove filler words
        text = cls._FILLERS.sub(' ', text)

        # Collapse repeated words (stutters): "the the the" → "the"
        text = cls._REPEAT_WORDS.sub(r'\1', text)

        # Normalise whitespace
        text = cls._MULTI_SPACE.sub(' ', text).strip()

        # Ensure sentence endings
        if text and text[-1] not in '.!?':
            text += '.'

        return text


# ── Phase X — STAR Evaluation (behavioral questions only) ────────────────────

class STAREvaluationStage:
    """
    Phase X: Evaluate behavioral answers against the STAR framework.
    Only runs when question_type == "behavioral".

    Checks for each STAR component (Situation / Task / Action / Result)
    via embedding similarity to canonical STAR descriptions.

    Returns star_score (0-100) stored on PerQuestionResult.
    Missing components are logged for feedback.
    """

    STAR_SENTENCES = {
        "Situation": "setting the context, background, or situation of the story",
        "Task":      "describing the task, responsibility, or challenge assigned",
        "Action":    "explaining specific actions taken, steps followed, personal contribution",
        "Result":    "describing the outcome, result, impact, or what was achieved",
    }
    STAR_THRESHOLD = 0.28

    async def evaluate(self, question: str, answer_clean: str) -> Dict[str, Any]:
        """
        Returns a dict with:
          star_score     int  0-100
          components     dict per-component presence + similarity
          missing        list of missing STAR components
          word_count     int
        """
        loop = asyncio.get_event_loop()

        def _compute():
            model    = _get_embedding_model()
            ans_emb  = model.encode([answer_clean], normalize_embeddings=True)[0]
            star_embs = model.encode(
                list(self.STAR_SENTENCES.values()), normalize_embeddings=True
            )
            sims = np.dot(star_embs, ans_emb)

            components: Dict[str, Any] = {}
            for (component, _), sim in zip(self.STAR_SENTENCES.items(), sims):
                components[component] = {
                    "present":    bool(sim >= self.STAR_THRESHOLD),
                    "similarity": round(float(sim), 3),
                }

            present_count = sum(1 for v in components.values() if v["present"])
            base_score    = int((present_count / 4) * 100)

            # Bonus: behavioral answers should be substantive
            word_count   = len(answer_clean.split())
            length_bonus = min(10, max(0, (word_count - 50) / 10))
            score        = int(min(100, base_score + length_bonus))

            missing = [k for k, v in components.items() if not v["present"]]

            return {
                "star_score":   score,
                "components":   components,
                "present_count": present_count,
                "missing":      missing,
                "word_count":   word_count,
            }

        try:
            result = await loop.run_in_executor(None, _compute)
            logger.debug(
                f"[PIPELINE] Phase X (STAR): score={result['star_score']} "
                f"present={result['present_count']}/4 missing={result['missing']}"
            )
            return result
        except Exception as e:
            logger.warning(f"[PIPELINE] Phase X (STAR) error: {e}")
            return {"star_score": 0, "components": {}, "missing": list(self.STAR_SENTENCES), "word_count": 0}


# ── Result schema ─────────────────────────────────────────────────────────────

@dataclass
class PerQuestionResult:
    # Identity
    question_id:  str
    interview_id: str
    topic:        str
    difficulty:   str          # easy / medium / hard
    question:     str
    answer:       str          # alias for answer_raw (kept for backward compat)

    # Phase 0 — cleaned transcript fields
    answer_raw:   str = ""     # original STT transcript (unmodified)
    answer_clean: str = ""     # after TranscriptCleaner (fillers removed, etc.)

    # Sub-scores (0-10)
    relevance_score:    float = 0.0
    correctness_score:  float = 0.0
    depth_score:        float = 0.0
    coverage_score:     float = 0.0
    clarity_score:      float = 0.0
    anti_gaming_penalty:float = 0.0

    # Aggregated
    question_score:   float = 0.0    # Phase 10 — 0-100 before difficulty
    weighted_score:   float = 0.0    # Phase 12 — after difficulty multiplier

    # Phase X — STAR (populated only when question_type == "behavioral")
    question_type:   str        = ""               # e.g. "behavioral", "technical", "general"
    star_score:      float      = 0.0              # 0-100; 0.0 means STAR not run
    star_components: Dict[str, Any] = field(default_factory=dict)  # per-component presence + similarity
    star_missing:    List[str]  = field(default_factory=list)      # missing STAR components

    # Status
    status:     str = "processed"    # "filtered" | "processed"
    filter_reason: Optional[str] = None

    # Ideal answer (Phase 4)
    ideal_answer: str = ""

    # Concepts extracted from ideal answer (Phase 7)
    key_concepts: List[str] = field(default_factory=list)

    # LLM reasons (for audit)
    depth_reason:   str = ""
    clarity_reason: str = ""

    # Feedback (Phase 16)
    strengths:              List[str] = field(default_factory=list)
    weaknesses:             List[str] = field(default_factory=list)
    improvement_suggestions:List[str] = field(default_factory=list)

    # Timing
    total_elapsed_ms: int = 0


# ── Anti-gaming helpers ───────────────────────────────────────────────────────

_VAGUE_PHRASES = re.compile(
    r"\b(i (don'?t|do not) know|not sure|i am not sure|no idea|"
    r"maybe|i think so|possibly|it depends|hard to say|"
    r"it'?s complicated)\b",
    re.IGNORECASE,
)

_FILLER = re.compile(
    r"\b(um+|uh+|like|you know|basically|literally|whatever|stuff|things)\b",
    re.IGNORECASE,
)


def _anti_gaming_penalty(answer: str) -> float:
    """
    Phase 9: Detect low-quality signals and return a penalty (0-3 points off
    on a 0-10 sub-score scale).

    Checks:
    • Repetition: word-type/token ratio — very low ratio means padding
    • Vague stock phrases (I don't know, it depends…)
    • Filler density
    • Long but content-free (>200 words, low unique-word ratio)
    """
    words   = answer.lower().split()
    n       = len(words)
    penalty = 0.0

    if n == 0:
        return 0.0

    unique_ratio = len(set(words)) / n

    # Heavy word repetition
    if n > 30 and unique_ratio < 0.30:
        penalty += 2.0
    elif n > 15 and unique_ratio < 0.40:
        penalty += 1.0

    # Vague phrases
    vague_hits = len(_VAGUE_PHRASES.findall(answer))
    penalty += min(1.5, vague_hits * 0.5)

    # Filler density
    filler_hits = len(_FILLER.findall(answer))
    filler_rate = filler_hits / n
    if filler_rate > 0.15:
        penalty += 1.0

    # Long but meaningless (>150 words, unique ratio < 0.35)
    if n > 150 and unique_ratio < 0.35:
        penalty += 1.0

    return round(min(3.0, penalty), 2)


# ── Phase 2 — Basic Filtering ─────────────────────────────────────────────────

def _basic_filter(answer: str) -> Optional[str]:
    """Return a rejection reason string, or None if answer passes."""
    if not answer or not answer.strip():
        return "empty_answer"
    if len(answer.strip().split()) < 5:
        return "too_short"
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

class PerQuestionPipeline:
    """
    Evaluates a single question-answer pair through all 16 phases.
    Instantiate once per call; the object is not reused.
    """

    def __init__(
        self,
        question_id:     str,
        interview_id:    str,
        question:        str,
        answer:          str,
        job_description: str,
        topic:           str = "",
        difficulty:      str = "medium",
        resume_context:  str = "",
        question_type:   str = "",
    ):
        self.question_id     = question_id
        self.interview_id    = interview_id
        self.question        = question.strip()
        self.answer          = answer.strip()
        self.job_description = job_description
        self.topic           = topic or "General"
        self.difficulty      = difficulty.lower()
        self.resume_context  = resume_context
        self.question_type   = question_type.lower()  # "behavioral" triggers STAR

    async def run(self) -> PerQuestionResult:
        t0 = time.perf_counter()

        # ── Phase 0: Transcript Cleaning ──────────────────────────────────────
        # Runs BEFORE Phase 1 so the cleaned text is used throughout the pipeline.
        answer_clean = TranscriptCleaner.clean(self.answer)
        logger.debug(
            f"[PIPELINE] Phase 0 (clean): raw={len(self.answer)} chars "
            f"→ clean={len(answer_clean)} chars"
        )

        result = PerQuestionResult(
            question_id   = self.question_id,
            interview_id  = self.interview_id,
            topic         = self.topic,
            difficulty    = self.difficulty,
            question      = self.question,
            answer        = self.answer,
            answer_raw    = self.answer,       # preserved verbatim
            answer_clean  = answer_clean,      # cleaned version
            question_type = self.question_type,
        )

        # ── Phase 1: Input stored in result object (done above) ────────────
        logger.info(
            f"[PIPELINE] Start Q{self.question_id} | topic={self.topic} "
            f"| diff={self.difficulty} | interview={self.interview_id}"
        )

        # ── Phase 2: Basic Filtering ───────────────────────────────────────
        reject_reason = _basic_filter(self.answer)
        if reject_reason:
            result.status             = "filtered"
            result.filter_reason      = reject_reason
            # Keep all sub-scores and question_score at 0.0 so that
            # generate_interview_summary() includes this question in
            # averages as a true zero — no phantom scores.
            result.question_score     = 0.0
            result.weighted_score     = 0.0
            result.relevance_score    = 0.0
            result.correctness_score  = 0.0
            result.depth_score        = 0.0
            result.coverage_score     = 0.0
            result.clarity_score      = 0.0
            logger.info(f"[PIPELINE] Q{self.question_id} filtered: {reject_reason}")
            result.total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return result

        # ── Phase 4: Ideal Answer Generation (sequential — feeds phases 5-7) ──
        ideal_prompt = prompts.ideal_answer(
            question        = self.question,
            job_description = self.job_description,
            topic           = self.topic,
        )
        ideal_text = await llm.generate(ideal_prompt, temperature=0.3, max_tokens=300)
        if not ideal_text:
            ideal_text = f"A complete answer to: {self.question}"  # graceful fallback
        result.ideal_answer = ideal_text

        # ── Phases 3, 5, 6, 7, 8 — RUN IN PARALLEL ────────────────────────
        (
            relevance,
            correctness,
            depth_data,
            concepts_data,
            clarity_data,
        ) = await asyncio.gather(
            self._phase3_relevance(),
            self._phase5_correctness(ideal_text),
            self._phase6_depth(ideal_text),
            self._phase7_coverage(ideal_text),
            self._phase8_clarity(),
        )

        result.relevance_score    = relevance
        result.correctness_score  = correctness
        result.depth_score        = depth_data["score"]
        result.depth_reason       = depth_data["reason"]
        result.key_concepts       = concepts_data["concepts"]
        result.coverage_score     = concepts_data["score"]
        result.clarity_score      = clarity_data["score"]
        result.clarity_reason     = clarity_data["reason"]

        # ── Phase 9: Anti-Gaming ───────────────────────────────────────────
        penalty = _anti_gaming_penalty(self.answer)
        result.anti_gaming_penalty = penalty

        # Apply penalty evenly across sub-scores (clamped to 0)
        result.relevance_score   = max(0.0, result.relevance_score   - penalty)
        result.correctness_score = max(0.0, result.correctness_score - penalty)
        result.depth_score       = max(0.0, result.depth_score       - penalty)
        result.coverage_score    = max(0.0, result.coverage_score    - penalty)
        result.clarity_score     = max(0.0, result.clarity_score     - penalty)

        # ── Phase X: STAR Evaluation (behavioral only) ──────────────────────
        # Runs BEFORE Phase 10 so star_score is available to the weighted
        # formula for behavioral questions.
        if self.question_type == "behavioral":
            logger.info(f"[PIPELINE] Phase X (STAR): running for Q{self.question_id} (behavioral)")
            star_stage = STAREvaluationStage()
            star_data  = await star_stage.evaluate(self.question, result.answer_clean)
            result.star_score      = float(star_data.get("star_score", 0))
            result.star_components = star_data.get("components", {})
            result.star_missing    = star_data.get("missing", [])
            logger.info(
                f"[PIPELINE] Phase X (STAR): Q{self.question_id} "
                f"star_score={result.star_score} "
                f"missing={result.star_missing}"
            )
        else:
            logger.debug(
                f"[PIPELINE] Phase X (STAR): skipped for Q{self.question_id} "
                f"(question_type={self.question_type!r})"
            )

        # ── Phase 10: Per-Question Score ────────────────────────────────────
        q_score = compute_question_score(
            self.question_type,
            {
                "correctness": result.correctness_score,
                "relevance":   result.relevance_score,
                "depth":       result.depth_score,
                "coverage":    result.coverage_score,
                "clarity":     result.clarity_score,
                "star_score":  result.star_score,
            },
        )
        result.question_score = q_score

        # ── Phase 11: status already set to "processed" ────────────────────
        result.status = "processed"

        # ── Phase 12: Difficulty Weighting ────────────────────────────────
        result.weighted_score = apply_difficulty_weight(q_score, self.difficulty)

        # ── Phase 16: Feedback Generation ─────────────────────────────────
        await self._phase16_feedback(result)

        result.total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[PIPELINE] Done Q{self.question_id}: score={q_score:.1f} "
            f"weighted={result.weighted_score:.1f} elapsed={result.total_elapsed_ms}ms"
        )
        return result

    # ── Phase 3 ──────────────────────────────────────────────────────────────

    async def _phase3_relevance(self) -> float:
        """
        Semantic similarity between question and answer.
        Measures: does the answer actually address what was asked?
        Returns score 0–10.
        """
        try:
            sim = await cosine_similarity(self.question, self.answer)
            # cosine-sim 0-1 → scale to 0-10
            # Raw similarity between Q and A tends to be moderate even for good answers
            # (since questions are short and answers are long). Re-scale with a floor.
            score = min(10.0, max(0.0, sim * 12))
            return round(score, 2)
        except Exception as e:
            logger.warning(f"[PIPELINE] Phase 3 error: {e}")
            return 5.0

    # ── Phase 5 ──────────────────────────────────────────────────────────────

    async def _phase5_correctness(self, ideal: str) -> float:
        """
        Semantic similarity between candidate answer and ideal answer.
        Core accuracy signal. Returns score 0–10.
        """
        try:
            sim = await cosine_similarity(self.answer, ideal)
            score = min(10.0, max(0.0, sim * 13))
            return round(score, 2)
        except Exception as e:
            logger.warning(f"[PIPELINE] Phase 5 error: {e}")
            return 5.0

    # ── Phase 6 ──────────────────────────────────────────────────────────────

    async def _phase6_depth(self, ideal: str) -> Dict[str, Any]:
        """LLM-scored depth analysis. Returns {score, reason}."""
        prompt = prompts.depth_analysis(self.question, self.answer, ideal)
        data = await llm.generate_json(
            prompt,
            temperature=0.1,
            max_tokens=150,
            fallback={"depth_score": 5, "reason": "LLM unavailable"},
        )
        return {
            "score":  float(min(10, max(0, data.get("depth_score", 5)))),
            "reason": str(data.get("reason", "")),
        }

    # ── Phase 7 ──────────────────────────────────────────────────────────────

    async def _phase7_coverage(self, ideal: str) -> Dict[str, Any]:
        """
        Extract key concepts from ideal answer, then measure embedding
        coverage of those concepts in the candidate's answer.
        Returns {score, concepts}.
        """
        # Step A: extract concepts via LLM
        concept_prompt = prompts.extract_key_concepts(ideal, self.question)
        concept_data = await llm.generate_json(
            concept_prompt,
            temperature=0.1,
            max_tokens=200,
            fallback=[],
        )

        # LLM may return a list directly or {"concepts": [...]}
        if isinstance(concept_data, list):
            concepts = [str(c) for c in concept_data]
        elif isinstance(concept_data, dict):
            concepts = [str(c) for c in concept_data.get("concepts", [])]
        else:
            concepts = []

        concepts = concepts[:8]   # cap

        if not concepts:
            return {"score": 5.0, "concepts": []}

        # Step B: embedding-based coverage ratio
        try:
            ratio = await coverage_ratio(concepts, self.answer)
            score = round(ratio * 10, 2)
        except Exception as e:
            logger.warning(f"[PIPELINE] Phase 7 coverage error: {e}")
            score = 5.0

        return {"score": score, "concepts": concepts}

    # ── Phase 8 ──────────────────────────────────────────────────────────────

    async def _phase8_clarity(self) -> Dict[str, Any]:
        """LLM-scored clarity / communication. Returns {score, reason}."""
        prompt = prompts.clarity_scoring(self.question, self.answer)
        data = await llm.generate_json(
            prompt,
            temperature=0.1,
            max_tokens=150,
            fallback={"clarity_score": 5, "reason": "LLM unavailable"},
        )
        return {
            "score":  float(min(10, max(0, data.get("clarity_score", 5)))),
            "reason": str(data.get("reason", "")),
        }

    # ── Phase 16 ─────────────────────────────────────────────────────────────

    async def _phase16_feedback(self, result: PerQuestionResult) -> None:
        """Populate strengths, weaknesses, suggestions on the result object."""
        prompt = prompts.feedback_generation(
            question    = self.question,
            answer      = self.answer,
            ideal       = result.ideal_answer,
            final_score = result.question_score,
            breakdown   = {
                "correctness": result.correctness_score,
                "relevance":   result.relevance_score,
                "depth":       result.depth_score,
                "coverage":    result.coverage_score,
                "clarity":     result.clarity_score,
            },
            topic       = self.topic,
            difficulty  = self.difficulty,
        )
        data = await llm.generate_json(
            prompt,
            temperature=0.2,
            max_tokens=400,
            fallback={},
        )
        result.strengths               = _as_list(data.get("strengths", []))
        result.weaknesses              = _as_list(data.get("weaknesses", []))
        result.improvement_suggestions = _as_list(data.get("improvement_suggestions", []))


# ── Utility ───────────────────────────────────────────────────────────────────

def _as_list(val) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []