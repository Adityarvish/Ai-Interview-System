from __future__ import annotations


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def ideal_answer(question: str, job_description: str, topic: str = "") -> str:
    topic_line = f"Topic/Domain: {topic}\n" if topic else ""
    return f"""You are a senior technical interviewer with deep expertise.

{topic_line}Job Role Context:
{job_description[:600]}

Interview Question:
{question}

Generate a concise IDEAL ANSWER that a strong candidate should give.
Requirements:
- 3–5 sentences maximum
- Include 1–2 concrete examples or technical specifics
- Use precise, domain-accurate terminology
- Do NOT pad with clichés or generic filler

Respond with ONLY the ideal answer text. No preamble, no labels."""


# ── Phase 6 ───────────────────────────────────────────────────────────────────

def depth_analysis(question: str, answer: str, ideal: str) -> str:
    return f"""You are evaluating the DEPTH of a candidate's interview answer.

Question: {question}

Ideal Reference Answer:
{ideal}

Candidate's Answer:
{answer}

Score the candidate's answer on DEPTH (0–10) considering:
  - Level of detail and specificity (not just surface-level)
  - Quality of explanation (WHY and HOW, not just WHAT)
  - Presence of concrete examples, numbers, or real-world scenarios
  - Technical accuracy beyond buzzwords

Scoring guide:
  9–10 : Exceptional depth with specific examples and strong WHY/HOW
  7–8  : Good depth, mostly specific, minor gaps
  5–6  : Adequate but mostly surface-level, few examples
  3–4  : Shallow, vague, generic phrases
  0–2  : Almost no depth; one-liners or completely off-topic

Respond with ONLY this JSON (no markdown):
{{"depth_score": <integer 0-10>, "reason": "<one-sentence explanation>"}}"""


# ── Phase 7 ───────────────────────────────────────────────────────────────────

def extract_key_concepts(ideal_answer_text: str, question: str) -> str:
    return f"""Extract the KEY TECHNICAL CONCEPTS that must be present in a complete answer.

Question: {question}

Ideal Answer:
{ideal_answer_text}

List only the essential concepts, terms, or ideas — 3 to 8 items.
Do NOT include generic words like "explain", "discuss", "important".

Respond with ONLY a JSON array of strings:
["concept_1", "concept_2", "concept_3"]"""


# ── Phase 8 ───────────────────────────────────────────────────────────────────

def clarity_scoring(question: str, answer: str) -> str:
    return f"""You are evaluating the CLARITY AND COMMUNICATION quality of an interview answer.

Question: {question}

Candidate's Answer:
{answer}

Score on CLARITY (0–10) considering:
  - Sentence structure and logical flow (does it make sense on first read?)
  - Absence of filler words, rambling, or self-contradiction
  - Directness — does the answer address the question without wandering?
  - Grammar and professional language appropriate to the context

Scoring guide:
  9–10 : Exceptionally clear, structured, professional
  7–8  : Clear with minor readability issues
  5–6  : Understandable but disorganised or wordy
  3–4  : Hard to follow; confused structure
  0–2  : Incoherent or completely off-topic

Respond with ONLY this JSON (no markdown):
{{"clarity_score": <integer 0-10>, "reason": "<one-sentence explanation>"}}"""


# ── Phase 16 ──────────────────────────────────────────────────────────────────

def feedback_generation(
    question: str,
    answer: str,
    ideal: str,
    final_score: float,
    breakdown: dict,
    topic: str = "",
    difficulty: str = "medium",
) -> str:
    breakdown_str = "\n".join(
        f"  {k}: {v}/10" for k, v in breakdown.items()
    )
    topic_line = f"Topic: {topic} | " if topic else ""
    return f"""You are a supportive but honest technical interview coach.

{topic_line}Difficulty: {difficulty} | Question Score: {final_score:.1f}/100

Question: {question}

Ideal Answer:
{ideal}

Candidate's Answer:
{answer}

Score Breakdown:
{breakdown_str}

Generate specific, actionable feedback. Be honest but constructive.

Respond with ONLY this JSON (no markdown):
{{
  "strengths": ["<specific positive from this answer>", "<another specific positive>"],
  "weaknesses": ["<specific gap or error>", "<another gap>"],
  "improvement_suggestions": ["<concrete actionable advice>", "<another concrete suggestion>"]
}}

Rules:
- Each item must reference THIS specific answer, not generic advice
- strengths: 1–3 items; weaknesses: 1–3 items; suggestions: 1–3 items
- If score < 30, strengths may have 0 items"""
