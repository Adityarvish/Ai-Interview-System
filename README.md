# AI VOICE INTERVIEW SYSTEM
**Complete Technical Reference & Developer Documentation**

---

## Table of Contents
1. System Overview & Architecture
2. Project Structure — Every File Explained
3. Configuration & Environment Variables
4. Technology Stack & Dependencies
5. Interview Flow — Stages & State Machine
6. Question Deduplication System
7. Speech Pipeline (STT & TTS)
8. The 16-Phase Per-Question Evaluation Pipeline
9. The 6 Core Evaluation Metrics — Formulas & Detail
10. STAR Evaluation Framework (Behavioral Questions)
11. Anti-Gaming & Transcript Cleaning
12. Scoring Aggregation — From Sub-Scores to Final Decision
13. Database Schema — Every Table & Column
14. REST API Endpoints
15. Socket.IO Events — Real-Time Communication Protocol
16. Frontend Architecture
17. Language Support (English · Hindi · Marathi)
18. Setup & Installation Guide

---

## 1. System Overview & Architecture

### 1.1 What the System Does
The AI Voice Interview System is a full-stack, AI-powered platform that conducts automated job interviews via voice in **English, Hindi, and Marathi**. It asks questions using text-to-speech (TTS), captures candidate responses via microphone, transcribes them using speech-to-text (STT), and evaluates every answer through a 16-phase pipeline that produces per-question scores, skill-level breakdowns, and an overall hiring recommendation (SELECT / HOLD / REJECT).

Candidates select their preferred interview language on the landing page (`en` / `hi` / `mr`). The system then routes all TTS through the appropriate provider (Groq Orpheus for English, Sarvam AI for Hindi/Marathi), passes the language code to Whisper for accurate transcription, and generates questions and greeting/closing messages in the chosen language.

### 1.2 Architecture Layers

| Layer | Description |
|---|---|
| Frontend (Browser) | 3-page app: Landing (job setup + language picker), Interview (voice capture), Results (scores & feedback) |
| Socket.IO Layer | Bidirectional real-time channel for audio streaming, transcription, and AI responses |
| REST API (FastAPI) | HTTP endpoints for session creation, file upload, TTS, and report retrieval |
| Interview Engine | Stage-based conversation manager with duplicate detection, adaptive questioning, and per-language greeting/closing templates |
| Evaluation Pipeline | 16-phase async pipeline: transcript cleaning → embedding scoring → LLM scoring → aggregation |
| LLM Client (Groq) | Cloud inference for ideal-answer generation, depth/clarity scoring, concept extraction, feedback |
| Embedding Engine | all-MiniLM-L6-v2 sentence-transformer for semantic similarity and concept coverage |
| TTS — English | Groq Orpheus (`canopylabs/orpheus-v1-english`) |
| TTS — Hindi / Marathi | Sarvam AI (`bulbul:v1`) — free-tier Indian-language synthesis |
| STT | Groq Whisper (`whisper-large-v3-turbo`) with per-language code passed for improved accuracy |
| Database (PostgreSQL) | Two tables: `evaluations` (summary) and `per_question_evaluations` (per-Q detail) |

### 1.3 Data Flow Summary
1. Candidate uploads resume (PDF/TXT) and enters job description on the landing page.
2. `POST /api/start-interview` initialises `InterviewEngine`, extracts the resume, creates the in-memory session.
3. Browser connects via Socket.IO and joins a room for the interview (`join_interview`).
4. Candidate speaks an answer. Browser streams audio (`audio_chunk` events, then `audio_end`, or a single `audio_upload`).
5. Backend converts audio → WAV (ffmpeg), transcribes via Groq Whisper, generates the next question.
6. Concurrently, the per-question background dispatcher fires the 16-phase pipeline as an `asyncio` task.
7. After all questions, the `end_interview` socket event awaits in-flight evaluations, builds the summary, and saves it.
8. Frontend fetches the final report and renders the interactive results dashboard.

---

## 2. Project Structure — Every File Explained

| File / Directory | Role | Key Exports / Contents |
|---|---|---|
| backend/main.py | FastAPI app entry point | App factory, lifespan hooks, route mounting, Socket.IO ASGI binding |
| backend/config/settings.py | Centralised configuration | `Config` class — all env vars, paths, model names |
| backend/config/database.py | Database engine & session | `engine`, `SessionLocal`, `init_db()`, migration runner |
| backend/config/models.py | SQLAlchemy ORM — summary table | `Evaluation` model (`evaluations` table) |
| backend/config/per_question_models.py | SQLAlchemy ORM — detail table | `PerQuestionEvaluation` model (`per_question_evaluations` table), `create_per_question_tables()` |
| backend/evaluator/pipeline.py | 16-phase evaluation engine | `PerQuestionPipeline`, `PerQuestionResult`, `TranscriptCleaner`, `STAREvaluationStage` |
| backend/scoring/aggregator.py | Score formula definitions | `TECHNICAL/BEHAVIORAL/GENERAL_WEIGHTS`, `compute_question_score()`, `apply_difficulty_weight()`, `aggregate_overall_score()`, `compute_skill_scores()`, `make_decision()` |
| backend/embeddings/engine.py | Embedding & similarity engine | `cosine_similarity()`, `coverage_ratio()`, `encode()` |
| backend/llm/client.py | Groq LLM client wrapper | `generate()`, `generate_json()` — with primary/fallback model retry |
| backend/llm/prompts.py | LLM prompt templates | `ideal_answer()`, `depth_analysis()`, `extract_key_concepts()`, `clarity_scoring()`, `feedback_generation()` |
| backend/services/interview_engine.py | Stage-based conversation manager | `InterviewEngine`, `InterviewState`, `DuplicateDetector`, `STAGES`, `STAGE_MIN_QUESTIONS` — greeting, closing, and question-generation prompts are language-aware (`en`/`hi`/`mr`) |
| backend/services/per_question_background.py | Background eval dispatcher | `trigger_per_question_eval()`, `release_semaphore()`, per-interview concurrency semaphore |
| backend/services/per_question_db_service.py | Async DB layer for per-Q evals | `PerQuestionDBService.save()`, `.update_aggregate()`, `.get_aggregate()`, `.get_question_results()`, `generate_interview_summary()` |
| backend/services/speech_to_text.py | STT via Groq Whisper | `transcribe_audio()`, `SpeechToTextService` / `stt_service` — uses `whisper-large-v3-turbo`; accepts a `language` param (`"en"`, `"hi"`, `"mr"`) passed directly to the Whisper API for improved accuracy |
| backend/services/text_to_speech.py | TTS via Groq Orpheus (English) and Sarvam AI (Hindi/Marathi) | `text_to_speech()`, `TextToSpeechService` / `tts_service` — routes `"hi"`/`"mr"` to `_sarvam_text_to_speech()`, falls back to `canopylabs/orpheus-v1-english` for `"en"` |
| backend/services/database_service.py | High-level DB operations | `DatabaseService` — CRUD for `evaluations` table |
| backend/services/resume_parser.py | Resume text extraction | `ResumeParser` — supports PDF (pdfplumber) and TXT |
| backend/services/rag_service.py | RAG context builder | `RAGService` — FAISS + LangChain for resume-aware question generation |
| backend/services/llm_service.py | Groq inference wrapper (interview) | `GroqService`, `OllamaService` — question generation with retry, `check_connection()` |
| backend/services/warm_cache.py | Background warmup & cleanup | `warmup_all()`, `start_audio_cleanup_loop()` — pre-loads embedding model, starts audio cleanup loop |
| backend/routers/interview.py | HTTP REST endpoints | `/api/start-interview`, `/api/final-report/{id}`, `/api/tts`, `/api/tts/health`, `/api/health`, `/debug-tts` |
| backend/routers/per_question_router.py | Per-question HTTP endpoints | Mounted at `/api/pq-eval` — submit, sync-evaluate, aggregate, final-report |
| backend/sockets/handlers.py | Socket.IO event handlers | `connect`, `disconnect`, `join_interview`, `audio_chunk`, `audio_end`, `audio_upload`, `end_interview`, `cancel_interview` |
| backend/schemas/interview.py | Pydantic request/response schemas | `StartInterviewResponse`, `FinalReportResponse`, `TTSRequest`, `TTSResponse`, `HealthResponse`, `ErrorResponse` |
| backend/requirements.txt | Python dependencies | FastAPI, Groq, sentence-transformers, SQLAlchemy, pdfplumber, LangChain, FAISS |
| frontend/index.html | Landing page | Job description form, resume upload, **interview language selector (English / हिन्दी / मराठी)**, start-interview trigger |
| frontend/interview.html | Interview page | Microphone controls, live transcript display, stage progress indicator |
| frontend/result.html | Results dashboard | Score dials, per-question breakdown, skill heatmap, feedback |
| frontend/js/interview.js | Interview client (~60 KB) | Socket.IO client, audio recording (MediaRecorder), TTS playback, state machine |
| frontend/js/landing.js | Landing page logic | Form validation, resume file upload, session initialisation |
| frontend/js/result.js | Results renderer | Score gauge rendering, topic breakdown charts, feedback display |
| frontend/css/style.css | Global styles | Dark-mode interview UI, responsive layout |

---

## 3. Configuration & Environment Variables

### 3.1 Environment File Location
All runtime settings are loaded from a `.env` file in the project root (one level above `backend/`). `settings.py` uses `python-dotenv` to load this file at import time.

### 3.2 Complete Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| GROQ_API_KEY | (required) | Groq Cloud API key — used for LLM text generation, Whisper STT, and Orpheus TTS |
| GROQ_PRIMARY_MODEL | llama-3.3-70b-versatile | Primary LLM for question generation, scoring, ideal-answer synthesis, and feedback |
| GROQ_FALLBACK_MODEL | llama-3.1-8b-instant | Fallback LLM used when the primary model errors out |
| GROQ_TTS_VOICE | daniel | TTS voice for English. Female: autumn, diana, hannah. Male: austin, daniel, troy. *(Not used for Hindi/Marathi — Sarvam AI handles those.)* |
| SARVAM_API_KEY | (required for hi/mr) | Sarvam AI API key — required only when Hindi or Marathi is selected. Obtain a free-tier key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| POSTGRES_HOST | localhost | PostgreSQL server hostname |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| POSTGRES_USER | postgres | Database username |
| POSTGRES_PASSWORD | (empty) | Database password |
| POSTGRES_DB | ai_interview_db | Database name |
| DATABASE_URL | auto-constructed | Full SQLAlchemy URL — overrides individual `POSTGRES_*` vars if set |
| HOST | 0.0.0.0 | Interface to bind the FastAPI/Uvicorn server |
| PORT | 5000 | TCP port for the web server |
| MAX_INTERVIEW_DURATION | 2700 | Maximum interview session length in seconds (45 minutes) |

### 3.3 File Upload Limits

| Setting | Value |
|---|---|
| Maximum resume size | 10 MB |
| Maximum audio file size | 50 MB |
| Allowed resume extensions | .pdf, .txt |
| Allowed audio extensions | .webm, .wav, .ogg, .mp3 |
| Audio storage location | `uploads/audio/` (auto-created) |
| Resume storage location | `uploads/resumes/` (auto-created) |


---

## 4. Technology Stack & Dependencies

(`backend/requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.0 | Async HTTP framework |
| uvicorn[standard] | 0.30.6 | ASGI server |
| python-socketio | 5.11.1 | Server-side Socket.IO |
| python-engineio | 4.9.1 | Engine.IO transport layer |
| pydantic | 2.8.2 | Request/response validation |
| python-multipart | 0.0.9 | Multipart form parsing (resume uploads) |
| starlette | 0.38.2 | ASGI toolkit |
| sqlalchemy | 2.0.31 | ORM / DB engine |
| psycopg2-binary | >=2.9.9 | PostgreSQL adapter |
| python-dotenv | 1.0.1 | `.env` loading |
| pdfplumber | 0.11.0 | PDF text extraction |
| langchain / langchain-community / langchain-core / langchain-text-splitters | 0.2.16 / 0.2.16 / 0.2.38 / 0.2.4 | RAG pipeline |
| faiss-cpu | 1.8.0 | Vector similarity for RAG |
| sentence-transformers | >=2.7.0 | all-MiniLM-L6-v2 embeddings |
| groq | >=0.9.0 | LLM / Whisper / Orpheus SDK |
| numpy | 1.26.4 | Embedding math |
| requests | 2.31.0 | HTTP client |

System dependencies: `ffmpeg`, PostgreSQL ≥ 14, Python ≥ 3.10.

---

## 5. Interview Flow — Stages & State Machine


| Stage | Min Questions | Purpose |
|---|---|---|
| greeting | 1 | Welcomes the candidate by name. No evaluation. |
| introduction | 2 | Background, motivation, role interest. |
| resume | 2 | RAG-driven questions about the candidate's actual resume content. |
| technical | 5 | Core technical depth; adaptive difficulty. |
| behavioral | 2 | STAR-framework questions; STAR scoring activates only here. |
| closing | 1 | Conversational wrap-up. No evaluation score. |


---

## 6. Question Deduplication System


**Layer 1 — Exact Normalised Match:** lowercase, punctuation stripped to spaces, whitespace collapsed; identical strings = duplicate (score 1.0).

**Layer 2 — Topic Cluster Match:** 8 hard-coded clusters (`self_introduction`, `role_interest`, `technical_challenge`, `learning_technology`, `team_disagreement`, `explaining_technical`, `career_goals`, `resume_project`); same cluster as a previous question = duplicate (score 0.9).

**Layer 3 — Keyword Jaccard Similarity:** stop words removed, Jaccard similarity of remaining keyword sets computed; `≥ 0.45` = duplicate.

Stop word list (`_STOP_WORDS`) has ~90 common English words, plus interview verbs (tell, describe, explain, talk, walk, give). Only words with length > 2 are used for Jaccard comparison.

---

## 7. Speech Pipeline (STT & TTS)

### 7.1 Speech-to-Text

| Property | Detail |
|---|---|
| Provider | Groq Cloud API |
| Model | whisper-large-v3-turbo |
| Input format | WAV, 16 kHz, mono (ffmpeg pre-processing) |
| Language | Passed per-session from candidate's choice: `en` (English), `hi` (Hindi), `mr` (Marathi) |
| Minimum audio size | 500 bytes — smaller is rejected as too short |
| Client singleton | Module-level Groq client, created once |

### 7.2 Text-to-Speech

TTS is **routed by language**. English uses Groq Orpheus; Hindi and Marathi use Sarvam AI (`bulbul:v1`), because Orpheus is English-only.

**English (Groq Orpheus)**

| Property | Detail |
|---|---|
| Provider | Groq Cloud API |
| Model | canopylabs/orpheus-v1-english |
| Default voice | daniel (`GROQ_TTS_VOICE` env var) |
| Available voices | Female: autumn, diana, hannah · Male: austin, daniel, troy |
| Output format | WAV |
| Maximum input length | 1800 characters per call — longer text is automatically chunked at sentence/clause boundaries and the resulting WAV chunks are concatenated |
| Fallback | Browser's built-in `speechSynthesis` API if TTS fails |

**Hindi / Marathi (Sarvam AI)**

| Property | Detail |
|---|---|
| Provider | Sarvam AI Cloud API |
| Model | bulbul:v1 |
| Speaker | meera |
| Language codes | `hi-IN` (Hindi), `mr-IN` (Marathi) |
| Output format | WAV (22 050 Hz, 16-bit PCM) |
| Chunking | Same sentence-boundary chunking as Orpheus; WAV chunks are concatenated |
| Requires | `SARVAM_API_KEY` environment variable |

### 7.3 Audio Processing Chain
1. Browser records candidate speech (MediaRecorder API, WebM).
2. Audio is sent either as streamed chunks (`audio_chunk` → `audio_end`) or as one base64 blob (`audio_upload`).
3. Backend buffers/decodes the bytes and writes a temp file.
4. ffmpeg converts the file to 16 kHz mono WAV (30-second timeout).
5. Groq Whisper transcribes the WAV file to text.
6. Temp files are deleted after transcription.
7. The transcript is passed to `InterviewEngine.process_answer()`.
8. The per-question background evaluation is triggered concurrently with next-question generation.

---

## 8. The 16-Phase Per-Question Evaluation Pipeline

 `backend/evaluator/pipeline.py` 

**Concurrency model:** Phases 3, 5, 6, 7, 8 run concurrently via `asyncio.gather()` after Phase 4 (ideal-answer generation) completes, since 5/6/7 consume Phase 4's output. Embedding work runs inside a thread pool executor via `run_in_executor`.

| Phase | Name | What Happens |
|---|---|---|
| 0 | Transcript Cleaning | Regex filler removal, stutter collapse, contraction expansion, punctuation normalisation. Produces `answer_clean`; `answer_raw` preserved. |
| 1 | Input Storage | Stores question/answer/topic/difficulty/type into `PerQuestionResult`. |
| 2 | Basic Filtering | Rejects empty answers or answers under 5 words → `status='filtered'`, all scores 0.0, counted as a zero in the aggregate. |
| 3 | Semantic Relevance | `cosine(embed(question), embed(answer))`; `score = min(10, max(0, sim×12))`. |
| 4 | Ideal Answer Generation | LLM generates a 3–5 sentence reference answer; temperature 0.3, max_tokens 300. |
| 5 | Semantic Correctness | `cosine(embed(answer), embed(ideal))`; `score = min(10, max(0, sim×13))`. |
| 6 | Depth Analysis | LLM JSON `{depth_score, reason}`, temperature 0.1. |
| 7 | Coverage Check | LLM extracts 3–8 key concepts (capped at 8); coverage ratio = concepts with similarity ≥ 0.38 ÷ total concepts; `score = ratio×10`. |
| 8 | Clarity Scoring | LLM JSON `{clarity_score, reason}`, temperature 0.1. |
| 9 | Anti-Gaming Check | Computes a 0–3 penalty subtracted evenly from the five 0–10 sub-scores (clamped at 0). |
| X | STAR Evaluation | Only when `question_type == "behavioral"`. Runs **before** Phase 10. |
| 10 | Per-Question Score | Weighted sum via question-type weight map → `question_score` (0–100). |
| 11 | Store Result | `status='processed'`. DB persistence happens in the caller (`PerQuestionDBService.save()`). |
| 12 | Difficulty Weighting | `question_score × {easy:1.0, medium:1.1, hard:1.2}`, clamped to 100 → `weighted_score`. |
| 13 | Score Aggregation | Computed on-the-fly from all `per_question_evaluations` rows. |
| 14 | Skill Grouping | Per-topic average of `weighted_score`, excluding generic stage-name topics when real topics exist. |
| 15 | Final Decision | `≥75 SELECT`, `≥40 HOLD`, `<40 REJECT`. |
| 16 | Feedback Generation | LLM produces 1–3 strengths/weaknesses/suggestions; temperature 0.2, max_tokens 400. |

---

## 9. The 6 Core Evaluation Metrics — Formulas & Detail

`pipeline.py` / `embeddings/engine.py` 

**Correctness** (0–10): `cosine(embed(answer), embed(ideal_answer))`; `score = min(10, max(0, sim×13))`.

**Relevance** (0–10): `cosine(embed(question), embed(answer))`; `score = min(10, max(0, sim×12))`.

**Depth** (0–10): LLM judging temperature 0.1; JSON `{depth_score, reason}`.

**Coverage** (0–10): LLM extracts ≤8 key concepts from the ideal answer; a concept is "covered" if its embedding similarity to the answer is ≥ 0.38; `score = (covered/total)×10`.

**Clarity** (0–10): LLM judging temperature 0.1; JSON `{clarity_score, reason}`.

**STAR Score** (0–100, behavioral only): see Section 10.

---

## 10. STAR Evaluation & Scoring Weight Maps



| STAR Component | Canonical Description |
|---|---|
| Situation | Setting the context, background, or situation of the story |
| Task | Describing the task, responsibility, or challenge assigned |
| Action | Explaining specific actions taken, steps followed, personal contribution |
| Result | Describing the outcome, result, impact, or what was achieved |

Threshold: cosine similarity ≥ 0.28 marks a component "present". `base_score = (present/4)×100`. `length_bonus = min(10, max(0, (word_count-50)/10))`. `star_score = min(100, base_score + length_bonus)`.

### Weight Maps (`TECHNICAL_WEIGHTS` / `BEHAVIORAL_WEIGHTS` / `GENERAL_WEIGHTS`)

| Metric | Technical | Behavioral | General |
|---|---|---|---|
| correctness | 0.60 | 0.10 | 0.25 |
| relevance | 0.13 | 0.20 | 0.25 |
| depth | 0.10 | 0.15 | 0.15 |
| coverage | 0.10 | 0.05 | 0.15 |
| clarity | 0.07 | 0.20 | 0.20 |
| star_score | 0.00 | 0.30 | 0.00 |

`compute_question_score()`: each 0–10 sub-score is multiplied by 10 to put it on the 0–100 scale (`star_score` is already 0–100), then weighted and summed, clamped to [0, 100].

---

## 11. Anti-Gaming & Transcript Cleaning



| Check | Trigger | Penalty |
|---|---|---|
| Heavy word repetition | >30 words AND unique-word ratio < 0.30 | +2.0 |
| Moderate repetition | >15 words AND unique-word ratio < 0.40 | +1.0 |
| Vague stock phrases | "I don't know", "not sure", "maybe", "it depends", "hard to say", "it's complicated" | +0.5/hit, max 1.5 |
| Filler density | Filler-word rate > 15% | +1.0 |
| Long but meaningless | >150 words AND unique-word ratio < 0.35 | +1.0 |
| Maximum | — | clamped to 3.0 |

The penalty is subtracted from each of the five 0–10 sub-scores after the parallel phases complete, clamped to a minimum of 0.0.

**Transcript Cleaner (Phase 0):** 17 contraction expansions, filler-word regex removal (um, uh, like, you know, basically, literally, actually, honestly, etc.), 3+ repeated-word collapse, whitespace normalisation, and forced sentence-ending punctuation. All regex — no LLM call, near-zero latency.

---

## 12. Scoring Aggregation — From Sub-Scores to Final Decision



```
weighted_score = round(min(100.0, question_score × DIFFICULTY_MULTIPLIER[difficulty]), 2)
# easy:1.0  medium:1.1  hard:1.2

overall_score = round(sum(weighted_score of PROCESSED rows) / TOTAL rows (processed + filtered), 2)
# Filtered (too-short/empty) answers count as 0 in the denominator — this
# penalises candidates who refuse to answer.

skill_scores = per-topic mean of weighted_score, excluding generic stage names
               (greeting, introduction, closing, general, technical, behavioral)
               when at least one real topic exists.

decision:  overall_score ≥ 75 → SELECT
           overall_score ≥ 40 → HOLD
           overall_score < 40 → REJECT
```

---

## 13. Database Schema — Every Table & Column

### 13.1 Table: `evaluations` (interview summary)

| Column | Type | Description |
|---|---|---|
| interview_id | VARCHAR(36) PK | UUID4, matches session ID |
| candidate_name | VARCHAR(255) | From landing page |
| interview_status | VARCHAR(50) | `in_progress` / `completed` / `cancelled` |
| overall_score | FLOAT | Phase 13 aggregate, 0–100 |
| confidence_score | FLOAT | Average of `depth_score` and `clarity_score` across all questions |
| communication_score | FLOAT | Average `clarity_score` across all questions |
| problem_solving_score | FLOAT | Average `depth_score` across all questions |
| technical_knowledge_score | FLOAT | Average `correctness_score` across all questions |
| role_fitment_score | FLOAT | Average of `coverage_score` and `relevance_score` |
| clarity_score | FLOAT | Average `clarity_score` across all questions |
| recommendation | VARCHAR(50) | `SELECT`→"Hire", `HOLD`→"Hold", `REJECT`→"Reject" |
| summary | TEXT | Short narrative summary of the interview (LLM-generated when available, templated fallback otherwise) |
| strengths | JSONB | Aggregated strengths from all per-question evaluations |
| improvement_areas | JSONB | Aggregated improvement suggestions |
| decision_score | FLOAT | Mirrors `overall_score` |
| created_at | TIMESTAMPTZ | UTC timestamp |

### 13.2 Table: `per_question_evaluations` (per-answer detail)

One row per question-answer pair.

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment primary key |
| interview_id | VARCHAR(36) | FK to the interview |
| question_id | VARCHAR(64) | Sequential question identifier |
| question_type | VARCHAR(32) | `technical` / `behavioral` / `general` |
| topic | VARCHAR(128) | Question topic (e.g. "Python") |
| difficulty | VARCHAR(16) | `easy` / `medium` / `hard` |
| question_text | TEXT | Full question text |
| answer_text | TEXT | Candidate's raw answer |
| ideal_answer | TEXT | LLM-generated reference answer |
| key_concepts | JSONB | Concepts extracted in Phase 7 |
| relevance_score | FLOAT | Phase 3 score, 0–10 (post-penalty) |
| correctness_score | FLOAT | Phase 5 score, 0–10 (post-penalty) |
| depth_score | FLOAT | Phase 6 score, 0–10 (post-penalty) |
| coverage_score | FLOAT | Phase 7 score, 0–10 (post-penalty) |
| clarity_score | FLOAT | Phase 8 score, 0–10 (post-penalty) |
| anti_gaming_penalty | FLOAT | Phase 9 penalty subtracted, 0–3 |
| star_score | FLOAT | 0–100 (behavioral only; 0.0 otherwise) |
| star_components | JSONB | Per-component presence + similarity |
| star_missing | JSONB | Missing STAR components |
| question_score | FLOAT | Phase 10 weighted score, 0–100 |
| weighted_score | FLOAT | Phase 12 score after difficulty multiplier |
| depth_reason | TEXT | LLM explanation for depth score |
| clarity_reason | TEXT | LLM explanation for clarity score |
| strengths | JSONB | Phase 16 strengths |
| weaknesses | JSONB | Phase 16 weaknesses |
| improvement_suggestions | JSONB | Phase 16 suggestions |
| status | VARCHAR(20) | `processing` / `processed` / `filtered` |
| filter_reason | VARCHAR(64) | Set if filtered |
| eval_ms | INTEGER | Evaluation time in ms |
| evaluated_at | TIMESTAMPTZ | UTC timestamp |

**Index:** composite `ix_pqeval_interview_qid` on `(interview_id, question_id)`.

**Notes:**
- Default `status` on row creation is `"processing"`, not `"processed"` — it's updated once the pipeline finishes (or set to `"filtered"` if Phase 2 rejects the answer).
- There is no separate `interview_aggregates` table. A code comment in `per_question_db_service.py` notes it "has been removed" — aggregates are computed on the fly from this table instead.

---

## 14. REST API Endpoints

`backend/main.py`, `backend/routers/interview.py`, and `backend/routers/per_question_router.py`.

### 14.1 Core Interview Endpoints (`routers/interview.py`)

| Method + Path | Description |
|---|---|
| GET `/` | Serves `frontend/index.html` |
| GET `/{path}` | SPA fallback — serves matching static file or `index.html` |
| GET `/docs`, GET `/redoc` | Auto-generated API docs |
| GET `/api/health` | Service health/version check |
| POST `/api/start-interview` | multipart/form-data: `candidate_name`, `job_description`, `resume` (file). Returns `interview_id`, `candidate_id`, `first_question`, `timing` |
| GET `/api/final-report/{interview_id}` | Returns the saved `evaluations` row; if missing, rebuilds it on the fly from per-question data |
| POST `/api/tts` | Body `{text, interview_id}` → `{audio: base64-WAV, audio_format: "wav"}` |
| GET `/api/tts/health` | TTS configuration/health report (200 if ok, 503 otherwise) |
| GET `/debug-tts` | Generates a test audio clip and returns size/hash/timing metadata |

### 14.2 Per-Question Endpoints (`routers/per_question_router.py`, prefix `/api/pq-eval`)

| Method + Path | Description |
|---|---|
| POST `/api/pq-eval/submit` | Queues a background 16-phase evaluation (202 Accepted) |
| POST `/api/pq-eval/evaluate-sync` | Runs the pipeline synchronously and returns the full result — for testing/admin re-scoring |
| GET `/api/pq-eval/{interview_id}/questions` | All per-question rows for an interview |
| GET `/api/pq-eval/{interview_id}/question/{question_id}` | A single question's row (202 if still `processing`) |
| GET `/api/pq-eval/{interview_id}/aggregate` | Phases 13–15 output: `overall_score`, `skill_scores`, `decision` |
| GET `/api/pq-eval/{interview_id}/final-report` | Combined aggregate + all per-question rows — used for the results page |

---

## 15. Socket.IO Events — Real-Time Communication Protocol

`@sio.event` decorators in `backend/sockets/handlers.py`

### 15.1 Client → Server Events

| Event | Payload | Server Action |
|---|---|---|
| `connect` | (auto) | Logs the connection; no session created yet (session is created by `POST /api/start-interview`) |
| `join_interview` | `{interview_id}` | Joins the Socket.IO room for that interview, maps `sid → interview_id`, emits `joined_interview` back |
| `audio_chunk` | `{interview_id, data: base64}` | Buffers a raw audio chunk in `AUDIO_BUFFERS[interview_id]` |
| `audio_end` | `{interview_id}` | Concatenates buffered chunks and begins processing (convert → STT → next question → background eval) |
| `audio_upload` | `{interview_id, audio_data: base64, extension, elapsed_time}` | Single-shot alternative to chunk streaming; same processing pipeline |
| `end_interview` | `{interview_id}` | Awaits in-flight processing/eval tasks, builds and saves the final summary, emits `evaluation_ready` |
| `cancel_interview` | `{interview_id}` | Same as `end_interview` but marks `interview_status='cancelled'` |
| `disconnect` | (auto) | Cleans up the `sid → interview_id` mapping |

### 15.2 Server → Client Events

| Event | Payload | When Emitted |
|---|---|---|
| `joined_interview` | `{interview_id, sid}` | Response to `join_interview` |
| `status` | `{interview_id, message}` | Progress updates ("Transcribing your answer…", "Generating next question…", etc.) |
| `heartbeat` | `{interview_id, ts}` | Every 5 seconds while a long-running task is in flight, to keep the connection warm |
| `transcript` | `{interview_id, transcript}` | After STT completes |
| `next_question` | `{interview_id, question, audio, audio_format, is_final, question_count, interview_stage, timing}` | After the next question (or closing statement) is generated |
| `next_question_audio` | `{interview_id, audio, audio_format, is_final}` | Backup TTS delivery if audio wasn't ready in time for `next_question` |
| `evaluation_ready` | `{interview_id, evaluation}` | After `end_interview`/`cancel_interview` finishes building the final summary |
| `error` | `{interview_id, message}` | On any error during audio handling, STT, or evaluation |

**Rooms:** each interview gets its own Socket.IO room (named after `interview_id`); events are emitted to the room so delivery survives reconnects with a new `sid`.

---

## 16. Frontend Architecture

| Page | File |
|---|---|
| Landing Page | `frontend/index.html` + `frontend/js/landing.js` |
| Interview Page | `frontend/interview.html` + `frontend/js/interview.js` (~60 KB) |
| Results Dashboard | `frontend/result.html` + `frontend/js/result.js` |

`interview.js` manages the Socket.IO client connection, MediaRecorder-based audio capture, base64 audio chunking/upload, TTS playback (decoding base64 WAV via the Web Audio API, falling back to `speechSynthesis`), stage progress display, live transcript display, the interview timer, and end/cancel controls.

`result.js` fetches the final report, renders the score gauge and SELECT/HOLD/REJECT badge, shows per-topic skill breakdown, and lists per-question detail (scores, strengths, weaknesses, suggestions, ideal answer alongside the candidate's answer).

---

## 17. Language Support (English · Hindi · Marathi)

### 17.1 Supported Languages

| Code | Language | Script | TTS Provider | STT Language Hint |
|---|---|---|---|---|
| `en` | English | Latin | Groq Orpheus (`canopylabs/orpheus-v1-english`) | `en` |
| `hi` | Hindi | Devanagari | Sarvam AI (`bulbul:v1`, speaker: meera) | `hi` |
| `mr` | Marathi | Devanagari | Sarvam AI (`bulbul:v1`, speaker: meera) | `mr` |

### 17.2 How Language Flows Through the System

1. **Landing page** — candidate selects language from the dropdown (English / हिन्दी / मराठी). The value (`en`, `hi`, or `mr`) is submitted as the `language` form field in `POST /api/start-interview`.
2. **Session initialisation** — `InterviewEngine.initialize_interview()` stores the language code in `candidate_info['language']`. The router validates the value; unknown codes default to `"en"`.
3. **Greeting & closing** — `InterviewEngine` has hard-coded greeting and closing templates in all three languages (`_GREETING_TEMPLATES`, `_CLOSING_TEMPLATES`). The correct template is selected at runtime from `candidate_info['language']`.
4. **Question generation** — when language is not `"en"`, the LLM prompt appended to every question-generation request includes: `"LANGUAGE: Write the question in <Language Name> script."` The LLM then generates the question text in that language.
5. **TTS** — `text_to_speech(text, output_file, language=lang)` routes to `_sarvam_text_to_speech()` for `"hi"` or `"mr"`, and to the Groq Orpheus path for `"en"`.
6. **STT** — `stt_service.transcribe(audio_path, language=lang)` passes the language code to Groq Whisper, improving transcription accuracy for Devanagari speech.
7. **Results page** — scores and feedback are language-agnostic (numeric + English labels); no additional changes are needed on the results side.

### 17.3 Adding a New Language

To add another language (e.g. Tamil `ta`):

1. **TTS** — add an entry to `SARVAM_LANG_CODES` in `text_to_speech.py`: `"ta": "ta-IN"`. Verify that Sarvam AI's `bulbul:v1` model supports `ta-IN`.
2. **Interview Engine** — add a `"ta"` entry to `_GREETING_TEMPLATES` and `_CLOSING_TEMPLATES` in `interview_engine.py`. Add `"ta": "Tamil"` to `_LANGUAGE_NAMES`.
3. **Router validation** — update the allowed-language check in `routers/interview.py` to include `"ta"`.
4. **Frontend** — add `<option value="ta">தமிழ் (Tamil)</option>` to the `#interviewLanguage` select in `frontend/index.html`.
5. **Schema** — no change needed; `language` is a free-form string field with server-side validation.

---

## 18. Setup & Installation Guide

### 18.1 Prerequisites
- Python 3.10+
- PostgreSQL 14+ (running and accessible)
- ffmpeg in system PATH
- Groq API key (console.groq.com/keys)
- Sarvam AI API key (dashboard.sarvam.ai) — **required only if you plan to offer Hindi or Marathi interviews**
- Node.js (optional)

### 18.2 Step-by-Step Installation

```bash
git clone <repository-url>
cd Ai-interview-system-main

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt

createdb ai_interview_db
```

Create `.env` in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here   # Required for Hindi / Marathi interviews
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=ai_interview_db
GROQ_PRIMARY_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_TTS_VOICE=daniel
HOST=0.0.0.0
PORT=5000
```

```bash
cd backend
python main.py
```

Open `http://localhost:5000`.

> Note: `main.py` registers `uvicorn.run("main:socket_app", ...)` — when running with an external uvicorn/gunicorn command instead of `python main.py`, target `main:socket_app`, not `main:app`, or Socket.IO will not be mounted.

### 18.3 Startup Sequence
1. FastAPI lifespan context manager runs.
2. `asyncio` exception handler is registered.
3. `init_db()` creates tables and runs `ALTER TABLE … ADD COLUMN IF NOT EXISTS` migrations.
4. `create_per_question_tables()` ensures `per_question_evaluations` exists.
5. Groq API connectivity check (non-blocking warning if it fails).
6. Background warmup task: loads `all-MiniLM-L6-v2`, starts the audio cleanup loop, runs the TTS health check.
7. Server begins accepting requests; docs at `/docs`.

---