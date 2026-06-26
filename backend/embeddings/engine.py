from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"   


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"[EMB] Embedding model loaded: {MODEL_NAME}")
    return model


# ── Public helpers ────────────────────────────────────────────────────────────

def _encode_sync(texts: List[str]) -> np.ndarray:
    
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cosine_sync(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors (already normalised)."""
    return float(np.dot(a, b))


async def encode(texts: List[str]) -> np.ndarray:
   
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode_sync, texts)


async def cosine_similarity(text_a: str, text_b: str) -> float:
    
    embs = await encode([text_a, text_b])
    return _cosine_sync(embs[0], embs[1])


async def coverage_ratio(
    concepts: List[str],
    answer: str,
    threshold: float = 0.38,
) -> float:
    
    if not concepts or not answer.strip():
        return 0.0

    loop = asyncio.get_event_loop()
    all_texts = concepts + [answer]
    embs = await loop.run_in_executor(None, _encode_sync, all_texts)

    concept_embs = embs[: len(concepts)]
    answer_emb   = embs[-1]

    sims    = np.dot(concept_embs, answer_emb)          
    covered = int(np.sum(sims >= threshold))
    return covered / len(concepts)
