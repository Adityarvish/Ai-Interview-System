from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional

from groq import Groq

from config.settings import Config

logger = logging.getLogger(__name__)

_PRIMARY  = Config.GROQ_PRIMARY_MODEL
_FALLBACK = Config.GROQ_FALLBACK_MODEL

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


# ── Sync internals ────────────────────────────────────────────────────────────

def _generate_sync(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model       = model,
        messages    = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens  = max_tokens,
        timeout     = timeout,
    )
    return resp.choices[0].message.content or ""


def _try_generate(
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    
    try:
        return _generate_sync(prompt, _PRIMARY, temperature, max_tokens, timeout)
    except Exception as e:
        logger.warning(f"[LLM] Primary model failed ({e}), trying fallback")
        return _generate_sync(prompt, _FALLBACK, temperature, max_tokens, timeout)


# ── Public async API ──────────────────────────────────────────────────────────

async def generate(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    
    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _try_generate(prompt, temperature, max_tokens, timeout),
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[LLM] generate: {elapsed}ms, {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"[LLM] generate failed: {e}")
        return ""


async def generate_json(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 60,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    raw = await generate(prompt, temperature, max_tokens, timeout)
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning(f"[LLM] JSON parse failed. Raw snippet: {raw[:200]}")
        return fallback if fallback is not None else {}
    return parsed


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1)

    # Locate the outermost JSON object or array
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Attempt light repair: trailing commas, single quotes
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                repaired = repaired.replace("'", '"')
                try:
                    return json.loads(repaired)
                except Exception:
                    pass
    return None
