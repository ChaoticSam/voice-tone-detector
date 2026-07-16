"""Thin LLM provider abstraction for the lexical-fusion step.

Keeps the model a one-line swap (config `LLM_PROVIDER` / `LLM_MODEL`) — matching the
pipeline's "replaceable" principle. Currently implements OpenAI; the transcript (derived
text) is what gets sent, disclosed per the trial's data-handling rules (audio stays local).

Raises `LLMUnavailable` when the provider can't be reached (missing key, network, API
error) so callers can fall back to the acoustic-only verdict rather than crash the batch.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from app.config import LLM_MODEL, LLM_PROVIDER


class LLMUnavailable(Exception):
    """Raised when the LLM cannot be called (no key, network, or API error)."""


@lru_cache(maxsize=1)
def _openai_client() -> Any:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise LLMUnavailable("OPENAI_API_KEY is not set")
    return OpenAI()


def complete_json(system: str, user: str, *, max_tokens: int = 200) -> dict:
    """Call the configured LLM and parse a JSON object from its reply.

    Uses JSON response mode so the reply is always a JSON object. Raises LLMUnavailable
    on any provider/parse failure.
    """
    if LLM_PROVIDER != "openai":
        raise LLMUnavailable(f"unsupported LLM provider: {LLM_PROVIDER}")

    try:
        client = _openai_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except LLMUnavailable:
        raise
    except Exception as err:  # noqa: BLE001 - any provider/parse error -> fall back
        raise LLMUnavailable(str(err)) from err
