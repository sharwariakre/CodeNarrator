"""Ollama-backed implementation of :class:`LLMProvider`."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import ollama

from codenarrator.services.llm.base import (
    ChatResponse,
    LLMProvider,
    extract_tool_calls,
)

LOGGER = logging.getLogger(__name__)

# Three attempts total with linear backoff (1s, 2s) between them — preserves
# the historical contract of both _call_model_with_retry (agentic loop) and
# _call_ollama (ai_interpreter).
_RETRY_ATTEMPTS = 3


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local or remote Ollama server."""

    def __init__(self, model: str, host: Optional[str] = None):
        self.model = model
        self.host = host
        # When ``host`` is None, the ``ollama`` module's default Client reads
        # the OLLAMA_HOST env var or falls back to ``http://localhost:11434``.
        # When ``host`` is provided we use an explicit Client so the env var
        # cannot override the caller's intent.
        if host:
            self._client: Any = ollama.Client(host=host)
        else:
            self._client = ollama

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        timeout: int = 180,
    ) -> ChatResponse:
        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            if attempt > 0:
                # Linear backoff: 1s before attempt 2, 2s before attempt 3.
                time.sleep(attempt)
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    options={"temperature": temperature},
                )
                return ChatResponse(
                    content=(response.message.content or ""),
                    tool_calls=extract_tool_calls(
                        content=response.message.content,
                        native_tool_calls=response.message.tool_calls,
                    ),
                )
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt + 1, _RETRY_ATTEMPTS, exc,
                )
        raise RuntimeError(
            f"Ollama request failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc
