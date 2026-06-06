"""
LLM provider base classes and shared tool-call extraction.

The two value types — :class:`ToolCall` and :class:`ChatResponse` — are
flat dataclasses that callers consume directly: no `.message.tool_calls`
indirection, no provider-specific shapes leak past this module.

The shared :func:`extract_tool_calls` helper is ported verbatim from the
old ``agentic_analysis_service._extract_tool_calls``. It handles both the
"native" path (the provider returned a structured tool-call list) and the
"prose-JSON" fallback path (some models emit tool calls as a JSON object
embedded in message content — qwen2.5-coder does this consistently).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A single tool call produced by the model."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Provider-agnostic LLM response."""

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract LLM transport. Subclasses wire in a specific backend."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        timeout: int = 180,
    ) -> ChatResponse:
        """
        Send ``messages`` (in OpenAI/Ollama chat shape) to the model and
        return a :class:`ChatResponse`.

        ``tools``, if not None, follows the OpenAI-style function-tool schema
        (``[{"type": "function", "function": {"name", "description",
        "parameters"}}]``).

        On retry exhaustion or any underlying transport failure the
        implementation must raise; the two services translate that into
        their respective failure paths (the agentic loop stops the run,
        the interpreter falls back to deterministic edges).
        """
        ...


def extract_tool_calls(
    content: Optional[str],
    native_tool_calls: Optional[List[Any]],
) -> List[ToolCall]:
    """
    Normalise a provider's tool-call output into a flat ``list[ToolCall]``.

    Ported verbatim from ``agentic_analysis_service._extract_tool_calls``:

    1. If the provider populated a structured tool-call list (Ollama's
       ``response.message.tool_calls``), convert each entry's
       ``.function.name`` / ``.function.arguments`` into a flat :class:`ToolCall`.
    2. Otherwise, scrape JSON out of the message content. Handles markdown
       code fences (\\\\\\`\\\\\\`\\\\\\`json … \\\\\\`\\\\\\`\\\\\\`) and bare objects/arrays. Single-dict
       outputs (``{"name", "arguments"}``) get wrapped to a one-element list,
       and both ``{"name", "arguments"}`` and ``{"function": {"name", "arguments"}}``
       envelope shapes are accepted.

    Behaviour is preserved exactly; only the return shape is flat (no
    nested ``.function`` indirection on the consumer side).
    """
    # Native path — provider returned a structured tool-call list.
    if native_tool_calls:
        return [
            ToolCall(
                name=tc.function.name,
                arguments=dict(tc.function.arguments or {}),
            )
            for tc in native_tool_calls
        ]

    # Prose-JSON fallback — model embedded the tool call(s) as JSON in content.
    text = (content or "").strip()
    if not text:
        return []

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    else:
        match = re.search(r"(\{|\[)", text)
        if not match:
            return []
        raw = text[match.start():]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Single call -> wrap to list.
    if isinstance(parsed, dict) and "name" in parsed:
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    calls: List[ToolCall] = []
    for item in parsed:
        name = item.get("name") or item.get("function", {}).get("name")
        args = item.get("arguments") or item.get("function", {}).get("arguments") or {}
        if name:
            calls.append(ToolCall(name=name, arguments=args))
    return calls
