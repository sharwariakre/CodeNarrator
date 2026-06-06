"""
LLM provider abstraction.

The two LLM-touching services (`agentic_analysis_service` and `ai_interpreter`)
talk only to the :class:`LLMProvider` interface defined in :mod:`.base`.
Concrete providers live in sibling modules (currently :mod:`.ollama_provider`).
"""
from codenarrator.services.llm.base import (
    ChatResponse,
    LLMProvider,
    ToolCall,
    extract_tool_calls,
)
from codenarrator.services.llm.ollama_provider import OllamaProvider

__all__ = [
    "ChatResponse",
    "LLMProvider",
    "OllamaProvider",
    "ToolCall",
    "extract_tool_calls",
]
