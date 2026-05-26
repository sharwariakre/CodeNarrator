import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

from codenarrator.services.analysis_snapshot_service import _compute_dependency_graph_summary
from codenarrator.core.config import settings

LOGGER = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = settings.OLLAMA_MODEL


def interpret_architecture(analysis_state: Dict) -> Dict:
    """
    AI-assisted interpretation with deterministic fallbacks.

    Always returns a dict so the report has something to render. Tries the AI
    path first; on timeout, parse failure, or validation producing no usable
    key_dependencies, falls back to deterministic edges derived from
    graph_summary.internal_edges (top 5 by target in-degree).
    """
    graph_summary = _compute_dependency_graph_summary(analysis_state)
    fallback_deps = _build_fallback_key_dependencies(graph_summary)

    try:
        payload = _build_interpretation_payload(analysis_state, graph_summary)
        prompt = _build_prompt(payload)
        response_text = _call_ollama(prompt)
        parsed = _parse_interpretation_json(response_text)
        if parsed is not None:
            explored_paths = {
                fact["file_path"]
                for fact in analysis_state.get("inspected_facts", [])
                if fact.get("file_path")
            }
            parsed = _validate_interpretation(parsed, explored_paths)
            # AI returned valid JSON but validation stripped all edges
            # (model pointed only at external libs / phantom files).
            if not parsed.get("key_dependencies"):
                parsed["key_dependencies"] = fallback_deps
            return parsed
    except Exception as exc:
        LOGGER.warning("AI interpretation failed: %s", exc)

    # AI path failed entirely (timeout, parse error, or returned None).
    return {
        "architecture_pattern": "Not available",
        "main_components": [],
        "key_dependencies": fallback_deps,
        "confidence": 0.0,
    }


def _build_fallback_key_dependencies(graph_summary: Dict) -> list:
    """
    Top 5 internal edges by target in-degree, formatted as key_dependencies
    entries with the deterministic-fallback reason string.
    """
    edges = graph_summary.get("internal_edges", [])
    in_degree: Dict[str, int] = {}
    for edge in edges:
        target = edge.get("to")
        if target:
            in_degree[target] = in_degree.get(target, 0) + 1
    top_edges = sorted(
        (e for e in edges if e.get("from") and e.get("to")),
        key=lambda e: in_degree.get(e["to"], 0),
        reverse=True,
    )[:5]
    return [
        {
            "from": e["from"],
            "to": e["to"],
            "reason": "high-import-count dependency (deterministic fallback)",
        }
        for e in top_edges
    ]


def _build_interpretation_payload(analysis_state: Dict, graph_summary: Dict) -> Dict:
    inspected_facts = analysis_state.get("inspected_facts", [])[:30]

    compact_facts = [
        {
            "file_path": fact.get("file_path"),
            "language": fact.get("language"),
            "role_hint": fact.get("role_hint"),
            "imported_modules": fact.get("imported_modules", [])[:12],
        }
        for fact in inspected_facts
    ]

    return {
        "internal_edges": graph_summary.get("internal_edges", [])[:120],
        "clusters": graph_summary.get("clusters", [])[:20],
        "highest_dependency_files": graph_summary.get("highest_dependency_files", [])[:20],
        "inspected_facts": compact_facts,
    }


def _build_prompt(payload: Dict) -> str:
    schema = {
        "architecture_pattern": "string",
        "main_components": [
            {
                "name": "string",
                "files": ["string"],
                "description": "string",
            }
        ],
        "key_dependencies": [
            {
                "from": "string",
                "to": "string",
                "reason": "string",
            }
        ],
        "summary_for_new_developer": "string (3-5 sentences)",
    }

    return (
        "You are analyzing repository architecture from dependency evidence.\n"
        "Use only the provided data. Do not invent files.\n"
        "\n"
        "key_dependencies must be edges between two files that appear in the "
        "inspected_facts list. Both 'from' and 'to' must be actual file paths "
        "from that list — not external libraries (fastapi, react, numpy etc.), "
        "not invented paths, not directories. If you are unsure, omit the edge. "
        "For example, an edge from app/services/auth.py to app/models/user.py "
        "is valid; an edge from app/main.py to 'fastapi' is not.\n"
        "\n"
        "Return JSON only, matching this schema:\n"
        f"{json.dumps(schema)}\n\n"
        "Evidence:\n"
        f"{json.dumps(payload)}"
    )


_RETRY_DELAYS = [1, 2]  # seconds between attempts; total = 3 tries


def _call_ollama(prompt: str) -> str:
    request_body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt, _ in enumerate(["first"] + _RETRY_DELAYS):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            LOGGER.warning("Ollama call failed (attempt %d), retrying in %ds…", attempt, delay)
            time.sleep(delay)

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            return payload.get("response", "")
        except urllib.error.URLError as exc:
            last_exc = exc

    raise RuntimeError(f"Ollama request failed after {len(_RETRY_DELAYS) + 1} attempts: {last_exc}") from last_exc


def _parse_interpretation_json(response_text: str) -> Optional[Dict]:
    if not response_text:
        return None

    parsed = _load_json_loose(response_text)
    if parsed is None:
        return None

    required_keys = {
        "architecture_pattern",
        "main_components",
        "key_dependencies",
        "summary_for_new_developer",
    }
    if not isinstance(parsed, dict):
        return None
    if not required_keys.issubset(parsed.keys()):
        return None

    return parsed


def _validate_interpretation(interpretation: Dict, explored_paths: set) -> Dict:
    """
    Strip phantom file references from the AI output.
    - Components: filter file list to explored files only, but keep the component
      even if no files survive (name + description are still meaningful).
    - Key dependencies: only keep edges where both endpoints were actually explored.
    Deterministic key_dependencies fallback lives in interpret_architecture so it
    also fires when the AI call itself fails (timeout / parse error).
    """
    components = []
    for component in interpretation.get("main_components", []):
        valid_files = [f for f in component.get("files", []) if f in explored_paths]
        # Keep component regardless — just trim the file list.
        components.append({**component, "files": valid_files})
    interpretation["main_components"] = components

    interpretation["key_dependencies"] = [
        dep for dep in interpretation.get("key_dependencies", [])
        if dep.get("from") in explored_paths and dep.get("to") in explored_paths
    ]

    return interpretation


def _load_json_loose(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
