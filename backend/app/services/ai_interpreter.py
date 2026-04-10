import json
import logging
import re
import urllib.error
import urllib.request
from typing import Dict, Optional

from app.services.analysis_snapshot_service import _compute_dependency_graph_summary

LOGGER = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"


def interpret_architecture(analysis_state: Dict) -> Optional[Dict]:
    """
    Optional AI interpretation for architecture understanding.
    Returns None on any failure.
    """
    try:
        graph_summary = _compute_dependency_graph_summary(analysis_state)
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
        return parsed
    except Exception as exc:  # pragma: no cover - fallback safety for optional layer
        LOGGER.warning("AI interpretation failed: %s", exc)
        return None


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
        "Return JSON only, matching this schema:\n"
        f"{json.dumps(schema)}\n\n"
        "Evidence:\n"
        f"{json.dumps(payload)}"
    )


def _call_ollama(prompt: str) -> str:
    request_body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    payload = json.loads(raw)
    return payload.get("response", "")


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
