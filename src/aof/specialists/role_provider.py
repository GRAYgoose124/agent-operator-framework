"""`role:<name>` provider: any configured AOF model role (GGUF, Liquid Nano, llama-server) as a specialist.

Roles use the schema-constrained JSON path (`complete_json`), then validate the result. Invalid or
missing fields abstain so the registry escalates. Confidence is not available from these models
(no calibrated score), so it is left unset.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Sequence

from aof.config import AppConfig, resolve_role_path
from aof.specialists.base import (
    ANNOTATE,
    CLASSIFY,
    GENERATE,
    JUDGE,
    VERDICTS,
    SpecialistResult,
    label_schema,
    validate_fields,
)

logger = logging.getLogger(__name__)

BackendGetter = Callable[[str], Awaitable[Any]]

_JUDGE_SCHEMA = {
    "name": "judge_claim",
    "description": "Judge whether the evidence supports the claim.",
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "rationale": {"type": "string", "description": "one short sentence"},
            "query": {"type": "string", "description": "a web search query if verdict is needs_lookup or unsupported"},
        },
        "required": ["verdict", "rationale"],
    },
}


def _fields_prompt(schema: dict) -> str:
    params = schema.get("parameters", schema)
    lines = []
    for key, spec in params.get("properties", {}).items():
        detail = spec.get("description", "")
        if "enum" in spec:
            detail = (detail + " " if detail else "") + "One of: " + ", ".join(spec["enum"]) + "."
        lines.append(f"- {key}: {detail}".rstrip())
    return "\n".join(lines)


class RoleProvider:
    """Wraps one configured model role."""

    capabilities = (ANNOTATE, CLASSIFY, JUDGE, GENERATE)

    def __init__(self, role: str, config: AppConfig, get_backend: BackendGetter) -> None:
        self.role = role
        self.name = f"role:{role}"
        self._config = config
        self._get_backend = get_backend

    def available(self) -> bool:
        path = resolve_role_path(self._config, self.role)
        return bool(path) and os.path.exists(path)

    async def _json(self, system: str, user: str, schema: dict) -> dict | None:
        backend = await self._get_backend(self.role)
        try:
            data = await backend.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                schema.get("parameters", schema),
            )
        except Exception as e:
            logger.warning("%s complete_json failed (%s: %s); escalating", self.name, type(e).__name__, str(e)[:120])
            return None
        return data if isinstance(data, dict) else None

    async def annotate(self, text: str, schema: dict) -> SpecialistResult | None:
        system = (
            "You extract fields from text. Use only what the text states. "
            "Reply with a single JSON object with exactly these fields:\n" + _fields_prompt(schema)
        )
        data = await self._json(system, f"Text:\n{text}", schema)
        if data is None or not validate_fields(schema, data):
            return None
        return SpecialistResult(data, self.name)

    async def classify(self, text: str, labels: Sequence[str], *, description: str = "") -> SpecialistResult | None:
        schema = label_schema(labels, description or "Pick the single best label for the text.")
        system = (
            f"{schema['description']} Reply with JSON like {{\"label\": \"<one of: {', '.join(labels)}>\"}}."
        )
        data = await self._json(system, f"Text:\n{text}", schema)
        if data is None or not validate_fields(schema, data):
            return None
        return SpecialistResult(data["label"], self.name)

    async def judge(self, claim: str, evidence: str) -> SpecialistResult | None:
        system = (
            "You verify claims against evidence. Verdicts: supported (the evidence states or clearly implies it), "
            "contradicted (the evidence says otherwise), unsupported (the evidence does not address it), "
            "needs_lookup (it cannot be judged without outside information). "
            "Use only the evidence given. Reply with JSON: " + json.dumps(
                {"verdict": "|".join(VERDICTS), "rationale": "one sentence", "query": "optional search query"}
            )
        )
        data = await self._json(system, f"Claim: {claim}\n\nEvidence:\n{evidence}", _JUDGE_SCHEMA)
        if data is None or not validate_fields(_JUDGE_SCHEMA, data):
            return None
        return SpecialistResult(data, self.name)

    async def generate(self, system: str, user: str, *, max_tokens: int = 400) -> SpecialistResult | None:
        from aof.inference.parsing import parse_response

        backend = await self._get_backend(self.role)
        try:
            result = await backend.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3, max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("%s generate failed (%s: %s); escalating", self.name, type(e).__name__, str(e)[:120])
            return None
        text = parse_response(result.text).text.strip()
        return SpecialistResult(text, self.name) if text else None

    async def close(self) -> None:
        return None
