"""Specialist capabilities, results and the provider contract.

A *specialist* is any model (or model-free component) that can do one narrow job well:
fielded extraction, classification, embedding, claim judging. Providers implement the capability
methods they support; the registry cascades cheap -> capable and records what happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

ANNOTATE = "annotate"  # text + tool-shaped schema -> dict of fields
CLASSIFY = "classify"  # text + labels -> label
EMBED = "embed"  # texts -> vectors
JUDGE = "judge"  # claim + evidence -> verdict dict
GENERATE = "generate"  # system + user prompt -> free text (query planning, rewriting)

CAPABILITIES = (ANNOTATE, CLASSIFY, EMBED, JUDGE, GENERATE)

VERDICTS = ("supported", "contradicted", "unsupported", "needs_lookup")


@dataclass(frozen=True)
class SpecialistResult:
    """Outcome of a capability call: the value plus who produced it and how sure they were."""

    value: Any
    provider: str
    confidence: float | None = None  # provider-native scale; do not compare across providers


class SpecialistError(RuntimeError):
    """A provider failed in a way that should escalate to the next one."""


class SpecialistExhausted(SpecialistError):
    """Every provider in the chain abstained, errored, or was unavailable."""


@runtime_checkable
class Provider(Protocol):
    """A named provider. Implement any of annotate/classify/embed/judge; others are skipped.

    Capability methods return a `SpecialistResult`, or `None` to *abstain* (escalate without error).
    """

    name: str

    def available(self) -> bool:
        """False if a dependency or model file is missing; the registry then skips this provider."""
        ...

    async def close(self) -> None: ...


def validate_fields(schema: dict, fields: dict) -> bool:
    """True if `fields` satisfies a tool-shaped schema's required keys and enums."""
    params = schema.get("parameters", schema)
    props = params.get("properties", {})
    for key in params.get("required", ()):
        if key not in fields:
            return False
    for key, value in fields.items():
        enum = props.get(key, {}).get("enum")
        if enum is not None and value not in enum:
            return False
    return True


def label_schema(labels: Sequence[str], description: str = "Pick the single best label.") -> dict:
    """Tool-shaped schema whose only field is a label restricted to `labels`."""
    return {
        "name": "classify",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(labels)}},
            "required": ["label"],
        },
    }
