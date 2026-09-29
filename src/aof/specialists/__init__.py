"""Specialist models: capability -> provider chains (Needle, Liquid Nanos, any AOF role, embeddings)."""

from aof.specialists.base import (
    ANNOTATE,
    CAPABILITIES,
    CLASSIFY,
    EMBED,
    JUDGE,
    VERDICTS,
    Provider,
    SpecialistError,
    SpecialistExhausted,
    SpecialistResult,
)
from aof.specialists.registry import ProviderStats, SpecialistRegistry, build_registry

__all__ = [
    "ANNOTATE", "CAPABILITIES", "CLASSIFY", "EMBED", "JUDGE", "VERDICTS",
    "Provider", "ProviderStats", "SpecialistError", "SpecialistExhausted", "SpecialistRegistry",
    "SpecialistResult", "build_registry",
]
