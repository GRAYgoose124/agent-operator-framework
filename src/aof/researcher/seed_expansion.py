"""Seed expansion: one-shot LLM call to generate many follow-up questions from a seed."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from aof.inference.parsing import extract_json, parse_response

if TYPE_CHECKING:
    from aof.config import AppConfig
    from aof.researcher.workspace import ResearcherWorkspace

logger = logging.getLogger(__name__)

SEED_SYSTEM = (
    "You are a research strategist. Given a single seed question, you produce a structured list "
    "of distinct follow-up research questions that explore philosophical, scientific, empirical, "
    "and lateral angles. Avoid duplicates and trivial rephrases. Output only valid JSON."
)
SEED_USER_TEMPLATE = (
    "Seed question: {seed}\n\n"
    "Generate between {min_count} and {max_count} distinct research questions that branch from this seed. "
    "Cover multiple perspectives: conceptual, empirical, comparative, critical, and lateral. "
    "Output a single JSON object with one key, \"questions\", whose value is an array of strings. "
    "Example: {{\"questions\": [\"First question?\", \"Second question?\"]}}"
)
MAX_QUESTIONS_CAP = 50
SEED_EXPANSION_MAX_TOKENS = 2048

# Phrases that indicate a parsed "question" is actually meta/thinking, not a real question
_META_OPENERS = (
    "okay,",
    "let me",
    "i need to",
    "the user wants",
    "so,",
    "first,",
    "alright,",
    "i will",
    "i should",
    "let me start",
    "let me break",
)
_META_PHRASES = ("the user wants me", "i should", "let me start", "i need to use", "maybe the")


def _is_meta_question(q: str) -> bool:
    """True if q looks like meta-commentary or thinking, not a real research question."""
    if len(q) > 200:
        return True
    lower = q.lower().strip()
    if any(lower.startswith(op) for op in _META_OPENERS):
        return True
    if any(phrase in lower for phrase in _META_PHRASES):
        return True
    return False


def _topical_summary(questions: list[str]) -> str:
    """Derive a short topical summary from question texts (categories/themes)."""
    categories = []
    for label in ("Philosophical", "Scientific", "Comparative", "Critical", "Lateral", "Empirical", "Conceptual"):
        if any(label.lower() in q.lower() for q in questions):
            categories.append(label)
    if categories:
        return "Themes: " + ", ".join(categories) + "."
    if len(questions) >= 3:
        return "Topics span multiple angles from the seed question."
    return "Follow-up questions generated from seed."


def _parse_questions_from_response(text: str) -> list[str]:
    """Extract list of questions from LLM response. Prefer JSON; fallback to numbered/bullet list."""
    data = extract_json(text)
    if isinstance(data.get("questions"), list):
        return [str(q).strip() for q in data["questions"] if q]
    if isinstance(data.get("steps"), list):
        return [str(s).strip() for s in data["steps"] if s]
    # Fallback: numbered lines or bullet lines
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading number or bullet
        m = re.match(r"^(?:\d+[.)]\s*|[-\*]\s*)(.+)$", line)
        if m:
            line = m.group(1).strip()
        if len(line) > 10 and "?" in line:
            questions.append(line)
    return questions


async def run_seed_expansion(
    config: AppConfig,
    queue_path: str,
    seed_question: str,
    workspace: "ResearcherWorkspace | None" = None,
) -> int:
    """Add seed to queue, run one-shot expansion with seed_expansion_role, add generated questions.

    Returns the number of follow-up questions added to the queue (excluding the seed).
    Optionally writes a seed summary note to workspace memory when workspace is provided.
    """
    from aof.research import ResearchQueue
    from aof.researcher.runner import _make_multi_backends

    queue = ResearchQueue(queue_path)
    existing = {
        item.question.strip().lower()
        for items in queue.list_all().values()
        for item in items
    }
    if seed_question.strip().lower() not in existing:
        queue.add(seed_question.strip())
    role = config.research.seed_expansion_role or "medium"
    target = min(MAX_QUESTIONS_CAP, config.research.seed_expansion_count or 30)
    min_count = max(15, target - 15)
    max_count = target

    backends = await _make_multi_backends(config, {role, "general", "default"})
    backend = backends.get(role) or backends.get("general") or backends.get("default")
    if not backend:
        backend = next(iter(backends.values()))

    user_content = SEED_USER_TEMPLATE.format(
        seed=seed_question,
        min_count=min_count,
        max_count=max_count,
    )
    messages = [
        {"role": "system", "content": SEED_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        result = await backend.complete(messages, max_tokens=SEED_EXPANSION_MAX_TOKENS)
        # Strip <think> blocks so we parse JSON from the actual response, not thinking
        clean_text = parse_response(result.text).text
        questions = _parse_questions_from_response(clean_text)
        added = 0
        seen = existing | {seed_question.strip().lower()}
        added_list: list[str] = []
        for q in questions[:MAX_QUESTIONS_CAP]:
            q = q.strip()
            if not q or len(q) < 8:
                continue
            if _is_meta_question(q):
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            queue.add(q)
            added += 1
            added_list.append(q)
        logger.info("Seed expansion: added %d follow-up questions (role=%s)", added, role)

        if workspace and added > 0:
            from aof.memory.store import MemoryStore

            memory = MemoryStore(config.memory)
            await memory.initialize()
            try:
                summary = _topical_summary(added_list)
                content = (
                    f"Seed: {seed_question}\n\n"
                    f"Model: {role} (used for seed expansion).\n"
                    f"Generated {added} follow-up questions.\n\n"
                    f"{summary}"
                )
                await memory.create_note(
                    title="Seed expansion",
                    content=content,
                    tags=["seed", "research"],
                    source="seed_expansion",
                    agent_id="runner",
                )
            finally:
                await memory.close()

        return added
    finally:
        for be in set(backends.values()):
            await be.shutdown()
