"""Lateral thinking: proactive tangential questions added to the research queue."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aof.inference.parsing import extract_json

if TYPE_CHECKING:
    from aof.config import AppConfig

logger = logging.getLogger(__name__)

LATERAL_SYSTEM = (
    "You suggest lateral or tangential research questions. Given recent research findings, "
    "you propose 3–5 questions that connect or diverge from the main thread. "
    "Output only valid JSON: {\"questions\": [\"...\", \"...\"]}."
)
LATERAL_USER_TEMPLATE = (
    "Recent done (do not duplicate):\n{done_preview}\n\n"
    "Current backlog (do not duplicate):\n{backlog_preview}\n\n"
    "Suggest 3–5 lateral or tangential research questions. Output JSON: {{\"questions\": [\"q1\", \"q2\"]}}."
)


def _parse_questions(text: str) -> list[str]:
    """Extract questions list from JSON or fallback lines."""
    data = extract_json(text)
    if isinstance(data.get("questions"), list):
        return [str(q).strip() for q in data["questions"] if q]
    lines = text.strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*")) and "?" in line:
            out.append(line.lstrip("-* ").strip())
        elif line[0].isdigit() and "?" in line:
            idx = 0
            while idx < len(line) and (line[idx].isdigit() or line[idx] in ".)"):
                idx += 1
            out.append(line[idx:].strip())
    return out[:5]


async def run_lateral_thinking_loop(
    config: AppConfig,
    queue_path: str,
    control: dict,
    *,
    interval_seconds: int = 300,
    min_backlog_target: int = 5,
    backend_role: str = "medium",
) -> None:
    """Every interval_seconds, when backlog < min_backlog_target, add 3–5 lateral questions from LLM."""
    from aof.research import ResearchQueue
    from aof.researcher.runner import _make_multi_backends

    queue = ResearchQueue(queue_path)
    backends = await _make_multi_backends(config, {backend_role, "general", "default"})
    backend = backends.get(backend_role) or backends.get("general") or backends.get("default")
    if not backend:
        backend = next(iter(backends.values()))

    try:
        while not control.get("quit", False):
            await asyncio.sleep(interval_seconds)
            if control.get("quit", False):
                break
            grouped = queue.list_all()
            backlog = grouped.get("backlog", [])
            done = grouped.get("done", [])
            if len(backlog) >= min_backlog_target:
                continue
            done_preview = "\n".join(f"- {i.question[:80]}" for i in (done[-10:] if len(done) > 10 else done))
            backlog_preview = "\n".join(f"- {i.question[:80]}" for i in backlog[:20])
            user_content = LATERAL_USER_TEMPLATE.format(
                done_preview=done_preview or "(none)",
                backlog_preview=backlog_preview or "(none)",
            )
            messages = [
                {"role": "system", "content": LATERAL_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            try:
                result = await backend.complete(messages)
                text = getattr(result, "text", str(result))
                questions = _parse_questions(text)
            except Exception as e:
                logger.warning("Lateral thinking LLM call failed: %s", e)
                continue
            seen = {i.question.strip().lower() for i in backlog + done[-20:]}
            added = 0
            for q in questions:
                q = q.strip()
                if not q or len(q) < 8:
                    continue
                if q.lower() in seen:
                    continue
                seen.add(q.lower())
                queue.add(q)
                added += 1
            if added:
                logger.info("Lateral thinking: added %d questions.", added)
    except asyncio.CancelledError:
        pass
    finally:
        for be in set(backends.values()):
            await be.shutdown()
