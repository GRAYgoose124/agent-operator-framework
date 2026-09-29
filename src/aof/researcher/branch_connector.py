"""Branch connector: periodic synthesis note linking themes across research items."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aof.config import AppConfig
    from aof.researcher.workspace import ResearcherWorkspace

logger = logging.getLogger(__name__)

CONNECTOR_SYSTEM = (
    "You synthesize research. Given recent research summaries and themes, you write a short "
    "'Connections' note that links recurring ideas, contrasts, and open gaps across the branches. "
    "Output only the note body (no title). Be concise."
)
CONNECTOR_USER_TEMPLATE = (
    "Recent research artifacts (titles and snippets):\n{artifacts}\n\n"
    "Recent memory/synthesis titles: {memory_titles}\n\n"
    "Write a Connections note linking themes, contrasts, and gaps."
)


async def run_branch_connector_loop(
    config: AppConfig,
    workspace: "ResearcherWorkspace",
    control: dict,
    *,
    interval_seconds: int = 600,
    backend_role: str = "medium",
) -> None:
    """Every interval_seconds, gather recent artifacts + memory, run one LLM call, create_note(connections)."""
    from aof.memory.store import MemoryStore
    from aof.researcher.runner import _make_multi_backends

    memory = MemoryStore(config.memory)
    await memory.initialize()
    backends = await _make_multi_backends(config, {backend_role, "general", "default"})
    backend = backends.get(backend_role) or backends.get("general") or backends.get("default")
    if not backend:
        backend = next(iter(backends.values()))

    last_fingerprint: tuple | None = None
    try:
        while not control.get("quit", False):
            await asyncio.sleep(interval_seconds)
            if control.get("quit", False):
                break

            art_dir = workspace.artifacts_dir
            artifact_parts = []
            if art_dir.exists():
                main_mds = [
                    p for p in art_dir.glob("*.md")
                    if not p.stem.endswith("_progress") and not p.stem.endswith("_sources")
                ]
                by_mtime = sorted(main_mds, key=lambda p: p.stat().st_mtime, reverse=True)
                for p in by_mtime[:8]:
                    try:
                        content = p.read_text(encoding="utf-8").strip()
                        first = content.split("\n")[0][:80] if content else ""
                        snippet = content[:300].replace("\n", " ") if content else ""
                        artifact_parts.append(f"- {p.stem}: {first}\n  {snippet}...")
                    except Exception:
                        artifact_parts.append(f"- {p.stem}")
            artifacts_text = "\n".join(artifact_parts) if artifact_parts else "(none yet)"
            fingerprint = tuple(artifact_parts)
            if not artifact_parts or fingerprint == last_fingerprint:
                continue  # nothing new since the last Connections note

            try:
                notes = await memory.get_recent(limit=5)
                memory_titles = "; ".join(n.title[:50] for n in notes) or "(none)"
            except Exception:
                memory_titles = "(none)"

            user_content = CONNECTOR_USER_TEMPLATE.format(
                artifacts=artifacts_text,
                memory_titles=memory_titles,
            )
            messages = [
                {"role": "system", "content": CONNECTOR_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            try:
                result = await backend.complete(messages)
                text = getattr(result, "text", str(result))
            except Exception as e:
                logger.warning("Branch connector LLM call failed: %s", e)
                continue
            if not text or len(text.strip()) < 20:
                continue
            await memory.create_note(
                title="Connections",
                content=text.strip(),
                tags=["connections", "research"],
                source="branch_connector",
                agent_id="runner",
            )
            last_fingerprint = fingerprint
            logger.info("Branch connector: wrote Connections note.")
    except asyncio.CancelledError:
        pass
    finally:
        for be in set(backends.values()):
            await be.shutdown()
        await memory.close()
