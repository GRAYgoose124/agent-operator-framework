"""Memory consolidation: cluster related notes and synthesize summaries."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aof.config import AppConfig
    from aof.inference.backend import InferenceBackend
    from aof.memory.store import MemoryStore
    from aof.memory.zettel import ZettelNote

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Clusters related notes and synthesizes summaries.

    Notes tagged 'reflection' or 'auto' (but not 'archived') are grouped by their
    primary tag. Clusters of 3+ notes get synthesized into a consolidated summary
    via a micro model. Source notes are tagged 'archived'; the consolidated note
    links back to them.
    """

    def __init__(
        self,
        memory: MemoryStore,
        config: AppConfig,
        backend: InferenceBackend | None = None,
    ) -> None:
        self._memory = memory
        self._config = config
        self._backend = backend
        self._min_cluster_size = 3

    async def run_once(self) -> int:
        """Run one consolidation pass. Returns number of consolidated clusters."""
        min_notes = self._config.memory.consolidation_min_notes
        total = await self._memory.count()
        if total < min_notes:
            logger.debug("Consolidation: %d notes < min %d, skipping", total, min_notes)
            return 0

        # Get candidate notes (tagged auto/reflection, not archived)
        from aof.researcher.artifacts import _is_overflow_message

        raw = await self._memory.get_notes_by_tags(
            ["reflection", "auto", "research", "synthesis"],
            limit=200,
            exclude_tags=["archived", "consolidated", "knowledge_summary", "topic_map"],
        )
        # Exclude notes that are only context-overflow error markers
        candidates = [n for n in raw if not _is_overflow_message(n.content)]
        if len(candidates) < self._min_cluster_size:
            return 0

        # Cluster by primary tag
        clusters = self._cluster_by_tag(candidates)
        consolidated = 0

        for tag, notes in clusters.items():
            if len(notes) < self._min_cluster_size:
                continue
            try:
                await self._consolidate_cluster(tag, notes)
                consolidated += 1
            except Exception as e:
                logger.warning("Consolidation failed for cluster '%s': %s", tag, e)

        # Generate topic map after consolidation
        await self._generate_topic_map()

        return consolidated

    async def run_loop(self, control: dict, interval: int = 600) -> None:
        """Background consolidation loop. Respects control['quit']."""
        while not control.get("quit", False):
            try:
                n = await self.run_once()
                if n > 0:
                    logger.info("Consolidated %d clusters", n)
            except Exception as e:
                logger.warning("Consolidation loop error: %s", e)
            await asyncio.sleep(interval)

    def _cluster_by_tag(self, notes: list[ZettelNote]) -> dict[str, list[ZettelNote]]:
        """Group notes by their first non-system tag."""
        system_tags = {"auto", "reflection", "research", "synthesis", "archived", "consolidated"}
        clusters: dict[str, list[ZettelNote]] = defaultdict(list)
        for note in notes:
            primary = "general"
            for t in note.tags:
                if t not in system_tags:
                    primary = t
                    break
            clusters[primary].append(note)
        return dict(clusters)

    async def _consolidate_cluster(self, tag: str, notes: list[ZettelNote]) -> None:
        """Synthesize a cluster of notes into one consolidated note."""
        # Build a summary prompt from note titles and content snippets
        snippets = []
        source_ids = []
        for note in notes[:10]:  # Cap at 10 notes per cluster
            snippet = f"- [{note.title}]: {note.content[:150].replace(chr(10), ' ')}"
            snippets.append(snippet)
            source_ids.append(note.id)

        combined = "\n".join(snippets)

        if self._backend:
            # Use the model to synthesize
            prompt = (
                f"Synthesize these {len(snippets)} research notes about '{tag}' into a concise summary. "
                f"Identify key themes, findings, and open questions:\n\n{combined}"
            )
            try:
                response = await self._backend.generate(
                    messages=[
                        {"role": "system", "content": "You synthesize research notes into concise summaries."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                )
                summary = response.get("content", "") if isinstance(response, dict) else str(response)
            except Exception as e:
                logger.warning("Model synthesis failed, using simple concat: %s", e)
                summary = f"Consolidated {len(snippets)} notes about '{tag}':\n\n{combined}"
        else:
            # No model available — simple concatenation
            summary = f"Consolidated {len(snippets)} notes about '{tag}':\n\n{combined}"

        # Create consolidated note
        await self._memory.create_note(
            title=f"Consolidated: {tag}",
            content=summary,
            tags=["consolidated", tag],
            links=source_ids,
            source="consolidator",
            agent_id="consolidator",
        )

        # Mark source notes as archived
        for note in notes[:10]:
            if "archived" not in note.tags:
                note.tags.append("archived")
                await self._memory.update_note(note)

        logger.info("Consolidated %d notes for tag '%s'", len(source_ids), tag)

    async def _generate_topic_map(self) -> None:
        """Generate/update a topic_map note listing clusters and key findings."""
        # Get all non-archived notes grouped by primary tag
        all_notes = await self._memory.get_notes_by_tags(
            ["research", "synthesis", "consolidated", "reflection"],
            limit=200,
            exclude_tags=["archived"],
        )
        if not all_notes:
            return

        system_tags = {"auto", "reflection", "research", "synthesis", "archived", "consolidated",
                        "knowledge_summary", "topic_map"}
        topic_counts: dict[str, list[str]] = defaultdict(list)
        for note in all_notes:
            primary = "general"
            for t in note.tags:
                if t not in system_tags:
                    primary = t
                    break
            topic_counts[primary].append(note.title[:60])

        # Build topic map content
        lines = ["# Topic Map\n"]
        for topic, titles in sorted(topic_counts.items(), key=lambda x: -len(x[1])):
            lines.append(f"## {topic} ({len(titles)} notes)")
            for title in titles[:5]:
                lines.append(f"- {title}")
            if len(titles) > 5:
                lines.append(f"- ... and {len(titles) - 5} more")
            lines.append("")

        content = "\n".join(lines)

        # Update or create topic_map note
        existing = await self._memory.search_by_tag("topic_map", limit=1)
        if existing:
            note = existing[0]
            note.content = content
            await self._memory.update_note(note)
        else:
            await self._memory.create_note(
                title="Topic Map",
                content=content,
                tags=["topic_map", "auto"],
                source="consolidator",
                agent_id="consolidator",
            )
