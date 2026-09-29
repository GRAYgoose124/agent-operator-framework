"""Storing claims in the vault: atomic claim notes, lossless merge, archival with supersession links."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import Claim
from aof.refine.dedupe import pick_canonical

AGENT_ID = "refine"


def question_slug(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:48]


def _note_id(claim: Claim) -> str:
    """Stable id per (claim text, source): identical sentences from different sources must not collide."""
    digest = hashlib.sha256(f"{claim.text}\n{claim.url}".encode()).hexdigest()[:8]
    return f"claim-{digest}"


def _title(claim: Claim, limit: int = 90) -> str:
    text = claim.standalone
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _evidence_line(claim: Claim) -> str:
    cite = claim.title or claim.url
    year = f" ({claim.year})" if claim.year else ""
    return f'- "{claim.text}" — {cite}{year}, {claim.url}'


def _canonical_content(claims: Sequence[Claim], members: Sequence[int], canonical: int) -> str:
    lines = [claims[canonical].standalone, "", "**Evidence**"]
    ordered = [canonical] + [i for i in members if i != canonical]
    lines += [_evidence_line(claims[i]) for i in ordered]
    return "\n".join(lines)


def _annotation_tags(claim: Claim) -> list[str]:
    tags = []
    if claim.topic and claim.topic != "other":
        tags.append(f"topic:{claim.topic}")
    tags.extend(f"entity:{e.lower()}" for e in claim.entities[:3])
    return tags


async def store_clusters(
    store: MemoryStore,
    question: str,
    claims: Sequence[Claim],
    clusters: Sequence[Sequence[int]],
    *,
    extra_tags: Sequence[str] = (),
) -> tuple[list[ZettelNote], list[ZettelNote]]:
    """Write one canonical note per cluster and archive every other member (nothing is discarded).

    Returns (canonical notes, archived notes). A canonical note lists the archived ids it supersedes; each archived
    note points back via `superseded_by`. A cluster backed by >= 2 distinct sources is `canonical`; a single-source
    claim is `refined` until something corroborates it.
    """
    slug = f"q:{question_slug(question)}"
    canonical_notes: list[ZettelNote] = []
    archived_notes: list[ZettelNote] = []
    for members in clusters:
        best = pick_canonical(claims, members)
        canonical_id = _note_id(claims[best])
        archived_ids: list[str] = []
        for i in members:
            if i == best:
                continue
            archived_id = _note_id(claims[i])
            if archived_id == canonical_id or archived_id in archived_ids:
                continue  # same sentence from the same source twice
            note = await store.create_note(
                _title(claims[i]), claims[i].text, ["claim", "archived", slug, *extra_tags],
                source=claims[i].url, agent_id=AGENT_ID, note_id=archived_id,
                kind="claim", status="archived", sources=[claims[i].url],
            )
            note.superseded_by = canonical_id
            await store.update_note(note)
            archived_notes.append(note)
            archived_ids.append(archived_id)
        urls = list(dict.fromkeys(claims[i].url for i in [best, *members]))  # best first, no repeats
        canonical_notes.append(await store.create_note(
            _title(claims[best]), _canonical_content(claims, members, best),
            ["claim", slug, *extra_tags, *_annotation_tags(claims[best])],
            source=claims[best].url, agent_id=AGENT_ID, note_id=canonical_id,
            kind="claim", status="canonical" if len(urls) >= 2 else "refined",
            sources=urls, supersedes=archived_ids,
        ))
    return canonical_notes, archived_notes
