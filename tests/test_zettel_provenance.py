"""ZettelNote refinement fields: markdown round-trip, backward compatibility, store persistence."""

from __future__ import annotations

from aof.config import MemoryConfig
from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote


def test_legacy_note_serialisation_is_unchanged():
    note = ZettelNote(id="n1", title="T", content="Body", tags=["a"], links=["n2"])
    md = note.to_markdown()
    for key in ("kind:", "status:", "confidence:", "superseded_by:", "## Provenance", "## Merged from"):
        assert key not in md
    back = ZettelNote.from_markdown(md)
    assert (back.kind, back.status, back.sources, back.supersedes) == ("note", "raw", [], [])
    assert back.content == "Body" and back.links == ["n2"]


def test_refinement_fields_round_trip():
    note = ZettelNote(
        id="c1", title="TRN is GABAergic", content="The TRN is a GABAergic shell around the thalamus.",
        tags=["claim", "thalamus"], links=["c2"], kind="claim", status="canonical",
        sources=["https://example.org/a?x=1,2", "doi:10.1000/xyz"], supersedes=["old1", "old2"],
        superseded_by="", confidence=0.83,
    )
    back = ZettelNote.from_markdown(note.to_markdown())
    assert back.kind == "claim" and back.status == "canonical" and back.confidence == 0.83
    assert back.sources == ["https://example.org/a?x=1,2", "doi:10.1000/xyz"]  # commas/colons survive
    assert back.supersedes == ["old1", "old2"]
    assert back.links == ["c2"]
    assert back.content == "The TRN is a GABAergic shell around the thalamus."


def test_archived_note_points_at_canonical():
    note = ZettelNote(id="old1", title="t", content="c", status="archived", superseded_by="c1")
    back = ZettelNote.from_markdown(note.to_markdown())
    assert (back.status, back.superseded_by) == ("archived", "c1")


async def test_store_persists_refinement_fields(tmp_path):
    store = MemoryStore(MemoryConfig(db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")))
    await store.initialize()
    try:
        made = await store.create_note(
            "Claim", "Spindles are 11-16 Hz.", ["claim"], kind="claim", status="refined",
            sources=["https://example.org/s"], supersedes=["a", "b"], confidence=0.5,
        )
        got = await store.get_note(made.id)
    finally:
        await store.close()
    assert (got.kind, got.status, got.sources, got.supersedes, got.confidence) == (
        "claim", "refined", ["https://example.org/s"], ["a", "b"], 0.5,
    )


def test_report_style_sources_heading_in_content_is_left_alone():
    """Research reports already end with '## Sources'; that is content, not provenance."""
    body = "Key findings.\n\n## Sources\n- https://python.org"
    back = ZettelNote.from_markdown(ZettelNote(id="r", title="t", content=body).to_markdown())
    assert back.content == body and back.sources == []
