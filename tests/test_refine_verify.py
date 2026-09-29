"""Verification and curation with real evidence sources and real judge models."""

from __future__ import annotations

import os

import pytest

from aof.config import MemoryConfig, load_config, resolve_role_path
from aof.memory.store import MemoryStore
from aof.refine.pipeline import registry_embedder
from aof.refine.sources import PubMedSource
from aof.refine.verify import Verification, apply_verification, claim_text, curate, verify_note
from aof.specialists import build_registry
from aof.specialists.backends import RoleBackends

CONFIG = load_config("config.toml")
HAVE_SMALL = os.path.exists(resolve_role_path(CONFIG, "small"))
needs_small = pytest.mark.skipif(not HAVE_SMALL, reason="local Qwen3-4B role model not available")


@pytest.fixture
async def env(tmp_path):
    store = MemoryStore(MemoryConfig(db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")))
    await store.initialize()
    backends = RoleBackends(CONFIG)
    registry = build_registry(CONFIG, backends.get)
    yield store, registry
    await registry.close()
    await backends.close()
    await store.close()


def test_claim_text_strips_evidence_block():
    from aof.memory.zettel import ZettelNote

    note = ZettelNote(id="n", title="t", content='The TRN is GABAergic.\n\n**Evidence**\n- "x" — T, u')
    assert claim_text(note) == "The TRN is GABAergic."


@needs_small
async def test_real_verification_corroborates_a_true_claim_with_independent_evidence(env):
    store, registry = env
    note = await store.create_note(
        "TRN is GABAergic", "The thalamic reticular nucleus is composed of GABAergic neurons that inhibit thalamic relay cells.",
        ["claim"], kind="claim", status="refined", sources=["https://example.org/own-source"],
    )
    v = await verify_note(note, registry=registry, sources=[PubMedSource()], embed=registry_embedder(registry))
    if v.verdict == "no_evidence":
        pytest.skip("no independent evidence retrieved (source unreachable?)")
    assert v.verdict == "supported", (v.verdict, v.rationale, [s for s, _ in v.evidence])
    assert v.evidence and all(d.url != "https://example.org/own-source" for _, d in v.evidence)

    updated = await apply_verification(store, note, v)
    assert updated.status == "canonical"
    assert len(updated.sources) > 1 and "**Corroboration**" in updated.content
    reloaded = await store.get_note(note.id)
    assert reloaded.status == "canonical" and reloaded.sources == updated.sources


@needs_small
async def test_real_verification_does_not_corroborate_a_false_claim(env):
    store, registry = env
    note = await store.create_note(
        "False TRN claim", "The thalamic reticular nucleus sends excitatory glutamatergic projections directly to the neocortex.",
        ["claim"], kind="claim", status="refined", sources=["https://example.org/own-source"],
    )
    v = await verify_note(note, registry=registry, sources=[PubMedSource()], embed=registry_embedder(registry))
    if v.verdict == "no_evidence":
        pytest.skip("no independent evidence retrieved (source unreachable?)")
    assert v.verdict != "supported", (v.verdict, v.rationale)
    updated = await apply_verification(store, note, v)
    assert updated.status == "refined"  # never promoted on the strength of contradicting evidence
    if v.verdict == "contradicted":
        assert "conflict" in updated.tags and v.escalated in (True, v.judged_by == "role:large")


async def test_apply_verification_is_a_no_op_without_usable_evidence(env):
    store, _ = env
    note = await store.create_note("t", "A claim.", ["claim"], kind="claim", status="refined")
    for verdict in ("no_evidence", "unsupported", "needs_lookup", "supported"):  # 'supported' with no evidence too
        out = await apply_verification(store, note, Verification(verdict))
        assert out.status == "refined" and out.content == "A claim."


@needs_small
async def test_real_curation_archives_irrelevant_claims_but_keeps_them(env):
    store, registry = env
    good = await store.create_note(
        "TRN", "The thalamic reticular nucleus is a GABAergic shell around the dorsal thalamus.", ["claim"],
        kind="claim", status="refined",
    )
    bad = await store.create_note(
        "Stocks", "The stock market fell three percent on Tuesday after the interest rate announcement.", ["claim"],
        kind="claim", status="refined",
    )
    graded = await curate(
        "What is the thalamic reticular nucleus?", [good, bad], store=store, registry=registry, only="role:small",
    )
    assert good in graded["core"] + graded["supporting"]
    assert bad in graded["irrelevant"]
    kept = await store.get_note(bad.id)  # archived, not deleted
    assert kept is not None and kept.status == "archived" and "off-topic" in kept.tags
    assert (await store.get_note(good.id)).status == "refined"


def test_same_statement_and_own_quotes():
    from aof.memory.zettel import ZettelNote
    from aof.refine.verify import own_quotes, same_statement, url_work_key

    assert same_statement("The TRN is GABAergic.", "the TRN is  GABAergic")
    assert not same_statement("The TRN is GABAergic.", "Grid cells fire in hexagonal patterns.")
    note = ZettelNote(id="n", title="t", content='Claim.\n\n**Evidence**\n- "Quote one." \u2014 Paper, https://x.org/1\n\n**Corroboration**\n- "Other." \u2014 P2, https://y.org/2')
    assert own_quotes(note) == ["Claim.", "Quote one."]  # corroboration quotes are not "own"
    assert url_work_key("https://doi.org/10.1/ABC") == url_work_key("http://dx.doi.org/10.1/abc") == "doi:10.1/abc"
    assert url_work_key("https://pubmed.ncbi.nlm.nih.gov/1/") == "url:https://pubmed.ncbi.nlm.nih.gov/1/"


async def test_repair_strips_corroboration_that_repeats_the_same_paper(tmp_path):
    from aof.refine.repair import repair_false_corroboration

    store = MemoryStore(MemoryConfig(db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")))
    await store.initialize()
    try:
        claim = "Current knowledge about NMDAR-independent LTP is limited."
        fake = await store.create_note(
            "fake", f'{claim}\n\n**Evidence**\n- "{claim}" \u2014 Paper, https://doi.org/10.1/p\n\n**Corroboration**\n- "{claim}" \u2014 Paper, https://pubmed.ncbi.nlm.nih.gov/9/',
            ["claim"], note_id="fake", kind="claim", status="canonical",
            sources=["https://doi.org/10.1/p", "https://pubmed.ncbi.nlm.nih.gov/9/"],
        )
        real = await store.create_note(
            "real", 'Spindles are 11-16 Hz.\n\n**Evidence**\n- "Spindles are 11-16 Hz." \u2014 A, https://a.org/1\n\n**Corroboration**\n- "Sleep spindles occur at 11 to 16 Hz in humans." \u2014 B, https://b.org/2',
            ["claim"], note_id="real", kind="claim", status="canonical", sources=["https://a.org/1", "https://b.org/2"],
        )
        examined, repaired = await repair_false_corroboration(store)
        assert (examined, repaired) == (2, 1)
        fixed = await store.get_note("fake")
        assert fixed.status == "refined" and fixed.sources == ["https://doi.org/10.1/p"]
        assert "**Corroboration**" not in fixed.content and "**Evidence**" in fixed.content
        untouched = await store.get_note("real")
        assert untouched.status == "canonical" and len(untouched.sources) == 2 and "**Corroboration**" in untouched.content
    finally:
        await store.close()


async def test_weak_conflicts_are_removed_but_real_ones_survive():
    """Uses real sentence embeddings: on-subject evidence keeps a conflict flag, off-subject evidence loses it."""
    from aof.memory.zettel import ZettelNote
    from aof.refine.repair import strip_weak_conflict
    from aof.refine.vectors import cosine
    from aof.specialists.embed_provider import SentenceTransformerProvider

    claim = "The thalamic reticular nucleus is composed of GABAergic neurons."
    off_subject = "Cerebellar output is directed mainly to the ventrolateral thalamus and the red nucleus."
    on_subject = "The thalamic reticular nucleus is composed of glutamatergic neurons in this preparation."
    embedded = await SentenceTransformerProvider().embed([claim, off_subject, on_subject])
    if embedded is None:
        pytest.skip("embedding model unavailable")
    vec = dict(zip([claim, off_subject, on_subject], embedded.value))
    sim = lambda a, b: cosine(vec[a], vec[b])  # noqa: E731

    def note(quote):
        return ZettelNote(
            id="n", title="t", tags=["claim", "conflict"], content=(
                claim + "\n\n**Evidence**\n" + f'- "{claim}" — A, https://a.org' + "\n\n"
                + "**Conflicting evidence**\n" + f'- "{quote}" — B, https://b.org'
            ),
        )

    weak = note(off_subject)
    assert strip_weak_conflict(weak, sim) and "conflict" not in weak.tags
    assert "Conflicting evidence" not in weak.content and "**Evidence**" in weak.content
    real = note(on_subject)
    assert not strip_weak_conflict(real, sim) and "conflict" in real.tags
