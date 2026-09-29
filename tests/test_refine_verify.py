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
