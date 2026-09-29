"""Standalone-ness check with a real small model."""

from __future__ import annotations

import os

import pytest

from aof.config import MemoryConfig, load_config, resolve_role_path
from aof.memory.store import MemoryStore
from aof.refine.standalone import CHECKED_TAG, first_evidence, is_standalone, is_verbatim, make_standalone
from aof.specialists import build_registry
from aof.specialists.backends import RoleBackends

CONFIG = load_config("config.toml")
needs_small = pytest.mark.skipif(
    not os.path.exists(resolve_role_path(CONFIG, "small")), reason="local Qwen3-4B role model not available"
)


def _content(quote: str, title: str = "Paper", url: str = "https://x.org/p") -> str:
    return f'{quote}\n\n**Evidence**\n- "{quote}" \u2014 {title} (2020), {url}'


def test_first_evidence_and_verbatim_detection():
    from aof.memory.zettel import ZettelNote

    note = ZettelNote(id="n", title="t", content=_content("Spindles are 11-16 Hz oscillations."))
    assert first_evidence(note) == ("Spindles are 11-16 Hz oscillations.", "Paper (2020)", "https://x.org/p")
    assert is_verbatim(note)
    note.content = note.content.replace("Spindles are 11-16 Hz oscillations.\n\n**Evidence**", "Sleep spindles are 11-16 Hz.\n\n**Evidence**", 1)
    assert not is_verbatim(note)  # headline was rewritten
    assert first_evidence(ZettelNote(id="m", title="t", content="No evidence block here.")) is None


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


@needs_small
async def test_real_classifier_separates_standalone_from_fragment(env):
    _, registry = env
    full = "The thalamic reticular nucleus is a shell of GABAergic neurons surrounding the dorsal thalamus."
    fragment = "Increases happened in the ventral CA1 hippocampus and the basolateral amygdala."
    assert await is_standalone(full, registry, only="role:small") is True
    assert await is_standalone(fragment, registry, only="role:small") is False


@needs_small
async def test_real_make_standalone_never_loses_the_quote_and_is_idempotent(env):
    store, registry = env
    fragment = "Increases happened in the ventral CA1 hippocampus and the basolateral amygdala."
    full = "The thalamic reticular nucleus is a shell of GABAergic neurons surrounding the dorsal thalamus."
    a = await store.create_note(
        "frag", _content(fragment, "Fear memory retrieval engages ventral CA1 and basolateral amygdala in mice"),
        ["claim"], note_id="a", kind="claim", status="refined", sources=["https://x.org/p"],
    )
    b = await store.create_note("full", _content(full), ["claim"], note_id="b", kind="claim", status="refined",
                                sources=["https://x.org/p"])
    report = await make_standalone([a, b], store=store, registry=registry, only="role:small")
    assert report.checked == 2 and report.already_fine >= 1
    assert report.rewritten + report.unresolved == 1  # the fragment is either repaired or flagged, never ignored

    got = await store.get_note("a")
    assert CHECKED_TAG in got.tags
    assert first_evidence(got)[0] == fragment  # the verbatim quote survives regardless
    assert (await store.get_note("b")).content.startswith(full)

    again = await make_standalone([got, await store.get_note("b")], store=store, registry=registry, only="role:small")
    assert again.checked == 0 and again.skipped == 2  # idempotent: already-checked notes are skipped
