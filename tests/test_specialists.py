"""Specialist registry: cascade logic (fixture providers) and real Needle / role providers."""

from __future__ import annotations

import os

import pytest

from aof.config import DEFAULT_SPECIALIST_CHAINS, load_config, resolve_role_path
from aof.specialists import (
    ANNOTATE,
    CLASSIFY,
    JUDGE,
    SpecialistExhausted,
    SpecialistRegistry,
    SpecialistResult,
    build_registry,
)
from aof.specialists.backends import RoleBackends


# -- cascade logic ---------------------------------------------------------------------------------
# Fixture providers implement the Provider contract with fixed behaviour so the cascade itself is
# tested deterministically; the real providers are exercised further down.

class Fixed:
    def __init__(self, name, behaviour, *, available=True):
        self.name, self._behaviour, self._available = name, behaviour, available

    def available(self):
        return self._available

    async def classify(self, text, labels):
        if self._behaviour == "raise":
            raise RuntimeError("boom")
        return None if self._behaviour == "abstain" else SpecialistResult(self._behaviour, self.name)

    async def close(self):
        return None


class NoClassify:
    name = "no-classify"

    def available(self):
        return True

    async def close(self):
        return None


async def test_cascade_escalates_on_abstain_and_error():
    reg = SpecialistRegistry(chains={CLASSIFY: [
        Fixed("a", "abstain"), Fixed("b", "raise"), Fixed("c", "thalamus"),
    ]})
    result = await reg.call(CLASSIFY, "text", ["thalamus", "other"])
    assert (result.value, result.provider) == ("thalamus", "c")
    assert reg.stats[(CLASSIFY, "a")].abstained == 1
    assert reg.stats[(CLASSIFY, "b")].errors == 1
    assert reg.stats[(CLASSIFY, "c")].accepted == 1
    assert reg.escalation_rate(CLASSIFY) == 1.0


async def test_first_provider_settles_means_no_escalation():
    reg = SpecialistRegistry(chains={CLASSIFY: [Fixed("a", "sleep"), Fixed("b", "other")]})
    for _ in range(3):
        assert (await reg.call(CLASSIFY, "t", ["sleep"])).provider == "a"
    assert (CLASSIFY, "b") not in reg.stats
    assert reg.escalation_rate(CLASSIFY) == 0.0


async def test_unavailable_and_unsupported_providers_are_skipped():
    reg = SpecialistRegistry(chains={CLASSIFY: [
        Fixed("gone", "x", available=False), NoClassify(), Fixed("ok", "sleep"),
    ]})
    assert [p.name for p in reg.providers_for(CLASSIFY)] == ["ok"]
    assert (await reg.call(CLASSIFY, "t", ["sleep"])).provider == "ok"


async def test_exhausted_chain_raises_with_providers_named():
    reg = SpecialistRegistry(chains={CLASSIFY: [Fixed("a", "abstain"), Fixed("b", "abstain")]})
    with pytest.raises(SpecialistExhausted, match="a, b"):
        await reg.call(CLASSIFY, "t", ["x"])
    with pytest.raises(SpecialistExhausted, match="no available provider"):
        await reg.call(JUDGE, "claim", "evidence")


# -- config ----------------------------------------------------------------------------------------

def test_specialist_chains_default_and_override(tmp_path):
    assert load_config(None).specialists.chains == DEFAULT_SPECIALIST_CHAINS
    (tmp_path / "c.toml").write_text('[specialists.annotate]\nproviders = ["role:fast"]\n')
    chains = load_config(tmp_path / "c.toml").specialists.chains
    assert chains["annotate"] == ("role:fast",)
    assert chains["classify"] == DEFAULT_SPECIALIST_CHAINS["classify"]  # untouched capabilities keep defaults


def test_build_registry_rejects_unknown_provider_kind(tmp_path):
    (tmp_path / "c.toml").write_text('[specialists.annotate]\nproviders = ["magic:1"]\n')
    with pytest.raises(ValueError, match="unknown specialist provider"):
        build_registry(load_config(tmp_path / "c.toml"), get_backend=None)


# -- real providers --------------------------------------------------------------------------------

try:
    import needle  # noqa: F401
    HAVE_NEEDLE = True
except ImportError:
    HAVE_NEEDLE = False

CONFIG = load_config("config.toml")
HAVE_SMALL = os.path.exists(resolve_role_path(CONFIG, "small"))
HAVE_FAST = os.path.exists(resolve_role_path(CONFIG, "fast"))

TOPICS = ["hippocampus", "thalamus", "cortex", "sleep", "other"]
TRN_TEXT = "The thalamic reticular nucleus is a GABAergic shell around the dorsal thalamus."
OFF_TOPIC = "The stock market fell three percent on Tuesday."


def _needle_or_skip(reg):
    provider = next((p for p in reg.providers_for(CLASSIFY) if p.name == "needle:3"), None)
    if provider is None:
        pytest.skip("cactus-needle not installed (uv sync --extra needle)")
    return provider


@pytest.mark.skipif(not HAVE_NEEDLE, reason="cactus-needle not installed (uv sync --extra needle)")
async def test_real_needle3_classifies_and_abstains_off_topic():
    from aof.specialists.needle_provider import NeedleProvider

    needle3 = NeedleProvider(3)
    assert needle3.available()
    try:
        on_topic = await needle3.classify(TRN_TEXT, TOPICS)
    except Exception as e:  # weights are fetched from Hugging Face on first use
        pytest.skip(f"Needle 3 weights unavailable: {e}")
    assert on_topic is not None and on_topic.value == "thalamus"
    # The engine withholds a call it is unsure about -> abstain -> registry escalates.
    assert await needle3.classify(OFF_TOPIC, TOPICS) is None


@pytest.mark.skipif(not HAVE_SMALL, reason="local Qwen3-4B role model not available")
async def test_real_role_provider_judges_claims():
    backends = RoleBackends(CONFIG)
    reg = build_registry(CONFIG, backends.get)
    try:
        supported = await reg.call(
            JUDGE, "The TRN projects to thalamic relay nuclei.",
            "The thalamic reticular nucleus sends GABAergic projections to thalamic relay nuclei.",
        )
        contradicted = await reg.call(
            JUDGE, "The TRN projects to neocortex.",
            "The thalamic reticular nucleus projects only within the thalamus and does not project to neocortex.",
        )
    finally:
        await reg.close()
        await backends.close()
    assert supported.value["verdict"] == "supported"
    assert contradicted.value["verdict"] == "contradicted"


@pytest.mark.skipif(not (HAVE_NEEDLE and HAVE_FAST), reason="needs cactus-needle and the local LFM2.5 'fast' role")
async def test_real_cascade_from_needle_to_role_model():
    backends = RoleBackends(CONFIG)
    reg = build_registry(CONFIG, backends.get)
    try:
        first = await reg.call(CLASSIFY, TRN_TEXT, TOPICS)
        second = await reg.call(CLASSIFY, OFF_TOPIC, TOPICS)
    except SpecialistExhausted as e:
        pytest.skip(f"no provider could answer (weights offline?): {e}")
    finally:
        await reg.close()
        await backends.close()
    assert first.value == "thalamus"
    assert second.value in TOPICS
    assert reg.stats[(CLASSIFY, "needle:3")].abstained >= 1  # off-topic text escalated past Needle
    assert second.provider.startswith("role:") or second.provider == "needle:2"
