"""VRAM planning and stage-aware model residency.

Planning reads real GGUF headers; the residency test runs real llama-server processes on the CPU (so it does not
fight for the GPU). Opt in with AOF_LLAMA_SERVER (binary or directory) and AOF_TEST_LARGE_MODEL (any GGUF path).
"""

from __future__ import annotations

import os

import pytest

from aof.config import AppConfig, LlamaServerConfig, RolesConfig, VramConfig
from aof.inference.residency import ManagedBackend, ResidencyManager
from aof.inference.vram import model_size, plan_vram, read_gguf_meta

MODEL = os.environ.get("AOF_TEST_LARGE_MODEL", "")
HAVE_MODEL = bool(MODEL) and os.path.exists(MODEL)
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="set AOF_TEST_LARGE_MODEL to a GGUF file")
needs_server = pytest.mark.skipif(
    not HAVE_MODEL or not os.environ.get("AOF_LLAMA_SERVER"), reason="set AOF_LLAMA_SERVER and AOF_TEST_LARGE_MODEL"
)


def _config(per_role: dict[str, dict], budget_mb: int, **vram) -> AppConfig:
    roles = RolesConfig(**{r: MODEL for r in per_role})
    return AppConfig(
        roles=roles,
        llama_server=LlamaServerConfig(roles=tuple(per_role), per_role=per_role, n_parallel=1),
        vram=VramConfig(budget_mb=budget_mb, **vram),
    )


@needs_model
def test_gguf_header_and_size_estimate():
    meta = read_gguf_meta(MODEL)
    assert meta["general.architecture"]
    size = model_size(MODEL)
    assert size.n_layers > 0 and size.weights_mb > 100 and size.kv_bytes_per_token > 0
    assert size.need_mb(4096) > size.weights_mb


@needs_model
def test_shares_split_the_budget_and_partial_offload():
    size = model_size(MODEL)
    need = size.need_mb(8192)
    # Two pinned roles, 1:3 shares, and room for only ~1.2 models in total: the heavier share wins the GPU.
    cfg = _config({"small": {"vram_share": 1}, "large": {"vram_share": 3}}, int(need * 1.2))
    cfg = AppConfig(**{**cfg.__dict__, "role_context": type(cfg.role_context)(small=8192, large=8192)})
    plan = plan_vram(cfg, vram=(24000, 24000))
    small, large = plan.roles["small"], plan.roles["large"]
    assert large.gpu_mb > small.gpu_mb
    assert large.n_gpu_layers > small.n_gpu_layers
    assert small.offloaded  # squeezed: some layers stay on the CPU
    assert small.gpu_mb + large.gpu_mb <= plan.budget_mb + 1


@needs_model
def test_everything_fits_when_the_budget_is_generous():
    cfg = _config({"small": {}, "large": {}}, 200_000)
    plan = plan_vram(cfg, vram=(24000, 24000))
    assert all(not p.offloaded for p in plan.roles.values())


@needs_model
def test_swap_roles_share_one_pool():
    size = model_size(MODEL)
    cfg = _config({"small": {"residency": "swap"}, "large": {"residency": "swap"}}, int(size.need_mb(8192) * 1.5))
    plan = plan_vram(cfg, vram=(24000, 24000))
    a, b = plan.roles["small"], plan.roles["large"]
    assert a.gpu_mb == b.gpu_mb  # each gets the whole pool when it is the active one
    assert a.gpu_mb <= plan.budget_mb


def test_budget_detection_uses_free_or_total():
    cfg = AppConfig(vram=VramConfig(reserve_mb=1000))
    assert plan_vram(cfg, roles=(), vram=(12000, 4000)).budget_mb == 11000
    cfg = AppConfig(vram=VramConfig(reserve_mb=1000, use_free=True))
    assert plan_vram(cfg, roles=(), vram=(12000, 4000)).budget_mb == 3000


@needs_server
async def test_swap_roles_take_turns_on_a_small_pool():
    """With room for one model, using B parks A (its server process ends), and A wakes again on demand."""
    from aof.researcher.runner import _make_multi_backends

    cfg = AppConfig(
        roles=RolesConfig(small=MODEL, large=MODEL),
        llama_server=LlamaServerConfig(
            binary=os.environ["AOF_LLAMA_SERVER"], roles=("small", "large"), n_gpu_layers=0,  # CPU: keeps the GPU free
            startup_timeout=180,
        ),
    )
    cfg = AppConfig(**{**cfg.__dict__, "role_context": type(cfg.role_context)(small=512, large=1024)})  # distinct n_ctx: identical (path, n_ctx) share one server
    made = await _make_multi_backends(cfg, {"small", "large"}, None, frozenset({"small", "large"}))
    mgr = ResidencyManager(pool_mb=100, park_mode="unload")  # smaller than any model: one resident at a time
    for role, backend in made.items():
        mgr.register(role, backend, swappable=True, need_mb=100)
    a, b = ManagedBackend("small", made["small"], mgr), ManagedBackend("large", made["large"], mgr)
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    try:
        assert not made["small"].resident and not made["large"].resident  # created parked
        assert (await a.complete(messages, max_tokens=8)).text
        assert made["small"].resident and not made["large"].resident
        assert (await b.complete(messages, max_tokens=8)).text
        assert made["large"].resident and not made["small"].resident  # A was parked to make room
        await mgr.stage({"small"})  # stage-aware: load A up front, park B
        assert made["small"].resident and not made["large"].resident
        assert mgr.stats()["small"]["wakes"] == 2 and mgr.stats()["large"]["parks"] == 1
    finally:
        await a.shutdown()
        await b.shutdown()
