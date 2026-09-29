"""VRAM awareness: detect the GPU budget, size models from their GGUF headers, plan GPU layers per role.

Roles compose by ratio. Each managed role has a `vram_share` (its weight) and a `residency`:
  * `pinned` roles stay loaded; the budget is split between them by share (a role that needs less than its
    share gives the rest back to the others).
  * `swap` roles take turns in one pool (sized by the largest share among them); the residency manager
    (`aof.inference.residency`) loads one when it is needed and parks the others. Parked weights stay in the
    OS file cache, so waking is a reload from system memory rather than from disk.
A role whose allotment is smaller than the model keeps the remaining layers on the CPU (`-ngl` partial offload).
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass

from aof.config import AppConfig, resolve_role_path

logger = logging.getLogger(__name__)

MB = 1024 * 1024
OVERHEAD_MB = 450  # CUDA context, compute buffers
_KV_BYTES = 2  # f16 cache


def detect_vram() -> tuple[int, int] | None:
    """(total_mb, free_mb) of the first NVIDIA GPU, or None when it cannot be read."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.splitlines()[0]
        total, free = (int(x.strip()) for x in out.split(","))
        return total, free
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


# --- GGUF header -------------------------------------------------------------------------------------------------

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _read_value(f, vtype: int):
    if vtype in _SCALARS:
        fmt = "<" + _SCALARS[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:  # string
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")
    if vtype == 9:  # array: skip its contents (we only need scalar hyper-parameters)
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _read_value(f, etype)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_meta(path: str) -> dict:
    """Scalar/string metadata from a GGUF header (arrays are skipped)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        (_version,) = struct.unpack("<I", f.read(4))
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        meta: dict = {}
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode("utf-8", errors="replace")
            (vtype,) = struct.unpack("<I", f.read(4))
            value = _read_value(f, vtype)
            if value is not None:
                meta[key] = value
        return meta


@dataclass(frozen=True)
class ModelSize:
    weights_mb: int
    n_layers: int
    kv_bytes_per_token: int

    def need_mb(self, n_ctx_total: int) -> int:
        return self.weights_mb + self.kv_bytes_per_token * n_ctx_total // MB + OVERHEAD_MB


def model_size(path: str) -> ModelSize:
    """Weights = file size; KV cache from the header's attention geometry (falls back to a size-based guess)."""
    weights_mb = max(1, os.path.getsize(path) // MB)
    try:
        meta = read_gguf_meta(path)
    except (OSError, ValueError, struct.error):
        meta = {}
    arch = str(meta.get("general.architecture", ""))

    def get(name: str, default=0):
        return meta.get(f"{arch}.{name}", default)

    n_layers = int(get("block_count", 0)) or max(1, int(weights_mb / 130))
    heads = int(get("attention.head_count", 0))
    kv_heads = int(get("attention.head_count_kv", 0)) or heads
    head_dim = int(get("attention.key_length", 0)) or (int(get("embedding_length", 0)) // heads if heads else 0)
    if kv_heads and head_dim:
        interval = int(get("full_attention_interval", 0)) or 1  # hybrid models cache only their attention layers
        attn_layers = max(1, n_layers // interval)
        kv_per_token = 2 * kv_heads * head_dim * _KV_BYTES * attn_layers
    else:
        kv_per_token = int(weights_mb * 26)  # ~26 KB per token per GB of weights: a conservative dense-model guess
    return ModelSize(weights_mb, n_layers, kv_per_token)


# --- planning ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RolePlan:
    role: str
    residency: str  # pinned | swap
    share: float
    need_mb: int  # everything on the GPU
    gpu_mb: int  # what the plan puts on the GPU
    n_gpu_layers: int
    n_layers: int

    @property
    def offloaded(self) -> bool:
        return self.n_gpu_layers < self.n_layers


@dataclass(frozen=True)
class VramPlan:
    budget_mb: int
    total_mb: int | None
    roles: dict[str, RolePlan]

    def render(self) -> str:
        head = f"VRAM budget {self.budget_mb} MB" + (f" of {self.total_mb} MB" if self.total_mb else " (GPU not detected)")
        lines = [head, f"{'role':<14}{'residency':<10}{'share':>6}{'need MB':>9}{'GPU MB':>8}{'layers':>10}"]
        for p in self.roles.values():
            layers = "all" if not p.offloaded else f"{p.n_gpu_layers}/{p.n_layers}"
            lines.append(f"{p.role:<14}{p.residency:<10}{p.share:>6.2f}{p.need_mb:>9}{p.gpu_mb:>8}{layers:>10}")
        return "\n".join(lines)


def _split(budget: float, wants: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Water-fill `budget` over {key: (share, need)}: proportional to share, no key gets more than its need."""
    alloc: dict[str, float] = {}
    left = dict(wants)
    while left:
        total_share = sum(s for s, _ in left.values()) or 1.0
        capped = {k: need for k, (s, need) in left.items() if budget * s / total_share >= need}
        if not capped:
            return {**alloc, **{k: budget * s / total_share for k, (s, _) in left.items()}}
        for k, need in capped.items():
            alloc[k] = need
            budget -= need
            del left[k]
    return alloc


def _layers_for(gpu_mb: float, size: ModelSize, n_ctx_total: int) -> int:
    if gpu_mb >= size.need_mb(n_ctx_total):
        return size.n_layers
    fixed = size.kv_bytes_per_token * n_ctx_total // MB + OVERHEAD_MB
    frac = max(0.0, (gpu_mb - fixed) / max(1, size.weights_mb))
    return min(size.n_layers - 1, int(frac * size.n_layers))


def plan_vram(config: AppConfig, roles: tuple[str, ...] | None = None, vram: tuple[int, int] | None = None) -> VramPlan:
    """Decide each managed role's GPU layers from the budget, shares and residency (see module docstring)."""
    if vram is None:
        vram = detect_vram()
    total = vram[0] if vram else None
    cfg = config.vram
    if cfg.budget_mb > 0:
        budget = cfg.budget_mb
    elif vram:
        budget = max(0, (vram[1] if cfg.use_free else vram[0]) - cfg.reserve_mb)
    else:
        budget = 0
    roles = roles if roles is not None else config.llama_server.roles
    sizes: dict[str, tuple[ModelSize, int, float, str]] = {}
    for role in roles:
        path = resolve_role_path(config, role)
        if not path or not os.path.exists(path):
            continue
        rc = config.llama_server.for_role(role)
        n_ctx_total = config.role_context.get(role, config.model.n_ctx) * max(1, rc.n_parallel)
        sizes[role] = (model_size(path), n_ctx_total, max(0.01, rc.vram_share), rc.residency)

    if not vram and cfg.budget_mb <= 0:  # nothing to plan against: keep the configured behaviour
        return VramPlan(0, total, {
            r: RolePlan(r, res, sh, size.need_mb(nc), 0, config.llama_server.for_role(r).n_gpu_layers, size.n_layers)
            for r, (size, nc, sh, res) in sizes.items()
        })

    def need(r: str) -> float:
        size, nc, _, _ = sizes[r]
        return size.need_mb(nc)

    pinned = [r for r in sizes if sizes[r][3] != "swap"]
    swapped = [r for r in sizes if sizes[r][3] == "swap"]
    wants: dict[str, tuple[float, float]] = {r: (sizes[r][2], need(r)) for r in pinned}
    if swapped:  # the swap roles share one pool, sized like the biggest of them
        wants["\0swap"] = (max(sizes[r][2] for r in swapped), max(need(r) for r in swapped))
    alloc = _split(float(budget), wants)

    plans: dict[str, RolePlan] = {}
    for r, (size, nc, sh, res) in sizes.items():
        gpu = alloc["\0swap"] if res == "swap" else alloc[r]
        gpu = min(gpu, need(r))
        plans[r] = RolePlan(r, res, sh, int(need(r)), int(gpu), _layers_for(gpu, size, nc), size.n_layers)
    return VramPlan(int(budget), total, plans)


def apply_plan(config: AppConfig, role: str, plan: VramPlan):
    """The role's LlamaServerConfig with `n_gpu_layers` filled in from the plan when it is set to auto (-1)."""
    from dataclasses import replace

    rc = config.llama_server.for_role(role)
    if rc.n_gpu_layers >= 0:
        return rc
    p = plan.roles.get(role)
    return replace(rc, n_gpu_layers=p.n_gpu_layers if p and plan.budget_mb > 0 else 99)


def log_plan(plan: VramPlan) -> None:
    for line in plan.render().splitlines():
        logger.info(line)


__all__ = ["detect_vram", "model_size", "plan_vram", "apply_plan", "read_gguf_meta", "VramPlan", "RolePlan", "ModelSize"]
