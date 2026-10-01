"""Configuration loading from TOML files."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    path: str = ""
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    chat_format: str = "chatml"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass(frozen=True)
class PoolConfig:
    max_agents: int = 8
    inference_pool_size: int = 2
    thread_pool_workers: int = 4


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str = "data/memory.db"
    notes_dir: str = "data/memory"
    auto_commit: bool = True
    memory_strategy: str = "search"  # search | recent | agent_notes | hybrid | agentic
    memory_search_limit: int = 5
    vector_db_path: str = ""  # Default: {notes_dir}/chroma when hybrid
    embedding_model: str = "all-MiniLM-L6-v2"
    fts_preview_chars: int = 500  # Chars of note content indexed for FTS search
    consolidation_enabled: bool = False
    consolidation_interval: int = 600  # seconds between consolidation runs
    consolidation_min_notes: int = 10  # minimum notes before consolidation triggers
    # Optional A-MEM (agentic memory): LLM-generated metadata, linking, evolution
    agentic_enabled: bool = False
    agentic_llm_backend: str = "ollama"  # openai | ollama | openrouter | sglang
    agentic_llm_model: str = ""  # e.g. llama2, gpt-4o-mini
    agentic_embedding_model: str = "all-MiniLM-L6-v2"
    agentic_api_key: str = ""  # for openai/openrouter; or set env


@dataclass(frozen=True)
class ToolsConfig:
    scripts_dir: str = "data/tools"
    crawlers_dir: str = "data/crawlers"
    sandbox_timeout: int = 30
    blocked_imports: tuple[str, ...] = ("os", "subprocess", "socket", "shutil", "ctypes")


@dataclass(frozen=True)
class PipelineConfig:
    default_max_steps: int = 10
    max_tool_rounds: int = 5  # Max LLM rounds per step when model returns tool calls


@dataclass(frozen=True)
class EvaluatorConfig:
    check_interval_seconds: float = 5.0
    health_threshold: float = 0.5
    guidance_cooldown_seconds: float = 30.0


@dataclass(frozen=True)
class ResearchConfig:
    """Research queue persistence path and extended mode settings."""
    queue_path: str = "data/research_queue.json"
    interrupt_flag_path: str = "data/research_interrupt.flag"
    extended_mode: bool = False
    director_role: str = "medium"
    expert_role: str = "micro"
    expert_count: int = 4
    poll_seconds: int = 3
    route_gen_enabled: bool = False
    route_gen_interval: int = 300
    route_gen_min_backlog: int = 2
    parallel_items: int = 1
    refinement_pipeline: str = ""
    refinement_passes: int = 0
    # Seed expansion (--seed-question): role for one-shot expansion, target count
    seed_expansion_role: str = "medium"
    seed_expansion_count: int = 30
    # Branch connector: periodic synthesis note linking themes across items
    branch_connector_enabled: bool = False
    branch_connector_interval: int = 600
    # Lateral thinking: proactive tangential questions into queue
    lateral_thinking_enabled: bool = False
    lateral_interval: int = 300
    lateral_min_backlog: int = 5


@dataclass(frozen=True)
class SotaConfig:
    """`aof sota`: the research harness (evidence gathering, vault graph, long-form reports, live console).

    Model settings name roles from [roles]; `writer_ctx` runs the writer with its own context size (e.g. the large
    model at 8192 to leave VRAM for other roles, or at 32768 for bigger evidence packs). 0 = [role_context] value.
    """
    writer: str = "large"  # writes outlines, report sections, synthesis and `ask` answers
    writer_ctx: int = 0
    writer_parallel: int = 0  # llama-server slots for the writer (0 = [llama_server] setting); 2 lets `ask` run beside a report
    writer_thinking: bool = False  # let a thinking model reason before writing (slower, more tokens)
    judge: str = "small"  # cheap support judge for cited sentences
    judge_ctx: int = 4096  # judge prompts are short: a small context keeps its KV cache (VRAM) small
    judge_parallel: int = 2  # llama-server slots for the judge (each slot holds judge_ctx of KV cache)
    confirm: str = "large"  # confirms every negative verdict before a sentence is flagged
    check: str = "cheap"  # none | cheap | strict
    report_kind: str = "review"  # review | investigation
    report_sections: int = 7
    section_words: int = 650
    section_out_tokens: int = 1800
    evidence_max: int = 45  # evidence items per section (also bounded by writer_ctx)
    per_work: int = 3  # evidence items per source work per section
    research_thin: bool = True  # research sections whose evidence is thin before writing them
    ask_evidence: int = 24
    sources: tuple[str, ...] = ("pubmed", "openalex", "wikipedia")
    per_source: int = 5
    claims_per_doc: int = 6
    curate: bool = True
    verify_top: int = 2
    concurrency: int = 4  # concurrent small-model calls (match the judge role's n_parallel)
    auto: bool = False  # when idle, keep working: queue-file questions, then gap research, then reports
    auto_structure_every: int = 4  # rebuild the vault graph after this many research jobs
    auto_report_every: int = 0  # write a report on the least-covered area after N research jobs (0 = never)
    graph_max_hub_size: int = 40
    graph_link_k: int = 2


@dataclass(frozen=True)
class DiscoveryConfig:
    """Discovery queue (breadth-first web navigation) settings."""
    queue_path: str = "data/discovery_queue.json"


@dataclass(frozen=True)
class LocalServerConfig:
    """Config for local OpenAI-compatible server (LM Studio, Ollama)."""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen3-0.6b"
    timeout: int = 300  # seconds per request
    # None = leave the model's default; False/True sends chat_template_kwargs.enable_thinking
    enable_thinking: bool | None = None
    # Sampling overrides (None = server default). min_temperature floors any requested temperature:
    # some reasoning models loop at T <= 0.3 (see docs/llama-cpp.md).
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    min_temperature: float = 0.0


@dataclass(frozen=True)
class LlamaServerConfig:
    """Managed llama.cpp `llama-server` (OpenAI-compatible) for models llama-cpp-python cannot load.

    The binary is found via `binary`, then the AOF_LLAMA_SERVER env var, then PATH.
    See docs/llama-cpp.md for the minimum llama.cpp build.
    """
    binary: str = ""
    host: str = "127.0.0.1"
    n_gpu_layers: int = -1  # -1 = plan from the VRAM budget ([vram], vram_share, residency); >= 0 forces a value
    n_parallel: int = 1
    startup_timeout: int = 300
    request_timeout: int = 180  # seconds; requests here take seconds, so a longer wait means a hung server
    # Skip llama-server's host-RAM prompt cache and keep few context checkpoints: our prompts are short and varied,
    # and the defaults (8 GiB cache, 32 checkpoints per slot) ballooned a server to ~15 GB of RAM. Needs a recent build.
    lean_cache: bool = True
    extra_args: tuple[str, ...] = ()
    enable_thinking: bool = False  # most agent steps want the answer, not untagged chain-of-thought
    # Sampling overrides (None = server default). min_temperature floors any requested temperature:
    # some reasoning models loop at T <= 0.3 (see docs/llama-cpp.md).
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    min_temperature: float = 0.0
    roles: tuple[str, ...] = ()  # roles served through llama-server instead of llama-cpp-python
    # VRAM composition (see docs/llama-cpp.md): `vram_share` is this role's weight when the budget is split;
    # `residency` = "pinned" (stay loaded) or "swap" (loaded on demand, parked when another swap role needs the pool).
    vram_share: float = 1.0
    residency: str = "pinned"
    # Per-role overrides of any field above, e.g. [llama_server.per_role.large] min_temperature = 0.6
    per_role: dict[str, dict] = field(default_factory=dict)

    def for_role(self, role: str) -> "LlamaServerConfig":
        """This config with the role's overrides applied (unknown keys are ignored)."""
        overrides = {k: v for k, v in self.per_role.get(role, {}).items() if k in LlamaServerConfig.__dataclass_fields__}
        for k, v in overrides.items():
            if isinstance(v, list):
                overrides[k] = tuple(v)
        return replace(self, **overrides)


# Default provider chains, cheapest first. Provider specs: "needle:<generation>", "role:<role>",
# "sentence-transformers". Unavailable providers (package or model file missing) are skipped.
DEFAULT_SPECIALIST_CHAINS: dict[str, tuple[str, ...]] = {
    "annotate": ("needle:3", "needle:2", "role:lfm2_extract", "role:small"),
    "classify": ("needle:3", "needle:2", "role:small", "role:large"),
    "embed": ("sentence-transformers",),
    "judge": ("role:small", "role:large"),
    "generate": ("role:small", "role:large"),
}


@dataclass(frozen=True)
class SpecialistsConfig:
    """Capability -> ordered provider chain."""
    chains: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_SPECIALIST_CHAINS))
    embed_device: str = "cpu"  # sentence-transformers device; cpu keeps VRAM for the language models (Needle is CPU-only)


@dataclass(frozen=True)
class VramConfig:
    """GPU memory budget shared by the llama-server roles (see aof.inference.vram)."""
    budget_mb: int = 0  # 0 = detect: total (or free, with use_free) minus reserve_mb
    reserve_mb: int = 1024  # left for the desktop, display and other processes
    use_free: bool = False  # budget from currently free VRAM instead of total (e.g. when gaming alongside)
    park: str = "unload"  # what parking a swap role does: "unload" (frees RAM too) or "cpu" (keeps it warm on the CPU)
    idle_park_seconds: int = 0  # park swap roles unused for this long (0 = only park when another role needs the room)


@dataclass(frozen=True)
class ModelsConfig:
    """Local model directory for discovery."""
    directory: str = ""


@dataclass(frozen=True)
class RolesConfig:
    """Model path mappings for agent roles (relative to models.directory).

    Model families:
      - Qwen3: thinking models, <tool_call> XML format
      - LFM2/LFM2.5: LiquidAI hybrid conv+attn, <|tool_call_start|> format, 32K context
    """
    micro: str = "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    small: str = "Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    medium: str = "Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
    # Qwen3.5 (hybrid attention; needs an upstream llama.cpp build, so serve it via [llama_server] roles)
    qwen35_4b: str = "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
    qwen35_0_8b: str = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
    vision: str = "Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
    # LiquidAI LFM2.5 — efficient hybrid conv+attn, 32K ctx, great for agentic/RAG
    fast: str = "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"
    reasoning: str = "LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf"
    orchestrator: str = "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"
    thinker: str = "LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf"
    # Liquid Nanos — task-specific LFM2 models (https://huggingface.co/collections/LiquidAI/liquid-nanos)
    lfm2_tool: str = "LFM2-1.2B-Tool-GGUF/LFM2-1.2B-Tool-Q4_K_M.gguf"
    lfm2_rag: str = "LFM2-1.2B-RAG-GGUF/LFM2-1.2B-RAG-Q4_K_M.gguf"
    lfm2_extract: str = "LFM2-1.2B-Extract-GGUF/LFM2-1.2B-Extract-Q4_K_M.gguf"
    lfm2_extract_350m: str = "LFM2-350M-Extract-GGUF/LFM2-350M-Extract-Q4_K_M.gguf"
    lfm2_math: str = "LFM2-350M-Math-GGUF/LFM2-350M-Math-Q4_K_M.gguf"
    lfm2_transcript: str = "LFM2-2.6B-Transcript-GGUF/LFM2-2.6B-Transcript-Q4_K_M.gguf"
    # LFM2.5-VL (vision-language): https://huggingface.co/collections/LiquidAI/lfm25-vl
    lfm2_vl: str = "LFM2.5-VL-1.6B-GGUF/LFM2.5-VL-1.6B-Q4_0.gguf"
    # LFM2.5-JP (Japanese): https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-GGUF
    lfm2_jp: str = "LFM2.5-1.2B-JP-GGUF/LFM2.5-1.2B-JP-Q4_K_M.gguf"
    # Judgement-heavy role (verify, reconcile, structure) for a larger local model; empty = unset.
    # Typically served via [llama_server] roles = ["large"]; absolute paths are allowed.
    large: str = ""
    # Fallback / best performance — use when health is low or explicit --role fallback
    fallback: str = "Qwen3-14B-Instruct-GGUF/Qwen3-14B-Instruct-Q4_K_M.gguf"
    # Long-context steps (same model as medium/fallback, use with role_context for larger n_ctx)
    report_large: str = ""
    analyze_large: str = ""


@dataclass(frozen=True)
class RoleContextConfig:
    """Per-role n_ctx overrides. 0 means use global model.n_ctx default."""
    micro: int = 4096
    small: int = 8192
    medium: int = 16384
    qwen35_4b: int = 8192
    qwen35_0_8b: int = 8192
    vision: int = 8192
    fast: int = 8192
    reasoning: int = 8192
    orchestrator: int = 8192
    thinker: int = 8192
    fallback: int = 16384
    large: int = 8192
    report_large: int = 16384  # Use for report/synthesis steps to avoid context overflow
    analyze_large: int = 16384
    lfm2_tool: int = 0
    lfm2_rag: int = 0
    lfm2_extract: int = 0
    lfm2_extract_350m: int = 0
    lfm2_math: int = 0
    lfm2_transcript: int = 0
    lfm2_vl: int = 0
    lfm2_jp: int = 0

    def get(self, role: str, default: int = 0) -> int:
        """Get n_ctx for a role, returning default if not set or 0."""
        val = getattr(self, role, 0)
        return val if val > 0 else default


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    local_server: LocalServerConfig = field(default_factory=LocalServerConfig)
    llama_server: LlamaServerConfig = field(default_factory=LlamaServerConfig)
    vram: VramConfig = field(default_factory=VramConfig)
    specialists: SpecialistsConfig = field(default_factory=SpecialistsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    role_context: RoleContextConfig = field(default_factory=RoleContextConfig)
    sota: SotaConfig = field(default_factory=SotaConfig)


def _merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _make_config(section_cls, data: dict):
    """Instantiate a frozen dataclass, ignoring unknown keys."""
    valid = {f.name for f in section_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid}
    # Convert lists to tuples for frozen dataclasses
    for k, v in filtered.items():
        if isinstance(v, list):
            filtered[k] = tuple(v)
    return section_cls(**filtered)


def config_with_workspace(config: AppConfig, workspace_root: Path) -> AppConfig:
    """Return a new AppConfig with memory and research paths overridden for the workspace."""
    root = Path(workspace_root).resolve()
    notes_dir = str(root / "memory")
    db_path = str(root / "memory.db")
    queue_path = str(root / "queue.json")
    interrupt_path = str(root / "interrupt.flag")

    memory = MemoryConfig(
        db_path=db_path,
        notes_dir=notes_dir,
        auto_commit=config.memory.auto_commit,
        memory_strategy=config.memory.memory_strategy,
        memory_search_limit=config.memory.memory_search_limit,
        vector_db_path=config.memory.vector_db_path or str(root / "memory" / "chroma"),
        embedding_model=config.memory.embedding_model,
        fts_preview_chars=config.memory.fts_preview_chars,
        consolidation_enabled=config.memory.consolidation_enabled,
        consolidation_interval=config.memory.consolidation_interval,
        consolidation_min_notes=config.memory.consolidation_min_notes,
        agentic_enabled=config.memory.agentic_enabled,
        agentic_llm_backend=config.memory.agentic_llm_backend,
        agentic_llm_model=config.memory.agentic_llm_model,
        agentic_embedding_model=config.memory.agentic_embedding_model,
        agentic_api_key=config.memory.agentic_api_key,
    )
    research = ResearchConfig(
        queue_path=queue_path,
        interrupt_flag_path=interrupt_path,
        extended_mode=config.research.extended_mode,
        director_role=config.research.director_role,
        expert_role=config.research.expert_role,
        expert_count=config.research.expert_count,
        poll_seconds=config.research.poll_seconds,
        route_gen_enabled=config.research.route_gen_enabled,
        route_gen_interval=config.research.route_gen_interval,
        route_gen_min_backlog=config.research.route_gen_min_backlog,
        parallel_items=config.research.parallel_items,
        refinement_pipeline=config.research.refinement_pipeline,
        refinement_passes=config.research.refinement_passes,
        seed_expansion_role=config.research.seed_expansion_role,
        seed_expansion_count=config.research.seed_expansion_count,
        branch_connector_enabled=config.research.branch_connector_enabled,
        branch_connector_interval=config.research.branch_connector_interval,
        lateral_thinking_enabled=config.research.lateral_thinking_enabled,
        lateral_interval=config.research.lateral_interval,
        lateral_min_backlog=config.research.lateral_min_backlog,
    )
    return replace(config, memory=memory, research=research)


def _make_specialists(data: dict) -> SpecialistsConfig:
    """`[specialists.<capability>] providers = [...]` overrides the default chain for that capability."""
    chains = dict(DEFAULT_SPECIALIST_CHAINS)
    for capability, section in data.items():
        providers = section.get("providers") if isinstance(section, dict) else None
        if providers:
            chains[capability] = tuple(providers)
    return SpecialistsConfig(chains=chains, embed_device=str(data.get("embed_device", "cpu")))


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from a TOML file, falling back to defaults."""
    raw: dict = {}
    if path is not None:
        p = Path(path)
        # config.toml is deep-merged with an untracked config.local.toml (machine-specific paths etc.)
        for candidate in (p, p.with_name(f"{p.stem}.local{p.suffix}")):
            if candidate.exists():
                with open(candidate, "rb") as f:
                    raw = _merge(raw, tomllib.load(f))

    for section, key in (("model", "path"), ("models", "directory")):
        value = raw.get(section, {}).get(key)
        if isinstance(value, str) and value:
            raw[section][key] = os.path.expanduser(value)

    return AppConfig(
        model=_make_config(ModelConfig, raw.get("model", {})),
        pool=_make_config(PoolConfig, raw.get("pool", {})),
        memory=_make_config(MemoryConfig, raw.get("memory", {})),
        tools=_make_config(ToolsConfig, raw.get("tools", {})),
        pipeline=_make_config(PipelineConfig, raw.get("pipeline", {})),
        evaluator=_make_config(EvaluatorConfig, raw.get("evaluator", {})),
        local_server=_make_config(LocalServerConfig, raw.get("local_server", {})),
        llama_server=_make_config(LlamaServerConfig, raw.get("llama_server", {})),
        vram=_make_config(VramConfig, raw.get("vram", {})),
        specialists=_make_specialists(raw.get("specialists", {})),
        models=_make_config(ModelsConfig, raw.get("models", {})),
        roles=_make_config(RolesConfig, raw.get("roles", {})),
        research=_make_config(ResearchConfig, raw.get("research", {})),
        discovery=_make_config(DiscoveryConfig, raw.get("discovery", {})),
        role_context=_make_config(RoleContextConfig, raw.get("role_context", {})),
        sota=_make_config(SotaConfig, raw.get("sota", {})),
    )


def resolve_role_path(config: AppConfig, role: str) -> str:
    """Resolve the model path for a role (absolute role paths are kept; others join models.directory)."""
    if role in ("general", "default"):
        return config.model.path
    role_path = getattr(config.roles, role, "") if role in RolesConfig.__dataclass_fields__ else ""
    if not role_path:
        return config.model.path
    models_dir = config.models.directory
    if models_dir:
        return str(Path(models_dir) / role_path)  # absolute role_path wins over models_dir
    return str(role_path)
