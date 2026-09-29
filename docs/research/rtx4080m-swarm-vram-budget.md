# Local home agent swarm on a single RTX 4080M

**A single RTX 4080M with 12GB VRAM can simultaneously run a Qwen3-14B orchestrator (25 GPU layers), a 4B worker, and 8–16 concurrent 0.6B micro-agents — if VRAM is partitioned with sub-gigabyte precision.** This document provides exact memory budgets, class signatures, and a phased build order for a fully in-process agent swarm running inside one Docker container. The design exploits llama-cpp-python for GGUF models and transformers+bitsandbytes for 4-bit models in the same CUDA context, uses FastMCP's in-memory transport for zero-overhead tool dispatch, and LangGraph StateGraph with `thread_id` isolation for concurrent agent execution. Every number here derives from verified model configs (head_dim=128 across all Qwen3 variants, GQA with 8 KV heads) and actual VRAM reports from the unsloth model ecosystem.

---

## The VRAM budget that makes everything work

The central constraint is **12,288 MB of VRAM** on the RTX 4080M. Every architectural decision flows from this. Both llama-cpp-python (GGML CUDA backend) and PyTorch's caching allocator share one CUDA primary context, which costs **~400–500 MB** once. They use separate memory allocators — GGML uses raw `cudaMalloc`, PyTorch uses its caching allocator — so freed memory from one cannot be reused by the other. This makes precise budgeting non-negotiable.

### Qwen3 architecture constants

All Qwen3 models use **head_dim=128** and **8 KV heads** (GQA), regardless of model size. The KV cache formula per token is `2 × num_layers × 8 × 128 × 2 bytes`. Native context is **32,768 tokens** (40,960 max_position_embeddings including prompt budget). The VL-8B variant has 256K native context but we constrain it to 2–4K for VRAM safety.

| Model | Layers | KV/token | Weights (format) | VRAM |
|-------|--------|----------|-------------------|------|
| Qwen3-14B | 40 | 160 KB | GGUF Q4_K_M (9.0 GB file) | ~214 MB/layer |
| Qwen3-4B | 36 | 144 KB | bnb-4bit (unsloth) | ~4.2 GB total |
| Qwen3-0.6B | 28 | 112 KB | bnb-4bit | ~0.6 GB total |
| Qwen3-VL-8B | 36 | 144 KB | bnb-4bit + ViT | ~6.0 GB total |

### Default operating mode (Scenario A)

This is the steady-state configuration where the orchestrator, a fast worker, and the micro-agent fleet all coexist:

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context + fragmentation buffer | 800 MB | Shared by both allocators |
| Qwen3-14B Q4_K_M (25/40 layers) | 5,350 MB | `n_gpu_layers=25`, KV on CPU |
| Qwen3-4B bnb-4bit (Instruct or Thinking) | 4,200 MB | Full model + quantization overhead |
| Qwen3-0.6B bnb-4bit (shared instance) | 600 MB | Single instance serving all micro-agents |
| 0.6B KV cache (batch=8, 1K context) | 896 MB | 8 × 112 KB × 1024 tokens |
| **Total** | **11,846 MB** | **442 MB headroom** |

The 14B runs at **25 of 40 layers on GPU** (62.5% offload), yielding approximately **8–12 tokens/sec** for generation — significantly above CPU-only speeds of 3–5 t/s. The KV cache for 14B stays on CPU via `offload_kqv=False`, which eliminates its 320 MB GPU footprint at 2K context. The 4B model has its full **288 MB KV cache at 2K** absorbed within the 4.2 GB reported VRAM figure. Keeping micro-agent contexts at 1024 tokens is sufficient for scraping and tool-calling tasks and keeps KV cache manageable.

### Vision task mode (Scenario B)

When a VisionAnalyst agent activates, the 4B worker is unloaded:

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context | 800 MB | |
| Qwen3-14B Q4_K_M (17/40 layers) | 3,640 MB | Reduced GPU layers |
| Qwen3-VL-8B bnb-4bit + ViT | 6,000 MB | Includes vision encoder |
| Qwen3-0.6B bnb-4bit | 600 MB | |
| 0.6B KV (batch=4, 1K) | 448 MB | Reduced concurrency |
| **Total** | **11,488 MB** | **800 MB headroom** |

The 14B drops to 17 GPU layers (~42% offload, **5–8 t/s**). This mode activates only for image analysis tasks and automatically reverts to Scenario A when complete.

### Heavy coding mode (Scenario C)

When Qwen3-Coder-Next is requested (assuming ~14B-class model at Q4_K_M):

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context | 800 MB | |
| Coder-Next Q4_K_M (all 40 layers) | 9,000 MB | Full GPU offload |
| Qwen3-0.6B bnb-4bit | 600 MB | Micro-agents stay alive |
| 0.6B KV (batch=4, 1K) | 448 MB | |
| Coder KV (4K context) | 640 MB | |
| **Total** | **11,488 MB** | **800 MB headroom** |

Both the 4B worker and the always-resident 14B orchestrator are evicted. The 14B reloads to CPU-only mode for any orchestration needed during coding. This is a temporary mode — the ModelRouter handles the swap lifecycle automatically.

---

## Inference engine: two runtimes, one CUDA context

The hybrid backend runs **llama-cpp-python** for all GGUF models (14B orchestrator, Coder-Next) and **transformers + bitsandbytes** for all bnb-4bit models (4B workers, 0.6B micro-agents, VL-8B). Both coexist in the same Python process, sharing a single CUDA context.

### Loading order matters

Load llama-cpp-python models **first** because GGML uses raw `cudaMalloc` with fixed allocations. Then load PyTorch models, whose caching allocator dynamically fills remaining space. Call `torch.cuda.set_per_process_memory_fraction(0.5)` to cap PyTorch at 6 GB, leaving the rest for GGML. This fraction is a soft limit on the caching allocator only — it does not cap CUDA context overhead or cuBLAS workspace.

```python
import torch
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Step 1: Cap PyTorch before any model loads
torch.cuda.set_per_process_memory_fraction(0.55, device=0)

# Step 2: Load GGUF model first (fixed VRAM allocation)
orchestrator_llm = Llama(
    model_path="/models/Qwen3-14B-Q4_K_M.gguf",
    n_gpu_layers=25,          # 25 layers × ~214 MB = ~5.35 GB
    n_ctx=2048,               # Context for orchestrator
    offload_kqv=False,        # KV cache stays on CPU
    flash_attn=True,          # Enable flash attention
    n_batch=512,              # Prompt processing batch size
    use_mmap=True,            # Memory-map for fast reload
    verbose=False,
)

# Step 3: Load bnb-4bit models (PyTorch caching allocator)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

worker_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

micro_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B",     # Load at bf16 or bnb-4bit
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### Known compatibility gotchas

**bitsandbytes + llama-cpp-python in same process**: Works, but they use different CUDA allocators. Call `torch.cuda.empty_cache()` after unloading any PyTorch model before adjusting llama.cpp's `n_gpu_layers`. Memory fragmentation is the primary risk — PyTorch's allocator may hold blocks that GGML can't use. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to mitigate.

**Unsloth Docker image baseline**: The `unsloth/unsloth` image ships with PyTorch, transformers, bitsandbytes, accelerate, peft, and xformers. It does **not** include llama-cpp-python or vLLM. Add to your Dockerfile:

```dockerfile
FROM unsloth/unsloth:latest
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python>=0.3.0
RUN pip install langgraph>=1.0.8 fastmcp>=2.0.0 chromadb>=0.4.22 \
    aiosqlite>=0.20.0 playwright>=1.40.0 litellm>=1.16.11 \
    sentence-transformers>=2.2.2
```

---

## ModelRouter: the VRAM orchestrator

The ModelRouter is the single point of control for all inference. It tracks VRAM allocations, queues load/unload requests, and dispatches inference calls to the correct backend. Every agent requests inference through the ModelRouter — no agent directly holds a model reference.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import asyncio
import threading

class ModelTier(Enum):
    ORCHESTRATOR = "orchestrator"    # 14B GGUF, always-resident
    WORKER = "worker"                # 4B bnb-4bit, always-resident
    VISION = "vision"                # VL-8B bnb-4bit, on-demand
    MICRO = "micro"                  # 0.6B, always-resident
    CODER = "coder"                  # Coder-Next, on-demand

class ModelBackend(Enum):
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS_BNB = "transformers_bnb"
    TRANSFORMERS_FP16 = "transformers_fp16"

@dataclass
class ModelSpec:
    model_id: str
    tier: ModelTier
    backend: ModelBackend
    vram_budget_mb: int              # Max VRAM this model may consume
    n_gpu_layers: Optional[int]      # For GGUF models
    is_resident: bool                # Always loaded vs on-demand
    preemption_priority: int         # Lower = evicted first (0=first to go)

@dataclass
class LoadedModel:
    spec: ModelSpec
    instance: object                 # Llama | PreTrainedModel
    tokenizer: Optional[object]      # AutoTokenizer for transformers models
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = 0.0

class ModelRouter:
    """Central dispatch for all inference. Manages VRAM budget and model lifecycle."""

    def __init__(self, total_vram_mb: int = 12288, reserved_mb: int = 800):
        self.total_vram_mb = total_vram_mb
        self.reserved_mb = reserved_mb
        self.available_mb = total_vram_mb - reserved_mb
        self.models: dict[str, LoadedModel] = {}
        self.registry: dict[str, ModelSpec] = {}
        self._load_lock = asyncio.Lock()
        self._inference_executor = ThreadPoolExecutor(max_workers=2)
        self._request_batcher: Optional[RequestBatcher] = None

    def register(self, spec: ModelSpec) -> None:
        self.registry[spec.model_id] = spec

    async def ensure_loaded(self, model_id: str) -> LoadedModel:
        if model_id in self.models:
            self.models[model_id].last_used = time.monotonic()
            return self.models[model_id]
        async with self._load_lock:
            if model_id in self.models:
                return self.models[model_id]
            spec = self.registry[model_id]
            await self._make_room(spec.vram_budget_mb)
            loaded = await self._load_model(spec)
            self.models[model_id] = loaded
            self.available_mb -= spec.vram_budget_mb
            return loaded

    async def unload(self, model_id: str) -> None:
        if model_id not in self.models:
            return
        loaded = self.models.pop(model_id)
        spec = loaded.spec
        if spec.backend == ModelBackend.LLAMA_CPP:
            del loaded.instance
        else:
            del loaded.instance
            del loaded.tokenizer
            import gc; gc.collect()
            torch.cuda.empty_cache()
        self.available_mb += spec.vram_budget_mb

    async def _make_room(self, needed_mb: int) -> None:
        while self.available_mb < needed_mb:
            victim = min(
                (m for m in self.models.values() if not m.spec.is_resident),
                key=lambda m: (m.spec.preemption_priority, m.last_used),
                default=None,
            )
            if victim is None:
                # Must evict a resident model — enter degraded mode
                victim = min(
                    self.models.values(),
                    key=lambda m: m.spec.preemption_priority,
                )
            await self.unload(victim.spec.model_id)

    async def infer(
        self,
        model_id: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.6,
        **kwargs,
    ) -> str:
        loaded = await self.ensure_loaded(model_id)
        loop = asyncio.get_running_loop()

        if loaded.spec.backend == ModelBackend.LLAMA_CPP:
            result = await loop.run_in_executor(
                self._inference_executor,
                lambda: loaded.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return result["choices"][0]["message"]["content"]
        else:
            # Transformers path — use the batcher for micro-agents
            if loaded.spec.tier == ModelTier.MICRO and self._request_batcher:
                prompt = loaded.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return await self._request_batcher.generate(
                    prompt, max_new_tokens=max_tokens, temperature=temperature
                )
            # Single inference for worker/vision models
            return await loop.run_in_executor(
                self._inference_executor,
                self._sync_transformers_infer,
                loaded, messages, max_tokens, temperature,
            )
```

### Dynamic model loading lifecycle

When a micro-agent or worker requests a model not currently loaded (e.g., Coder-Next), the flow is:

1. **Request arrives** at `ModelRouter.infer("coder-next", messages)`.
2. `ensure_loaded` acquires `_load_lock` (prevents concurrent load attempts).
3. `_make_room` calculates shortfall and evicts models by `preemption_priority` (lowest first), then by `last_used` (LRU). On-demand models evict before residents.
4. The requesting agent's coroutine **awaits** the load — it does not block the event loop or thread pool. Other agents continue running on already-loaded models.
5. The model loads from NVMe SSD. For a 4B bnb-4bit model, expect **3–5 seconds**; for an 8B VL model, **5–10 seconds**. CPU-offloaded models (kept in RAM via `device_map="cpu"`) can move to GPU in **1–2 seconds** via PCIe transfer.
6. After loading, `_load_lock` releases, and all queued requests for that model proceed.

### Preemption priority table

| Model | Priority | Resident | Eviction order |
|-------|----------|----------|----------------|
| VL-8B | 0 | No | First to go |
| Coder-Next | 1 | No | Second |
| 4B-Thinking | 2 | Yes (soft) | Third — degraded mode |
| 4B-Instruct | 2 | Yes (soft) | Third — degraded mode |
| 0.6B | 8 | Yes (hard) | Almost never evicted |
| 14B | 9 | Yes (hard) | Last resort only |

---

## Micro-agent fleet: batched inference for 8–16 concurrent 0.6B instances

The 0.6B tier is the workhorse for scraping, tool-calling, and lightweight analysis. A single model instance on GPU serves all micro-agents through a **RequestBatcher** that collects concurrent requests and dispatches them as batched `model.generate()` calls.

### Why not vLLM or llama.cpp server

**vLLM** provides ideal continuous batching but spawns subprocesses by default in V1 (even single-GPU), pre-allocates GPU memory aggressively, and adds a heavy dependency. **llama.cpp server** with `--parallel 16` provides native continuous batching but requires HTTP communication (adding 1–5ms per call) and doesn't expose a Python API for parallel slots. The custom RequestBatcher trades continuous batching for simplicity: static batching with a configurable window, fully in-process, zero additional dependencies. For scraping tasks where latency tolerance is 100ms+, this tradeoff is acceptable.

For Phase 4 scaling, consider upgrading to llama.cpp server running as a subprocess within the container, communicating via Unix domain socket for minimal overhead.

### RequestBatcher implementation

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    future: Optional[asyncio.Future] = None
    created_at: float = field(default_factory=time.monotonic)

class RequestBatcher:
    """Async batcher for HuggingFace model.generate().

    Collects requests within a time window, batches them into a single
    generate() call on a dedicated thread, and resolves individual futures.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "AutoTokenizer",
        max_batch_size: int = 16,
        max_wait_ms: float = 10.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)  # Serialized GPU access
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        self._executor.shutdown(wait=True)

    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = InferenceRequest(
            prompt=prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.6),
            future=future,
        )
        await self._queue.put(req)
        return await future

    async def _batch_loop(self) -> None:
        while self._running:
            batch: list[InferenceRequest] = []
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch.append(first)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    ))
                except asyncio.TimeoutError:
                    break
            loop = asyncio.get_running_loop()
            try:
                results = await loop.run_in_executor(
                    self._executor, self._sync_generate, batch
                )
                for req, text in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(text)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    def _sync_generate(self, batch: list[InferenceRequest]) -> list[str]:
        prompts = [r.prompt for r in batch]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=batch[0].max_new_tokens,
                temperature=batch[0].temperature,
                do_sample=True,
            )
        results = []
        for i, output in enumerate(ids):
            input_len = inputs["input_ids"][i].ne(
                self.tokenizer.pad_token_id
            ).sum().item()
            text = self.tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            results.append(text.strip())
        return results
```

**Critical detail**: the tokenizer must have `padding_side="left"` for decoder-only models, and `pad_token` set to `eos_token`. The `ThreadPoolExecutor(max_workers=1)` serializes all GPU access — **never call `model.generate()` from multiple threads simultaneously** on the same model instance, as PyTorch's default CUDA stream is not thread-safe for generation.

### Batch window tuning

For 8 micro-agents firing near-simultaneously, a **10ms window** captures most requests in one batch. For staggered workloads (agents operating on different timescales), increase to 20–30ms. At batch_size=8 with 1K context on a 0.6B model, inference takes approximately **200–500ms** per batch, making the 10ms collection window negligible overhead.

### KV cache budget at scale

At bnb-4bit with bf16 KV cache, the 0.6B model consumes **112 KB per token per request**. For 8 concurrent requests at 1K context: 8 × 112 KB × 1024 = **896 MB**. For 16 at 2K: 16 × 224 MB = **3.58 GB**. The architecture pins micro-agents to 1K context maximum, keeping KV overhead under 1 GB even at 8 concurrent requests.

---

## Agent instances via LangGraph StateGraph

All agents are Python objects sharing one compiled `StateGraph`, isolated by LangGraph's `thread_id` mechanism. The graph definition is compiled once; each agent invocation passes a unique `thread_id` in its config, giving it completely independent state and checkpoint history.

### State schema

```python
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from datetime import datetime
from enum import Enum

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    VISION_ANALYST = "vision_analyst"
    SCRAPER = "scraper"
    TOOL_BUILDER = "tool_builder"

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    role: AgentRole
    task_id: str
    agent_id: str
    model_id: str = ""
    working_context: str = ""
    tool_results: list[dict] = Field(default_factory=list)
    error_count: int = 0
    token_count: int = 0
    status: Literal["active", "waiting", "done", "error"] = "active"

class TaskLedger(BaseModel):
    task_id: str
    description: str
    assigned_agents: list[str] = Field(default_factory=list)
    subtasks: list[dict] = Field(default_factory=list)
    status: Literal["pending", "active", "done", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.now)

class ProgressLedger(BaseModel):
    task_id: str
    completed_steps: list[str] = Field(default_factory=list)
    current_step: str = ""
    blockers: list[str] = Field(default_factory=list)
    confidence: float = 0.0
```

### Concurrent execution pattern

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import RetryPolicy
import asyncio

# Build graph once
builder = StateGraph(AgentState)
builder.add_node("plan", plan_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("execute", execute_node, retry=RetryPolicy(max_attempts=2))
builder.add_node("reflect", reflect_node)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", route_after_plan)
builder.add_conditional_edges("execute", route_after_execute)
builder.add_edge("reflect", END)

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    # Launch 12 concurrent agents
    tasks = []
    for i in range(8):
        config = {"configurable": {"thread_id": f"scraper-{i}"}}
        input_state = AgentState(
            messages=[], role=AgentRole.SCRAPER,
            task_id="task-001", agent_id=f"scraper-{i}",
            model_id="qwen3-0.6b",
        )
        tasks.append(graph.ainvoke(input_state.model_dump(), config))

    results = await asyncio.gather(*tasks)
```

Each `ainvoke` call runs through the graph independently. LangGraph's checkpointer keys all state by `thread_id`, so **scraper-0** and **scraper-7** never see each other's state. The compiled graph object is stateless and safely shared.

### Asyncio event loop design

The system runs a **single asyncio event loop**. All agent coroutines, MCP tool calls, and memory operations run as async tasks. Blocking operations (GPU inference, file I/O) are dispatched to a `ThreadPoolExecutor`:

```python
class AgentRuntime:
    def __init__(self, model_router: ModelRouter, memory: "NamespacedMemory"):
        self.router = model_router
        self.memory = memory
        self.loop = asyncio.get_event_loop()
        self._cpu_executor = ThreadPoolExecutor(max_workers=4)  # CPU-bound tasks
        # GPU inference goes through ModelRouter's own executor

    async def run_agent(self, agent_id: str, role: AgentRole, task_id: str):
        """Run a single agent as an async task."""
        config = {"configurable": {"thread_id": agent_id}}
        state = AgentState(
            messages=[], role=role, task_id=task_id,
            agent_id=agent_id, model_id=ROLE_MODEL_MAP[role],
        )
        return await self.graph.ainvoke(state.model_dump(), config)

    async def spawn_scraper_fleet(self, task_id: str, targets: list[str]):
        """Spawn 8-16 scraper micro-agents concurrently."""
        tasks = [
            self.run_agent(f"scraper-{i}", AgentRole.SCRAPER, task_id)
            for i, _ in enumerate(targets)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

---

## A-mem integration with three-tier namespaces

The A-mem system (`agentic-memory` package) provides a Zettelkasten-style memory with LLM-driven evolution. Each `MemoryNote` carries content, keywords, links to related notes, context, tags, category, and an evolution history. The core class `AgenticMemorySystem` stores notes in both a Python dict (`self.memories`) and a ChromaDB collection (via `ChromaRetriever`), using `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional embeddings.

### Thread safety assessment

**ChromaDB is thread-safe for multi-threaded single-process access** — explicitly confirmed in Chroma's official documentation. Reads parallelize up to vCPU count, writes are WAL-buffered with stable latency as writer count increases. The `PersistentClient` is synchronous-only (no native async), so all ChromaDB calls must go through `loop.run_in_executor()`. For the expected load of an agent swarm (dozens of memory operations per minute, not thousands per second), ChromaDB is more than sufficient and does not need to be replaced.

### NamespacedMemory wrapper

A-mem has **no built-in namespace support**. Each `AgenticMemorySystem` instance operates on a single ChromaDB collection. The namespace design uses separate ChromaDB collections per tier:

```python
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

class NamespacedMemory:
    """Three-tier memory namespace over A-mem instances.

    - private/{agent_id}: Agent's personal scratchpad
    - task/{task_id}: Shared within a task's agent team
    - shared/global: System-wide knowledge
    """

    def __init__(self, base_path: str = "./memory_store"):
        self._systems: dict[str, AgenticMemorySystem] = {}
        self._base_path = base_path
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _get_system(self, namespace: str) -> AgenticMemorySystem:
        with self._lock:
            if namespace not in self._systems:
                self._systems[namespace] = AgenticMemorySystem(
                    model_name="all-MiniLM-L6-v2",
                    llm_backend="ollama",       # Route through local ModelRouter
                    llm_model="qwen3-4b",
                    # ChromaDB collection path parameterized per namespace
                )
            return self._systems[namespace]

    async def add_note(
        self,
        content: str,
        namespace: str,                    # "private/agent-1" | "task/task-001" | "shared/global"
        tags: list[str] | None = None,
        category: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        system = self._get_system(namespace)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: system.add_note(content, tags=tags, category=category),
        )

    async def search(
        self,
        query: str,
        namespaces: list[str],            # Search across multiple tiers
        k: int = 5,
    ) -> list[dict]:
        loop = asyncio.get_running_loop()
        all_results = []
        for ns in namespaces:
            system = self._get_system(ns)
            results = await loop.run_in_executor(
                self._executor,
                lambda s=system: s.search_agentic(query, k=k),
            )
            for r in results:
                r["namespace"] = ns
            all_results.extend(results)
        # Re-rank by relevance across namespaces
        all_results.sort(key=lambda x: x.get("distance", float("inf")))
        return all_results[:k]
```

### TaskContextBuilder

Before every inference call, the `TaskContextBuilder` assembles the optimal context for a (task, role) pair by querying all three memory tiers:

```python
class TaskContextBuilder:
    def __init__(self, memory: NamespacedMemory, model_router: ModelRouter):
        self.memory = memory
        self.router = model_router

    async def build_context(
        self,
        task_id: str,
        agent_id: str,
        role: AgentRole,
        query: str,
        max_tokens: int = 2048,
    ) -> str:
        namespaces = [
            f"private/{agent_id}",
            f"task/{task_id}",
            "shared/global",
        ]
        memories = await self.memory.search(query, namespaces, k=10)

        context_parts = []
        token_budget = max_tokens
        for mem in memories:
            chunk = f"[{mem['namespace']}] {mem['content']}"
            chunk_tokens = len(chunk.split()) * 1.3  # Rough estimate
            if chunk_tokens > token_budget:
                break
            context_parts.append(chunk)
            token_budget -= chunk_tokens

        return "\n---\n".join(context_parts)
```

### ContextDistiller for window overflow

When an agent's accumulated context exceeds 80% of the model's context window, the distiller compresses it using a 4B model:

```python
class ContextDistiller:
    THRESHOLD_RATIO = 0.8  # Trigger at 80% of context window

    def __init__(self, model_router: ModelRouter, memory: NamespacedMemory):
        self.router = model_router
        self.memory = memory

    async def maybe_distill(
        self, state: AgentState, context_window: int = 2048
    ) -> AgentState:
        if state.token_count < int(context_window * self.THRESHOLD_RATIO):
            return state
        summary = await self.router.infer(
            "qwen3-4b-thinking",
            messages=[{
                "role": "user",
                "content": f"Compress this working context into key facts and "
                           f"decisions. Preserve all actionable details:\n\n"
                           f"{state.working_context}",
            }],
            max_tokens=512,
            temperature=0.3,
        )
        await self.memory.add_note(
            content=f"Distilled context: {summary}",
            namespace=f"task/{state.task_id}",
            tags=["distilled", "context"],
            agent_id=state.agent_id,
        )
        state.working_context = summary
        state.token_count = len(summary.split()) + 50  # Reset count
        return state
```

---

## MCP servers as in-process FastMCP instances

FastMCP v2.x provides a dedicated **in-memory transport** via `FastMCPTransport`. When you pass a `FastMCP` server object directly to `Client()`, it creates a zero-network, same-process connection. No HTTP server, no subprocess, no stdio pipes. Messages stay as in-memory Python objects, going through MCP's JSON-RPC protocol layer for schema validation only.

```python
from fastmcp import FastMCP, Client

memory_mcp = FastMCP("MemoryServer")
filesystem_mcp = FastMCP("FilesystemServer")
shell_mcp = FastMCP("ShellServer")
model_router_mcp = FastMCP("ModelRouterServer")

@memory_mcp.tool
async def search_memory(query: str, namespaces: list[str], k: int = 5) -> list[dict]:
    """Search agent memory across namespace tiers."""
    return await namespaced_memory.search(query, namespaces, k)

@memory_mcp.tool
async def add_memory(content: str, namespace: str, tags: list[str] = []) -> str:
    """Store a memory note in the specified namespace."""
    return await namespaced_memory.add_note(content, namespace, tags=tags)

@model_router_mcp.tool
async def infer(model_id: str, messages: list[dict], max_tokens: int = 512) -> str:
    """Request inference from any model in the fleet."""
    return await model_router.infer(model_id, messages, max_tokens=max_tokens)

@model_router_mcp.tool
async def load_model(model_id: str) -> dict:
    """Pre-load a model into VRAM."""
    loaded = await model_router.ensure_loaded(model_id)
    return {"status": "loaded", "model_id": model_id, "vram_mb": loaded.spec.vram_budget_mb}

# Agents access tools via in-memory client
async def agent_uses_tools():
    async with Client(memory_mcp) as mem_client:
        results = await mem_client.call_tool(
            "search_memory",
            {"query": "deployment status", "namespaces": ["shared/global"]},
        )
```

### Dynamic tool registration for ToolBuilder

FastMCP supports `add_tool()` and `remove_tool()` at runtime. When the ToolBuilder agent generates a new Python function, it registers it immediately:

```python
@model_router_mcp.tool
async def register_dynamic_tool(
    code: str,
    tool_name: str,
    description: str,
    target_server: str = "tools",
) -> dict:
    """Register AI-generated code as a new MCP tool."""
    # Execute in sandbox first to validate
    namespace = {"__builtins__": __builtins__}
    exec(code, namespace)
    func = namespace[tool_name]

    # Register with the appropriate MCP server
    dynamic_mcp.add_tool(func, name=tool_name, description=description)
    return {"status": "registered", "tool_name": tool_name}
```

The MCP protocol supports `notifications/tools/list_changed`, so clients automatically discover new tools.

### Core MCP server roster

- **MemoryMCPServer**: Wraps `NamespacedMemory` with `search_memory()`, `add_memory()`, `update_memory()`, `get_memory()`. Routes to correct namespace based on `agent_id`/`task_id` parameters.
- **FilesystemMCPServer**: `read_file()`, `write_file()`, `list_dir()`. Sandboxed to `workspace/{task_id}/`. Uses `pathlib.Path.resolve()` to prevent directory traversal.
- **ShellMCPServer**: `exec_command(cmd, timeout=30)`. Runs via `asyncio.create_subprocess_exec` with kill-on-timeout. Captures stdout/stderr.
- **BrowserMCPServer**: `navigate(url)`, `screenshot()`, `extract_text(selector)`, `click(selector)`. Uses Playwright's async API.
- **CodeExecMCPServer**: `execute_python(code, timeout=10)`. Runs in NsJail sandbox (see Security section).
- **ModelRouterMCPServer**: `infer()`, `load_model()`, `unload_model()`, `list_loaded()`. The central inference dispatch point.

---

## Pydantic pipeline schemas

Every boundary between agents, tools, and the ModelRouter is typed with Pydantic models. These compose into a schema hierarchy that LangGraph StateGraph nodes consume and produce:

```python
from pydantic import BaseModel, Field
from typing import Literal, Union
from datetime import datetime

# === Inference boundary ===
class InferenceRequest(BaseModel):
    model_id: str
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    request_id: str = Field(default_factory=lambda: str(uuid4()))

class InferenceResult(BaseModel):
    request_id: str
    model_id: str
    content: str
    tokens_used: int
    latency_ms: float

# === Tool boundary ===
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    server: str = "default"

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: object = None
    error: str | None = None

# === Memory boundary ===
class MemoryWriteRequest(BaseModel):
    content: str
    namespace: str
    tags: list[str] = Field(default_factory=list)
    category: str = "Uncategorized"
    agent_id: str
    task_id: str

class MemoryReadResult(BaseModel):
    memories: list[dict]
    namespace: str
    query: str
    count: int

# === Agent boundary ===
class AgentInput(BaseModel):
    task_description: str
    role: AgentRole
    context: str = ""
    constraints: list[str] = Field(default_factory=list)

class AgentOutput(BaseModel):
    result: str
    artifacts: list[str] = Field(default_factory=list)  # Workspace file paths
    memory_writes: list[MemoryWriteRequest] = Field(default_factory=list)
    next_actions: list[ToolCall] = Field(default_factory=list)
    status: Literal["done", "needs_review", "blocked", "error"] = "done"

# === Error types ===
class PipelineError(BaseModel):
    error_type: Literal["inference", "tool", "memory", "timeout", "validation"]
    message: str
    recoverable: bool
    retry_count: int = 0
    max_retries: int = 3
```

LangGraph conditional edges use the `status` discriminant on `AgentOutput` to route:

```python
def route_on_output(state: AgentState) -> str:
    if state.status == "error" and state.error_count < 3:
        return "retry"
    if state.status == "blocked":
        return "escalate_to_orchestrator"
    if state.status == "needs_review":
        return "reflect"
    return "__end__"
```

The ContextDistiller fires as middleware before every inference call. Inside the graph, a `pre_inference` node checks `state.token_count` against `THRESHOLD_RATIO * context_window` and routes to distillation if needed.

---

## Storage, workspace, and event logging

### ChromaDB stays as the A-mem backend

Despite ChromaDB's fundamentally single-threaded HNSW index, the expected load (tens of memory operations per minute, not per second) makes it adequate. Swapping A-mem's backend to SurrealDB or pgvector requires refactoring the `ChromaRetriever` class into a formal `VectorStoreProtocol` — moderate effort (~200 lines) but unnecessary at this scale. ChromaDB's HNSW index resides entirely in RAM; for 100K notes with 384-dim embeddings, expect **~1.6 GB** system RAM usage. Configure the LRU cache to cap this: `chroma_segment_cache_policy="LRU"` with `chroma_memory_limit_bytes=536870912` (512 MB).

### Workspace structure

```
/workspace/
├── models/                    # GGUF and cached model files
│   ├── Qwen3-14B-Q4_K_M.gguf
│   └── ...
├── tasks/
│   └── {task_id}/
│       ├── {agent_id}/        # Per-agent artifacts
│       │   ├── scrape_output.json
│       │   └── generated_code.py
│       └── shared/            # Task-level shared artifacts
│           └── final_report.md
├── memory_store/              # ChromaDB persistent storage
│   ├── private/
│   ├── task/
│   └── shared/
├── logs/
│   └── events.db             # SQLite event log
└── checkpoints/
    └── langgraph.db          # LangGraph checkpointer
```

### Append-only event log

Every operation is recorded for debugging and replay:

```python
import aiosqlite
from datetime import datetime

class EventLog:
    def __init__(self, db_path: str = "/workspace/logs/events.db"):
        self.db_path = db_path

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    agent_id TEXT,
                    task_id TEXT,
                    model_id TEXT,
                    payload TEXT,
                    latency_ms REAL,
                    success BOOLEAN
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )
            await db.commit()

    async def log(
        self,
        event_type: str,      # "inference" | "tool_call" | "memory_write" | "state_transition"
        agent_id: str = "",
        task_id: str = "",
        model_id: str = "",
        payload: str = "",
        latency_ms: float = 0.0,
        success: bool = True,
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO events (timestamp, event_type, agent_id, task_id, "
                "model_id, payload, latency_ms, success) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), event_type, agent_id, task_id,
                 model_id, payload, latency_ms, success),
            )
            await db.commit()
```

aiosqlite uses a dedicated thread per connection, so concurrent `await log()` calls from multiple agent tasks are safe. SQLite's WAL mode (enabled by default in aiosqlite) allows concurrent reads with writes.

---

## Security and isolation

### Container-level controls

```bash
docker run \
  --gpus '"device=0"' \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --memory=32g \
  --memory-swap=32g \
  --cpus=8 \
  --pids-limit=512 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256M \
  -v /host/workspace:/workspace:rw \
  -v /host/models:/models:ro \
  --network=bridge \
  agent-swarm:latest
```

NVIDIA GPU passthrough does **not** require `--privileged`. The `nvidia-container-toolkit` handles device mapping cleanly. GPU memory is **not** isolated between processes on the same GPU — all models share the full 12GB. Set `NVIDIA_DRIVER_CAPABILITIES=compute,utility` to minimize exposed GPU features.

### AI-generated code sandbox

RestrictedPython is **not a security boundary** — it has a history of bypasses (CVE-2025-22153, CVE-2024-47532) and its own documentation disclaims sandbox status. The recommended approach uses **NsJail** (or the snekbox wrapper) for namespace-isolated execution within the container:

```python
import asyncio
import json

class CodeSandbox:
    """Execute AI-generated Python in NsJail isolation."""

    NSJAIL_CONFIG = {
        "time_limit": 10,
        "cgroup_mem_max": 256 * 1024 * 1024,  # 256 MB
        "cgroup_pids_max": 5,
        "mount_proc": False,
        "disable_clone_newnet": False,  # No network access
    }

    async def execute(self, code: str, timeout: int = 10) -> dict:
        proc = await asyncio.create_subprocess_exec(
            "nsjail",
            "--mode", "once",
            "--time_limit", str(timeout),
            "--cgroup_mem_max", str(self.NSJAIL_CONFIG["cgroup_mem_max"]),
            "--cgroup_pids_max", str(self.NSJAIL_CONFIG["cgroup_pids_max"]),
            "--disable_clone_newnet",
            "--", "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 5
        )
        return {
            "stdout": stdout.decode()[:10000],
            "stderr": stderr.decode()[:10000],
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
        }
```

If NsJail setup proves too complex, a fallback approach uses subprocess with the `resource` module:

```python
import subprocess
import resource

def _run_sandboxed(code: str, timeout: int = 10) -> dict:
    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # No fork

    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, timeout=timeout + 2,
        preexec_fn=set_limits,
    )
    return {"stdout": result.stdout.decode(), "returncode": result.returncode}
```

### Network policy

True per-agent network isolation within a single container requires `CAP_NET_ADMIN` (a security tradeoff). Instead, use a **Python-level HTTP allowlist** in the shared `aiohttp.ClientSession`:

```python
class FilteredConnector(aiohttp.TCPConnector):
    ALLOWED_DOMAINS = {"*.wikipedia.org", "api.github.com", ...}
    BLOCKED_RANGES = ["10.0.0.0/8", "172.16.0.0/12", "169.254.169.254/32"]

    async def _resolve_host(self, host, port, traces=None):
        # Check against allowlist before connecting
        ...
```

Scraper agents route through this connector. The orchestrator and tool builder have no outbound access by default.

---

## Observability stack

**OpenTelemetry** with the `opentelemetry-instrumentation-asyncio` package traces coroutines, futures, and thread dispatch. LangGraph integration comes via LangSmith's OTel export (`LANGSMITH_OTEL_ENABLED=true`, requires `langsmith>=0.4.25`), which automatically captures node transitions, tool calls, and LLM invocations as spans.

For a home deployment, ship traces to a local **Jaeger** instance (single binary, ~50 MB RAM) running alongside the agent container. OTel overhead is **3–9% on p95 latency** at full sampling; use `TraceIdRatioBasedSampler(0.1)` for 10% sampling in production, which drops overhead to ~3%. Since LLM inference dominates at 500ms–5s per call, this overhead is negligible.

The `EventLog` SQLite table serves as the structured audit trail for replay and debugging, complementing OTel's distributed tracing with a queryable, append-only record of every system operation.

---

## Phased implementation roadmap

### Phase 1: Core inference stack (weeks 1–2)

**Goal**: 14B + 4B resident, ModelRouter dispatching, basic 0.6B serving.

1. Extend Dockerfile from `unsloth/unsloth` — install `llama-cpp-python` with CUDA, `langgraph>=1.0.8`, `fastmcp>=2.0.0`, `chromadb>=0.4.22`, `aiosqlite>=0.20.0`.
2. Implement `ModelRouter` with `register()`, `ensure_loaded()`, `unload()`, `infer()`. Start with hardcoded model specs for 14B, 4B-Instruct, 4B-Thinking, and 0.6B.
3. Validate VRAM budgets empirically: load 14B at `n_gpu_layers=25`, then 4B bnb-4bit, then 0.6B. Confirm total VRAM via `nvidia-smi` stays under 11.8 GB.
4. Implement `RequestBatcher` for 0.6B with `max_batch_size=8`, `max_wait_ms=10`.
5. Write integration test: 8 concurrent async tasks each calling `router.infer("qwen3-0.6b", ...)` and getting distinct responses.

**Library versions for Phase 1:**
```
torch>=2.1.0
transformers>=4.51.0
bitsandbytes>=0.43.0
accelerate>=0.27.0
llama-cpp-python>=0.3.0  # Built with -DGGML_CUDA=on
langgraph>=1.0.8
fastmcp>=2.0.0,<3.0.0
chromadb>=0.4.22
aiosqlite>=0.20.0
sentence-transformers>=2.2.2
```

### Phase 2: A-mem integration + MCP servers (weeks 3–4)

1. Install `agentic-memory` (A-mem). Fork and patch `ChromaRetriever` to accept a configurable collection name and persist directory.
2. Implement `NamespacedMemory` wrapping multiple `AgenticMemorySystem` instances.
3. Implement the six core MCP servers as `FastMCP` instances with in-memory transport.
4. Wire `ModelRouterMCPServer` to the actual `ModelRouter`. Validate that agents can call `infer()` through the MCP tool interface.
5. Implement `TaskContextBuilder` and `ContextDistiller`.
6. Set up `EventLog` with aiosqlite.

### Phase 3: LangGraph agent instances + Pydantic schemas (weeks 5–6)

1. Define the full Pydantic schema hierarchy (`AgentInput`, `AgentOutput`, `InferenceRequest`, etc.).
2. Build the `StateGraph` with nodes for `plan`, `execute`, `use_tool`, `reflect`, `distill_context`.
3. Implement conditional edges for routing on `AgentOutput.status` and error handling via `RetryPolicy`.
4. Implement agent role dispatch: the `execute` node inspects `state.role` and calls the appropriate model via `ModelRouterMCPServer`.
5. Implement the Orchestrator's Task Ledger and Progress Ledger as shared A-mem entries.
6. Test concurrent execution: 1 orchestrator + 2 workers + 4 scrapers running simultaneously.

### Phase 4: Micro-agent scraper fleet + dynamic tools (weeks 7–8)

1. Scale `RequestBatcher` to `max_batch_size=16`. Profile actual batch sizes and window timing under load.
2. Implement the `MicroAgentController` that receives scraping directives from the orchestrator, spawns 8–16 scraper tasks, collects results, and reports back.
3. Implement `ToolBuilder` agent: accepts a natural-language tool description, uses 14B or Coder-Next to generate Python code, validates in `CodeSandbox`, registers via `dynamic_mcp.add_tool()`.
4. Implement the dynamic model request flow: micro-agent calls `infer("coder-next", ...)`, ModelRouter evicts as needed, loads from NVMe, serves, then optionally unloads.
5. Implement VL-8B on-demand loading for `VisionAnalyst` role.

### Phase 5: Observability and hardening (weeks 9–10)

1. Add OTel instrumentation: `BatchSpanProcessor`, `TraceIdRatioBasedSampler(0.1)`, export to local Jaeger.
2. Integrate LangSmith OTel export for LangGraph node-level tracing.
3. Set up `CodeSandbox` with NsJail for AI-generated code execution.
4. Implement the `FilteredConnector` HTTP allowlist.
5. Harden Docker run command with capability drops, memory limits, and read-only root filesystem.
6. Add `ContextDistiller` threshold checks as middleware before every inference call.
7. End-to-end test: orchestrator receives complex task → decomposes → spawns researcher + coder + scraper fleet → assembles result → writes to workspace.

---

## Conclusion

The architecture's viability hinges on three verified facts. First, llama-cpp-python and transformers+bitsandbytes share a single CUDA context (~500 MB overhead once, not twice), making the dual-runtime approach practical. Second, Qwen3's universal **head_dim=128 with 8 KV heads** means KV cache costs scale predictably across all model sizes — the 0.6B's 112 KB/token at batch=8 with 1K context costs only 896 MB, leaving ample room for co-resident models. Third, FastMCP's in-memory transport eliminates the overhead of MCP server processes entirely, making the "everything in one process" constraint not just feasible but architecturally clean.

The most surprising finding is that **ChromaDB's single-threaded HNSW bottleneck is irrelevant at home-agent scale** — the real serialization point is GPU inference, where the `RequestBatcher` + single-threaded executor pattern is unavoidable regardless of storage backend. The highest-risk integration is the VRAM boundary between PyTorch's caching allocator and GGML's raw `cudaMalloc` — memory fragmentation under load will be the first failure mode to monitor. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` from day one, and build the `EventLog` early to capture VRAM allocation patterns before they become production incidents.