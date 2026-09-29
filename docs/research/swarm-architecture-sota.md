# Building a composable home agent swarm: SOTA architecture guide

**A multi-agent home swarm built on A-mem, MCP, and Docker is technically viable today, but the architecture demands careful constraint.** The most critical finding from DeepMind's December 2025 scaling study is the **"45% threshold"**: multi-agent systems only reliably outperform single agents when base-model task accuracy falls below ~45%, and accuracy gains saturate beyond 4 agents in any topology. This means your swarm should be lean—3-4 specialist agents maximum, with a hierarchical orchestrator using a blackboard-style shared workspace (A-mem), communicating exclusively through MCP's Streamable HTTP transport. The structured output discipline (Pydantic everywhere) is arguably the single most impactful reliability decision, as it transforms emergent agent behavior into verifiable, replayable pipelines.

---

## 1. Orchestration topology: hierarchical blackboard with ≤4 agents

The dominant orchestration pattern in 2025 combines **hierarchical delegation** with a **shared blackboard** (workspace). Microsoft's Magentic-One exemplifies this: an Orchestrator agent maintains a Task Ledger (facts + guesses) and Progress Ledger (self-reflection per step), delegating to specialist workers (WebSurfer, Coder, FileSurfer, ComputerTerminal). When stuck, an outer loop triggers replanning. CAMEL's OWL project, built on this pattern, achieved **#1 open-source score on GAIA** (58.18% average) and was accepted at NeurIPS 2025.

**Framework selection matters less than architecture.** LangGraph provides the most mature graph-based state machine with built-in checkpointing, Pydantic state schemas, and MCP integration via `langchain-mcp-adapters`. AutoGen v0.4 offers an event-driven actor model with distributed runtime support (cross-process, cross-machine) and natively supports Swarm, SelectorGroupChat, and GraphFlow patterns. CrewAI provides the fastest path to role-based multi-agent prototypes with its dual Crews (autonomous) + Flows (event-driven DAG) architecture. For your stack, **LangGraph is the strongest fit** because its `StateGraph` natively enforces typed Pydantic state contracts at every node boundary, its checkpoint system enables time-travel debugging, and its subgraph composition maps cleanly to MCP tool servers.

The benchmarks tell a sobering story about multi-agent versus single-agent performance. On SWE-bench Verified, the top systems (Claude Opus 4.5 at **80.9%**) use single-agent + rich tool environments—multi-agent approaches like ChatDev score as low as 25%. On GAIA, the best systems hit ~90% (near human baseline of 92%), but OpenHands-Versa showed that a well-designed single-agent with proper tools beats most multi-agent specialist systems by **9.1+ points**. The UC Berkeley MAST taxonomy identified **14 failure modes** across 7 MAS frameworks, with the most critical being rigid task decomposition, echo chambers (agents validating each other's hallucinations), and missing verification steps. DeepMind measured that multi-agent systems consume **15x more tokens** than equivalent single-agent approaches.

**Concrete recommendation for your swarm:** Start with a single orchestrator agent using LangGraph's `StateGraph` with Pydantic state schema. Add specialist agents only when you identify tasks where single-agent accuracy is genuinely below 45%. Cap at 4 agents total. Implement Magentic-One's dual-loop pattern: inner loop for step execution with progress checking, outer loop for replanning when stuck.

```
Orchestrator (reasoning model, Task+Progress Ledgers)
├── Researcher (web search, document analysis)
├── Coder (code generation, execution in sandbox)
└── Executor (computer use, file operations, shell)
    ↕ A-mem shared workspace (via MCP)
    ↕ Verifier agent or validation node (structured output checks)
```

---

## 2. A-mem as the swarm's associative memory backbone

A-mem (NeurIPS 2025, Rutgers University, arXiv:2502.12110) implements a **Zettelkasten-inspired memory graph** where each memory is a structured "note" containing `content`, `context` (LLM-generated situational description), `keywords`, `tags`, `timestamp`, `embedding` (via `all-MiniLM-L6-v2`), and `links` (bidirectional connections to related notes). It uses **ChromaDB** as its vector store backbone, with an associative graph layer built on top through explicit link fields. What makes A-mem distinctive is **memory evolution**: when new notes link to existing ones, the LLM may update contextual descriptions of those existing notes, allowing the knowledge network to continuously self-refine.

A-mem's creation pipeline works as follows: raw content is ingested → LLM generates context, keywords, and tags → embedding is computed → note is stored in ChromaDB → system retrieves top-k similar historical notes → LLM decides whether meaningful connections exist → bidirectional links are established → linked notes may evolve. This is fundamentally different from vector-store-only approaches (flat embedding retrieval, no organization) and from MemGPT/Letta's OS-inspired tiered hierarchy (core/recall/archival with self-editing). Compared to **Zep/Graphiti**, A-mem lacks temporal reasoning (Zep's temporal knowledge graph can invalidate superseded facts). Compared to **Mem0**, A-mem lacks the mature multi-tenant scoping and dynamic forgetting/decay mechanisms. However, A-mem's Zettelkasten associative linking and memory evolution make it uniquely suited for a swarm where agents collaboratively build knowledge—it's the only system where storing new knowledge actively improves the organization of existing knowledge.

**Three-tier namespace architecture for multi-agent A-mem:**

The shared workspace should implement three memory tiers, inspired by G-Memory (NeurIPS 2025 spotlight) and Microsoft's Multi-Agent Reference Architecture. **Tier 1: Per-agent private memory** stores agent-specific knowledge and learned procedures in namespaced ChromaDB collections (`agent_{id}_private`). **Tier 2: Per-task ephemeral context** creates task-scoped shared memory at task start, accessible to all agents on that task, with TTL-based cleanup after completion. **Tier 3: Global shared workspace** persists organizational knowledge, cross-task insights, and user profiles indefinitely, with write access controlled by agent role. Each tier maps to a separate ChromaDB collection, and the A-mem MCP server routes requests based on the `namespace` parameter.

For **concurrent memory access** (the hardest problem), implement event sourcing as the primary strategy. Every memory operation (create, update, delete, link, evolve) is an append-only event with `agent_id`, `timestamp`, `reasoning`, and `payload`. This provides a complete audit trail of who wrote what and why, supports replay with different strategies, and naturally handles A-mem's evolution steps as events. Layer optimistic concurrency control on top: each note carries a version number, updates specify expected version, and stale writes trigger a **semantic arbiter agent** that uses an LLM to determine whether to merge, prefer one write, or keep both as separate notes. For high-contention fields, use CRDT semantics: grow-only sets for tags and keywords, last-writer-wins registers for content, observed-remove sets for links.

**Critical implementation detail:** A-mem's LLM-dependent operations (note construction, linking, evolution) are the latency bottleneck. Agents should not block on memory evolution. Use a background worker pattern: the agent writes raw content to an event queue immediately (fast), and a background worker processes LLM enrichment, linking, and evolution asynchronously. For the production backend, ChromaDB is adequate for development but should be migrated to **pgvector/PostgreSQL** for ACID transactions and concurrent write safety, or to **Neo4j** if you want native graph traversal queries.

---

## 3. MCP as the universal integration protocol

MCP has achieved explosive adoption since Anthropic open-sourced it in November 2024: **97 million monthly SDK downloads**, 10,000+ active servers, and first-class client support in Claude, ChatGPT, Cursor, Gemini, and VS Code. In December 2025, it was donated to the Agentic AI Foundation under the Linux Foundation. The protocol exposes three capability types: **Tools** (executable functions with JSON Schema `inputSchema` and optional `outputSchema`), **Resources** (file-like data for loading into LLM context), and **Prompts** (reusable templates). The June 2025 spec update introduced `outputSchema` for structured content, and the November 2025 update added async operations, statelessness, and server identity.

**Transport selection is critical for your Docker topology.** Use **Streamable HTTP** (single `/mcp` endpoint, session IDs via `Mcp-Session-Id` header, OAuth 2.1 auth) between containers. Use **stdio** for local tool servers within the same container. SSE transport is deprecated since March 2025. Each MCP server container exposes a single `/mcp` endpoint, and agents connect through an **MCP proxy/gateway** that aggregates all server tools into a unified interface. MCProxy (Rust-based, by igrigorik) or Docker's MCP Gateway are the strongest options—they support dynamic tool updates via `toolListChanged` notifications, middleware for logging and security, and namespace prefixing to prevent tool shadowing.

**Wrapping A-mem as an MCP server** requires five core tools: `memory_add` (write note to namespace), `memory_search` (keyword + semantic query), `memory_read` (retrieve specific note by ID), `memory_update` (modify existing note), and `memory_delete` (remove note). Using FastMCP with Pydantic models:

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("a-mem")

class MemoryNote(BaseModel):
    content: str
    namespace: str  # "agent_{id}_private" | "task_{id}" | "shared"
    tags: list[str] = Field(default_factory=list)
    category: str = "semantic"  # episodic | semantic | procedural
    agent_id: str

@mcp.tool()
async def memory_add(note: MemoryNote) -> dict:
    """Write a structured memory note with automatic linking and evolution"""
    system = get_memory_system(note.namespace)
    return system.add_note(note.content, tags=note.tags, category=note.category)
```

**Dynamic tool registration** is natively supported in MCP. Agents can generate new tool specifications at runtime, register them via `notifications/tools/list_changed`, and other agents discover them through `tools/list`. The security implications are significant: tool poisoning (malicious descriptions manipulating LLM behavior), cross-server tool shadowing, and privilege escalation through generated tools. Mitigate with a validation pipeline: AST parsing → static analysis (Semgrep) → sandboxed test execution → human review for sensitive operations. Version-control generated tools with hash-based naming (SHA256 of tool code) in a git-backed registry.

Anthropic's own recommended pattern for scaling MCP tool usage is **code execution mode**: instead of loading all tool definitions upfront (which consumes **150K+ tokens** for thousands of tools), agents write code that interacts with MCP servers via filesystem-like tool discovery, reducing context usage from 150K to **2K tokens**—a 98.7% reduction.

---

## 4. Structured output as the composability primitive

The structured output discipline is what transforms a bag of agents into a verifiable pipeline. **Every agent boundary—input, output, inter-agent message, memory write, tool call—should be typed with Pydantic models.** This provides schema validation at runtime, automatic retry on validation failure, typed error propagation, and deterministic replay.

Four frameworks dominate structured output for agents. **Pydantic AI** (by the Pydantic team, 14.8K GitHub stars) provides generic typed agents (`Agent[Dependencies, OutputType]`) with automatic JSON Schema generation from Pydantic models, validation with `ModelRetry` on failure, and durable execution across transient API failures. **Instructor** (used at OpenAI/Google/Microsoft) patches the OpenAI SDK to add `response_model` with automatic Tenacity-based retry when Pydantic validation fails. **BAML** (BoundaryML) introduces a compile-time DSL that generates typed clients in Python/TypeScript/Ruby/Go with built-in json-repair—prompts become version-controlled, auditable text files with `@retry` directives. **Outlines** (by dottxt-ai) compiles JSON schemas into finite state machines that mask logits at the token level, **mathematically guaranteeing** valid structure on first generation—but only works with self-hosted models.

A critical finding from Instill AI benchmarking: a **multi-step pipeline** (reasoning step → structuring step) consistently achieves both correct reasoning AND valid structure where single-step approaches fail. For complex tasks, decouple the thinking from the formatting.

**Pipeline branching** uses discriminated unions in Pydantic. The router agent returns `Union[ContinueAction, BranchAction, TerminateAction]` with literal type discriminators, and Python's structural pattern matching dispatches to the appropriate handler. **Retry logic** feeds Pydantic validation errors back to the LLM as correction prompts—Instructor automates this with `max_retries`, and Pydantic AI raises `ModelRetry` exceptions that send errors back to the model. **Error propagation** should use Result types (inspired by Rust): each pipeline step returns `Union[PipelineSuccess[T], PipelineError]` with typed error information including `recoverable` flag and `retry_hint`.

For execution model selection: **LangGraph's graph-based state machine** is the strongest fit for your stack. It provides explicit state transitions validated by Pydantic, built-in checkpointing for durable execution, `Command` objects for state updates + routing, and subgraph composition for hierarchical agent roles. The key tradeoff: prompt chaining is most reliable but least flexible; agentic loops (ReAct) are most flexible but hardest to debug and cost-unpredictable; graph-based execution (LangGraph) offers the best balance of composability, reliability, and debuggability.

**Deterministic replay** is achievable. Docker's `cagent` tool implements the VCR (Video Cassette Recorder) pattern: `--record` captures full request/response cycles to YAML cassettes, `--fake` replays from cassettes with zero API calls and millisecond execution. Cassettes are version-controlled alongside test code. For property-based testing, use Hypothesis to generate random inputs and verify that pipeline invariants hold (schema validity, confidence ranges, required fields).

---

## 5. Computer use and dynamic tool extension in sandboxed containers

**Anthropic's Computer Use API** (`computer-use-2025-11-24` beta with Claude Opus 4.6) remains SOTA for general-purpose GUI interaction, providing `computer` (mouse/keyboard/screenshot), `text_editor`, and `bash` tools. Their official Docker reference image exposes VNC (5900), noVNC (6080), Streamlit UI (8501), and HTTP (8080). For web-specific tasks, **browser-use** (`uv add browser-use`, Python ≥3.11) offers a hybrid DOM + vision approach using Playwright that is **3-5x faster** than screenshot-based methods. For API-driven browser interaction without vision overhead, **Playwright MCP** (`@playwright/mcp@latest`) uses accessibility snapshots.

On benchmarks, AGI Inc.'s OSAgent achieved **76.26% on OSWorld** (superhuman, vs. human baseline of 72.36%) using online RL with self-verification. OpenAI's CUA scored 58% on WebArena and 38% on OSWorld. The key insight: scaffolding and harness design now rival model quality for benchmark improvements—modular multi-agent systems with specialized planning, UI reading, and action modules outperform monolithic models on long sequences.

**Sandboxing is non-negotiable for AI-generated code.** Standard Docker containers are insufficient—the shared kernel attack surface is too large (Dirty COW, Dirty Pipe, CVE-2022-0185). The minimum viable isolation for untrusted code is **gVisor** (`docker run --runtime=runsc`), which intercepts syscalls in user space and reduces host kernel exposure from ~350 to ~68 syscalls with 10-30% I/O overhead. For maximum security, **Firecracker microVMs** (used by AWS Lambda) provide dedicated kernels per workload with ~125ms boot and <5 MiB overhead, but add operational complexity. For self-hosted deployments, **microsandbox** (Apache-2.0, libkrun-based, MCP-native, sub-200ms boot) is the most practical choice. **E2B** (cloud, Firecracker-based, ~150ms startup) offers the smoothest developer experience but requires a cloud dependency.

Mandatory security controls per NVIDIA's AI Red Team guidance:

- Network egress controls (block arbitrary outbound; use `--network=none` or allowlisted egress)
- Read-only root filesystem (`--read-only` with `tmpfs /tmp:rw,noexec,nosuid,size=100m`)
- All capabilities dropped (`--cap-drop=ALL`)
- No privilege escalation (`--security-opt=no-new-privileges`)
- PID limits (`--pids-limit=50`) and memory/CPU limits
- Block writes to configuration files (prevents exploitation of hooks, MCP configs, skills)

**Dynamic tool extension** follows a clear pipeline: agent identifies need → generates Python tool code → code is validated (AST parsing, Semgrep static analysis, sandboxed test execution) → tool is wrapped as a FastMCP server → `notifications/tools/list_changed` notification propagates to swarm → other agents discover via `tools/list`. Version-control with SHA256 content-addressable storage in a git-backed registry. Security risks include upstream drift (tool changes breaking agents), schema injection (prompt injection via tool descriptions), and dependency confusion (~20% hallucinated imports in AI-generated code). The vendored tool definitions pattern (Vercel's `mcp-to-ai-sdk`) generates static schemas from MCP servers that are version-controlled in your repo while the runtime still calls the live server.

---

## 6. Container architecture with uv workspaces and hybrid topology

**uv workspaces** (inspired by Cargo) are the ideal monorepo structure. All workspace members share a **single lockfile** (`uv.lock`), ensuring dependency consistency across agents. The recommended project layout:

```
agent-swarm/
├── pyproject.toml          # Workspace root (virtual package)
├── uv.lock                 # Single lock for ALL agents + MCP servers
├── agents/
│   ├── orchestrator/       # Per-agent pyproject.toml
│   ├── researcher/
│   └── coder/
├── packages/
│   └── shared-lib/         # A-mem client, common Pydantic models, utils
└── mcp-servers/
    ├── memory/             # A-mem MCP wrapper
    ├── filesystem/
    └── shell/
```

The multi-stage Docker build pattern pins uv to a specific version (`ghcr.io/astral-sh/uv:0.8.21`), installs dependencies in a cached layer using `--locked --no-install-project`, then copies source and installs the project. Key environment variables: `UV_COMPILE_BYTECODE=1` (faster startup), `UV_LINK_MODE=copy` (Docker filesystem compatibility), `UV_PYTHON_DOWNLOADS=0` (use system Python). Per-agent builds use `uv sync --package researcher-agent` to install only that agent's dependency tree.

**The hybrid container topology is optimal for a home swarm.** Core agents run as separate containers (independent resource limits, restart policies, log separation). MCP servers run as sidecar containers behind Docker's MCP Gateway (unified control plane, sandboxed isolation per Docker's MCP Toolkit pattern). Shared services (graph DB, LLM inference) are single instances on the Docker bridge network.

For **persistent graph storage**, two strong options exist. **Neo4j Community** (`neo4j:2026.01-community`) provides native graph queries via Cypher, the APOC plugin library, and mature Docker support with well-defined volume mounts (`/data`, `/logs`, `/plugins`). It requires 1-2 GB RAM. **SurrealDB** is a compelling lightweight alternative: it natively supports graph relationships, document storage, vector embeddings, AND time-series in a single database with 256-512 MB footprint—potentially eliminating the need for separate ChromaDB and graph databases. For A-mem specifically, migrating from ChromaDB to SurrealDB would unify the vector store and graph layers while adding ACID transactions.

**Home hardware requirements:** A minimum viable setup (API-only agents, no local LLM) needs 4+ cores, 16 GB RAM, and 256 GB NVMe SSD. For local LLM inference, target 8+ cores, 32-64 GB DDR5, and an **NVIDIA RTX 3090** (24 GB VRAM, ~$700 used)—this runs 7B-13B models at full precision or 70B quantized (Q4) at 15-35 tokens/sec. GPU passthrough in Docker requires the NVIDIA Container Toolkit (`nvidia-container-toolkit`) and the `deploy.resources.reservations.devices` configuration in docker-compose. On Apple Silicon, Docker Desktop does not expose GPU to containers—use Docker Model Runner (native host process with Metal acceleration) instead.

The complete resource budget for a 32 GB home server: each agent at 0.5-1.0 cores and 512M-1G, MCP Gateway at 0.25 cores and 256M, Neo4j at 1.0 core and 2G, Ollama with GPU at 4.0 cores and 12G, Traefik reverse proxy at 0.25 cores and 128M—totaling ~8 cores and 18-20 GB.

---

## 7. Reliability, observability, and evaluation as first-class concerns

Multi-agent reliability remains the critical unsolved problem. The CLEAR framework (November 2025) found that agent performance drops from **60% (single-run) to 25% (8-run consistency)**—reliability, not accuracy, is the deployment bottleneck. METR's March 2025 measurements show AI task-completion horizon doubling every ~7 months, with Claude 3.7 Sonnet at ~1 hour horizon.

**Deterministic execution** requires three controls: set `temperature=0` and `seed=<fixed_int>` on all LLM calls, wrap all calls in a deterministic proxy that logs the exact request/response, and implement LangGraph checkpointing with `PostgresSaver` for state snapshots at every super-step. This enables time-travel debugging (`graph.get_state_history(config)`), state modification before replay, and automatic resume after failures. For full deterministic replay, use Docker's `cagent` with `--record` to capture VCR cassettes.

**The observability stack should be OpenTelemetry-native.** Langfuse v3 (self-hosted via Docker Compose, MIT license, ClickHouse backend) provides the primary tracing and logging layer with native OTel support. Arize Phoenix (self-hosted, strongest eval library) provides the evaluation layer. OpenLLMetry (`traceloop-sdk`) auto-instruments OpenAI/Anthropic calls. An OpenTelemetry Collector container routes spans and metrics to both Langfuse and Phoenix, plus Prometheus for system metrics. Capture five categories of structured attributes: agent reasoning chains (`agent.thought`, `agent.plan`, `agent.decision`), tool calls with full inputs/outputs, memory operations (reads, writes, conflicts), inter-agent messages, and LLM call metadata (model, tokens, finish reason).

**Failure detection requires four layers.** Loop detection: maintain a deque of recent `(tool_name, input_hash)` tuples per agent; if the same action appears N times in M steps, trigger a loop breaker. Hallucinated tool call detection: validate every tool call against registered JSON Schema before execution, reject calls to non-existent tools. Memory corruption detection: optimistic concurrency control with version vectors, schema validation on all writes. Cascading failure prevention: circuit breakers between agent clusters (not individual connections) with adaptive thresholds based on interaction success rates; exponential backoff with jitter (base=1s, max=60s); bulkhead pattern to isolate agent resource pools.

For **guardrails**, NeMo Guardrails v0.20+ provides five rail types (input/output/dialog/retrieval/execution) with Colang DSL, reasoning-capable content safety models (Nemotron), and three NIM microservices for content safety, topic control, and jailbreak detection. Combine with LangGraph's native human-in-the-loop via `interrupt_before` for dangerous operations, and a `BudgetEnforcer` class that tracks cumulative cost, tokens, and API calls with hard limits.

**Evaluation should use the CLEAR framework dimensions:** Cost (normalized accuracy per dollar), Latency (end-to-end and planning-phase), Efficacy (task completion rate and intermediate step accuracy), Assurance (guardrail violation rate, prompt injection resistance), and Reliability (pass@k with k=8). For benchmarking your specific swarm, REALM-Bench tests planning/scheduling across LangGraph, AutoGen, CrewAI, and Swarm configurations, while MultiAgentBench/MARBLE (ACL 2025) evaluates collaboration across star/chain/tree/graph topologies.

---

## Synthesis: the recommended architecture stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Orchestration | LangGraph `StateGraph` + Pydantic state | Typed state contracts, checkpointing, subgraph composition |
| Memory | A-mem + event sourcing layer + pgvector or SurrealDB | Associative linking + evolution; event sourcing for concurrency |
| Memory topology | 3-tier namespace (private/task/shared) | Isolation where needed, collaboration where needed |
| Tool protocol | MCP (Streamable HTTP between containers, stdio local) | Universal, 10K+ servers, structured I/O, dynamic registration |
| Structured output | Pydantic AI for agents, FastMCP for tool servers | Runtime validation, retry-on-failure, typed pipelines |
| Computer use | Anthropic Computer Use (GUI) + browser-use (web) | Best general + best web-specific |
| Sandboxing | gVisor minimum; microsandbox for self-hosted microVMs | Defense-in-depth for AI-generated code |
| Container architecture | Hybrid: agents as containers, MCP as sidecars, Docker MCP Gateway | Isolation + unified control plane |
| Dependencies | uv workspaces, single lockfile, multi-stage Docker builds | Reproducible builds, per-agent dependency trees |
| Graph persistence | Neo4j Community or SurrealDB, Docker named volumes | Native graph queries, Docker-native persistence |
| Observability | OTel Collector → Langfuse v3 + Arize Phoenix + Prometheus/Grafana | Full-stack: traces, evals, system metrics |
| Guardrails | NeMo Guardrails v0.20+ + LangGraph HITL + budget enforcement | Input/output/execution rails + human approval |
| Evaluation | CLEAR framework (pass@k=8) + REALM-Bench | Cost-aware, reliability-focused |

**Known limitations and open problems:** A-mem has no built-in multi-agent support—the namespace and concurrency layers must be custom-built. ChromaDB is a single-instance bottleneck under concurrent writes (migrate to pgvector or SurrealDB). The MCP specification is still rapidly evolving, with the November 2025 update adding significant new features. Deterministic replay is imperfect because LLM providers may change model weights silently (log `system_fingerprint` to detect this). The 45% threshold from DeepMind suggests that as foundation models improve, the window where multi-agent outperforms single-agent narrows—your architecture should gracefully degrade to single-agent mode. Finally, **24% of top-50 SWE-bench leaderboard positions are incorrect** (UC Berkeley, December 2025), so treat all benchmark claims with appropriate skepticism.