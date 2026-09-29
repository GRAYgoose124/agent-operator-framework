# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Operator Framework (AOF) — async Python agent framework for small local LLMs (Qwen3 + LFM2.5 families). Dual inference backends: `llama-cpp-python` for direct GGUF loading, or OpenAI-compatible local servers (LM Studio/Ollama). Includes Zettelkasten memory (SQLite FTS5 + git-versioned markdown), tool registry with sandboxed execution, and multi-model role routing.

## Build & Run Commands

```bash
# Dependencies (always use uv, never pip)
uv sync
uv sync --extra dev          # include pytest

# Run CLI
uv run aof agent "prompt" --role micro --backend deep
uv run aof canary --model-path path/to/model.gguf
uv run aof pipeline examples/research.toml "query"
uv run aof research run --mode extended --limit 10
uv run aof researcher run --workspace myproject  # Continuous researcher with REPL
uv run aof researcher run --workspace test-project1 --seed-question "what is the reason for conscious existence?"  # New workspace + seed expansion (dozens of questions)
uv run aof researcher run --parallel 3           # Process 3 items concurrently
uv run aof researcher clone --from proj1 --to proj2  # Clone workspace
uv run aof researcher start --workspace myproject # Headless daemon mode
uv run aof researcher attach                      # Attach to running daemon

# Tests (if `uv run pytest` fails with "Failed to canonicalize script path" on Windows, use `uv run --extra dev python -m pytest`)
uv run pytest tests/                        # all tests
uv run pytest tests/test_parsing.py         # single file
uv run pytest tests/test_agent.py::test_xyz # single test
uv run pytest -v --tb=short tests/          # verbose
```

pytest-asyncio is configured with `asyncio_mode = "auto"` — async test functions just work.

## Package Layout

- Build backend: **hatchling** (not setuptools)
- Source layout: `src/aof/` with `[tool.hatch.build.targets.wheel] packages = ["src/aof"]`
- CLI entry point: `aof = "aof.cli:main"`
- Config: `config.toml` at project root, loaded via `src/aof/config.py` into frozen dataclasses (`~` expanded in `model.path` / `models.directory`); an untracked `config.local.toml` beside it is deep-merged over it for machine-specific paths

## Architecture

### Agent Lifecycle
`AgentState`: IDLE → PLANNING → EXECUTING → REFLECTING → STORING → DONE (or ERROR). Each agent carries an `AgentContext` (mutable state: goal, plan, history, results) and `AgentMetrics` (steps, tool accuracy, parse failures, health score = 0.4×completion + 0.3×tool_accuracy + 0.3×parse_score).

### Concurrency
- `AgentPool` runs up to `max_agents` (default 8) concurrent agents as `asyncio.Task`s
- `LlamaBackend` maintains a pool of 2-4 Llama instances behind `asyncio.Queue`, dispatched to `ThreadPoolExecutor` for sync llama-cpp calls
- `LocalServerBackend` delegates to LM Studio/Ollama via `ChatOpenAI` (no model pool needed)

### Multi-Model Roles
Agents have a `role` property; the pool maps role → backend/model:
- `micro` (Qwen3-0.6B), `small` (Qwen3-4B), `medium` (Qwen3-8B), `vision` (Qwen3-VL-8B)
- `fast` (LFM2.5-1.2B-Instruct), `reasoning` (LFM2.5-1.2B-Thinking), `orchestrator`, `thinker`; `fallback` (Qwen3-14B)
- Liquid Nanos: `lfm2_tool`, `lfm2_rag`, `lfm2_extract`, `lfm2_extract_350m`, `lfm2_math`, `lfm2_transcript`; `lfm2_vl` (LFM2.5-VL), `lfm2_jp` (Japanese)

### Inference & Parsing
Model family auto-detected by `detect_model_family()` in `inference/parsing.py`:
- **Qwen3**: `<tool_call>{"name":..., "arguments":...}</tool_call>`, thinking in `<think>...</think>`
- **LFM2.5**: `<|tool_call_start|>...<|tool_call_end|>` (Pythonic `[func(arg=val)]` or JSON), thinking in `<think>...</think>`
- Bare JSON fallback with regex extraction for malformed output
- Context budget: system_prompt + memory + history + task + generation must fit `n_ctx`. Uses 3 chars ≈ 1 token heuristic (conservative). Task reserve scales with n_ctx. Thinking models get +512 max_tokens for `<think>` blocks.
- Overflow protection: `LlamaBackend` pre-truncates the longest user message if estimated tokens exceed n_ctx, and catches `llama_decode` errors gracefully instead of crashing.
- No-tool pipeline steps use `skip_plan=True` (direct execution, no planning overhead). Goal context is only injected when the step differs from the goal, and is truncated to fit the task budget.

### Tool System
`ToolRegistry` holds `ToolDefinition`s (name, description, JSON schema, async handler). Sources: builtin (`tools/builtin/`), agent-created scripts (`data/tools/` with JSON stdin/stdout protocol). Sandbox blocks imports: os, subprocess, socket, shutil, ctypes. **Tool-call format**: The system prompt includes tool definitions plus an explicit "To call a tool, respond with ..." line (Qwen3: `<tool_call>{"name":..., "arguments":...}</tool_call>`; LFM2: `<|tool_call_start|>...<|tool_call_end|>`). The context builder keeps the full tools block (no truncation) so the model always sees the required format.

**add_citation**: Requires `current_research_item_id` (set only during `research run`). Pipelines with `add_citation` must be run via `aof research run --pipeline X.toml`, not `aof pipeline X.toml`. Outside research context, `add_citation` returns an error.

### Memory
SQLite FTS5 + markdown files in `data/memory/` (git-versioned). Strategies: search, recent, agent_notes, hybrid. `ZettelNote`: id, title, tags, links, content, created_at, agent_id. **Optional A-MEM (agentic memory)**: when `memory.agentic_enabled = true` and the `agentic` extra is installed (`uv sync --extra agentic`), notes are mirrored into A-mem-sys for LLM-generated keywords/context/tags and evolution; `search_memory` uses A-MEM search with FTS fallback. Behaviour summary notes (per completed research item) are stored with tags `behaviour`, `research` so agents can search past behaviour.

### Pipeline
`Pipeline.from_toml()` loads multi-step workflows. Steps specify system_prompt, goal_template (placeholders: `{input}`, `{prev_result}`, `{memory_context}`, `{workspace_context}`, `{signals}`), tools, role, and max_steps. `Pipeline.run(..., workspace_context="")` injects workspace context (e.g. recent artifacts and memory titles from the researcher workspace) when steps reference `{workspace_context}`. Use `prev_result_max_chars` per step to allow more of the previous step output (default 1500; e.g. 2500 for analyze steps). Each step can use a different model role. VL steps: `image_input = "{input}"` passes image paths to vision models. Parallel steps: `parallel_count > 1` runs N agents; `choose` = `first` | `best` | `concat` combines results. Per-step memory: `memory_strategy`, `memory_search_limit` override; `memory_context = true` pre-fetches memory and injects as `{memory_context}`. Error handling: `on_error` = `propagate` | `retry` | `abort`. Conditional steps: `when = "non_empty"` (default) skips if prev_result is empty or a failure message; `when = "always"` runs unconditionally. **Step repetition**: `max_repeats` + `repeat_until` re-runs a step if its output doesn't meet a quality bar (conditions: `non_empty`, `min_length:N`). **Pipeline signals**: steps can emit `[SIGNAL:key=value]` in output; collected into `{signals}` placeholder for downstream steps. **Step names**: the `name` of each step is used in progress files (`artifacts/<item_id>_progress.md`) and in artifact section headers (e.g. "### Step 1: search"). Use short, action-oriented names (e.g. "search", "analyze", "report") for clear artifacts and memory titles.

### Research Mode
- **Single**: one agent per question via pipeline TOML. Use `research_sota.toml` for citations; `mixed_model_research.toml` for general mixed-model research. Both include a `spawn` step that generates follow-up questions.
- **Extended**: 1× director (8B) decomposes into tasks → N× experts (0.6B) execute concurrently → director synthesizes. Director uses hardcoded prompts; does not load pipeline TOML.
- **Parallel items**: `research.parallel_items` (config; default 1) or `--parallel N` (CLI override) processes up to N queue items concurrently via `asyncio.gather`. Set > 1 for batched throughput; agents in the same run can see other items' activity via `{workspace_context}` when enabled.
- **Refinement**: `research.refinement_pipeline` + `research.refinement_passes` run a second pipeline (e.g. `examples/refine.toml`) on each completed item to evaluate and improve output.
- Kanban-style queue persisted to `data/research_queue.json`

### Continuous Researcher (`aof researcher run`)
- **Workspace**: Dedicated `data/workspaces/<name>/` with queue.json, memory.db, memory/, artifacts/, activity.log, behaviour.jsonl (structured step/item events)
- **REPL**: Interactive commands — `add`, `list`, `status`, `pause`, `resume`, `interrupt`, `note <text>`, `docs`, `quit`
- **Artifacts**: Per-item synthesis written to `artifacts/<item_id>.md`; activity stream in `activity.log`
- **Human notes**: `note <text>` appends to current/next item; injected into prompts as "Human guidance"
- **Clone**: `aof researcher clone --from X --to Y` copies workspace queue + memory with all items reset to backlog
- **Daemon mode**: `aof researcher start --daemon` runs headless; `aof researcher attach` connects via file-based IPC; `aof researcher stop` sends quit
- **Seed question**: `--seed-question "..."` adds the seed to the queue and runs a **seed expansion** phase: one LLM call with `research.seed_expansion_role` (default `medium` = Qwen3-8B) to generate many follow-up questions (target `research.seed_expansion_count`, default 30; cap 50). A "Seed expansion" note is stored in workspace memory. When `--seed-question` is used, consolidation is auto-enabled for that run if not already on. Use for "start a new workspace and spawn dozens of questions" workflows.
- **Branch connector**: When `research.branch_connector_enabled = true`, a background task every `research.branch_connector_interval` s writes a "Connections" note (tag `connections`, `research`) linking themes across recent artifacts and memory.
- **Lateral thinking**: When `research.lateral_thinking_enabled = true`, a background task every `research.lateral_interval` s adds 3–5 lateral/tangential questions to the queue when backlog is below `research.lateral_min_backlog` (default 5).

### Memory Consolidation
- **Background task**: When `memory.consolidation_enabled = true`, clusters related notes by tag and synthesizes summaries.
- `MemoryConsolidator` in `memory/consolidator.py`: `run_once()` clusters → synthesizes → archives sources. `run_loop()` runs on interval.
- **Topic map**: Auto-generated `topic_map` note listing clusters and key findings; included in workspace context.
- **Knowledge summary**: Running `knowledge_summary` note updated after each research item; provides situational awareness to agents.

### LangChain Integration (langchain_refactor branch)
- `--backend deep`: LangGraph + Deep Agents orchestration. Single model; pipeline TOML is flattened into one combined prompt. No per-step role routing. Use for `/memories/` StoreBackend.
- `--backend llama`: Legacy multi-role pipeline. Per-step role routing, memory_context, when, on_error. Use for full composability.
- `--backend server`: OpenAI-compatible local server
- Adapters in `src/aof/langchain_adapters/` convert ToolRegistry → LangChain tools, MemoryStore → LangChain store

## Key Modules

| Path | Purpose |
|------|---------|
| `agent/base.py` | AgentState, AgentContext, AgentMetrics, BaseAgent protocol |
| `agent/pool.py` | Concurrent agent pool with multi-model routing |
| `refine/pipeline.py` | Evidence-first refinement of one question (plan, gather, extract, cluster, decontextualise, store, curate, verify) |
| `refine/sources.py` | Evidence sources: PubMed, OpenAlex, Wikipedia, web; `doc_key` merges copies of one work |
| `refine/{link,hubs,structure,metrics,export,assess}.py` | Links, hub notes, vault metrics, Obsidian export, rubric assessment |
| `specialists/registry.py` | Capability -> provider-chain cascade (Needle 3/2, role models, embeddings); see `docs/VAULT_OVERHAUL.md` |
| `inference/llama_server.py` | Managed llama.cpp `llama-server` backend (roles in `[llama_server] roles`); see `docs/llama-cpp.md` |
| `agent/lifecycle.py` | Single agent state machine |
| `agent/evaluator.py` | Health scoring, canary tests |
| `inference/llama_backend.py` | llama-cpp-python pool + thread dispatch |
| `inference/openai_backend.py` | LocalServerBackend for LM Studio/Ollama |
| `inference/parsing.py` | Qwen3/LFM2.5 output parsing, model family detection |
| `inference/context.py` | Token budget management |
| `tools/registry.py` | Central tool registration |
| `tools/sandbox.py` | Subprocess sandbox for agent scripts |
| `memory/store.py` | SQLite FTS5 + Zettelkasten markdown |
| `pipeline/compose.py` | TOML-driven sequential pipeline |
| `research/director.py` | Extended research orchestration |
| `researcher/workspace.py` | Workspace model, directory layout |
| `researcher/runner.py` | Research loop with artifacts, activity log |
| `researcher/repl.py` | Interactive REPL (add, pause, note, etc.) |
| `researcher/daemon.py` | Headless daemon mode (file-based IPC) |
| `memory/consolidator.py` | Memory consolidation + topic map |
| `orchestrator/deep_agent_factory.py` | LangGraph agent factory |

## Config

`config.toml` sections: `[model]`, `[pool]`, `[memory]`, `[tools]`, `[pipeline]`, `[evaluator]`, `[local_server]`, `[models]`, `[roles]`, `[research]`, `[role_context]`. Loaded by `load_config()` → `AppConfig` (nested frozen dataclasses). TOML values override defaults via deep merge; unknown keys are filtered during instantiation.

Key new config fields:
- `[role_context]`: Per-role `n_ctx` overrides (e.g. `micro = 4096`, `medium = 16384`). 0 = use global default.
- `[research]`: `parallel_items`, `refinement_pipeline` + `refinement_passes`; `seed_expansion_role` (medium|fallback), `seed_expansion_count` (for --seed-question); `branch_connector_enabled` + `branch_connector_interval`; `lateral_thinking_enabled`, `lateral_interval`, `lateral_min_backlog`.
- `[memory]`: `consolidation_enabled`, `consolidation_interval`, `consolidation_min_notes` (background memory clustering).

## Testing Conventions

- Fixtures in `tests/conftest.py`: `temp_config`, `temp_memory`, `temp_tools`, `mock_chat_model`, `run_cli(args)`
- E2E tests in `tests/e2e/` test CLI subprocess invocation and full pipelines
- Windows asyncio pipe warnings in test output are harmless

## Project Conventions

- Refer to and update the TODO.md for goals and with progress.
- Keep any relevant documents updated in docs/
- Use `uv` to manage the Python project
- Write cohesive integration tests which test the full behaviour of the complex system.
- Do not mock or stub out functionality