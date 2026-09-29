# Agent Operator Framework (AOF)

Async Python agent framework for small local LLMs. Run concurrent agents with multi-model routing, tool calling, Zettelkasten memory, and research pipelines — all on your own hardware.

## Supported Models

- **Qwen3** family (0.6B, 4B, 8B, 14B, VL-8B) — via GGUF
- **LFM2.5** family (1.2B Instruct, 1.2B Thinking) — LiquidAI hybrid arch

## Quick Start

```bash
# Install
uv sync
# Optional: agentic memory (A-MEM) for LLM-enriched notes and behaviour recording
# uv sync --extra agentic

# Run a single agent
uv run aof agent "What is the capital of France?" --role micro

# Run a multi-step pipeline
uv run aof pipeline examples/research.toml "AI trends 2025"

# Full-featured pipeline (orchestrator, multi-role, memory, when/on_error)
uv run aof pipeline examples/complete_research.toml "AI trends 2025"

# Test a model
uv run aof canary --model-path path/to/model.gguf

# Research mode
uv run aof research add "Compare sparse vs dense attention"
uv run aof research run --mode extended
# Or with a specific pipeline: research_sota.toml (citations) or complete_research.toml
uv run aof research run --pipeline examples/complete_research.toml --limit 1

# Continuous researcher (workspace + interactive REPL, uses local GGUF by default)
uv run aof researcher run --workspace myproject
# In the REPL: add <question>, list, status, pause, resume, note <text>, docs, quit

# New workspace seeded from one question (expands into dozens of follow-ups)
uv run aof researcher run --workspace myproject --seed-question "Why is there something rather than nothing?"
```

## Continuous Researcher

Run a long-lived researcher with its own workspace and guide it via the REPL. **Defaults to local GGUF models** (Qwen3, LFM2.5) via `--backend legacy`; use `--backend deep` if you have LM Studio or Ollama running.

```bash
uv run aof researcher run --workspace myproject
```

- **Workspace**: `data/workspaces/myproject/` — queue, memory, `artifacts/` (synthesis reports), `activity.log`
- **REPL commands**: `add <question>`, `list`, `status`, `pause`, `resume`, `interrupt`, `note <text>`, `promote <id>`, `restart [--all] [id] [note]`, `docs`, `quit`
- **Human notes**: `note Focus on X` appends guidance to the current or next item and injects it into prompts
- **Queue reload**: The background runner reloads the queue from disk each loop; `add` or `restart` in the REPL is picked up on the next iteration.
- **Cohesive reports**: Each artifact is one markdown file (research question as title, report body, `## Sources`). A synthesis note with the same content is stored in workspace memory per item.
- **Route generator** (optional): `--route-gen` or `[research] route_gen_enabled = true` refills the backlog when low using web search and the current done/backlog context.
- **Seed expansion**: `--seed-question "..."` queues the seed and asks `research.seed_expansion_role` (default `medium`) for up to `research.seed_expansion_count` follow-up questions.
- **Branch connector / lateral thinking** (optional, off by default): `research.branch_connector_enabled` periodically writes a "Connections" note linking themes; `research.lateral_thinking_enabled` adds tangential questions when the backlog runs low.
- **Feature verification**: See `docs/researcher_verification.md` for how to confirm warm backends, activity log, config, and tests.

## Backends

| Flag | Backend | Description |
|------|---------|-------------|
| `--backend deep` | LangGraph + Deep Agents | Single model, flattened TOML (default) |
| `--backend llama` | Legacy Pipeline | Multi-role, per-step routing, full composability |
| `--backend server` | OpenAI-compatible API | LM Studio, Ollama, vLLM |

## Building a curated knowledge vault

`aof refine` builds a vault from a question set: it gathers peer-reviewed evidence, extracts claims as verbatim
quotes, merges duplicates losslessly, curates, verifies, links, builds hub notes, and exports Obsidian-ready markdown.
Small models do narrow jobs; the framework owns the loop. See [docs/VAULT_OVERHAUL.md](docs/VAULT_OVERHAUL.md).

```bash
uv run aof refine --log-level WARNING run --workspace neuro \
    --queue-file examples/queues/neuro_hippocampal_thalamic.toml --curate --verify-top 4
uv run aof refine structure --workspace neuro   # links + hub notes
uv run aof refine metrics   --workspace neuro   # duplicates, orphans, sourcing, verification
uv run aof refine assess    --workspace neuro --queue-file examples/queues/neuro_hippocampal_thalamic.toml
uv run aof refine export    --workspace neuro   # data/workspaces/neuro/export/
```

## Specialist models

Small task-specific models (Needle, Liquid Nanos, any configured role) are exposed as *capabilities* with cheap-first
fallback chains (`[specialists.<capability>] providers = [...]`). Needle is optional: `uv sync --extra needle`
(sets `NEEDLE_TELEMETRY=0`). See [docs/VAULT_OVERHAUL.md](docs/VAULT_OVERHAUL.md).

## Running through llama.cpp

Models `llama-cpp-python` can't load (e.g. the `qwen35` architecture) can be served through a managed
`llama-server`. See [docs/llama-cpp.md](docs/llama-cpp.md) for the llama.cpp build to install and the config.

## Configuration

Edit `config.toml` to set model paths, pool sizes, memory settings, and role assignments. `~` is expanded in `model.path` and `models.directory` (default: `~/.lmstudio/models/lmstudio-community`). See `examples/` for pipeline configs.

## Development

```bash
uv sync --extra dev
uv run pytest tests/
```

## License

MIT
