# Memory Tags Conventions

Pipelines and agents use tags to categorize notes. Hybrid memory (VectorHaSH Path A) combines FTS5 + vector similarity with task-specific tags for associative retrieval. Per-step `memory_tags` filter retrieval so agents receive context relevant to their task type.

## Tag Conventions

| Tag | Source | Use |
|-----|--------|-----|
| `research` | Director, research pipelines | Research findings and reports |
| `synthesis` | Director | Synthesized reports combining expert findings |
| `reflection` | BaseAgent auto-storage | Agent reflections on completion |
| `refined`, `merged` | refine_memory tool | Notes created by merging/consolidating |
| `update_tags` | refine_memory tool | Notes updated via update_tags action |
| `{role}` | BaseAgent | Agent role (e.g. `micro`, `small`, `lfm2_rag`) |
| `{item.id}` | Director | Research queue item ID for traceability |

## Pipeline Step: memory_tags

Filter memory retrieval so only notes with **any** of the given tags are returned:

```toml
[[steps]]
name = "synthesize"
role = "lfm2_rag"
memory_context = true
memory_tags = ["research", "synthesis"]
```

Steps without `memory_tags` receive all matching notes (no tag filter).

## Task-Specific Retrieval

- **Research synthesis steps**: `memory_tags = ["research", "synthesis"]` — only prior research reports
- **Reflection steps**: `memory_tags = ["reflection"]` — only agent reflections
- **General context**: omit `memory_tags` — full memory search

## Tool context requirements

**add_citation**: Pipelines that use `add_citation` must be run via `aof research run --pipeline X.toml`, not `aof pipeline X.toml`. The tool requires `current_research_item_id`, which is only set during research queue processing.

## refine_memory and create_note

Both support tags:

- **create_note**: `tags` is required; use task-specific tags like `["research", "synthesis"]`
- **refine_memory** (merge): `tags` defaults to `["refined", "merged"]`; pass explicit tags for task context
- **refine_memory** (update_tags): updates note tags for categorization

## RAG Synthesis Pattern

For memory-heavy synthesis steps, use `memory_context=true` with `role=lfm2_rag` and optional `memory_tags`:

```toml
[[steps]]
name = "synthesize"
role = "lfm2_rag"
memory_context = true
memory_search_limit = 8
memory_tags = ["research", "synthesis"]
goal_template = "Synthesize using this context:\n{memory_context}\n\nContent:\n{prev_result}"
```

1. **Gather step**: Uses tools (web_search, search_memory) to collect content.
2. **Synthesize step**: Pre-fetches memory, injects as `{memory_context}`, and runs an LFM2-RAG model to produce synthesized output.

See `examples/memory_rag.toml` for a full pipeline.
