# SOTA Research Behaviour — Next Stage Plan

**Version:** 1.0  
**Date:** 2026-02-11  
**Status:** Planning

---

## Executive Summary

This plan outlines the next stage of SOTA (State of the Art) behaviour for the Agent Operator Framework (AOF). The goal is an **autonomous research agent** that:

1. **Ingests a queue of research questions** and works through them systematically
2. **Self-improves** by gathering sources, citations, and updating scrapers on the fly
3. **Accepts user interrupts** at any time—putting current work back on the stack (Kanban-style)
4. **Uses associative, context-aware memory** inspired by VectorHaSH for retrieval and storage
5. **Leverages web + SOTA plans** for research and orchestration

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Research Orchestrator                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Kanban     │  │   SOTA       │  │   User       │  │  Research        │ │
│  │   Board      │  │   Planner    │  │   Input      │  │  Loop            │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │                 │                 │                    │          │
│         └─────────────────┴─────────────────┴────────────────────┘          │
│                                      │                                       │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VectorHaSH-Inspired Memory Layer                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Associative Lookup │ Semantic Search │ Episodic Context │ Links   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Backed by: Zettel (md) + SQLite FTS + [Vector Store / Embeddings]          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Research Agents (pool)  │  Tools: web_search, web_scrape, web_news         │
│  Dynamic tool creation   │  Citations, source tracking                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Research Queue & Kanban Board

| Column | Description | Transitions |
|--------|-------------|-------------|
| **Backlog** | Queued research questions, user-addable at any time | → In Progress |
| **In Progress** | Currently being researched | → Done, Blocked, Backlog (on interrupt) |
| **Blocked** | Paused (e.g. missing info, needs user input) | → Backlog, In Progress |
| **Done** | Completed with findings stored | — |

**Behaviour:**
- User can add items to **Backlog** at any time
- User can **interrupt** and push current item back to Backlog (or Blocked)
- Agent pulls next item from Backlog when ready
- Queue is persisted (TOML/JSON or SQLite) so it survives restarts

**Implementation sketch:**
```python
# src/aof/research/queue.py
@dataclass
class ResearchItem:
    id: str
    question: str
    status: Literal["backlog", "in_progress", "blocked", "done"]
    sources: list[str]
    citations: list[str]
    created_at: datetime
    updated_at: datetime

class ResearchQueue:
    async def add(question: str) -> str
    async def pop_next() -> ResearchItem | None
    async def interrupt(item_id: str) -> None  # push to backlog
    async def promote(item_id: str) -> None     # blocked → backlog
```

---

### 2.2 Associative Memory (VectorHaSH-Inspired)

**Important:** [VectorHaSH](https://github.com/FieteLab/VectorHaSH) is a *neuroscience research implementation* (hippocampal associative memory from Nature 2025). It is not a drop-in software library—it runs neural simulations. For AOF we have two paths:

| Path | Approach | Effort | Use Case |
|------|----------|--------|----------|
| **A: Practical** | Use embedding-based vector stores (ChromaDB, FAISS, sentence-transformers) for semantic similarity | Low–Medium | Associative lookup via embeddings |
| **B: Conceptual** | Extract/adapt VectorHaSH ideas (sparse codes, pattern completion) into a custom retrieval layer | High | Research project, closer to paper |

**Recommended:** Start with **Path A** for immediate associative lookup, then explore Path B if you want to align with the VectorHaSH paper.

**Path A (implemented):**
- `aof.memory.vector_store` with ChromaDB + sentence-transformers
- Zettel content embedded with `all-MiniLM-L6-v2` (configurable via `embedding_model`)
- Hybrid search: FTS5 + vector similarity in `MemoryStore.search()` when `memory_strategy=hybrid`
- `memory_tags` for task-specific associative retrieval; RAG synthesis pattern in `memory_rag.toml`

**Optional A-MEM (agentic memory):** When `memory.agentic_enabled = true` and the `agentic` extra is installed, the framework uses [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) as an agentic memory layer: new notes are mirrored into A-MEM for LLM-generated keywords, context, and tags and for memory evolution (linking). Search can use A-MEM first with FTS fallback. Behaviour recording (step_done and per-item behaviour summary notes) feeds into memory so agents can search for past research behaviour.

**Path B (conceptual):**
- VectorHaSH uses grid codes + sparse patterns for associative recall
- Could design a "sparse code" layer: hash content → sparse vectors → nearest-neighbor over codebook
- Would require porting/adapting `assoc_utils.py` from the repo

---

### 2.3 Self-Improving Research Loop

The agent should:

1. **Plan** (SOTA Planner): decompose research question into sub-goals and steps
2. **Gather**: use `web_search`, `web_news`, `web_scrape` to collect sources
3. **Track citations**: store URL, title, snippet for each source
4. **Update scrapers on the fly**: if a site blocks or changes structure, agent can propose/apply scraper updates
5. **Store findings**: write to memory with tags, links, citations
6. **Reflect**: update its approach based on what it finds

**New capabilities:**
- **Citation tool**: `add_citation(url, title, snippet)` — stored with research item
- **Dynamic scraper**: agent can call `create_crawler` or `propose_crawler_update` to add or refine site-specific crawling logic (with changes proposed via markdown for later review)
- **Research pipeline**: `research → synthesize → store` with optional `improve_scraper` step
- **Behaviour recording**: per-workspace `behaviour.jsonl` (step_start, step_done, item_done) and a behaviour summary memory note per completed item (tags: behaviour, research) so search_memory can retrieve how past research was conducted

---

### 2.4 User Interrupt Handling

- **CLI**: `aof research add "What is X?"` adds to queue
- **CLI**: `aof research interrupt` sends signal to current agent
- **Runtime**: Agent checks `ctx.metadata.get("interrupt_requested")` between steps
- **Effect**: Current work is checkpointed (partial findings saved), item moved to Backlog
- **Resume**: When item is pulled again, load checkpoint and continue

**Checkpoint format:**
```json
{
  "item_id": "...",
  "question": "...",
  "sources_gathered": [...],
  "partial_findings": "...",
  "step_index": 2
}
```

---

## 3. Integration with Existing Systems

| Component | Current | Planned |
|-----------|---------|---------|
| **Planner** | `Planner.decompose()` | Use for research sub-goals |
| **Pipeline** | `Pipeline.run()` sequential | Add `ResearchPipeline` with queue + interrupt |
| **Memory** | `MemoryStore` (SQLite FTS + md) | Add vector layer for associative search |
| **Tools** | `web_search`, `web_scrape`, `web_news` | Add `add_citation`, `create_tool` (dynamic) |
| **Pool** | `AgentPool` for concurrent agents | Research orchestrator uses pool |
| **CLI** | `agent`, `batch`, `pipeline` | Add `research` subcommand |

---

## 4. Implementation Phases

### Phase 1: Research Queue + Kanban (1–2 weeks)
- [ ] `ResearchQueue` class with backlog/in_progress/blocked/done
- [ ] Persistence (e.g. `data/research_queue.json`)
- [ ] `aof research add <question>`
- [ ] `aof research list` — show Kanban view
- [ ] `aof research run` — process queue with single agent

### Phase 2: Interrupt + Checkpointing (1 week)
- [ ] Interrupt signal handling (CLI + metadata)
- [ ] Checkpoint/resume for research items
- [ ] `aof research interrupt`

### Phase 3: Associative Memory (Path A) (1–2 weeks)
- [ ] Add `chromadb` or `faiss-cpu` + `sentence-transformers` to deps
- [ ] `VectorStore` wrapper: embed notes, query by similarity
- [ ] Hybrid search: FTS5 + vector for `memory.search()`
- [ ] Store research sources/citations in memory with embeddings

### Phase 4: Self-Improving Research (2 weeks)
- [ ] Citation tracking tool
- [ ] Research pipeline: search → scrape → synthesize → store
- [ ] Dynamic tool creation (agent can propose new scraper for a site)
- [ ] Integration with SOTA planner for decomposition

### Phase 5: VectorHaSH Concepts (Optional, Research)
- [ ] Evaluate VectorHaSH assoc_utils for sparse associative retrieval
- [ ] If useful: design adapter layer (embeddings → sparse codes → retrieval)

---

## 5. Example Workflows

### 5.1 Basic Research Queue
```bash
# Add research questions
aof research add "What are the latest advances in sparse attention?"
aof research add "Compare Llama 3.2 vs Qwen3 for code generation"

# Run research loop (processes queue)
aof research run --pipeline examples/research_sota.toml
```

### 5.2 Interrupt and Resume
```bash
# While agent is working on question 1...
aof research interrupt   # Pushes current work to backlog
aof research add "Urgent: summarize this paper"  # User adds priority item
aof research run         # Resumes; processes backlog in order
```

### 5.3 Kanban View
```bash
aof research list
# Backlog:      [3] What are the latest advances in sparse attention?
# In Progress: [1] Compare Llama 3.2 vs Qwen3...
# Done:        [2] Summarize VectorHaSH paper
```

---

## 6. Pipeline Config: Research SOTA

```toml
# examples/research_sota.toml
# Queue-aware research pipeline with citations and self-improvement

[queue]
persist_path = "data/research_queue.json"

[[steps]]
name = "plan"
system_prompt = "You decompose research questions into sub-goals and search strategies."
goal_template = "Plan research for: {input}"
tools = []
role = "reasoning"
max_steps = 1

[[steps]]
name = "gather"
system_prompt = "You search the web, scrape sources, and collect citations."
goal_template = "Research: {input}\n\nPlan: {prev_result}"
tools = ["web_search", "web_news", "web_scrape", "add_citation"]
role = "micro"
max_steps = 5

[[steps]]
name = "synthesize"
system_prompt = "You synthesize findings into a structured report with citations."
goal_template = "Synthesize:\n{prev_result}"
tools = []
role = "fast"
max_steps = 2

[[steps]]
name = "store"
system_prompt = "You store findings in memory with proper tags and links."
goal_template = "Store these findings: {prev_result}"
tools = []  # Uses memory.create_note internally
role = "micro"
max_steps = 1
```

---

## 7. Dependencies to Add

```toml
# For associative memory (Path A)
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Or alternatively:
# faiss-cpu>=1.7.0
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| VectorHaSH not directly usable | Use Path A (embeddings) first; Path B as research |
| Scraper updates could be unsafe | Sandbox tool creation; require user approval for new tools |
| Interrupt mid-inference | Check only between steps; graceful cancellation |
| Queue persistence conflicts | Use file locking or SQLite for queue |

---

## 9. Success Criteria

- [ ] User can add research questions to a queue and run autonomous research
- [ ] User can interrupt and push current work back to backlog
- [ ] Research findings are stored with citations and associative retrieval works
- [ ] Agent can update scrapers or add tools when needed (with safeguards)
- [ ] Kanban view shows backlog, in progress, blocked, done

---

## 10. References

- [VectorHaSH](https://github.com/FieteLab/VectorHaSH) — Episodic and associative memory from spatial scaffolds (Nature 2025)
- [ChromaDB](https://www.trychroma.com/) — Embedding store for semantic search
- [sentence-transformers](https://www.sbert.net/) — Local embedding models
- Existing AOF: `research.toml`, `mixed_model_research.toml`
