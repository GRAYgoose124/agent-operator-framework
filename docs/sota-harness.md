# SOTA 2.0 research harness (`aof sota`)

One long-running process that gathers evidence, keeps the vault structured, and writes long-form reports from it,
while listening to you. It builds on the evidence-first vault (`aof refine`): every report sentence cites vault
claims, and every claim is a verbatim quote from a paper.

```
          ┌──────────── console / inbox (aof sota send) ────────────┐
          │  ask · steer · report · research · now · set · status   │
          ▼                                                          │
   job queue (user > autopilot; `now` pauses anything) ──► running job
          │                                                          │
   research ─► claims ─► graph (hubs · concepts · sources) ─► retrieval ─► report writer
                                                                          (outline → evidence packs →
                                                                           sections → support check →
                                                                           synthesis → references)
```

## Quick start

```bash
# interactive console on an existing vault
uv run aof sota run -w neuro

# one review, then exit (resumes if interrupted)
uv run aof sota report -w neuro "The thalamic reticular nucleus in sleep and attention"

# an insight investigation with the large model at a smaller context
uv run aof sota report -w neuro "Do sleep spindles cause memory consolidation?" --kind investigation --writer-ctx 8192

# a new workspace from one seed question: research, gap research, structure, then a review
uv run aof sota run -w spindles --seed "How do slow oscillations, spindles and ripples interact during consolidation?"

# headless, with messages from another terminal
uv run aof sota run -w neuro --headless --auto --queue-file examples/queues/neuro_hippocampal_thalamic.toml
uv run aof sota send -w neuro "report thalamic control of cortical attention -i"
uv run aof sota tail -w neuro
```

## Console

Every message acts immediately. Long work goes to the job queue; `ask` runs beside the current job.

| Command | What happens |
|---|---|
| `ask <question>` or any text ending in `?` | Answered now from the vault with `[n]` citations (graph retrieval + writer); saved to `sota/answers.md` |
| `steer <text>` or any other plain text | Guidance for the running job (read before its next report section) and every later job |
| `report <topic> [-i] [--sections N] [--fresh] [--check none\|cheap\|strict]` | Queue a review (`-i`: insight investigation) |
| `investigate <question>` | Queue an insight investigation |
| `research <question>` / `deepen <question>` | Evidence pipeline for one question / gap research (decompose, find uncovered parts, research them) |
| `now <command>` | Run immediately; the current job pauses and resumes afterwards (reports resume from their saved state) |
| `map <topic>` / `find <text>` | Hubs and concepts the vault has around a topic / closest claims |
| `structure`, `export`, `metrics`, `verify [N]`, `assess` | Vault maintenance jobs |
| `status`, `jobs`, `cancel [id]`, `pause`, `resume`, `auto on\|off`, `focus <topic>`, `guidance [clear]` | Queue and autopilot control |
| `models`, `set <key>=<value>`, `config` | Roles, context sizes and VRAM plan; change any `[sota]` setting live |

User jobs outrank autopilot jobs and pause them. Model changes (`set writer=…`, `set writer_ctx=…`) reload the models
as soon as no job is using them.

## Models and context

`[sota]` in `config.toml` (every key can be changed with `set` or a CLI flag):

- `writer` (default `large`): the role that plans and writes. `writer_ctx` gives it its own context size, e.g. the 9B at
  8192 to leave VRAM for the judge, or 32768 for bigger evidence packs. The evidence pack of each section is budgeted
  from this number, so a smaller context means fewer (but still the most relevant) claims per section.
- `writer_parallel`: llama-server slots for the writer; 2 lets `ask` answer while a report section is being written.
- `writer_thinking`: allow a thinking model to reason before writing (adds up to 2048 tokens per call).
- `judge` (default `small`) checks each cited sentence against the claims it cites; `confirm` (default `large`)
  re-judges every negative before a sentence is flagged. `check = strict` uses `confirm` for every sentence.

`models` shows what is loaded and the VRAM plan. If the plan shows the writer partly offloaded (e.g. `27/32` layers),
writing is slower; lower `writer_ctx`/`writer_parallel` or give the judge `residency = "swap"` (see `docs/llama-cpp.md`).

## The vault graph (`aof refine structure`)

The first structure pass linked every claim to its most similar claims. On a 5.6k-claim vault that made one
hairball (every claim ~6 similarity links), hubs of 200+ claims, duplicate hub titles, and 1,300+ isolated archive
notes. Claim links are now derived from typed layers:

| Layer | What | Why |
|---|---|---|
| Areas and topic hubs | tree of hubs, ≤ ~40 direct claims each, unique titles; parents named from their subtopics | browse from the index down |
| Concepts | recurring specific terms ("place cells", "TRN", "sleep spindles"), with broader/narrower/related concepts | traverse by idea, not by wording |
| Sources | one note per paper, listing its claims | claims of one paper are one neighbourhood |
| Claim links | 0-2 mutual nearest neighbours from *other* papers | strong cross-paper relations only |

Concept extraction is deterministic (no model): candidate phrases that do not start or end on a generic word, kept
when recurring but not ubiquitous; acronyms defined in the text ("thalamic reticular nucleus (TRN)") merge into their
phrase. Hub names are the only model-written text.

The Obsidian export (`aof refine export`) uses readable file names (Obsidian labels graph nodes by file name), removes
files from the previous export, links archived claims to the note they were merged into, and writes a default
`.obsidian/graph.json` (colour by layer, archive hidden) unless one exists.

## Reports

`aof.refine.report.ReportWriter`, driven by the harness or `aof sota report`:

1. **Map**: graph retrieval around the topic lists the hubs and concepts the vault covers.
2. **Outline**: the writer plans sections (title, focus question, search queries); falls back to the top hubs.
3. **Evidence**: per section, graph-aware retrieval (similarity plus concept/hub activation, MMR diversity, at most
   `per_work` claims per paper, claims used by earlier sections pushed down), sized to the writer's context.
   Thin evidence can trigger research for that section first (`research_thin`).
4. **Draft** from the numbered pack only; `[n]` after every factual sentence.
5. **Check**: citations outside the pack are removed; each cited sentence is judged against its cited claims (cheap
   judge, negatives confirmed by the strong one). Failures go back for one revision; any still failing get a †.
6. **Synthesis, open questions, abstract** from the sections' cited key points; references renumbered by first use.

Output: `<workspace>/reports/<slug>.md`, and a `report` note in the vault linking every cited claim (so reports are
nodes in the graph). State is saved after each section (`reports/<slug>/state.json`): an interrupted or preempted
report resumes where it stopped; `--fresh` rewrites it.
