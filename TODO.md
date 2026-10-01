# TODO

## Recently Done

- [x] Seed expansion (`--seed-question`), branch connector, lateral thinking (see `researcher/`), reflection meta-filter, empty-search hints.
- [x] Repo prepared for public release: LICENSE, portable `config.toml`, `.gitignore` cleanup, single init commit.

- [x] Use http://github.com/WujiangXu/A-mem-sys
    - Optional agentic memory backend (`uv sync --extra agentic`); `memory.agentic_enabled`, adapter in `memory/agentic_adapter.py`; create_note mirrors to A-MEM, search uses A-MEM with FTS fallback.
    - Behaviour recording: `behaviour.jsonl` per workspace (step_start, step_done, item_done), plus behaviour summary note per completed research item (tags: behaviour, research).
- [x] Custom hippocampal routing / memory system
    - A-MEM provides the optional agentic/hippocampal-style memory path (LLM-generated metadata, linking, evolution). VectorHaSH remains a separate planned direction.

## Next (immediate focus)

- [ ] VectorHaSH-inspired associative memory
- [ ] Agents spend too many tokens on filler words. Agents need to spend more words directly. (Use /no_think in more cases it matters, or properly trim output for steps/contexts.)

## In Progress

- [ ] SOTA 2.0 harness (`aof sota`, docs/sota-harness.md). Done: layered vault graph (hubs capped + unique titles,
  concept notes, source notes, sparse claim links, clean export), graph-aware retrieval, report writer (reviews and
  insight investigations with cited, support-checked sections), harness with job queue, preemption, live steering,
  side-running `ask`, runtime model/context settings, inbox, autopilot. Next: faster writer throughput (see
  llama-server benchmark), reader-graded report quality on the neuro vault, per-section figures/tables.

- [ ] Knowledge-vault overhaul. Done: llama-server backend, specialist registry (Needle 3/2, role models),
  atomic notes with provenance, `aof refine` (gather, extract, merge, curate, verify, link, hubs, metrics, export, assess),
  standalone-ness check, gap-driven research. Next: raise rubric coverage further (recall of landmark papers, full-text
  evidence), corroboration beyond abstracts, hub quality (merge near-identical hubs), Needle-based field extraction.

## Planned

## Ongoing

## Ideas
