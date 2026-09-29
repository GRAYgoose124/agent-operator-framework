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

- [ ] VectorHaSH-inspired associative memory (see `docs/SOTA_RESEARCH_PLAN.md`)
- [ ] Agents spend too many tokens on filler words. Agents need to spend more words directly. (Use /no_think in more cases it matters, or properly trim output for steps/contexts.)

## In Progress

- [ ] Knowledge-vault overhaul: see `docs/VAULT_OVERHAUL.md` (llama-server backend, specialist registry incl. Needle 3/2 + LFM2 Nanos, atomic notes + provenance, refine passes, vault export + metrics).

## Planned

## Ongoing

## Ideas
