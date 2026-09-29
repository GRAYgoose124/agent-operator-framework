# Knowledge-vault overhaul

**Goal:** AOF produces *useful, curated, cohesive* knowledge vaults. Agents are good at gathering
and bad at refining; the framework's job is the refining: merging, splitting, linking, verifying,
compacting and structuring notes so the vault reads as one body of knowledge, not a pile of research
dumps. (Distillation is one tool for this, not the whole point.)

## Requirements (running list)

1. **Small models first, scaled wide.** Prefer 0.3B–4B models and many concurrent workers; escalate to a
   larger model (e.g. a 9B) only for roles that need judgement (verify, reconcile, structure).
2. **Lossless by construction.** Refinement never deletes information: originals are archived and linked,
   every claim in a refined note traces to a source, and a verifier checks that before archiving.
3. **Zettelkasten-informed.** Atomic notes (one claim each), provenance, links, hub/structure notes.
   Existing notes and the link graph inform every refine step.
4. **Generic specialist models.** A *capability → provider* registry, not hard-coded model names. Task-specific
   models (Liquid AI LFM2 Nanos: tool / RAG / extract / math / transcript; Needle) are first-class, and any other
   task-specific model can be registered the same way, with fallback to a general role.
5. **Needle support.** Needle 3 is the primary target; Needle 2 (`generation=2`) is used for testing and as a
   fallback. Both are verified to run on Windows (`pip install cactus-needle`). Roles: tool-call routing,
   structured extraction (claim/record extraction), embeddings (dedupe, link proposals). Adapter must set
   `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` (the binary reports telemetry by default).
6. **llama.cpp runtime support.** `llama-server` (OpenAI-compatible) is a supported backend for models the
   bundled `llama-cpp-python` cannot load (e.g. `qwen35` arch). AOF can spawn/stop it. The repo documents the
   minimum llama.cpp build; **no machine-specific paths in the repo** (binary via `AOF_LLAMA_SERVER` / `PATH`,
   local overrides in an untracked `config.local.toml`). Big-model tests are opt-in.
7. **Thinking control.** Some models (Qwen3.5-family) think in plain text without tags; each role must be able
   to disable thinking or have it split from the answer.
8. **Measurable.** Vault metrics (duplicate rate, orphan notes, sourced-claim ratio, link density) so refinement
   is judged on a real workspace before/after.
9. **Readable output.** Export as an Obsidian-style markdown vault (wikilinks, frontmatter, hub notes).
10. **Repo hygiene.** Real integration tests, no mocks; docs kept current; nothing personal in the public repo.

## Architecture sketch

```
                 gather (many small agents)
                          |
                    raw notes (atomic, sourced)
                          |
   +----------------------+-----------------------+
   |        refine passes (specialists by capability)
   |  extract -> dedupe/embed -> merge/split -> link -> verify -> structure
   +----------------------+-----------------------+
                          |
       canonical notes + hub notes + archived originals
                          |
                 vault export + metrics
```

### Specialist registry

`[specialists.<capability>]` maps a capability (`tool_call`, `extract`, `embed`, `rag`, `math`, `summarize`,
`verify`, ...) to an ordered provider chain, e.g.

```toml
[specialists.extract]
providers = ["needle:3", "role:lfm2_extract", "role:small"]   # first available wins
[specialists.embed]
providers = ["needle:3", "sentence-transformers"]
```

Provider kinds: `needle:<generation>`, `role:<role>` (existing GGUF/server roles), `llama_server`, plus a small
`Specialist` protocol so new task-specific models can be added without touching call sites.

Implemented in `src/aof/specialists/` (`annotate`, `classify`, `embed`, `judge`). A provider that abstains (returns
`None`) or errors escalates to the next in the chain; per-provider calls/abstains/errors/seconds are recorded and
`escalation_rate()` reports how often the cheapest tier could not settle a call. Providers whose package or model
file is missing are skipped, so the same config works on machines without Needle or the Nano models.

**Needle, as measured (cactus-needle 3.0.6, Windows):**
- Extraction is *span-level*: it returned the fragment `projects to thalamic relay nuclei` and dropped "but not to
  neocortex". Use it for fields (subject, topic, entities, year/number cues), not to write or split claims.
- The engine withholds low-confidence calls (`suppressed_calls`); we treat that as *abstain* and escalate. Off-topic
  text is therefore escalated rather than mislabelled. Numeric confidence scales differ by generation (Needle 3 ~0.07-0.3
  for correct calls, Needle 2 ~0.2-0.97), so no cross-provider threshold is used.
- Needle 2 often abstains or fails strict grounding validation; it is a fallback/test target, not the primary.
- Needle 3 embeddings work (3072-d) but barely separate related from unrelated sentences (cosine 0.94 vs 0.93), and
  Needle 2 has no embeddings. Embedding for dedupe/linking uses sentence-transformers; Needle embed is opt-in only.

### Note model additions

`ZettelNote` gains `status` (`raw | refined | canonical | archived`), `sources` (citation ids/urls),
`claim_ids`, `supersedes` / `superseded_by`, and `confidence`. Archival keeps the original file and links it.

## Model hierarchy (cheapest tier that can do the job)

Work is routed up a ladder and only escalates when a tier is unsure. Each specialist returns a confidence
(Needle reports a calibrated one; others use schema validity, self-agreement, or a verifier).

| Tier | What | Used for |
|------|------|----------|
| 0 | No model: hashes, FTS5, shingle/MinHash, link-graph metrics | exact/near-duplicate detection, orphan and link stats, ID/citation bookkeeping |
| 1 | Needle 3 (2 as fallback), 8-29 MB, CPU, hundreds of tok/s | tool-call routing, claim/record extraction into typed fields, enum classification (topic, note status), embeddings for dedupe and link candidates |
| 2 | Liquid Nanos / LFM2.5, 0.35-2.6B | `lfm2_extract` (documents to structure), `lfm2_rag` (is this claim supported by these sources?), `lfm2_tool`, `lfm2_math`, `lfm2_transcript`, LFM2.5 instruct for rewrites |
| 3 | Qwen3 0.6-4B | gather agents; drafting merges and splits; writing link rationales |
| 4 | 9B-class `large` via llama-server | only escalations: contradictions, hub/structure notes, final audit of merged notes |

Rules of thumb:
- **Verify cheaply, generate rarely.** Checking a claim against its source is much easier than producing text, so
  support checks run wide on tier 1-2. Only disagreements escalate.
- **Cascade on confidence.** Cheap tier first; escalate on low confidence, failed schema, or two cheap tiers disagreeing.
- **Scale out, not up.** Many parallel small workers beat one large model for extraction/dedupe/verification.
- **Track it.** Per-tier calls, tokens and escalation rate are vault metrics; the target is to keep tier 4 a small fraction.

## Grounding and light research during refinement

Refinement is not closed-book. Every tier can look things up, in proportion to its role:

- **Small models (tiers 1-3): light sanity research.** While merging or checking a note they may run a *budgeted*
  lookup (default 1-2 calls: `search_memory` first, then `web_search`/`web_scrape`) to compare a claim against
  another source. Results are attached as provenance, never silently folded in.
- **Final pass (tier 4, the 9B):** a whole-note/hub-level audit with a larger research budget to ground claims,
  fill gaps, and tie the note into related vault notes and external sources.

Finding (Qwythos-9B, llama-server, native tools): with a soft prompt ("use tools if needed") the model judged a
claim "unsupported by the excerpt" but **did not call the search tool**; with a direct instruction ("look up
the number") it called `web_search` correctly in both auto and forced (`tool_choice=required`) modes. So research
must be **scaffolded by the framework, not left to model initiative**:

1. The verifier returns a structured verdict: `supported | contradicted | unsupported | needs_lookup` (+ a proposed query).
2. On `unsupported`/`needs_lookup` the framework runs the lookup itself (within the tier's budget), then re-judges
   with the evidence in context. This works identically for a 0.6B model and the 9B.
3. New evidence becomes a cited source on the note (`add_citation` must therefore work in refine context, not
   only during `research run`).

## Phases

| # | Phase | Status |
|---|-------|--------|
| 0 | Repo prepared for release (single init commit on `github-release`) | done |
| 1 | `llama_server` backend + thinking control + docs (`docs/llama-cpp.md`) | done (verified with a real 9B) |
| 1b | Qwen3.5-family support: `<function=...>` tool-call parsing, lone `</think>` split, sampling floor/overrides | done |
| 2 | Specialist registry (capability → provider chain), Needle 3/2 adapter, role providers (LFM2 Nanos, Qwen, llama-server) | done |
| 3 | Atomic notes + provenance schema; refine passes (extract, dedupe, merge/split, verify) | planned |
| 4 | Link proposals + hub/structure notes | planned |
| 5 | Vault metrics + markdown export; before/after evaluation on a real seeded workspace | planned |

## Verified so far

- llama.cpp `llama-server` (build 8416, CUDA) loads a 9B `qwen35` GGUF fully on a 12 GB GPU, ~49 tok/s at 8k ctx;
  it emits untagged thinking, so thinking control is required.
- Needle 2 and Needle 3 (`cactus-needle` 3.0.6) both run on Windows. Needle 3 extracted a typed record correctly
  (~300 tok/s decode, ~150 MB RAM); Needle 2 declined the same request, so tool naming/description quality matters
  (one tool per action, names users would say).
