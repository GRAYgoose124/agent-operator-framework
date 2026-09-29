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

## Evidence-first pipeline (`aof refine`)

**Why not agent loops.** A baseline run of the existing SOTA pipeline on three neuro questions (0.6B gather agent,
LFM2.5 synthesise/store) lost the evidence before refinement could start: the gather agent ran one web search,
summarised *dictionary definitions* of "intrinsic circuit", never fetched a page, and the final artifact for a full
circuit question read "The findings from the search are stored as per the context... feel free to ask!". Tiny models
cannot be trusted with open-ended tool loops, so the framework owns the loop and models only do narrow jobs.

**Stages** (`src/aof/refine/`):

1. **Plan** queries: the question, a keyword form, and model-written sub-queries (`generate`).
2. **Gather** documents from pluggable sources (`sources.py`): PubMed, OpenAlex (peer-reviewed abstracts), Wikipedia,
   web fallback. Each carries an `authority` level; copies of one work are merged by DOI/title (`doc_key`), because the
   same paper arriving via PubMed *and* OpenAlex once produced false "independent corroboration".
3. **Extract** claims *extractively*: sentences are lifted verbatim, ranked by relevance to the question, and
   filtered (methods, questions, fragments, statements about the paper itself). A claim is grounded by construction.
4. **Cluster** near-duplicates (embeddings; token overlap fallback) and pick a canonical member by authority, abstract
   section, relevance.
5. **Decontextualise** claims that lean on context ("In turn, ..."): a small model rewrites, a judge must find the
   rewrite *supported by its own quote*, else the verbatim sentence stays. The quote is always kept as evidence.
6. **Store** one atomic claim note per cluster (`kind=claim`, `status=refined|canonical`, provenance list), archiving
   every other member with `superseded_by` (lossless: nothing is deleted).
7. **Curate** (optional): grade `core | supporting | irrelevant`; irrelevant claims are archived (kept, tagged
   `off-topic`). A negative from the cheap model is **confirmed by the large model** before archiving.
8. **Verify** (optional): look the claim up in *other* works; cheap judge first, a "contradicted" is confirmed by the
   large model before anything is flagged. Corroboration adds cited sources and promotes the note to `canonical`.
9. **Structure**: mutual links (semantic proximity + shared *specific* entities), extractive hub notes (only the title is
   model-written, and must reuse member vocabulary).
10. **Measure / export / assess**: vault metrics; Obsidian-style export; rubric coverage against a question set.

```bash
uv run aof refine --log-level WARNING run --workspace neuro \
    --queue-file examples/queues/neuro_hippocampal_thalamic.toml --curate --verify-top 4
uv run aof refine structure --workspace neuro
uv run aof refine metrics   --workspace neuro
uv run aof refine assess    --workspace neuro --queue-file examples/queues/neuro_hippocampal_thalamic.toml
uv run aof refine export    --workspace neuro          # -> data/workspaces/neuro/export/ (Obsidian-ready)
```

## Measured findings (12 GB RTX 4080 laptop, real models, real sentences)

**Topic classification** of 64 abstract sentences (reference = topic of the query that retrieved them; noisy but
identical for every provider):

| Provider | Coverage | Precision on accepted | Time (64 calls) |
|----------|----------|-----------------------|-----------------|
| needle:3 | 12% | 75% | 13.5 s |
| needle:2 | 33% | 71% | 10.5 s |
| role:fast (LFM2.5-1.2B) via llama-cpp-python | 3% (fails JSON) | 100% (n=2) | 2 s |
| role:small (Qwen3-4B) via llama-cpp-python (CPU) | 100% | 75% | 82 s |
| role:small (Qwen3-4B) via **llama-server (GPU)** | 100% | 78% | **12.7 s** |
| role:fast (LFM2.5-1.2B) via llama-server (GPU) | 100% | 62% | 6.3 s |
| role:large (Qwythos-9B) via llama-server (GPU) | 100% | 80% | 26 s |

Conclusions that shaped the defaults:
- The bundled `llama-cpp-python` wheel here is **CPU-only** (`llama_supports_gpu_offload() == False`), so every Qwen/LFM
  role ran on CPU while the GPU idled. Serving roles through `llama-server` (CUDA) made the 4B 6.5x faster and made the
  1.2B usable (JSON constraint works there). Route roles you use often through `[llama_server] roles`.
- Needle is not a general topic classifier: it abstains on most abstract-label calls and, when it answers, is no more
  precise than the 4B. Cheap-first still costs nothing in accuracy, but its real value is tool routing and fielded
  extraction. The 9B is only slightly more precise than the 4B on this task, so it is reserved for confirming negatives
  and for judging, not for bulk labelling.
- Curation with the 4B alone rejected 36 of 66 claims as irrelevant, including clear TRN anatomy ("Electrical
  synapses between TRN neurons were absent in Cx36-null mice"). Confirming every negative with the 9B cut rejects to
  11 of 57, all genuinely off-topic. A false reject hides knowledge, so negatives always get the strongest judge.
- Long-running work must survive a dead model server: `LlamaServerBackend` relaunches a dead `llama-server` and retries
  connection failures (after one crashed mid-run with no error in its log).

## First full evaluation (neuroscience demo queue, 28 questions)

`examples/queues/neuro_hippocampal_thalamic.toml` run end to end with local models (Qwen3-4B `small` and Qwythos-9B
`large` via CUDA llama-server, PubMed + OpenAlex + Wikipedia, `--curate --verify-top 4`, then `verify --top 150`,
`structure`, `assess`). Results are not committed; these are the numbers and what they taught us.

| Metric | Value |
|--------|-------|
| live claims / archived (retained) | 2,586 / 712 |
| duplicate rate (cosine >= 0.9 pairs per claim) | 0.01 |
| orphan rate / avg links per claim | 0% / 5.4 |
| hubs (two-level tree) / claims covered | 163 / 100% |
| sourced | 100% (every claim is a verbatim quote with a URL) |
| corroborated by >= 2 sources / verified by lookup | 3% / 2% |
| rubric coverage (9B judge, top-4 retrieval) | 45 / 74 key facts (61%) |

What the results showed, and what we changed because of it:

- **Recall gaps, not precision gaps.** `so-spindle-ripple` scored 0/3: the landmark papers (Staresina 2015,
  Latchoumane 2017) were never retrieved. PubMed relevance ranking plus model-written sub-queries misses specific
  well-known works. Next: gap-driven research (decompose the question, check each part against the vault, run targeted
  lookups for the uncovered parts).
- **The rubric mixes content with attribution.** Many misses are facts phrased as "(Sherman 2001)"; abstracts rarely
  state author-year attributions. Treat coverage as a lower bound and read the report's near-miss notes.
- **Verification produced false alarms until constrained.** Of 150 claims verified, 22 were corroborated, 31
  unsupported, 93 had no independent evidence, and all 4 "contradicted" flags were false (evidence about MDMA users,
  toad vision, the cerebellum), even after the 9B confirmed them. Fixes: the judge prompt now says evidence about a
  different subject is never a contradiction, a contradiction needs evidence with cosine >= 0.65 to the claim, and
  `aof refine repair` removes flags already stored. Separately, "corroboration" by the *same paper under another
  URL* was being counted; independence is now checked by DOI and by repeated quote text.
- **Context-free claims remain the main quality issue.** "Increases happened in the ventral CA1..." has no subject;
  the opener heuristic only catches discourse markers. A standalone-ness check on every claim is the next cheap win.
- **Hubs need a hierarchy.** A single-level clustering produced hubs of 627 and 480 claims. Large hubs now split into
  subtopic hubs recursively; the export index shows the whole tree.
- **Operational lessons.** Sources rate-limit (429): per-host throttling, backoff and cooldown are essential. Long
  runs must be resumable (`refined.json` marks completed questions) because processes do die; dead `llama-server`s
  are relaunched, and children are bound to the parent's lifetime so a crash cannot leave orphans holding VRAM/RAM.
  Similarity work must be matrix maths (`refine/vectors.py`), not pure-Python loops.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 0 | Repo prepared for release (single init commit on `github-release`) | done |
| 1 | `llama_server` backend + thinking control + docs (`docs/llama-cpp.md`) | done (verified with a real 9B) |
| 1b | Qwen3.5-family support: `<function=...>` tool-call parsing, lone `</think>` split, sampling floor/overrides | done |
| 2 | Specialist registry (capability → provider chain), Needle 3/2 adapter, role providers (LFM2 Nanos, Qwen, llama-server) | done |
| 3 | Atomic notes + provenance schema; refine passes (extract, dedupe, merge, curate, verify) | done |
| 4 | Link proposals + hub/structure notes | done |
| 5 | Vault metrics + markdown export + rubric assessment (`aof refine assess`) | done; first evaluation above |
| 6 | Gap-driven research (decompose, check coverage, targeted lookups), standalone-ness check for every claim | next |

## Verified so far

- llama.cpp `llama-server` (build 8416, CUDA) loads a 9B `qwen35` GGUF fully on a 12 GB GPU, ~49 tok/s at 8k ctx;
  it emits untagged thinking, so thinking control is required.
- Needle 2 and Needle 3 (`cactus-needle` 3.0.6) both run on Windows. Needle 3 extracted a typed record correctly
  (~300 tok/s decode, ~150 MB RAM); Needle 2 declined the same request, so tool naming/description quality matters
  (one tool per action, names users would say).
