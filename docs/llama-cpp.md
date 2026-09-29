# Running models through llama.cpp (`llama-server`)

AOF loads GGUF models in-process with `llama-cpp-python` by default. Some models need a newer runtime than
that wheel ships (for example the `qwen35` architecture), or you may want to run a model on the upstream
llama.cpp build you already use. For those, AOF can manage a local **`llama-server`** process and talk to it over
its OpenAI-compatible API.

## Which llama.cpp to install

- **Known-good builds (CUDA, Windows):** `b8416` and `b9803` (the latter ~15% faster on a 9B `qwen35` model:
  56 vs 49 tok/s on a 12 GB RTX 4080 laptop GPU). Older builds may not recognise recent architectures.
- **Get it** from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (pick the CUDA build
  for your OS), or build it yourself:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON   # drop the flag for CPU-only
cmake --build llama.cpp/build --config Release -j
```

You only need `llama-server` (`llama-server.exe` on Windows).

## Pointing AOF at it

AOF looks for the binary in this order: `[llama_server] binary` (file or directory), the `AOF_LLAMA_SERVER`
environment variable, then `PATH`.

Keep machine-specific paths out of `config.toml`. Put them in **`config.local.toml`** next to it — it is
gitignored and deep-merged over `config.toml`:

```toml
# config.local.toml
[llama_server]
binary = "/path/to/llama.cpp/build/bin"     # or set AOF_LLAMA_SERVER
roles  = ["large"]                          # roles served through llama-server

[roles]
large = "/path/to/models/some-9b-Q4_K_M.gguf"   # absolute paths are allowed
```

Any role listed in `[llama_server] roles` is served by its own managed `llama-server` (started on first use on a
free local port, stopped on shutdown). Other roles keep using `llama-cpp-python`. `large` is the suggested role
for judgement-heavy steps (verify, reconcile, structure); prefer small models elsewhere.

| Option | Default | Meaning |
|--------|---------|---------|
| `n_gpu_layers` | 99 | Layers offloaded to GPU |
| `n_parallel` | 1 | Concurrent slots; context is `role n_ctx × n_parallel` |
| `enable_thinking` | false | Sends `chat_template_kwargs.enable_thinking`. Some models (Qwen3.5-family) emit untagged chain-of-thought otherwise and can exhaust `max_tokens` before answering |
| `top_p`, `top_k`, `repeat_penalty` | unset | Sampling overrides sent with every request |
| `min_temperature` | 0.0 | Floor applied to any requested temperature (callers such as `complete_json` ask for 0.1) |
| `startup_timeout` | 300 | Seconds to wait for `/health` |
| `extra_args` | `[]` | Extra `llama-server` flags |

Per-role context comes from `[role_context]` (e.g. `large = 8192`).

## Use it for GPU speed, not only for new architectures

Check whether your `llama-cpp-python` wheel can offload to the GPU:

```bash
uv run python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

If that prints `False` (the default PyPI wheel is CPU-only), every role loaded in-process runs on CPU. Routing
frequently used roles through a CUDA `llama-server` was 6.5x faster for a 4B model in testing:

```toml
[llama_server]
roles = ["micro", "small", "fast", "large"]
```

Each listed role gets its own server on a free port (mind VRAM: a 9B Q4 is ~5.6 GB, a 4B Q4 ~2.7 GB). A dead server is
relaunched automatically, and connection failures are retried.

## Composing models by VRAM, and swapping through system memory

`n_gpu_layers = -1` (the default) lets AOF plan GPU layers from a VRAM budget. Give each role a `vram_share` (its weight)
and a `residency`, and the budget is split by ratio:

```toml
[vram]
reserve_mb = 1024          # kept free for the desktop; or set budget_mb explicitly
# use_free = true          # budget from currently free VRAM (handy when something else uses the GPU)
park = "unload"            # what parking does: "unload" or "cpu" (keep answering from system memory, slowly)
idle_park_seconds = 0      # >0 also parks swap roles that sat idle this long

[llama_server]
roles = ["small", "large"]

[llama_server.per_role.small]
vram_share = 1             # 1 : 2 -> `large` gets twice the GPU share of `small`
[llama_server.per_role.large]
vram_share = 2
residency = "swap"         # loaded on demand; parked when another swap role needs the pool
```

* `pinned` roles stay loaded. A role that needs less than its share gives the rest back to the others.
* `swap` roles take turns in one pool, sized like the largest swap share. Using one parks the least recently used
  (after its in-flight requests finish). Parked weights stay in the OS file cache, so waking is a reload from system
  memory, not from disk. Keep roles that a stage uses together in one pool that fits them both, or they will thrash.
* A role whose allotment is smaller than its model keeps the remaining layers on the CPU (partial `-ngl`).
* `aof refine` commands announce the roles they use up front (`--judge-with role:large`, ...), so a swap role is
  loaded once per phase instead of once per call.
* Run `aof vram` to see the detected budget and the layers planned for each role. Sizes come from each GGUF's header
  (weights = file size, KV cache from the attention geometry), so an estimate can be a few hundred MB off.

Needle (2 and 3) is a native CPU engine and never uses VRAM. The sentence-transformers embedder also runs on the CPU by
default (`[specialists] embed_device = "cpu"`; use `"cuda"` or `""` to change it).

## Qwen3.5 small models

Qwen3.5 (hybrid attention) needs an upstream llama.cpp build, so serve it through `llama-server`. `qwen35_4b` and
`qwen35_0_8b` are ready-made roles; point them at your GGUFs and use them in the chains:

```toml
# config.local.toml
[roles]
qwen35_4b = "path/to/Qwen3.5-4B-Q4_K_M.gguf"
[llama_server]
roles = ["qwen35_4b", "large"]
[specialists.judge]
providers = ["role:qwen35_4b", "role:large"]
```

## Recommended settings for reasoning models (e.g. Qwythos-9B / Qwen3.5-family)

The model card recommends `temperature 0.6, top_p 0.95, top_k 20, repeat_penalty 1.05` and warns that greedy
or `T <= 0.3` can cause repetition loops. Set them on the server role so callers cannot undercut them:

```toml
[llama_server]
enable_thinking = false      # cheap steps; use thinking for the final judgement pass

# Sampling advice is per model: apply it only to the reasoning role, not to small classifier roles.
[llama_server.per_role.large]
top_p = 0.95
top_k = 20
repeat_penalty = 1.05
min_temperature = 0.6
```

- **Tool calls:** Qwen3.5-family models emit `<tool_call><function=NAME><parameter=ARG>VALUE</parameter>...`.
  AOF detects this family (`qwen3.5`, `qwen35`, `qwythos` in the model path), prompts in that format and parses it.
- **Thinking:** with thinking on, the template pre-opens `<think>`, so output has only a closing `</think>`;
  AOF splits the reasoning from the answer. Allow a generous `max_tokens` (the card suggests 16384).
- **MTP variants** (`*-MTP-*.gguf`) need `--spec-type draft-mtp --spec-draft-n-max N` (pass via `extra_args`) and a
  build that has it (`b9803` does; `b8416` only offers `ngram-*` modes). Measured on a 12 GB laptop GPU with the
  card's sampling (T=0.6): plain 56 tok/s vs MTP 53 / 45 / 33 tok/s at `n-max` 2 / 3 / 6 (drafts mostly rejected),
  so MTP did not help there. Prefer the plain quant and re-measure on your own hardware.

## Tests

The real-model test is opt-in so contributors without the runtime are unaffected:

```bash
AOF_LLAMA_SERVER=/path/to/llama.cpp/build/bin \
AOF_TEST_LARGE_MODEL=/path/to/model.gguf \
uv run --extra dev python -m pytest tests/test_llama_server.py
```
