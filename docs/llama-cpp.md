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
