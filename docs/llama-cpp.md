# Running models through llama.cpp (`llama-server`)

AOF loads GGUF models in-process with `llama-cpp-python` by default. Some models need a newer runtime than
that wheel ships (for example the `qwen35` architecture), or you may want to run a model on the upstream
llama.cpp build you already use. For those, AOF can manage a local **`llama-server`** process and talk to it over
its OpenAI-compatible API.

## Which llama.cpp to install

- **Known-good build:** `b8416` (commit `6729d4920`), CUDA, Windows/MSVC. Newer builds should work; older
  builds may not recognise recent architectures.
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
| `startup_timeout` | 300 | Seconds to wait for `/health` |
| `extra_args` | `[]` | Extra `llama-server` flags |

Per-role context comes from `[role_context]` (e.g. `large = 8192`).

## Tests

The real-model test is opt-in so contributors without the runtime are unaffected:

```bash
AOF_LLAMA_SERVER=/path/to/llama.cpp/build/bin \
AOF_TEST_LARGE_MODEL=/path/to/model.gguf \
uv run --extra dev python -m pytest tests/test_llama_server.py
```
