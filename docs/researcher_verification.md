# Researcher feature verification

How to confirm that each implemented researcher feature is wired and used.

| Feature | Where it runs | How to verify |
|---------|----------------|----------------|
| **Queue reload** | Runner calls `queue.reload()` at start of each loop iteration | REPL: `add` or `restart --all`; within a few seconds `status` should show item moving to in progress or done. Tests: `tests/e2e/test_researcher_runner_e2e.py::test_queue_reload_sees_*`. |
| **Warm backends** (single-mode) | Runner builds backends once, reuses for all items, shuts down only on exit | Run 2+ items; logs should show model load once (e.g. "Loading ... model") at start, not per item. |
| **add_to_research_queue** in pipeline | Runner registers discovery_tools; TOML search step has the tool | Run researcher; if the search step discovers follow-ups, backlog can grow. Pipeline: `examples/mixed_model_research.toml` search step lists `add_to_research_queue`. |
| **Route generator** | CLI starts route_gen_task when `--route-gen` or `config.research.route_gen_enabled` | `uv run aof researcher run --workspace X --route-gen`; with low backlog, new items may appear. |
| **Cohesive artifact** (title + body + Sources) | `on_item_done` with `question`; `write_artifact(..., title=, sources=)` | After an item completes, open `data/workspaces/<name>/artifacts/<id>.md`: should start with `# <question>` and include `## Sources`. Test: `test_item_done_callback_writes_artifact_with_title_and_sources`. |
| **Synthesis memory note** (single-mode) | Runner calls `memory.create_note(...)` after writing artifact | Search workspace memory for tags `research`, `synthesis`, and the item id. Test: `test_synthesis_note_created_with_expected_tags`. |
| **Restart done → backlog** | REPL `restart` / queue `restart_done` | REPL: `restart <id>` or `restart --all`; `list` should show items back in backlog. Runner will pick them up after reload. Tests: `test_restart_done_item`, `test_restart_all_done`, `test_queue_reload_sees_restart_done`. |
| **Activity log** (step_start, step_done) | Pipeline `on_progress` in runner | Run one item; read `data/workspaces/<name>/activity.log` for lines with `step_start` and `step_done`. |
| **Config** (poll_seconds, route_gen_*) | Runner and CLI read `config.research` | Set `[research] poll_seconds = 5` or `route_gen_interval = 60` in `config.toml`; behaviour should match (e.g. poll interval, route gen interval). |

## Running the tests

```bash
uv run pytest tests/e2e/test_researcher_runner_e2e.py -v
uv run pytest tests/e2e/test_research_e2e.py tests/test_researcher.py -v
```
