"""Research runner: reusable loop with workspace, artifact write, and activity log."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aof.config import AppConfig, resolve_role_path
from aof.researcher.artifacts import (
    append_behaviour_event,
    append_progress_step,
    log_activity,
    write_artifact,
    write_sources_artifact,
)

if TYPE_CHECKING:
    from aof.researcher.workspace import ResearcherWorkspace

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 3


def _make_item_done_callback(workspace: ResearcherWorkspace):
    """Build callback for director/pipeline to write artifacts and log activity."""

    def on_done(
        item_id: str,
        content: str,
        sources: list[dict],
        *,
        question: str | None = None,
        step_results: list[str] | None = None,
        step_names: list[str] | None = None,
    ) -> None:
        write_artifact(
            workspace.artifacts_dir,
            item_id,
            content,
            title=question,
            sources=sources,
            step_results=step_results,
            step_names=step_names,
        )
        if sources:
            write_sources_artifact(workspace.artifacts_dir, item_id, sources)
        log_activity(
            workspace.activity_log_path,
            "runner",
            "item_done",
            f"Wrote artifacts/{item_id}.md",
        )

    return on_done


async def _process_single_item(
    item,
    *,
    config: AppConfig,
    pipeline_path: str,
    backend: str,
    backends: dict | None,
    memory,
    tools,
    workspace: "ResearcherWorkspace | None",
    on_item_done,
    queue,
    peer_item_ids: set[str] | None = None,
) -> bool:
    """Process one research item. Returns True on success, False on failure.

    Uses ContextVar current_research_item_id which is per-task safe with asyncio.gather.
    When peer_item_ids is set (e.g. other items in the same parallel batch), workspace_context
    includes them so agents can see other runners' activity.
    """
    from aof.research.context import current_research_item_id

    snippet = (item.question[:60] + "...") if len(item.question) > 60 else item.question
    if backend == "deep":
        logger.info("Processing [%s] %s (deep backend)", item.id, snippet)
    else:
        logger.info("Processing [%s] %s (pipeline)", item.id, snippet)

    if workspace:
        log_activity(
            workspace.activity_log_path,
            "runner",
            "item_start",
            f"[{item.id}] {item.question[:60]}...",
        )

    token = current_research_item_id.set(item.id)
    try:
        if backend == "deep":
            from aof.orchestrator.deep_agent_factory import create_aof_pipeline_agent

            agent = create_aof_pipeline_agent(config, tools, memory, pipeline_path)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": _input_with_notes(item)}]},
                config={"configurable": {"thread_id": item.id}},
            )
            final_output = ""
            if result and "messages" in result:
                last = result["messages"][-1]
                final_output = getattr(last, "content", str(last))
            queue.mark_done(item.id)
            if workspace and on_item_done:
                sources = [c.to_dict() for c in item.citation_entries]
                on_item_done(item.id, final_output, sources, question=item.question)
                await _store_synthesis_note(memory, item, final_output, sources)
        else:
            from aof.pipeline.compose import Pipeline

            default = backends.get("general") or backends.get("default") or next(iter(backends.values()))
            pipeline = Pipeline.from_toml(
                pipeline_path, default, memory, tools,
                backends=backends,
                max_tool_rounds=config.pipeline.max_tool_rounds,
            )
            on_progress = None
            if workspace:
                def on_progress(
                    step_index: int,
                    step_name: str,
                    event: str,
                    detail: str,
                    step_output: str | None = None,
                ) -> None:
                    log_activity(
                        workspace.activity_log_path,
                        "runner",
                        event,
                        f"[{item.id}] Step {step_index + 1} {step_name}: {detail}",
                    )
                    if event == "step_done" and step_output is not None:
                        append_progress_step(
                            workspace.artifacts_dir,
                            item.id,
                            step_index,
                            step_name,
                            step_output,
                            title=item.question,
                        )
                        from aof.researcher.artifacts import _is_failure_message
                        append_behaviour_event(
                            workspace.root,
                            "step_done",
                            {
                                "item_id": item.id,
                                "step_index": step_index,
                                "step_name": step_name,
                                "detail": detail,
                                "output_length": len(step_output),
                                "success": not _is_failure_message(step_output),
                            },
                        )
                    if event == "step_start":
                        append_behaviour_event(
                            workspace.root,
                            "step_start",
                            {"item_id": item.id, "step_index": step_index, "step_name": step_name},
                        )
            workspace_context = (
                await _build_workspace_context(
                    workspace, memory, in_progress_item_ids=peer_item_ids
                )
                if workspace
                else ""
            )
            result = await pipeline.run(
                _input_with_notes(item),
                on_progress=on_progress,
                workspace_context=workspace_context,
            )
            queue.reload()
            queue.mark_done(item.id)
            if workspace and on_item_done:
                sources = [c.to_dict() for c in item.citation_entries]
                on_item_done(
                    item.id,
                    result.final_output,
                    sources,
                    question=item.question,
                    step_results=result.results,
                    step_names=result.step_names,
                )
                from aof.researcher.artifacts import _is_failure_message, _is_overflow_message
                if _is_overflow_message(result.final_output) or _is_failure_message(result.final_output):
                    logger.warning(
                        "Skipping synthesis note for [%s]: final output is overflow or step failure",
                        item.id,
                    )
                else:
                    await _store_synthesis_note(memory, item, result.final_output, sources)
                step_tool_calls = result.metadata.get("step_tool_calls") or []
                append_behaviour_event(
                    workspace.root,
                    "item_done",
                    {
                        "item_id": item.id,
                        "question": item.question,
                        "step_names": result.step_names or [],
                        "step_tool_calls": step_tool_calls,
                        "success": True,
                    },
                )
                await _store_behaviour_summary_note(
                    memory, item, result.step_names or [], step_tool_calls,
                )

            # Refinement passes
            if config.research.refinement_passes > 0 and config.research.refinement_pipeline:
                await _run_refinement(
                    config, item, result.final_output, backends, memory, tools,
                    workspace, workspace_context,
                )
        return True
    except Exception as e:
        logger.exception("Research item failed: %s", e)
        queue.reload()
        queue.mark_blocked(item.id)
        if workspace:
            log_activity(
                workspace.activity_log_path,
                "runner",
                "item_done",
                f"[{item.id}] Error: {e}",
            )
        return False
    finally:
        current_research_item_id.reset(token)


async def _store_synthesis_note(memory, item, final_output: str, sources: list[dict]) -> None:
    """Store a single synthesis memory note for the completed research item."""
    cohesive = (f"# {item.question}\n\n" if item.question else "") + (final_output or "").strip()
    if sources:
        cohesive += "\n\n## Sources\n\n"
        cohesive += "".join(
            f"- [{s.get('title', '')}]({s.get('url', '')})\n  {s.get('snippet', '')}\n\n"
            for s in sources
        )
    await memory.create_note(
        title=f"Research: {item.question[:50]}..." if len(item.question) > 50 else f"Research: {item.question}",
        content=cohesive,
        tags=["research", "synthesis", item.id],
        source=f"research:{item.id}",
        agent_id="pipeline",
    )


async def _store_behaviour_summary_note(
    memory,
    item,
    step_names: list[str],
    step_tool_calls: list[int],
) -> None:
    """Store a short behaviour summary note so agents can search_memory for past research behaviour."""
    steps_str = ", ".join(step_names) if step_names else "—"
    tool_str = ", ".join(str(c) for c in step_tool_calls) if step_tool_calls else "—"
    content = (
        f"Research item: {item.question}\n\n"
        f"Steps: {steps_str}\n"
        f"Tool calls per step: {tool_str}\n"
        f"Outcome: success."
    )
    await memory.create_note(
        title=f"Behaviour: {item.question[:50]}..." if len(item.question) > 50 else f"Behaviour: {item.question}",
        content=content,
        tags=["behaviour", "research", item.id],
        source=f"research:{item.id}",
        agent_id="runner",
    )


async def _run_refinement(
    config, item, initial_output: str, backends, memory, tools,
    workspace, workspace_context: str,
) -> None:
    """Run refinement passes on completed research output."""
    from aof.pipeline.compose import Pipeline
    from aof.researcher.artifacts import append_progress_step

    refine_path = config.research.refinement_pipeline
    if not Path(refine_path).exists():
        logger.warning("Refinement pipeline not found: %s", refine_path)
        return

    current_output = initial_output
    for pass_num in range(config.research.refinement_passes):
        logger.info(
            "Refinement pass %d/%d for [%s]",
            pass_num + 1, config.research.refinement_passes, item.id,
        )
        default = backends.get("general") or backends.get("default") or next(iter(backends.values()))
        pipeline = Pipeline.from_toml(
            refine_path, default, memory, tools,
            backends=backends,
            max_tool_rounds=config.pipeline.max_tool_rounds,
        )
        refine_input = f"Original question: {item.question}\n\nCurrent draft:\n{current_output}"
        result = await pipeline.run(
            refine_input,
            workspace_context=workspace_context,
        )
        current_output = result.final_output

        # Write refinement artifact
        if workspace:
            refine_path_out = workspace.artifacts_dir / f"{item.id}_refine_{pass_num + 1}.md"
            refine_path_out.write_text(
                f"# Refinement Pass {pass_num + 1}: {item.question}\n\n{current_output}",
                encoding="utf-8",
            )


async def run_research_loop(
    config: AppConfig,
    pipeline_path: str,
    *,
    backend: str = "deep",
    mode: str = "single",
    limit: int | None = None,
    workspace: "ResearcherWorkspace | None" = None,
    control: dict | None = None,
) -> int:
    """Run research until queue empty or limit. Returns number processed.

    If control is provided, it may contain "quit" and "paused" keys (bool).
    When quit is True, exit. When paused is True, sleep until resumed.
    """
    from aof.research import ResearchQueue
    from aof.tools.builtin import citation, discovery_tools

    memory, tools, git_ops = await _make_services(config)
    queue = ResearchQueue(config.research.queue_path)
    citation.register(tools, queue)
    discovery_tools.register(tools, queue)

    on_item_done = (
        _make_item_done_callback(workspace) if workspace else None
    )
    interrupt_path = Path(config.research.interrupt_flag_path)
    ws_str = str(workspace.root) if workspace else "none"
    parallel_items = max(1, config.research.parallel_items)
    logger.info(
        "Researcher: backend=%s mode=%s pipeline=%s workspace=%s parallel=%d",
        backend, mode, pipeline_path, ws_str, parallel_items,
    )

    processed = 0
    try:
        if mode == "extended":
            from aof.agent.pool import AgentPool
            from aof.research.director import ResearchDirector

            roles = {config.research.director_role, config.research.expert_role, "general", "default"}
            backends = await _make_multi_backends(config, roles)
            director_path = resolve_role_path(config, config.research.director_role)
            expert_path = resolve_role_path(config, config.research.expert_role)
            logger.info(
                "Director role=%s -> %s, expert role=%s -> %s",
                config.research.director_role, director_path,
                config.research.expert_role, expert_path,
            )
            director_be = backends.get(config.research.director_role) or backends.get("default")
            expert_be = backends.get(config.research.expert_role) or backends.get("micro") or director_be

            pool = AgentPool(director_be, memory, tools, config, git_ops=git_ops)
            pool.register_backend(config.research.expert_role, expert_be)

            director = ResearchDirector(
                queue=queue,
                pool=pool,
                director_backend=director_be,
                memory=memory,
                tools=tools,
                config=config,
                interrupt_path=interrupt_path,
                on_item_done=on_item_done,
            )
            if control is None:
                control = {}
            while not control.get("quit", False):
                queue.reload()
                n = await director.run(limit=1)
                processed += n
                if n == 0:
                    logger.debug("Queue empty, waiting for items...")
                    poll = getattr(config.research, "poll_seconds", DEFAULT_POLL_SECONDS)
                    await asyncio.sleep(poll)
            for be in set(backends.values()):
                await be.shutdown()
        else:
            if control is None:
                control = {}
            backends = None
            while not control.get("quit", False):
                queue.reload()
                if control.get("paused", False):
                    await asyncio.sleep(1)
                    continue

                # Pop up to parallel_items items from the queue
                items = []
                for _ in range(parallel_items):
                    if limit is not None and processed + len(items) >= limit:
                        break
                    item = queue.pop_next()
                    if item is None:
                        break
                    items.append(item)

                if not items:
                    logger.debug("Queue empty, waiting for items...")
                    poll = getattr(config.research, "poll_seconds", DEFAULT_POLL_SECONDS)
                    await asyncio.sleep(poll)
                    continue

                if limit is not None and processed >= limit:
                    for item in items:
                        queue.interrupt(item.id)
                    break

                # Build backends once (kept warm for all items)
                if backend != "deep" and backends is None:
                    step_roles = _get_pipeline_step_roles(pipeline_path)
                    roles = step_roles | {"general", "default"}
                    backends = await _make_multi_backends(config, roles)

                # Process items (parallel if >1); pass peer item ids so workspace_context includes other runners
                batch_ids = {i.id for i in items}
                if len(items) == 1:
                    ok = await _process_single_item(
                        items[0],
                        config=config,
                        pipeline_path=pipeline_path,
                        backend=backend,
                        backends=backends,
                        memory=memory,
                        tools=tools,
                        workspace=workspace,
                        on_item_done=on_item_done,
                        queue=queue,
                        peer_item_ids=None,
                    )
                    if ok:
                        processed += 1
                else:
                    results = await asyncio.gather(
                        *[
                            _process_single_item(
                                item,
                                config=config,
                                pipeline_path=pipeline_path,
                                backend=backend,
                                backends=backends,
                                memory=memory,
                                tools=tools,
                                workspace=workspace,
                                on_item_done=on_item_done,
                                queue=queue,
                                peer_item_ids=batch_ids - {item.id},
                            )
                            for item in items
                        ],
                        return_exceptions=True,
                    )
                    for r in results:
                        if r is True:
                            processed += 1

            # Shut down backends when exiting single-mode loop (kept warm during loop)
            if backends is not None:
                for be in set(backends.values()):
                    await be.shutdown()
    finally:
        await memory.close()
    return processed


def _input_with_notes(item) -> str:
    """Build input string with human notes appended."""
    base = item.question
    if item.human_notes:
        notes = "\n".join(f"- {n}" for n in item.human_notes)
        return f"{base}\n\nHuman guidance:\n{notes}"
    return base


async def _build_workspace_context(
    workspace: "ResearcherWorkspace",
    memory,
    *,
    in_progress_item_ids: set[str] | None = None,
) -> str:
    """Build a rich workspace context string for pipeline goal_template {workspace_context}.

    Includes recent artifact content previews, knowledge_summary note, done/backlog counts,
    recent memory note titles, and (when running parallel items) other items in progress
    and the last few activity.log lines so agents can see other runners' activity.
    """
    parts = []

    # Other items in progress (same batch) so agents are aware of parallel work
    if in_progress_item_ids:
        parts.append(
            "Other items in progress (this batch): " + ", ".join(sorted(in_progress_item_ids))
        )

    # Last few activity lines so agents see recent runner activity
    try:
        if workspace.activity_log_path.exists():
            lines = workspace.activity_log_path.read_text(encoding="utf-8").strip().splitlines()
            recent_lines = [ln for ln in lines[-8:] if ln.strip()]
            if recent_lines:
                parts.append("Recent activity:\n" + "\n".join(recent_lines))
    except Exception:
        pass

    # Recent artifacts with content preview
    art_dir = workspace.artifacts_dir
    if art_dir.exists():
        main_mds = [
            p for p in art_dir.glob("*.md")
            if not p.stem.endswith("_progress") and not p.stem.endswith("_sources")
            and not p.stem.endswith("_refine_1") and not p.stem.endswith("_refine_2")
        ]
        by_mtime = sorted(main_mds, key=lambda p: p.stat().st_mtime, reverse=True)
        recent = []
        for p in by_mtime[:3]:
            try:
                content = p.read_text(encoding="utf-8").strip()
                first_line = content.split("\n")[0].strip()
                if first_line.startswith("#"):
                    first_line = first_line.lstrip("#").strip()
                # Include first 200 chars of content for richer context
                preview = content[:200].replace("\n", " ")
                recent.append(f"- {p.stem}: {first_line}\n  {preview}...")
            except Exception:
                recent.append(f"- {p.stem}")
        if recent:
            parts.append("Recent artifacts:\n" + "\n".join(recent))
        # Done / backlog counts
        done_count = len([p for p in art_dir.glob("*.md") if not p.stem.endswith(("_progress", "_sources"))])
        parts.append(f"Completed artifacts: {done_count}")

    # Knowledge summary note (if exists)
    try:
        summaries = await memory.search_by_tag("knowledge_summary", limit=1)
        if summaries:
            ks = summaries[0]
            parts.append(f"Knowledge summary: {ks.content[:500]}")
    except Exception:
        pass

    # Topic map note (if exists)
    try:
        topic_maps = await memory.search_by_tag("topic_map", limit=1)
        if topic_maps:
            tm = topic_maps[0]
            parts.append(f"Topic map: {tm.content[:500]}")
    except Exception:
        pass

    # Recent memory note titles
    try:
        notes = await memory.get_recent(limit=3)
        if notes:
            titles = [n.title[:50] + ("..." if len(n.title) > 50 else "") for n in notes]
            parts.append("Recent memory: " + "; ".join(titles))
    except Exception:
        pass
    return "\n".join(parts) if parts else ""


async def _update_knowledge_summary(memory, item, final_output: str) -> None:
    """Update (or create) the running knowledge_summary note with key findings."""
    try:
        existing = await memory.search_by_tag("knowledge_summary", limit=1)
        new_finding = f"\n- [{item.question[:60]}]: {final_output[:200].replace(chr(10), ' ')}"

        if existing:
            note = existing[0]
            updated_content = note.content + new_finding
            # Truncate if too long
            if len(updated_content) > 2000:
                updated_content = updated_content[-2000:]
            note.content = updated_content
            await memory.update_note(note)
        else:
            await memory.create_note(
                title="Knowledge Summary",
                content="Key findings from research:\n" + new_finding,
                tags=["knowledge_summary", "auto"],
                source="consolidator",
                agent_id="runner",
            )
    except Exception as e:
        logger.debug("Could not update knowledge summary: %s", e)


async def _make_services(config: AppConfig):
    """Initialize memory store, tool registry, and git ops."""
    from aof.memory.git_ops import GitOps
    from aof.memory.store import MemoryStore
    from aof.tools.builtin import file_tools, html_tools, math_tools, nlp, scraper, search
    from aof.tools.registry import ToolRegistry

    memory = MemoryStore(config.memory)
    await memory.initialize()

    git_ops = GitOps(Path(config.memory.notes_dir))
    await git_ops.ensure_init()

    tools = ToolRegistry()
    search.register(tools)
    scraper.register(tools)
    html_tools.register(tools)
    nlp.register(tools)
    math_tools.register(tools)
    file_tools.register(tools)
    from aof.tools.builtin import memory_tools, crawler_builder

    memory_tools.register(tools, memory)
    crawler_builder.register(tools, config.tools.crawlers_dir)
    tools.load_scripts(Path(config.tools.scripts_dir))

    return memory, tools, git_ops


async def _make_multi_backends(config: AppConfig, roles: set[str]):
    """Build backends for each role, applying per-role n_ctx overrides."""
    from dataclasses import replace

    from aof.inference.llama_backend import LlamaBackend

    path_to_backend: dict[str, object] = {}
    backends: dict[str, object] = {}

    for role in roles:
        path = resolve_role_path(config, role)
        role_n_ctx = config.role_context.get(role, config.model.n_ctx)
        # Cache key includes n_ctx so different context sizes get separate backends
        cache_key = f"{path}::{role_n_ctx}"
        if cache_key in path_to_backend:
            backends[role] = path_to_backend[cache_key]
            continue
        if role in config.llama_server.roles:
            from aof.inference.llama_server import LlamaServerBackend

            backend = LlamaServerBackend(path, config.llama_server.for_role(role), n_ctx=role_n_ctx)
        else:
            backend = LlamaBackend(replace(config.model, path=path, n_ctx=role_n_ctx), config.pool)
        await backend.start()
        logger.info("Model role=%s path=%s n_ctx=%d", role, path, role_n_ctx)
        path_to_backend[cache_key] = backend
        backends[role] = backend
    return backends


def _get_pipeline_step_roles(pipeline_path: str) -> set[str]:
    """Extract unique roles from a pipeline TOML file (including overflow_fallback_role)."""
    import tomllib
    roles: set[str] = set()
    with open(pipeline_path, "rb") as f:
        data = tomllib.load(f)
    for step in data.get("steps", []):
        roles.add(step.get("role", "general"))
        if step.get("overflow_fallback_role"):
            roles.add(step["overflow_fallback_role"])
    return roles
