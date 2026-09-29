"""Command-line interface for the Agent Operator Framework."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from aof import __version__
from aof.config import AppConfig, config_with_workspace, load_config, resolve_role_path


def main() -> None:
    # Common options shared by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default="config.toml", help="Path to config TOML")
    common.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    parser = argparse.ArgumentParser(
        prog="aof",
        description="Agent Operator Framework — concurrent small-LLM agents",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"aof {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # aof agent <prompt>
    agent_p = subparsers.add_parser("agent", help="Run a single agent with a prompt", parents=[common])
    agent_p.add_argument("prompt", type=str, help="The goal for the agent")
    agent_p.add_argument("--role", type=str, default="general")
    agent_p.add_argument(
        "--backend",
        choices=["deep", "llama", "server"],
        default="deep",
        help="Agent backend: deep (LangChain/Deep Agents), llama (local GGUF), server (LM Studio)",
    )
    agent_p.add_argument("--image", type=str, action="append", help="Image path(s) for VL models")

    # aof batch <prompts...>
    batch_p = subparsers.add_parser("batch", help="Run multiple agents concurrently", parents=[common])
    batch_p.add_argument("prompts", nargs="+", help="Goals for each agent")
    batch_p.add_argument("--role", type=str, default="general")
    batch_p.add_argument("--backend", choices=["llama", "server"], default="llama")

    # aof pipeline <pipeline.toml> <input>
    pipe_p = subparsers.add_parser("pipeline", help="Run a pipeline from a TOML definition", parents=[common])
    pipe_p.add_argument("pipeline_file", type=str, help="Path to pipeline TOML")
    pipe_p.add_argument("input", type=str, help="Initial input for the pipeline")
    pipe_p.add_argument(
        "--backend",
        choices=["deep", "legacy"],
        default="deep",
        help="Pipeline backend: deep (Deep Agents) or legacy (sequential agents)",
    )

    # aof canary [--model-path PATH]
    canary_p = subparsers.add_parser("canary", help="Run model capability checks", parents=[common])
    canary_p.add_argument("--backend", choices=["llama", "server"], default="llama")
    canary_p.add_argument("--model-path", type=str, default=None, help="Override model GGUF path for canary")

    # aof memory search <query>
    mem_p = subparsers.add_parser("memory", help="Memory operations", parents=[common])
    mem_sub = mem_p.add_subparsers(dest="memory_command")
    search_p = mem_sub.add_parser("search", help="Search memory notes")
    search_p.add_argument("query", type=str)
    search_p.add_argument("--limit", type=int, default=5)
    citations_p = mem_sub.add_parser("search_citations", help="Search citations by semantic similarity (hybrid)")
    citations_p.add_argument("query", type=str)
    citations_p.add_argument("--limit", type=int, default=5)
    recent_p = mem_sub.add_parser("recent", help="Show recent notes")
    recent_p.add_argument("--limit", type=int, default=10)

    # aof tools list | create | create crawler
    tools_p = subparsers.add_parser("tools", help="Tool operations", parents=[common])
    tools_sub = tools_p.add_subparsers(dest="tools_command")
    tools_sub.add_parser("list", help="List registered tools")
    create_p = tools_sub.add_parser("create", help="Scaffold a new agent-callable tool script")
    create_p.add_argument("name", type=str, help="Tool name (snake_case)")
    create_crawler_p = tools_sub.add_parser("create_crawler", help="Scaffold a Scrapy crawler for a site")
    create_crawler_p.add_argument("--site", type=str, required=True, help="Domain or site (e.g. example.com)")
    create_crawler_p.add_argument(
        "--template",
        type=str,
        default="generic",
        choices=["generic", "sitemap", "playwright"],
        help="Crawler template (default: generic)",
    )

    # aof research add | list | run
    research_p = subparsers.add_parser("research", help="Research queue (Kanban)", parents=[common])
    research_sub = research_p.add_subparsers(dest="research_command")
    research_add_p = research_sub.add_parser("add", help="Add a research question to backlog")
    research_add_p.add_argument("question", type=str, help="Research question")
    research_sub.add_parser("list", help="Show Kanban-style queue view")
    research_run_p = research_sub.add_parser("run", help="Process queue with pipeline")
    research_run_p.add_argument(
        "--pipeline",
        type=str,
        default="examples/mixed_model_research.toml",
        help=(
            "Pipeline TOML. Default: mixed_model_research.toml. "
            "Use research_sota.toml for single-mode with add_citation support."
        ),
    )
    research_run_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to process (default: all)",
    )
    research_run_p.add_argument(
        "--backend",
        choices=["deep", "legacy"],
        default="deep",
        help="Backend: deep (Deep Agents) or legacy (sequential pipeline)",
    )
    research_run_p.add_argument(
        "--mode",
        choices=["single", "extended"],
        default="single",
        help="Research mode: single (one pipeline) or extended (1 director + 4 experts)",
    )
    research_sub.add_parser("status", help="Show live research status (director, experts)")
    research_sub.add_parser("interrupt", help="Request interrupt; push current work to backlog")

    # aof discovery add | list | run
    discovery_p = subparsers.add_parser("discovery", help="Discovery queue (breadth-first web navigation)", parents=[common])
    discovery_sub = discovery_p.add_subparsers(dest="discovery_command")
    discovery_add_p = discovery_sub.add_parser("add", help="Add a seed (URL or topic) to backlog")
    discovery_add_p.add_argument("seed", type=str, help="URL or topic to explore")
    discovery_sub.add_parser("list", help="Show queue view")
    discovery_run_p = discovery_sub.add_parser("run", help="Process queue with pipeline")
    discovery_run_p.add_argument(
        "--pipeline",
        type=str,
        default="examples/discovery.toml",
        help="Pipeline TOML (default: examples/discovery.toml)",
    )
    discovery_run_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to process (default: all)",
    )
    discovery_run_p.add_argument(
        "--backend",
        choices=["deep", "legacy"],
        default="deep",
        help="Backend: deep (Deep Agents) or legacy (sequential pipeline)",
    )

    from aof.refine.cli import add_refine_parser

    add_refine_parser(subparsers, common)

    # aof researcher run [--workspace NAME] [--pipeline PATH] ...
    researcher_p = subparsers.add_parser(
        "researcher",
        help="Continuous researcher with workspace and interactive REPL",
        parents=[common],
    )
    researcher_sub = researcher_p.add_subparsers(dest="researcher_command")
    researcher_clone_p = researcher_sub.add_parser(
        "clone",
        help="Clone a workspace (queue + memory) resetting items to backlog",
        parents=[common],
    )
    researcher_clone_p.add_argument("--from", dest="from_ws", type=str, required=True, help="Source workspace name")
    researcher_clone_p.add_argument("--to", dest="to_ws", type=str, required=True, help="Destination workspace name")

    researcher_start_p = researcher_sub.add_parser(
        "start",
        help="Start researcher as a background daemon",
        parents=[common],
    )
    researcher_start_p.add_argument("--workspace", type=str, default="default")
    researcher_start_p.add_argument("--pipeline", type=str, default="examples/mixed_model_research.toml")
    researcher_start_p.add_argument("--backend", choices=["deep", "legacy"], default="legacy")
    researcher_start_p.add_argument("--mode", choices=["single", "extended"], default="single")

    researcher_sub.add_parser(
        "attach",
        help="Attach to a running daemon (tail activity + send commands)",
        parents=[common],
    )

    researcher_sub.add_parser(
        "stop",
        help="Send quit command to running daemon",
        parents=[common],
    )

    researcher_run_p = researcher_sub.add_parser(
        "run",
        help="Run continuous researcher with REPL (add, pause, note, etc.)",
        parents=[common],
    )
    researcher_run_p.add_argument(
        "--workspace",
        type=str,
        default="default",
        help="Workspace name (default: default)",
    )
    researcher_run_p.add_argument(
        "--pipeline",
        type=str,
        default="examples/mixed_model_research.toml",
        help="Pipeline TOML for single mode",
    )
    researcher_run_p.add_argument(
        "--backend",
        choices=["deep", "legacy"],
        default="legacy",
        help="Backend: legacy (local GGUF: Qwen3, LFM2.5) or deep (requires LM Studio/Ollama)",
    )
    researcher_run_p.add_argument(
        "--route-gen",
        action="store_true",
        help="Run route generator: refill backlog from done/backlog + web search when low",
    )
    researcher_run_p.add_argument(
        "--route-gen-interval",
        type=int,
        default=None,
        metavar="SECS",
        help="Route generator sleep between runs (default: config research.route_gen_interval)",
    )
    researcher_run_p.add_argument(
        "--route-gen-min-backlog",
        type=int,
        default=None,
        metavar="N",
        help="Run route generator when backlog below this (default: config research.route_gen_min_backlog)",
    )
    researcher_run_p.add_argument(
        "--mode",
        choices=["single", "extended"],
        default="single",
        help="Research mode: single (pipeline) or extended (director)",
    )
    researcher_run_p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Process N queue items concurrently (default: config research.parallel_items)",
    )
    researcher_run_p.add_argument(
        "--seed-question",
        type=str,
        default=None,
        metavar="TEXT",
        help="Seed question: add to queue and run expansion (thinking model) to spawn many follow-up questions before research loop",
    )
    researcher_run_p.add_argument(
        "--queue-file",
        type=str,
        default=None,
        metavar="PATH",
        help="TOML question set (see examples/queues/) to add to the workspace queue; already-queued questions are skipped",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging: console (INFO+ by default) + file (DEBUG always)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_level = getattr(logging, args.log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler("aof_debug.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s",
    ))
    root_logger.addHandler(file_handler)

    config = load_config(args.config)
    asyncio.run(_dispatch(args, config))


async def _dispatch(args: argparse.Namespace, config: AppConfig) -> None:
    """Route to the appropriate async handler."""
    if args.command == "agent":
        await _run_agent(args, config)
    elif args.command == "batch":
        await _run_batch(args, config)
    elif args.command == "pipeline":
        await _run_pipeline(args, config)
    elif args.command == "canary":
        await _run_canary(args, config)
    elif args.command == "memory":
        await _run_memory(args, config)
    elif args.command == "tools":
        await _run_tools(args, config)
    elif args.command == "research":
        await _run_research(args, config)
    elif args.command == "discovery":
        await _run_discovery(args, config)
    elif args.command == "researcher":
        await _run_researcher(args, config)
    elif args.command == "refine":
        from aof.refine.cli import run_refine_command

        await run_refine_command(args, config)


async def _make_backend(args: argparse.Namespace, config: AppConfig, model_path_override: str | None = None):
    """Create and start the appropriate inference backend."""
    backend_type = getattr(args, "backend", "llama")

    if backend_type == "server":
        from aof.inference.openai_backend import LocalServerBackend

        backend = LocalServerBackend(config.local_server)
    else:
        from aof.inference.llama_backend import LlamaBackend

        model_config = config.model
        if model_path_override:
            from dataclasses import replace
            model_config = replace(model_config, path=model_path_override)
        backend = LlamaBackend(model_config, config.pool)

    await backend.start()
    return backend


async def _make_multi_backends(
    config: AppConfig,
    roles: set[str],
    *,
    backend_type: str = "llama",
) -> dict[str, "InferenceBackend"]:
    """Create and start backends for each role, applying per-role n_ctx overrides."""
    from dataclasses import replace

    from aof.inference.backend import InferenceBackend
    from aof.inference.llama_backend import LlamaBackend

    backends: dict[str, InferenceBackend] = {}
    cache: dict[str, InferenceBackend] = {}

    for role in roles:
        path = resolve_role_path(config, role)
        role_n_ctx = config.role_context.get(role, config.model.n_ctx)
        cache_key = f"{path}::{role_n_ctx}"
        if cache_key in cache:
            backends[role] = cache[cache_key]
            continue
        model_config = replace(config.model, path=path, n_ctx=role_n_ctx)
        backend = LlamaBackend(model_config, config.pool)
        await backend.start()
        cache[cache_key] = backend
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


async def _run_agent(args: argparse.Namespace, config: AppConfig) -> None:
    backend_type = getattr(args, "backend", "deep")
    if backend_type == "deep":
        await _run_agent_deep(args, config)
        return
    if backend_type == "server":
        backend = await _make_backend(args, config)
        memory, tools, git_ops = await _make_services(config)
        try:
            from aof.agent.pool import AgentPool
            pool = AgentPool(backend, memory, tools, config, git_ops=git_ops)
            image_urls = getattr(args, "image", None) or []
            agent_id = await pool.submit(
                args.prompt,
                role=args.role,
                image_urls=image_urls,
            )
            result = await pool.wait(agent_id)
            if result:
                print(f"\n--- Agent {result.agent_id} ({result.role}) ---")
                print(f"State: {result.state.name}")
                print(f"Steps: {result.metrics.steps_completed}/{result.metrics.steps_attempted}")
                print(f"Health: {result.metrics.health_score:.2f}")
                print(f"Tokens: {result.metrics.total_tokens}")
                if result.results:
                    print(f"\nFinal output:\n{result.results[-1].text if hasattr(result.results[-1], 'text') else result.results[-1]}")
            else:
                print("Agent failed to complete.")
        finally:
            await backend.shutdown()
            await memory.close()
    else:
        image_urls = getattr(args, "image", None) or []
        effective_role = "vision" if image_urls else args.role
        roles = {effective_role, args.role, "general", "default"}
        backends = await _make_multi_backends(config, roles)
        memory, tools, git_ops = await _make_services(config)
        try:
            from aof.agent.pool import AgentPool
            default = backends.get(effective_role) or backends.get(args.role) or backends.get("general") or backends.get("default")
            pool = AgentPool(default, memory, tools, config, git_ops=git_ops)
            for role, be in backends.items():
                pool.register_backend(role, be)
            agent_id = await pool.submit(
                args.prompt,
                role=effective_role,
                image_urls=image_urls,
            )
            result = await pool.wait(agent_id)
            if result:
                print(f"\n--- Agent {result.agent_id} ({result.role}) ---")
                print(f"State: {result.state.name}")
                print(f"Steps: {result.metrics.steps_completed}/{result.metrics.steps_attempted}")
                print(f"Health: {result.metrics.health_score:.2f}")
                print(f"Tokens: {result.metrics.total_tokens}")
                if result.results:
                    print(f"\nFinal output:\n{result.results[-1].text if hasattr(result.results[-1], 'text') else result.results[-1]}")
            else:
                print("Agent failed to complete.")
        finally:
            for be in set(backends.values()):
                await be.shutdown()
            await memory.close()


async def _run_agent_deep(args: argparse.Namespace, config: AppConfig) -> None:
    """Run agent using Deep Agents (LangChain/LangGraph)."""
    memory, tools, _ = await _make_services(config)
    try:
        from aof.orchestrator.deep_agent_factory import create_aof_deep_agent

        agent = create_aof_deep_agent(config, tools, memory, role=args.role)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.prompt}]},
            config={"configurable": {"thread_id": "cli-session"}},
        )
        if result and "messages" in result:
            last = result["messages"][-1]
            content = getattr(last, "content", str(last))
            print(f"\n--- Deep Agent ---\n{content}")
        else:
            print(result)
    finally:
        await memory.close()


async def _run_batch(args: argparse.Namespace, config: AppConfig) -> None:
    backend_type = getattr(args, "backend", "llama")
    if backend_type == "server":
        backend = await _make_backend(args, config)
        memory, tools, git_ops = await _make_services(config)
        try:
            from aof.agent.pool import AgentPool
            pool = AgentPool(backend, memory, tools, config, git_ops=git_ops)
            await pool.start_evaluator()
            ids = await pool.submit_batch(args.prompts, role=args.role)
            results = await pool.wait_all()
            await pool.stop_evaluator()
            print(f"\n--- Batch Results ({len(results)} agents) ---")
            for agent_id, ctx in results.items():
                if ctx:
                    print(f"\n[{agent_id}] {ctx.state.name} | Health: {ctx.metrics.health_score:.2f} | Steps: {ctx.metrics.steps_completed}/{ctx.metrics.steps_attempted}")
                else:
                    print(f"\n[{agent_id}] FAILED")
            rankings = pool.get_rankings()
            if rankings:
                print("\n--- Rankings (worst first) ---")
                for agent_id, score, role in rankings:
                    print(f"  {agent_id} ({role}): {score:.2f}")
        finally:
            await backend.shutdown()
            await memory.close()
    else:
        roles = {args.role, "general", "default"}
        backends = await _make_multi_backends(config, roles)
        memory, tools, git_ops = await _make_services(config)
        try:
            from aof.agent.pool import AgentPool
            default = backends.get(args.role) or backends.get("general") or backends.get("default")
            pool = AgentPool(default, memory, tools, config, git_ops=git_ops)
            for role, be in backends.items():
                pool.register_backend(role, be)
            await pool.start_evaluator()
            ids = await pool.submit_batch(args.prompts, role=args.role)
            results = await pool.wait_all()
            await pool.stop_evaluator()
            print(f"\n--- Batch Results ({len(results)} agents) ---")
            for agent_id, ctx in results.items():
                if ctx:
                    print(f"\n[{agent_id}] {ctx.state.name} | Health: {ctx.metrics.health_score:.2f} | Steps: {ctx.metrics.steps_completed}/{ctx.metrics.steps_attempted}")
                else:
                    print(f"\n[{agent_id}] FAILED")
            rankings = pool.get_rankings()
            if rankings:
                print("\n--- Rankings (worst first) ---")
                for agent_id, score, role in rankings:
                    print(f"  {agent_id} ({role}): {score:.2f}")
        finally:
            for be in set(backends.values()):
                await be.shutdown()
            await memory.close()


async def _run_pipeline(args: argparse.Namespace, config: AppConfig) -> None:
    backend = getattr(args, "backend", "deep")
    memory, tools, _ = await _make_services(config)
    try:
        if backend == "deep":
            from aof.orchestrator.deep_agent_factory import create_aof_pipeline_agent

            agent = create_aof_pipeline_agent(config, tools, memory, args.pipeline_file)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": args.input}]},
                config={"configurable": {"thread_id": "pipeline-session"}},
            )
            if result and "messages" in result:
                last = result["messages"][-1]
                final_output = getattr(last, "content", str(last))
            else:
                final_output = str(result)
            print(f"\n--- Pipeline Complete (Deep Agent) ---\n{final_output}")
        else:
            roles = _get_pipeline_step_roles(args.pipeline_file)
            roles.add("general")
            roles.add("default")
            backends = await _make_multi_backends(config, roles)
            from aof.pipeline.compose import Pipeline

            default = backends.get("general") or backends.get("default") or next(iter(backends.values()))
            pipeline = Pipeline.from_toml(
                args.pipeline_file, default, memory, tools,
                backends=backends,
                max_tool_rounds=config.pipeline.max_tool_rounds,
            )
            result = await pipeline.run(args.input)
            print(f"\n--- Pipeline Complete ({result.steps_completed} steps) ---")
            print(f"\nFinal output:\n{result.final_output}")
            for be in set(backends.values()):
                await be.shutdown()
    finally:
        await memory.close()


async def _run_canary(args: argparse.Namespace, config: AppConfig) -> None:
    model_override = getattr(args, "model_path", None)
    backend = await _make_backend(args, config, model_path_override=model_override)

    try:
        from aof.agent.evaluator import AgentEvaluator

        evaluator = AgentEvaluator(config.evaluator)
        result = await evaluator.run_canary(backend)

        print(f"\n--- Canary Results ---")
        print(f"Math:         {'PASS' if result.math_ok else 'FAIL'}")
        print(f"  Response: {result.details.get('math_response', 'N/A')}")
        print(f"JSON:         {'PASS' if result.json_ok else 'FAIL'}")
        print(f"  Response: {result.details.get('json_response', 'N/A')}")
        print(f"Instruction:  {'PASS' if result.instruction_ok else 'FAIL'}")
        print(f"  Response: {result.details.get('instruction_response', 'N/A')}")
        print(f"\nOverall:      {result.overall_score * 100:.0f}%")
        print(f"Model:        {backend.model_info().get('model_path', backend.model_info().get('model', 'unknown'))}")
    finally:
        await backend.shutdown()


async def _run_memory(args: argparse.Namespace, config: AppConfig) -> None:
    from aof.memory.store import MemoryStore

    memory = MemoryStore(config.memory)
    await memory.initialize()

    try:
        if args.memory_command == "search":
            results = await memory.search(args.query, limit=args.limit)
            print(f"\n--- Memory Search: '{args.query}' ({len(results)} results) ---")
            for note in results:
                print(f"\n[{note.id}] {note.title}")
                print(f"  Tags: {', '.join(note.tags)}")
                print(f"  Agent: {note.agent_id}")
                print(f"  {note.content[:200]}")

        elif args.memory_command == "search_citations":
            results = await memory.search_citations(args.query, limit=args.limit)
            print(f"\n--- Citation Search: '{args.query}' ({len(results)} results) ---")
            for c in results:
                print(f"\n  {c.get('title', 'N/A')}")
                print(f"    URL: {c.get('url', 'N/A')}")
                print(f"    {c.get('snippet', '')[:150]}...")

        elif args.memory_command == "recent":
            results = await memory.get_recent(limit=args.limit)
            print(f"\n--- Recent Notes ({len(results)}) ---")
            for note in results:
                print(f"\n[{note.id}] {note.title} ({note.created_at.strftime('%Y-%m-%d %H:%M')})")
                print(f"  {note.content[:150]}")

        else:
            count = await memory.count()
            print(f"Memory store: {count} notes in {config.memory.notes_dir}")
    finally:
        await memory.close()


async def _run_research(args: argparse.Namespace, config: AppConfig) -> None:
    """Handle research queue commands: add, list, run."""
    from aof.research import ResearchQueue

    queue = ResearchQueue(config.research.queue_path)

    if args.research_command == "add":
        item_id = queue.add(args.question)
        print(f"Added to backlog: [{item_id}] {args.question[:60]}")

    elif args.research_command == "list":
        grouped = queue.list_all()
        print("\n--- Research Queue (Kanban) ---")
        for col in ["backlog", "in_progress", "blocked", "done"]:
            items = grouped.get(col, [])
            label = col.replace("_", " ").title()
            print(f"\n{label}:")
            if not items:
                print("  (empty)")
            for item in items:
                preview = item.question[:60] + "..." if len(item.question) > 60 else item.question
                print(f"  [{item.id}] {preview}")

    elif args.research_command == "status":
        grouped = queue.list_all()
        print("\n--- Research Status ---")
        in_progress = grouped.get("in_progress", [])
        if in_progress:
            for item in in_progress:
                print(f"In progress: [{item.id}] {item.question[:60]}...")
        else:
            print("No research in progress.")
        interrupt_path = Path(config.research.interrupt_flag_path)
        if interrupt_path.exists():
            print("Interrupt flag: active (will stop on next check)")
        else:
            print("Interrupt flag: inactive")

    elif args.research_command == "interrupt":
        interrupt_path = Path(config.research.interrupt_flag_path)
        interrupt_path.parent.mkdir(parents=True, exist_ok=True)
        interrupt_path.write_text("1", encoding="utf-8")
        print("Interrupt requested. Current research will stop at next checkpoint.")

    elif args.research_command == "run":
        pipeline_path = args.pipeline
        limit = args.limit
        processed = 0
        backend = getattr(args, "backend", "deep")
        mode = getattr(args, "mode", "single")

        memory, tools, git_ops = await _make_services(config)
        from aof.tools.builtin import citation

        citation.register(tools, queue)
        try:
            if mode == "extended":
                from aof.agent.pool import AgentPool
                from aof.research.director import ResearchDirector

                roles = {config.research.director_role, config.research.expert_role, "general", "default"}
                backends = await _make_multi_backends(config, roles)
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
                )
                processed = await director.run(limit=limit)
                print(f"\nProcessed {processed} items.")

                for be in set(backends.values()):
                    await be.shutdown()
            else:
                while True:
                    item = queue.pop_next()
                    if item is None:
                        print("\nQueue empty.")
                        break
                    if limit is not None and processed >= limit:
                        queue.interrupt(item.id)  # Put back on backlog
                        print(f"\nReached limit ({limit}). Stopping.")
                        break

                    print(f"\n--- Researching [{item.id}] {item.question[:60]}... ---")

                    from aof.research.context import current_research_item_id

                    token = current_research_item_id.set(item.id)
                    try:
                        if backend == "deep":
                            from aof.orchestrator.deep_agent_factory import create_aof_pipeline_agent

                            agent = create_aof_pipeline_agent(config, tools, memory, pipeline_path)
                            result = agent.invoke(
                                {"messages": [{"role": "user", "content": item.question}]},
                                config={"configurable": {"thread_id": item.id}},
                            )
                            final_output = ""
                            if result and "messages" in result:
                                last = result["messages"][-1]
                                final_output = getattr(last, "content", str(last))
                            queue.mark_done(item.id)
                            processed += 1
                            print(f"\nDone. Output length: {len(final_output)}")
                        else:
                            roles = _get_pipeline_step_roles(pipeline_path)
                            roles.add("general")
                            roles.add("default")
                            backends = await _make_multi_backends(config, roles)
                            from aof.pipeline.compose import Pipeline

                            default = backends.get("general") or backends.get("default") or next(iter(backends.values()))
                            pipeline = Pipeline.from_toml(
                                pipeline_path, default, memory, tools,
                                backends=backends,
                                max_tool_rounds=config.pipeline.max_tool_rounds,
                            )
                            result = await pipeline.run(item.question)
                            queue.mark_done(item.id)
                            processed += 1
                            print(f"\nDone. Output length: {len(result.final_output)}")
                            for be in set(backends.values()):
                                await be.shutdown()
                    except Exception as e:
                        print(f"\nError: {e}")
                        queue.mark_blocked(item.id)
                    finally:
                        current_research_item_id.reset(token)
        finally:
            await memory.close()

    else:
        print("Use: aof research add | list | run")


async def _run_discovery(args: argparse.Namespace, config: AppConfig) -> None:
    """Handle discovery queue commands: add, list, run."""
    from aof.discovery import DiscoveryQueue
    from aof.research import ResearchQueue

    discovery_queue = DiscoveryQueue(config.discovery.queue_path)
    research_queue = ResearchQueue(config.research.queue_path)

    if args.discovery_command == "add":
        item_id = discovery_queue.add(args.seed)
        print(f"Added to backlog: [{item_id}] {args.seed[:60]}")

    elif args.discovery_command == "list":
        grouped = discovery_queue.list_all()
        print("\n--- Discovery Queue ---")
        for col in ["backlog", "in_progress", "done"]:
            items = grouped.get(col, [])
            label = col.replace("_", " ").title()
            print(f"\n{label}:")
            if not items:
                print("  (empty)")
            for item in items:
                preview = item.seed[:60] + "..." if len(item.seed) > 60 else item.seed
                print(f"  [{item.id}] {preview}")

    elif args.discovery_command == "run":
        pipeline_path = args.pipeline
        limit = args.limit
        processed = 0
        backend = getattr(args, "backend", "deep")

        memory, tools, git_ops = await _make_services(config)
        from aof.tools.builtin import discovery_tools

        discovery_tools.register(tools, research_queue)
        try:
            while True:
                item = discovery_queue.pop_next()
                if item is None:
                    print("\nQueue empty.")
                    break
                if limit is not None and processed >= limit:
                    from aof.discovery.models import DiscoveryItemStatus

                    discovery_queue._items[item.id].status = DiscoveryItemStatus.BACKLOG
                    discovery_queue._save()
                    print(f"\nReached limit ({limit}). Stopping.")
                    break

                print(f"\n--- Discovering [{item.id}] {item.seed[:60]}... ---")

                try:
                    if backend == "deep":
                        from aof.orchestrator.deep_agent_factory import create_aof_pipeline_agent

                        agent = create_aof_pipeline_agent(config, tools, memory, pipeline_path)
                        result = agent.invoke(
                            {"messages": [{"role": "user", "content": item.seed}]},
                            config={"configurable": {"thread_id": item.id}},
                        )
                        final_output = ""
                        if result and "messages" in result:
                            last = result["messages"][-1]
                            final_output = getattr(last, "content", str(last))
                        discovery_queue.mark_done(item.id)
                        processed += 1
                        print(f"\nDone. Output length: {len(final_output)}")
                    else:
                        roles = _get_pipeline_step_roles(pipeline_path)
                        roles.add("general")
                        roles.add("default")
                        backends = await _make_multi_backends(config, roles)
                        from aof.pipeline.compose import Pipeline

                        default = backends.get("general") or backends.get("default") or next(iter(backends.values()))
                        pipeline = Pipeline.from_toml(
                            pipeline_path, default, memory, tools,
                            backends=backends,
                            max_tool_rounds=config.pipeline.max_tool_rounds,
                        )
                        result = await pipeline.run(item.seed)
                        discovery_queue.mark_done(item.id)
                        processed += 1
                        print(f"\nDone. Output length: {len(result.final_output)}")
                        for be in set(backends.values()):
                            await be.shutdown()
                except Exception as e:
                    print(f"\nError: {e}")
                    discovery_queue.mark_done(item.id)  # Still mark done to avoid blocking
        finally:
            await memory.close()

    else:
        print("Use: aof discovery add | list | run")


async def _run_researcher(args: argparse.Namespace, config: AppConfig) -> None:
    """Run continuous researcher with REPL and workspace."""
    cmd = getattr(args, "researcher_command", None)
    if cmd == "clone":
        from aof.researcher.workspace import clone_workspace
        try:
            ws = clone_workspace(args.from_ws, args.to_ws)
            print(f"Cloned '{args.from_ws}' -> '{args.to_ws}' at {ws.root}")
            print("All items reset to backlog. Memory and notes copied.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    if cmd == "stop":
        from aof.researcher.daemon import read_pid, send_command
        pid = read_pid()
        if pid is None:
            print("No daemon running.")
        else:
            send_command("quit")
            print(f"Sent quit to daemon (PID {pid}).")
        return

    if cmd == "attach":
        from aof.researcher.daemon import read_pid, read_status, send_command
        pid = read_pid()
        if pid is None:
            print("No daemon running. Use 'aof researcher start' first.")
            return
        status = read_status()
        ws_name = status.get("workspace", "default")
        ws_root = Path(f"data/workspaces/{ws_name}")
        activity_log = ws_root / "activity.log"
        print(f"Attached to daemon PID {pid} (workspace: {ws_name})")
        print("Commands: add <question>, pause, resume, quit, status")
        print()
        # Tail activity log + accept commands
        import sys
        last_pos = activity_log.stat().st_size if activity_log.exists() else 0
        try:
            while True:
                # Show new activity
                if activity_log.exists():
                    size = activity_log.stat().st_size
                    if size > last_pos:
                        with open(activity_log, encoding="utf-8") as f:
                            f.seek(last_pos)
                            new_lines = f.read()
                            if new_lines.strip():
                                print(new_lines, end="")
                            last_pos = f.tell()
                # Read command
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("daemon> ") if sys.stdin.isatty() else sys.stdin.readline(),
                    )
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                if line == "quit":
                    send_command("quit")
                    print("Sent quit.")
                    break
                send_command(line)
        except KeyboardInterrupt:
            print("\nDetached.")
        return

    if cmd == "start":
        from aof.researcher.daemon import cleanup_pid, poll_commands, read_pid, write_pid, write_status
        from aof.researcher.runner import run_research_loop
        from aof.researcher.workspace import ensure_workspace

        existing_pid = read_pid()
        if existing_pid is not None:
            print(f"Daemon already running (PID {existing_pid}). Use 'aof researcher stop' first.")
            return

        workspace = ensure_workspace(args.workspace)
        config = config_with_workspace(config, workspace.root)
        write_pid()
        write_status({"workspace": args.workspace, "pipeline": args.pipeline, "backend": args.backend})
        control: dict = {"quit": False, "paused": False}
        print(f"Daemon started (PID {os.getpid()}). Workspace: {workspace.root}")
        print("Use 'aof researcher attach' to connect, 'aof researcher stop' to quit.")

        async def _daemon_cmd_poller():
            """Poll for commands from attach clients."""
            while not control.get("quit", False):
                cmds = poll_commands()
                for c in cmds:
                    if c == "quit":
                        control["quit"] = True
                    elif c == "pause":
                        control["paused"] = True
                    elif c == "resume":
                        control["paused"] = False
                    elif c.startswith("add "):
                        from aof.research import ResearchQueue
                        q = ResearchQueue(str(config.research.queue_path))
                        item_id = q.add(c[4:])
                        logger = logging.getLogger(__name__)
                        logger.info("Daemon: added [%s] %s", item_id, c[4:60])
                await asyncio.sleep(1)

        try:
            research_task = asyncio.create_task(run_research_loop(
                config, args.pipeline, backend=args.backend, mode=args.mode,
                workspace=workspace, control=control,
            ))
            cmd_task = asyncio.create_task(_daemon_cmd_poller())
            await asyncio.gather(research_task, cmd_task)
        except (asyncio.CancelledError, KeyboardInterrupt):
            control["quit"] = True
        finally:
            cleanup_pid()
        return

    if cmd != "run":
        print("Use: aof researcher run [--workspace NAME] [--pipeline PATH] ...")
        print("     aof researcher clone --from X --to Y")
        print("     aof researcher start --workspace NAME (daemon)")
        print("     aof researcher attach / stop")
        return

    from aof.researcher.repl import run_repl
    from aof.researcher.runner import run_research_loop
    from aof.researcher.workspace import ensure_workspace
    from dataclasses import replace as _replace

    workspace = ensure_workspace(args.workspace)
    # Apply --parallel override to config
    parallel_override = getattr(args, "parallel", None)
    if parallel_override is not None:
        config = _replace(config, research=_replace(config.research, parallel_items=parallel_override))
    config = config_with_workspace(config, workspace.root)
    control: dict = {"quit": False, "paused": False}

    queue_file = getattr(args, "queue_file", None)
    if queue_file:
        from aof.research import ResearchQueue
        from aof.research.queue_file import enqueue_question_set, load_question_set

        qset = load_question_set(queue_file)
        n_queued = enqueue_question_set(ResearchQueue(str(config.research.queue_path)), qset)
        print(f"Queue file '{qset.name}': added {n_queued} of {len(qset.questions)} questions.")

    seed_question = getattr(args, "seed_question", None)
    if seed_question:
        from aof.researcher.seed_expansion import run_seed_expansion
        n_added = await run_seed_expansion(
            config, str(config.research.queue_path), seed_question, workspace
        )
        print(f"Seed expansion: added {n_added} follow-up questions.")
        if not config.memory.consolidation_enabled:
            config = _replace(config, memory=_replace(config.memory, consolidation_enabled=True))
        # Default to 2 parallel items when queue is pre-filled by seed expansion (unless --parallel set)
        if parallel_override is None and config.research.parallel_items == 1:
            config = _replace(config, research=_replace(config.research, parallel_items=2))

    print(f"Workspace: {workspace.root}")
    backend_desc = "legacy (local GGUF)" if args.backend == "legacy" else "deep (LM Studio/Ollama)"
    print(f"Backend: {backend_desc}. Pipeline: {args.pipeline}. Use --log-level DEBUG for more.")
    route_gen_on = getattr(args, "route_gen", False) or getattr(config.research, "route_gen_enabled", False)
    consolidation_on = config.memory.consolidation_enabled
    if route_gen_on:
        print("Route generator: on (refill backlog when low).")
    if consolidation_on:
        print(f"Memory consolidation: on (every {config.memory.consolidation_interval}s).")
    if config.research.parallel_items > 1:
        print(f"Parallel items: {config.research.parallel_items}")
    if config.research.refinement_passes > 0:
        print(f"Refinement: {config.research.refinement_passes} pass(es) via {config.research.refinement_pipeline}")
    branch_connector_on = config.research.branch_connector_enabled
    lateral_on = config.research.lateral_thinking_enabled
    if branch_connector_on:
        print(f"Branch connector: on (every {config.research.branch_connector_interval}s).")
    if lateral_on:
        print(f"Lateral thinking: on (every {config.research.lateral_interval}s).")
    print("Type 'help' for commands. Research runs in background.")
    print()

    async def research_task():
        try:
            n = await run_research_loop(
                config,
                args.pipeline,
                backend=args.backend,
                mode=args.mode,
                workspace=workspace,
                control=control,
            )
            if not control.get("quit"):
                print(f"\nProcessed {n} items. Queue empty.")
        except asyncio.CancelledError:
            pass

    async def repl_task():
        await run_repl(
            str(config.research.queue_path),
            control,
            workspace=workspace,
        )

    async def route_gen_task():
        from aof.researcher.route_generator import run_route_generator_loop
        try:
            await run_route_generator_loop(
                config,
                str(config.research.queue_path),
                control,
                interval_seconds=getattr(args, "route_gen_interval", None) or config.research.route_gen_interval,
                min_backlog_target=getattr(args, "route_gen_min_backlog", None) or config.research.route_gen_min_backlog,
            )
        except asyncio.CancelledError:
            pass

    async def consolidation_task():
        from aof.memory.consolidator import MemoryConsolidator
        from aof.memory.store import MemoryStore
        try:
            mem = MemoryStore(config.memory)
            await mem.initialize()
            consolidator = MemoryConsolidator(mem, config)
            await consolidator.run_loop(control, interval=config.memory.consolidation_interval)
        except asyncio.CancelledError:
            pass

    async def branch_connector_task():
        from aof.researcher.branch_connector import run_branch_connector_loop
        try:
            await run_branch_connector_loop(
                config, workspace, control,
                interval_seconds=config.research.branch_connector_interval,
            )
        except asyncio.CancelledError:
            pass

    async def lateral_task():
        from aof.researcher.lateral_thinking import run_lateral_thinking_loop
        try:
            await run_lateral_thinking_loop(
                config, str(config.research.queue_path), control,
                interval_seconds=config.research.lateral_interval,
                min_backlog_target=config.research.lateral_min_backlog,
            )
        except asyncio.CancelledError:
            pass

    research_coro = asyncio.create_task(research_task())
    route_gen_coro = asyncio.create_task(route_gen_task()) if route_gen_on else None
    consolidation_coro = asyncio.create_task(consolidation_task()) if consolidation_on else None
    branch_connector_coro = asyncio.create_task(branch_connector_task()) if branch_connector_on else None
    lateral_coro = asyncio.create_task(lateral_task()) if lateral_on else None
    bg_tasks = [t for t in [research_coro, route_gen_coro, consolidation_coro, branch_connector_coro, lateral_coro] if t is not None]
    try:
        await repl_task()
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nShutting down...")
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
    else:
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass


async def _run_tools(args: argparse.Namespace, config: AppConfig) -> None:
    from aof.tools.builtin import file_tools, html_tools, math_tools, nlp, scraper, search
    from aof.tools.registry import ToolRegistry

    if args.tools_command == "create":
        _run_tools_create(args, config)
        return
    if args.tools_command == "create_crawler":
        _run_tools_create_crawler(args, config)
        return

    tools = ToolRegistry()
    search.register(tools)
    scraper.register(tools)
    html_tools.register(tools)
    nlp.register(tools)
    math_tools.register(tools)
    file_tools.register(tools)
    scripts_loaded = tools.load_scripts(Path(config.tools.scripts_dir))

    if args.tools_command == "list":
        all_tools = tools.list_tools()
        print(f"\n--- Registered Tools ({len(all_tools)}) ---")
        for t in all_tools:
            print(f"\n  {t.name} [{t.source}]")
            print(f"    {t.description}")
        if scripts_loaded:
            print(f"\n  ({scripts_loaded} agent-created scripts loaded)")


def _run_tools_create(args: argparse.Namespace, config: AppConfig) -> None:
    """Scaffold a new tool script with @tool metadata."""
    name = args.name.strip().lower().replace("-", "_")
    if not name.replace("_", "").isalnum():
        print("Error: Tool name must be alphanumeric with underscores (snake_case)")
        return
    scripts_dir = Path(config.tools.scripts_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / f"{name}.py"
    if path.exists():
        print(f"Error: {path} already exists")
        return
    template = f'''"""Agent-callable tool: {name}."""

"""
@tool
name: {name}
description: Describe what this tool does.
parameters: {{"type": "object", "properties": {{"input": {{"type": "string", "description": "Input parameter"}}}}, "required": ["input"]}}
"""

import json
import sys


def main():
    args = json.loads(sys.stdin.read())
    # Implement your logic here
    result = {{"output": args.get("input", "")}}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
'''
    path.write_text(template, encoding="utf-8")
    print(f"Created {path}")
    print("Edit the docstring (name, description, parameters) and implement main().")


def _run_tools_create_crawler(args: argparse.Namespace, config: AppConfig) -> None:
    """Scaffold a Scrapy crawler from template."""
    site = args.site.strip().lower()
    template_name = getattr(args, "template", "generic")

    # Normalize domain: example.com -> example_com
    site_slug = site.replace(".", "_").replace("-", "_")
    if not site_slug.replace("_", "").isalnum():
        print("Error: Site must be alphanumeric with dots/dashes (e.g. example.com)")
        return

    crawlers_dir = Path(config.tools.crawlers_dir)
    crawlers_dir.mkdir(parents=True, exist_ok=True)
    path = crawlers_dir / f"site_{site_slug}.py"

    if path.exists():
        print(f"Error: {path} already exists")
        return

    # Load template
    templates_dir = Path(__file__).resolve().parent / "tools" / "crawlers" / "templates"
    template_path = templates_dir / f"{template_name}.py"
    if not template_path.exists():
        print(f"Error: Template '{template_name}' not found")
        return

    template = template_path.read_text(encoding="utf-8")
    from urllib.parse import urlparse
    if site.startswith("http"):
        parsed = urlparse(site)
        domain = parsed.netloc or parsed.path
        start_url = site
    else:
        domain = site if "." in site else f"{site}.com"
        start_url = f"https://{domain}/"
    sitemap_url = f"https://{domain}/sitemap.xml"

    content = template.replace("{site_slug}", site_slug)
    content = content.replace("{domain}", domain)
    content = content.replace("{start_url}", start_url)
    content = content.replace("{sitemap_url}", sitemap_url)

    path.write_text(content, encoding="utf-8")
    print(f"Created {path}")
    print(f"  Template: {template_name}")
    print(f"  Domain: {domain}")
    print("Edit start_urls/sitemap_urls if needed. Use crawl_site or load via CrawlerRegistry.")
