"""`aof sota ...`: the SOTA 2.0 research harness."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aof.config import AppConfig


def _model_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("models (override [sota] in config.toml)")
    g.add_argument("--writer", help="Role that writes reports and answers (e.g. large, medium, small)")
    g.add_argument("--writer-ctx", type=int, help="Context size for the writer (e.g. 8192 to run a large model lean)")
    g.add_argument("--writer-parallel", type=int, help="llama-server slots for the writer (2 lets `ask` run beside a report)")
    g.add_argument("--writer-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="Let a thinking model reason before writing")
    g.add_argument("--judge", help="Cheap judge role for the citation support check")
    g.add_argument("--confirm", help="Strong judge role that confirms negatives")
    g.add_argument("--check", choices=["none", "cheap", "strict"])
    g.add_argument("--sources", help="Evidence sources, comma-separated (pubmed,openalex,wikipedia,web)")
    g.add_argument("--concurrency", type=int)


def add_sota_parser(subparsers, common: argparse.ArgumentParser) -> None:
    sota_p = subparsers.add_parser(
        "sota", help="SOTA 2.0 research harness: research, vault graph, long-form reports, live console",
        parents=[common],
    )
    sub = sota_p.add_subparsers(dest="sota_command")

    run_p = sub.add_parser("run", help="Start the harness with an interactive console (or --headless)")
    run_p.add_argument("--workspace", "-w", required=True)
    run_p.add_argument("--queue-file", default="", help="TOML question set the autopilot works through")
    run_p.add_argument("--seed", default="", help="Seed question: research, deepen, structure, then write a review")
    run_p.add_argument("--auto", action=argparse.BooleanOptionalAction, default=None, help="Autopilot when idle")
    run_p.add_argument("--headless", action="store_true",
                       help="No console; take commands from the inbox (`aof sota send`) and log to sota/events.log")
    _model_flags(run_p)

    report_p = sub.add_parser("report", help="Write one review / investigation and exit")
    report_p.add_argument("--workspace", "-w", required=True)
    report_p.add_argument("topic")
    report_p.add_argument("--kind", choices=["review", "investigation"], default=None)
    report_p.add_argument("--sections", type=int, default=None)
    report_p.add_argument("--fresh", action="store_true", help="Discard saved progress and write from scratch")
    report_p.add_argument("--guidance", action="append", default=[], help="Extra instruction (repeatable)")
    report_p.add_argument("--research-thin", action=argparse.BooleanOptionalAction, default=None,
                          help="Research sections whose vault evidence is thin before writing them")
    _model_flags(report_p)

    ask_p = sub.add_parser("ask", help="Answer one question from the vault and exit")
    ask_p.add_argument("--workspace", "-w", required=True)
    ask_p.add_argument("question")
    _model_flags(ask_p)

    send_p = sub.add_parser("send", help="Send a message to a running harness (any console command or free text)")
    send_p.add_argument("--workspace", "-w", required=True)
    send_p.add_argument("text", nargs="+")

    status_p = sub.add_parser("status", help="Show the job queue and recent events of a workspace's harness")
    status_p.add_argument("--workspace", "-w", required=True)
    status_p.add_argument("--lines", type=int, default=20)

    tail_p = sub.add_parser("tail", help="Follow a running harness's events (Ctrl+C to stop)")
    tail_p.add_argument("--workspace", "-w", required=True)


def _settings(args: argparse.Namespace, config: AppConfig):
    from aof.sota.harness import settings_from_args

    keys = ("writer", "writer_ctx", "writer_parallel", "writer_thinking", "judge", "confirm", "check", "sources",
            "concurrency", "auto", "research_thin")
    return settings_from_args(config.sota, {k: getattr(args, k, None) for k in keys})


async def run_sota_command(args: argparse.Namespace, config: AppConfig) -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):  # piped/redirected output defaults to the ANSI code page on Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cmd = args.sota_command
    if cmd is None:
        print("Use: aof sota run|report|ask|send|status|tail --workspace NAME")
        return
    if cmd == "send":
        from aof.sota.harness import send_message

        path = send_message(args.workspace, " ".join(args.text))
        print(f"sent to {path}")
        return
    if cmd in ("status", "tail"):
        _print_status(args) if cmd == "status" else await _tail(args)
        return

    from aof.sota.console import Console, run_console
    from aof.sota.harness import Harness

    console = Console()
    headless = cmd != "run" or args.headless
    harness = Harness(config, args.workspace, queue_file=getattr(args, "queue_file", ""),
                      emit=(lambda m: print(m, flush=True)) if headless else console.print,
                      settings=_settings(args, config))
    await harness.start()
    try:
        if cmd == "report":
            flags = {"kind": args.kind or harness.settings.report_kind}
            if args.sections:
                flags["sections"] = args.sections
            if args.fresh:
                flags["fresh"] = True
            harness.guidance.extend(args.guidance)
            # one-shot: run only this report (reuse a queued one for the same topic, e.g. left by a stopped run)
            job = next((j for j in harness.jobs.queued() if j.kind == "report" and j.arg == args.topic), None)
            if job is None:
                job = harness.jobs.add("report", args.topic, **flags)
            else:
                job.options.update(flags)
            await harness.run_job(job)
            return
        if cmd == "ask":
            await harness._ask(args.question)
            return
        if args.seed:
            for kind in ("research", "deepen", "structure"):
                harness.jobs.add(kind, "" if kind == "structure" else args.seed)
            harness.jobs.add("report", args.seed, kind=harness.settings.report_kind)
            harness.emit(f"seed queued: research, deepen, structure, report for: {args.seed}")
        if headless:
            harness.emit(f"headless; send commands with: aof sota send -w {args.workspace} \"<command>\"")
            await harness.run()
        else:
            scheduler = asyncio.create_task(harness.run())
            await run_console(harness, console)
            harness.quit = True
            await scheduler
    finally:
        await harness.close()


def _print_status(args: argparse.Namespace) -> None:
    from aof.researcher.workspace import ensure_workspace

    root = ensure_workspace(args.workspace).root / "sota"
    jobs_path = root / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
    active = [j for j in jobs if j["status"] in ("running", "queued")]
    print(f"{len(active)} active job(s):")
    for j in active:
        print(f"  #{j['id']} {j['kind']} {j['arg'][:70]} [{j['status']}, {j['origin']}]")
    events = root / "events.log"
    if events.exists():
        print("\nrecent events:")
        print("\n".join(events.read_text(encoding="utf-8").splitlines()[-args.lines:]))


async def _tail(args: argparse.Namespace) -> None:
    from aof.researcher.workspace import ensure_workspace

    path = Path(ensure_workspace(args.workspace).root / "sota" / "events.log")
    pos = path.stat().st_size if path.exists() else 0
    print(f"following {path} (Ctrl+C to stop)")
    try:
        while True:
            if path.exists() and path.stat().st_size > pos:
                with open(path, encoding="utf-8") as f:
                    f.seek(pos)
                    print(f.read(), end="", flush=True)
                    pos = f.tell()
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        return
