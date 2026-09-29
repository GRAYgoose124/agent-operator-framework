"""Interactive REPL for the continuous researcher."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING

from aof.researcher.artifacts import log_activity

if TYPE_CHECKING:
    from aof.researcher.workspace import ResearcherWorkspace

HELP_TEXT = """
Commands:
  help          Show this help
  status        Queue summary, current item, pause state
  add <q>       Add question to backlog
  list          Kanban view (backlog, in_progress, blocked, done)
  pause         Pause after current item
  resume        Resume processing
  interrupt     Interrupt current item, push to backlog
  note <text>   Add human note (to current item or next)
  promote <id> Move blocked item back to backlog
  restart [--all] [--clear] [id] [note]  Move done item(s) to backlog; optional note; --clear resets citations
  docs         Print path to artifacts/
  quit, exit   Graceful shutdown
"""


async def run_repl(
    queue_path: str,
    control: dict,
    workspace: ResearcherWorkspace | None = None,
    *,
    prompt: str = "researcher> ",
) -> None:
    """Run the REPL loop. Reads commands and updates control/queue."""
    from aof.research import ResearchQueue

    queue = ResearchQueue(queue_path)

    while not control.get("quit", False):
        try:
            line = await asyncio.to_thread(input, prompt)
        except EOFError:
            control["quit"] = True
            break
        except asyncio.CancelledError:
            control["quit"] = True
            raise
        line = line.strip()
        if not line:
            continue

        parts = shlex.split(line)
        if not parts:
            continue
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit"):
            control["quit"] = True
            if workspace:
                log_activity(
                    workspace.activity_log_path,
                    "repl",
                    "quit",
                    "User requested shutdown",
                )
            print("Shutting down...")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "status":
            grouped = queue.list_all()
            in_progress = grouped.get("in_progress", [])
            backlog_count = len(grouped.get("backlog", []))
            in_progress_count = len(in_progress)
            blocked_count = len(grouped.get("blocked", []))
            done_count = len(grouped.get("done", []))
            print(
                f"Backlog: {backlog_count} | In progress: {in_progress_count} | "
                f"Blocked: {blocked_count} | Done: {done_count}"
            )
            if in_progress:
                item = in_progress[0]
                print(f"In progress: [{item.id}] {item.question[:60]}...")
            else:
                print("In progress: (none)")
            print(f"Paused: {control.get('paused', False)}")

        elif cmd == "add":
            if not args:
                print("Usage: add <question>")
            else:
                q = " ".join(args)
                item_id = queue.add(q)
                print(f"Added [{item_id}] {q[:60]}...")
                if workspace:
                    log_activity(
                        workspace.activity_log_path,
                        "repl",
                        "add",
                        f"[{item_id}] {q[:60]}...",
                    )

        elif cmd == "list":
            grouped = queue.list_all()
            for col in ["backlog", "in_progress", "blocked", "done"]:
                items = grouped.get(col, [])
                label = col.replace("_", " ").title()
                print(f"\n{label}:")
                if not items:
                    print("  (empty)")
                for item in items:
                    preview = item.question[:60] + "..." if len(item.question) > 60 else item.question
                    print(f"  [{item.id}] {preview}")

        elif cmd == "pause":
            control["paused"] = True
            print("Pausing after current item...")
            if workspace:
                log_activity(workspace.activity_log_path, "repl", "pause", "User requested pause")

        elif cmd == "resume":
            control["paused"] = False
            print("Resumed.")
            if workspace:
                log_activity(workspace.activity_log_path, "repl", "resume", "User resumed")

        elif cmd == "interrupt":
            interrupt_path = workspace.interrupt_path if workspace else None
            if interrupt_path is None:
                print("Interrupt requires workspace.")
            else:
                interrupt_path.parent.mkdir(parents=True, exist_ok=True)
                interrupt_path.write_text("1", encoding="utf-8")
                print("Interrupt requested. Will stop at next checkpoint.")
                if workspace:
                    log_activity(
                        workspace.activity_log_path,
                        "repl",
                        "interrupt",
                        "User requested interrupt",
                    )

        elif cmd == "note":
            if not args:
                print("Usage: note <text>")
            else:
                note_text = " ".join(args)
                item = queue.get_in_progress_item()
                if item:
                    queue.add_human_note(item.id, note_text)
                    print(f"Note added to current item [{item.id}]")
                else:
                    grouped = queue.list_all()
                    backlog = grouped.get("backlog", [])
                    if backlog:
                        queue.add_human_note(backlog[0].id, note_text)
                        print(f"Note added to next item [{backlog[0].id}]")
                    else:
                        print("No item to attach note to. Add a question first.")
                if workspace:
                    log_activity(
                        workspace.activity_log_path,
                        "repl",
                        "note",
                        note_text[:80] + "..." if len(note_text) > 80 else note_text,
                    )

        elif cmd == "promote":
            if not args:
                print("Usage: promote <item_id>")
            else:
                item_id = args[0].strip()
                grouped = queue.list_all()
                blocked = grouped.get("blocked", [])
                item = next((i for i in blocked if i.id == item_id), None)
                if item is None:
                    all_ids = [
                        i.id for col in ("backlog", "in_progress", "blocked", "done")
                        for i in grouped.get(col, [])
                    ]
                    if item_id in all_ids:
                        print(f"Item [{item_id}] is not blocked. Use 'list' to see status.")
                    else:
                        print(f"Item [{item_id}] not found.")
                else:
                    queue.promote(item_id)
                    print(f"Moved [{item_id}] back to backlog.")
                    if workspace:
                        log_activity(
                            workspace.activity_log_path,
                            "repl",
                            "promote",
                            f"[{item_id}] -> backlog",
                        )

        elif cmd == "restart":
            use_all = "--all" in args
            args_rest = [a for a in args if a != "--all"]
            clear_citations = "--clear" in args_rest
            args_rest = [a for a in args_rest if a != "--clear"]
            refinement_note = " ".join(args_rest[1:]).strip() if len(args_rest) > 1 else None
            if use_all:
                n = queue.restart_all_done(
                    refinement_note=refinement_note or None,
                    clear_citations=clear_citations,
                )
                print(f"Restarted {n} done item(s) to backlog." + (
                    f" Note: {refinement_note[:60]}..." if refinement_note and len(refinement_note) > 60 else
                    f" Note: {refinement_note}" if refinement_note else ""
                ))
                if workspace and n:
                    log_activity(
                        workspace.activity_log_path,
                        "repl",
                        "restart",
                        f"restart_all_done: {n} items" + (f" ({refinement_note[:40]}...)" if refinement_note else ""),
                    )
            elif args_rest:
                item_id = args_rest[0].strip()
                ok = queue.restart_done(
                    item_id,
                    refinement_note=refinement_note,
                    clear_citations=clear_citations,
                )
                if not ok:
                    grouped = queue.list_all()
                    done_ids = [i.id for i in grouped.get("done", [])]
                    if item_id in [i.id for col in ("backlog", "in_progress", "blocked") for i in grouped.get(col, [])]:
                        print(f"Item [{item_id}] is not done. Use 'list' to see status.")
                    else:
                        print(f"Item [{item_id}] not found.")
                else:
                    print(f"Restarted [{item_id}] to backlog." + (
                        f" Note: {refinement_note[:60]}..." if refinement_note and len(refinement_note) > 60 else
                        f" Note: {refinement_note}" if refinement_note else ""
                    ))
                    if workspace:
                        log_activity(
                            workspace.activity_log_path,
                            "repl",
                            "restart",
                            f"[{item_id}] -> backlog" + (f" ({refinement_note[:40]}...)" if refinement_note else ""),
                        )
            else:
                print("Usage: restart <item_id> [note]  OR  restart --all [note]")

        elif cmd == "docs":
            if workspace:
                arts = workspace.artifacts_dir.resolve()
                print(f"Artifacts: {arts}")
                print("Open the directory to browse synthesis reports.")
            else:
                print("No workspace (docs only available in researcher run)")

        else:
            print(f"Unknown command: {cmd}. Type 'help' for commands.")
