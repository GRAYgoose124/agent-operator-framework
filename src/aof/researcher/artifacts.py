"""Artifact writer and activity logger for the continuous researcher."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _is_failure_message(text: str) -> bool:
    """True if text looks like a propagated step failure."""
    return "[Step " in text and " failed:" in text


def _is_overflow_message(text: str) -> bool:
    """True if text is the context-overflow error from the Llama backend."""
    return bool(text) and "[Error: context overflow —" in text and "token window]" in text


def _looks_like_json(text: str) -> bool:
    """True if stripped content looks like raw JSON (starts with { or [)."""
    t = text.strip()
    return (t.startswith("{") or t.startswith("[")) and len(t) > 1


def append_behaviour_event(workspace_root: Path, event_type: str, payload: dict) -> None:
    """Append a structured behaviour event to behaviour.jsonl.

    One JSON object per line. Payload should include item_id, and for step_done:
    step_name, detail, output_length, success (bool). For item_done: question, step_names, step_tool_calls.
    """
    path = Path(workspace_root) / "behaviour.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_activity(
    path: Path,
    agent_id: str,
    action: str,
    summary: str,
) -> None:
    """Append a line to the activity log.

    Format: {iso_ts} | {agent_id} | {action} | {summary}
    """
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts} | {agent_id} | {action} | {summary}\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def write_artifact(
    artifacts_dir: Path,
    item_id: str,
    content: str,
    *,
    title: str | None = None,
    sources: list[dict] | None = None,
    step_results: list[str] | None = None,
    step_names: list[str] | None = None,
) -> Path:
    """Write synthesis report to artifacts/<item_id>.md.

    Optionally prepend title and append ## Sources. When step_results/step_names
    are provided and content is a failure message, body is built as ## Progress
    (per-step subsections), ## Error (failure message), then ## Sources.
    When content looks like raw JSON, it is wrapped in ## Raw output and a code block.
    """
    path = Path(artifacts_dir) / f"{item_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if title:
        prefix = f"# {title}\n\n"
    else:
        prefix = ""

    raw = content.strip()
    if step_results is not None and step_names is not None and _is_failure_message(raw):
        # Build Progress + Error body so artifact shows all step outputs and clear error.
        parts = ["## Progress\n\n"]
        for k, out in enumerate(step_results):
            name = step_names[k] if k < len(step_names) else f"Step {k + 1}"
            parts.append(f"### Step {k + 1}: {name}\n\n")
            parts.append(out.strip())
            parts.append("\n\n")
        parts.append("## Error\n\n")
        parts.append(raw)
        body = "".join(parts)
    else:
        body = raw
        if _looks_like_json(body):
            body = "## Raw output\n\n```json\n" + body + "\n```"

    if sources:
        lines = ["\n\n## Sources\n\n"]
        for s in sources:
            url = s.get("url", "")
            tit = s.get("title", "")
            snippet = s.get("snippet", "")
            lines.append(f"- [{tit}]({url})\n  {snippet}\n\n")
        body = body + "".join(lines)
    path.write_text(prefix + body, encoding="utf-8")
    return path


def append_progress_step(
    artifacts_dir: Path,
    item_id: str,
    step_index: int,
    step_name: str,
    step_output: str,
    *,
    title: str | None = None,
) -> Path:
    """Append a step block to artifacts/<item_id>_progress.md. Creates file with optional title on first write."""
    path = Path(artifacts_dir) / f"{item_id}_progress.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"## Step {step_index + 1}: {step_name}\n\n{step_output.strip()}\n\n"
    if not path.exists():
        header = (f"# {title}\n\n" if title else "")
        path.write_text(header + block, encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    return path


def write_sources_artifact(artifacts_dir: Path, item_id: str, sources: list[dict]) -> Path:
    """Write gathered sources to artifacts/<item_id>_sources.md."""
    path = Path(artifacts_dir) / f"{item_id}_sources.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sources\n\n"]
    for s in sources:
        url = s.get("url", "")
        title = s.get("title", "")
        snippet = s.get("snippet", "")
        lines.append(f"- [{title}]({url})\n  {snippet}\n\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path
