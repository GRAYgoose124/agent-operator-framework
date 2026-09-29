"""Headless daemon mode: run researcher in background with file-based IPC.

- `aof researcher start --daemon` writes PID file + command queue path
- `aof researcher attach` reads activity.log + writes commands to queue file
- IPC via file polling (simple, cross-platform)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DAEMON_DIR = Path("data/daemon")
PID_FILE = DAEMON_DIR / "daemon.pid"
CMD_QUEUE = DAEMON_DIR / "commands.jsonl"
STATUS_FILE = DAEMON_DIR / "status.json"


def write_pid() -> None:
    """Write current PID to daemon PID file."""
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def read_pid() -> int | None:
    """Read daemon PID. Returns None if not running."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        # Check if process exists (cross-platform)
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            # Process doesn't exist
            PID_FILE.unlink(missing_ok=True)
            return None
    except (ValueError, OSError):
        return None


def cleanup_pid() -> None:
    """Remove PID file on shutdown."""
    PID_FILE.unlink(missing_ok=True)


def write_status(status: dict) -> None:
    """Write daemon status to file."""
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def read_status() -> dict:
    """Read daemon status."""
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def send_command(cmd: str) -> None:
    """Append a command to the daemon command queue."""
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({"cmd": cmd, "ts": time.time()})
    with open(CMD_QUEUE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def poll_commands() -> list[str]:
    """Read and clear pending commands from the command queue."""
    if not CMD_QUEUE.exists():
        return []
    try:
        lines = CMD_QUEUE.read_text(encoding="utf-8").strip().split("\n")
        CMD_QUEUE.write_text("", encoding="utf-8")
        cmds = []
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    cmds.append(data.get("cmd", ""))
                except json.JSONDecodeError:
                    pass
        return cmds
    except OSError:
        return []
