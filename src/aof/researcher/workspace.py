"""Workspace model for continuous researcher: dedicated directory with queue, memory, artifacts."""

from __future__ import annotations

from pathlib import Path


def _default_workspaces_root() -> Path:
    """Default root for workspaces (data/workspaces)."""
    return Path("data/workspaces")


class ResearcherWorkspace:
    """Dedicated directory for a researcher session: queue, memory, artifacts, activity log."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def queue_path(self) -> Path:
        return self._root / "queue.json"

    @property
    def memory_db_path(self) -> Path:
        return self._root / "memory.db"

    @property
    def notes_dir(self) -> Path:
        return self._root / "memory"

    @property
    def artifacts_dir(self) -> Path:
        return self._root / "artifacts"

    @property
    def activity_log_path(self) -> Path:
        return self._root / "activity.log"

    @property
    def interrupt_path(self) -> Path:
        return self._root / "interrupt.flag"

    def ensure_dirs(self) -> None:
        """Create workspace directory and subdirs."""
        self._root.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def ensure_workspace(name: str, root: Path | None = None) -> ResearcherWorkspace:
    """Create or resolve a workspace by name. Returns ResearcherWorkspace with directories created."""
    if root is None:
        root = _default_workspaces_root()
    ws_root = Path(root) / name
    ws = ResearcherWorkspace(ws_root)
    ws.ensure_dirs()
    return ws


def clone_workspace(from_name: str, to_name: str, root: Path | None = None) -> ResearcherWorkspace:
    """Clone a workspace: copy queue.json + memory.db, reset all items to BACKLOG.

    Returns the new ResearcherWorkspace.
    """
    import json
    import shutil

    if root is None:
        root = _default_workspaces_root()
    src = Path(root) / from_name
    if not src.exists():
        raise FileNotFoundError(f"Source workspace not found: {src}")

    dest_ws = ensure_workspace(to_name, root)

    # Copy queue.json and reset statuses
    src_queue = src / "queue.json"
    if src_queue.exists():
        data = json.loads(src_queue.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            item["status"] = "backlog"
        (dest_ws.root / "queue.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    # Copy memory.db
    src_db = src / "memory.db"
    if src_db.exists():
        shutil.copy2(str(src_db), str(dest_ws.root / "memory.db"))

    # Copy memory notes
    src_notes = src / "memory"
    if src_notes.exists():
        for md in src_notes.glob("*.md"):
            shutil.copy2(str(md), str(dest_ws.notes_dir / md.name))

    return dest_ws
