"""Git operations for auto-committing memory changes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GitOps:
    """Manages a git repository inside the data directory for memory versioning.

    The data/memory/ directory is its own git repo, separate from the project repo.
    This keeps agent-generated content versioned and browsable.
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir
        self._initialized = False

    async def ensure_init(self) -> None:
        """Initialize git repo if not already initialized."""
        if self._initialized:
            return
        git_dir = self.repo_dir / ".git"
        if not git_dir.exists():
            self.repo_dir.mkdir(parents=True, exist_ok=True)
            await self._run("git", "init")
            # Create initial commit with .gitkeep
            gitkeep = self.repo_dir / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
            await self._run("git", "add", ".")
            await self._run("git", "commit", "-m", "Initialize memory store")
            logger.info("Initialized git repo at %s", self.repo_dir)
        self._initialized = True

    async def auto_commit(
        self,
        message: str = "Auto-commit memory update",
        agent_id: str = "",
    ) -> bool:
        """Stage all changes and commit if there are any.

        Returns True if a commit was made, False if nothing to commit.
        """
        await self.ensure_init()
        await self._run("git", "add", ".")

        # Check for staged changes
        result = await self._run("git", "status", "--porcelain")
        if not result.strip():
            return False

        # Build commit message with agent attribution
        full_message = message
        if agent_id:
            full_message = f"[{agent_id}] {message}"

        await self._run("git", "commit", "-m", full_message)
        logger.debug("Git commit: %s", full_message)
        return True

    async def log(self, n: int = 10) -> list[dict[str, str]]:
        """Get recent commit log entries."""
        await self.ensure_init()
        result = await self._run(
            "git", "log", f"--max-count={n}",
            "--pretty=format:%H|%s|%ai",
        )
        entries = []
        for line in result.strip().splitlines():
            if "|" in line:
                parts = line.split("|", 2)
                entries.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else "",
                    "date": parts[2] if len(parts) > 2 else "",
                })
        return entries

    async def _run(self, *args: str) -> str:
        """Run a git command asynchronously."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self.repo_dir),
            # Avoid inheriting the REPL stdin pipe — on Windows that deadlocks
            # create_subprocess_exec while asyncio.to_thread(input) is waiting.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stderr_text = stderr.decode(errors="replace")

        if proc.returncode != 0:
            # Ignore "nothing to commit" which is expected
            if "nothing to commit" in stderr_text or "nothing to commit" in stdout.decode(errors="replace"):
                return stdout.decode(errors="replace")
            logger.warning("Git command %s failed: %s", args, stderr_text)

        return stdout.decode(errors="replace")
