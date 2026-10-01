"""Tests for memory GitOps."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.memory.git_ops import GitOps


@pytest.mark.asyncio
async def test_ensure_init_creates_repo(tmp_path: Path):
    ops = GitOps(tmp_path / "memory")
    await ops.ensure_init()
    assert (tmp_path / "memory" / ".git").is_dir()


async def test_run_does_not_inherit_stdin(tmp_path: Path):
    """Git subprocesses get DEVNULL stdin (inheriting the REPL pipe deadlocks on Windows)."""
    ops = GitOps(tmp_path / "memory")
    await ops.ensure_init()
    # `git hash-object --stdin` reads stdin; with DEVNULL it hashes the empty blob immediately.
    out = await ops._run("git", "hash-object", "--stdin")
    assert "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" in str(out)
