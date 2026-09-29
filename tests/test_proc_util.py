"""A child bound to its parent's lifetime must die when the parent is killed hard."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from aof.inference.proc_util import bind_to_parent_lifetime

CHILD_LAUNCHER = """
import subprocess, sys, time
from aof.inference.proc_util import bind_to_parent_lifetime, parent_death_preexec
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], preexec_fn=parent_death_preexec())
bound = bind_to_parent_lifetime(child)
print(child.pid, bound, flush=True)
time.sleep(600)
"""


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.mark.skipif(sys.platform not in ("win32",) and not sys.platform.startswith("linux"),
                    reason="lifetime binding is implemented for Windows and Linux")
def test_child_dies_when_parent_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", CHILD_LAUNCHER], stdout=subprocess.PIPE, text=True)
    try:
        pid_text, bound_text = parent.stdout.readline().split()
        child_pid = int(pid_text)
        assert _alive(child_pid)
        if sys.platform == "win32":
            assert bound_text == "True"
        parent.kill()  # hard kill: no atexit, no cleanup code runs in the parent
        parent.wait()
        deadline = time.time() + 15
        while _alive(child_pid) and time.time() < deadline:
            time.sleep(0.25)
        assert not _alive(child_pid), "orphaned child survived its parent"
    finally:
        parent.kill()


def test_binding_returns_false_off_windows_without_raising():
    if sys.platform == "win32":
        pytest.skip("Windows binds via a Job Object")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert bind_to_parent_lifetime(proc) is False
    finally:
        proc.wait()
