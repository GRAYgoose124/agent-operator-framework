"""Process helpers: make a child process die with its parent, even if the parent crashes hard."""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

_job = None  # one Windows Job Object per Python process, created on first use


def _windows_job():
    """A Job Object whose processes are killed when this Python process ends (however it ends)."""
    global _job
    if _job is not None:
        return _job
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    _job = handle
    return _job


def bind_to_parent_lifetime(proc: subprocess.Popen) -> bool:
    """Kill `proc` automatically when this process exits or crashes. Returns False if it could not be arranged.

    Windows: assigns it to a kill-on-close Job Object. POSIX children should instead be started with
    `parent_death_preexec()`, since the guarantee has to be set up before exec.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        job = _windows_job()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):  # type: ignore[attr-defined]
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return True
    except Exception as e:  # never let lifetime binding break starting the server
        logger.warning("could not bind child process lifetime to this process: %s", e)
        return False


def parent_death_preexec():
    """`preexec_fn` for POSIX children: SIGTERM the child when the parent dies (Linux only; no-op elsewhere)."""
    if not sys.platform.startswith("linux"):
        return None

    def _set() -> None:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG

    return _set
