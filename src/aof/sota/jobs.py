"""Persistent job queue for the harness: user jobs outrank autonomous ones, `now` jumps the queue."""

from __future__ import annotations

import json
import itertools
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

JOB_KINDS = ("research", "deepen", "report", "structure", "verify", "export", "metrics", "assess")
PRIORITY = {"auto": 0, "user": 10, "now": 100}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    arg: str = ""
    options: dict = field(default_factory=dict)
    origin: str = "user"  # user | auto
    priority: int = PRIORITY["user"]
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: str = field(default_factory=_now)
    started: str = ""
    finished: str = ""
    result: str = ""
    error: str = ""
    preempted: int = 0  # times it was paused for a more urgent job

    def label(self) -> str:
        arg = f" {self.arg[:70]}" if self.arg else ""
        return f"#{self.id} {self.kind}{arg}"


class JobQueue:
    """Jobs persisted to `path` (JSON). Order: priority, then creation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.jobs: list[Job] = []
        self._ids = itertools.count(1)
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.jobs = [Job(**{k: v for k, v in j.items() if k in Job.__dataclass_fields__}) for j in data]
            for j in self.jobs:
                if j.status == "running":  # the harness stopped mid-job: run it again (reports resume from state)
                    j.status = "queued"
            top = max((int(j.id) for j in self.jobs if j.id.isdigit()), default=0)
            self._ids = itertools.count(top + 1)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs], indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, job_kind: str, arg: str = "", *, origin: str = "user", urgent: bool = False, **options) -> Job:
        """Queue a job. `options` are job-specific (e.g. a report's kind, sections, fresh)."""
        if job_kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {job_kind!r}")
        prio = PRIORITY["now"] if urgent else PRIORITY[origin]
        job = Job(str(next(self._ids)), job_kind, arg, dict(options), origin, prio)
        self.jobs.append(job)
        self.save()
        return job

    def queued(self) -> list[Job]:
        order = {j.id: i for i, j in enumerate(self.jobs)}
        return sorted((j for j in self.jobs if j.status == "queued"), key=lambda j: (-j.priority, order[j.id]))

    def next(self) -> Job | None:
        q = self.queued()
        return q[0] if q else None

    def get(self, job_id: str) -> Job | None:
        job_id = job_id.lstrip("#")
        return next((j for j in self.jobs if j.id == job_id), None)

    def mark(self, job: Job, status: str, *, result: str = "", error: str = "") -> None:
        job.status = status
        if status == "running":
            job.started = job.started or _now()
        elif status in ("done", "failed", "cancelled"):
            job.finished = _now()
        if result:
            job.result = result
        if error:
            job.error = error
        self.save()

    def has_pending(self, kind: str, arg: str = "") -> bool:
        return any(j.kind == kind and j.arg == arg and j.status in ("queued", "running") for j in self.jobs)

    def recent(self, n: int = 8) -> list[Job]:
        return [j for j in self.jobs if j.status in ("done", "failed", "cancelled")][-n:]
