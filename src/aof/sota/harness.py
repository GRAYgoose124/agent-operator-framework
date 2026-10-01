"""The SOTA 2.0 harness: one long-running process that researches, structures and writes, and listens while it works.

- **Jobs** (research, deepen, structure, report, verify, export, metrics, assess) run one at a time from a persistent
  queue. User jobs outrank autonomous ones and pause them; `now <command>` pauses anything. A paused report resumes
  from its saved state, so preemption loses at most one section.
- **Messages** arrive from the console or the workspace inbox (`aof sota send`) and act immediately: `ask` answers
  from the vault beside the running job, steering text is read by the running report before its next section,
  `set` changes models, context sizes and settings (model changes apply as soon as no job is using the models).
- **Autopilot** (`auto on`) keeps working when the queue is empty: questions from the queue file, graph rebuilds,
  gap research, and reports on well-covered areas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from aof.config import AppConfig, SotaConfig, config_with_workspace, resolve_role_path
from aof.sota.commands import HELP, Command, parse_command
from aof.sota.jobs import Job, JobQueue

logger = logging.getLogger(__name__)

WRITER_KEYS = {"writer", "writer_ctx", "writer_parallel", "writer_thinking", "judge", "judge_ctx", "judge_parallel", "confirm"}


def effective_config(config: AppConfig, settings: SotaConfig) -> AppConfig:
    """Apply the harness's model settings: context sizes, llama-server slots and the writer's thinking switch.

    The judge's context and slots matter for VRAM: its KV cache is n_ctx x slots, and a 4B judge at 4 x 4096 needs
    ~2.4 GB on top of its weights, which pushed a 9B writer partly off a 12 GB GPU (measured: 7 tok/s instead of ~45).
    """
    rc = config.role_context
    fields_ = type(rc).__dataclass_fields__
    ls = config.llama_server
    per_role = {k: dict(v) for k, v in ls.per_role.items()}
    if settings.judge != settings.writer:
        if settings.judge_ctx and settings.judge in fields_:
            rc = replace(rc, **{settings.judge: settings.judge_ctx})
        if settings.judge_parallel:
            per_role.setdefault(settings.judge, {})["n_parallel"] = settings.judge_parallel
    if settings.writer_ctx and settings.writer in fields_:
        rc = replace(rc, **{settings.writer: settings.writer_ctx})
    writer = per_role.setdefault(settings.writer, {})
    if settings.writer_parallel:
        writer["n_parallel"] = settings.writer_parallel
    writer["enable_thinking"] = settings.writer_thinking
    return replace(config, role_context=rc, llama_server=replace(ls, per_role=per_role))


def _coerce(value: str, current):
    if isinstance(current, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected on/off, got {value!r}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    return value


class Harness:
    def __init__(
        self,
        config: AppConfig,
        workspace: str,
        *,
        queue_file: str = "",
        emit: Callable[[str], None] | None = None,
        settings: SotaConfig | None = None,
        workspaces_root: str | Path | None = None,
    ) -> None:
        from aof.researcher.workspace import ensure_workspace

        self.workspace = ensure_workspace(workspace, Path(workspaces_root) if workspaces_root else None)
        self.root = self.workspace.root
        self.base_config = config_with_workspace(config, self.root)
        self.settings = settings or config.sota
        self.queue_file = queue_file
        self.dir = self.root / "sota"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs = JobQueue(self.dir / "jobs.json")
        self._emit_out = emit or print
        self.events_path = self.dir / "events.log"
        self.inbox_path = self.dir / "inbox.jsonl"
        self._inbox_offset = self._load_offset()
        self.guidance: list[str] = []  # session guidance: every writer prompt from now on
        self._fresh_guidance: dict[str, list[str]] = {}  # per running job: lines not yet picked up
        self.focus = ""
        self.paused = False
        self.auto = self.settings.auto
        self.quit = False
        self.current: Job | None = None
        self._task: asyncio.Task | None = None
        self._side: set[asyncio.Task] = set()
        self._wake = asyncio.Event()
        self._reconfigure = False
        self._last_progress = ""
        self._research_since_structure = 0
        self._research_since_report = 0
        self.store = self.backends = self.registry = self.sources = None
        self._graph = None
        self._embed = None

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        from aof.memory.store import MemoryStore

        self.store = MemoryStore(self.base_config.memory)
        await self.store.initialize()
        await self._build_models()
        self.emit(f"harness ready: workspace {self.root.name}; writer {self.settings.writer} "
                  f"(ctx {self.writer_ctx()}), judge {self.settings.judge}, confirm {self.settings.confirm}")

    async def _build_models(self) -> None:
        from aof.refine.embcache import cached_embedder
        from aof.refine.sources import build_sources
        from aof.specialists import build_registry
        from aof.specialists.backends import RoleBackends

        self.config = effective_config(self.base_config, self.settings)
        self.backends = RoleBackends(self.config)
        self.registry = build_registry(self.config, self.backends.get)
        self.sources = build_sources(list(self.settings.sources))
        self._embed = cached_embedder(self.registry, self.root)

    async def _close_models(self) -> None:
        if self.registry is not None:
            await self.registry.close()
        if self.backends is not None:
            await self.backends.close()

    async def close(self) -> None:
        for t in list(self._side):
            t.cancel()
        if self._task and not self._task.done():
            self._task.cancel()
        await self._close_models()
        if self.store is not None:
            await self.store.close()

    # ------------------------------------------------------------ output

    def emit(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        self._emit_out(line)

    def progress(self, msg: str) -> None:
        self._last_progress = msg
        self.emit(f"  {self.current.label() if self.current else ''}: {msg}")

    # ------------------------------------------------------------ models

    def writer_ctx(self) -> int:
        return self.config.role_context.get(self.settings.writer, self.config.model.n_ctx) if hasattr(self, "config") \
            else (self.settings.writer_ctx or 0)

    def writer_name(self) -> str:
        return f"{self.settings.writer} ({Path(resolve_role_path(self.config, self.settings.writer)).name})"

    async def generate(self, system: str, user: str, max_tokens: int) -> str:
        from aof.inference.parsing import parse_response

        backend = await self.backends.get(self.settings.writer)
        extra = 2048 if self.settings.writer_thinking else 0
        result = await backend.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4, max_tokens=max_tokens + extra,
        )
        return parse_response(result.text).text.strip()

    def _judge_fn(self, role: str):
        from aof.specialists.role_provider import RoleProvider

        provider = RoleProvider(role, self.config, self.backends.get)

        async def judge(statement: str, evidence: str) -> str:
            try:
                r = await provider.judge(statement, evidence)
            except Exception as e:  # a judge failure must not stop a report; the sentence counts as unchecked
                logger.warning("judge %s failed: %s", role, e)
                return "needs_lookup"
            return r.value["verdict"] if r is not None else "needs_lookup"

        return judge

    async def graph(self):
        """The loaded vault graph; structures the vault first if it has never been structured."""
        from aof.refine.retrieve import VaultGraph

        if self._graph is None:
            concepts = await self.store.get_notes_by_tags(["concept"], limit=1, exclude_tags=["archived"])
            if not concepts and await self.store.get_notes_by_tags(["claim"], limit=1):
                self.emit("vault has no graph yet; structuring first")
                await self._structure()
            self._graph = await VaultGraph(self.store, self._embed).load()
        return self._graph

    # ------------------------------------------------------------ messages

    async def submit(self, text: str, origin: str = "console") -> str:
        """Handle one user message. Returns the reply to show; long work is queued or started on the side."""
        text = text.strip()
        if not text:
            return ""
        try:
            cmd = parse_command(text)
        except ValueError as e:
            return str(e)
        if origin != "console":
            self.emit(f"message ({origin}): {text[:120]}")
        try:
            reply = await self._handle(cmd)
        except Exception as e:  # a bad command must never take the harness down
            logger.exception("command failed: %s", text)
            reply = f"error: {type(e).__name__}: {e}"
        self._wake.set()
        return reply

    async def _handle(self, cmd: Command) -> str:
        n, a = cmd.name, cmd.arg
        if n == "help":
            return HELP
        if n in ("quit", "exit"):
            self.quit = True
            if self._task and not self._task.done():
                self._task.cancel()
            return "shutting down (running job is paused and will resume next start)"
        if n == "now":
            inner = parse_command(a)
            if inner.name in ("ask", "map", "find", "status", "jobs"):
                return await self._handle(inner)
            return await self._enqueue(inner, urgent=True)
        if n == "ask":
            self._side_task(self._ask(a), f"ask: {a[:60]}")
            return f"answering beside the current job: {a[:80]}"
        if n == "steer":
            return self._steer(a)
        if n == "focus":
            self.focus = "" if a.lower() in ("", "off", "none") else a
            return f"focus: {self.focus or '(none)'}"
        if n == "guidance":
            if a.lower() == "clear":
                self.guidance.clear()
                return "guidance cleared"
            return "guidance:\n" + ("\n".join(f"- {g}" for g in self.guidance) or "(none)")
        if n == "map":
            return await self._map(a)
        if n == "find":
            return await self._find(a)
        if n in ("status", "jobs"):
            return self.status(full=n == "jobs")
        if n == "cancel":
            return self._cancel(a)
        if n == "pause":
            self.paused = True
            return "paused: the running job finishes its current step, then nothing new starts (resume to continue)"
        if n == "resume":
            self.paused = False
            return "resumed"
        if n == "auto":
            self.auto = a.lower() in ("on", "1", "true", "yes", "")
            return f"autopilot {'on' if self.auto else 'off'}"
        if n == "set":
            return self._set(a)
        if n == "config":
            return "\n".join(f"{f.name} = {getattr(self.settings, f.name)!r}" for f in fields(self.settings))
        if n == "models":
            return self._models()
        return await self._enqueue(cmd)

    async def _enqueue(self, cmd: Command, *, urgent: bool = False) -> str:
        kind_map = {"investigate": ("report", {"kind": "investigation"}), "review": ("report", {"kind": "review"})}
        kind, opts = kind_map.get(cmd.name, (cmd.name, {}))
        opts = {**opts, **cmd.flags}
        if kind in ("research", "deepen", "report") and not cmd.arg:
            return f"usage: {cmd.name} <text>"
        job = self.jobs.add(kind, cmd.arg, urgent=urgent, **opts)
        where = "now (current job pauses)" if urgent else f"position {self.jobs.queued().index(job) + 1}"
        if not urgent and self.current is not None and self.current.origin == "auto":
            where = "next (autopilot job pauses)"
        return f"queued {job.label()} — runs {where}"

    def _steer(self, text: str) -> str:
        if not text:
            return "usage: steer <guidance>"
        self.guidance.append(text)
        if self.current is not None:
            self._fresh_guidance.setdefault(self.current.id, []).append(text)
            return f"steering {self.current.label()} (read before its next step) and every later job: {text[:80]}"
        return f"guidance for every later job: {text[:80]}"

    def _cancel(self, arg: str) -> str:
        job = self.jobs.get(arg) if arg else self.current
        if job is None:
            return "no such job"
        if job is self.current and self._task and not self._task.done():
            job.options["_cancel"] = True
            self._task.cancel()
            return f"cancelling {job.label()}"
        if job.status == "queued":
            self.jobs.mark(job, "cancelled")
            return f"cancelled {job.label()}"
        return f"{job.label()} is {job.status}"

    def _set(self, arg: str) -> str:
        if "=" not in arg:
            return "usage: set <key>=<value>   (see `config` for keys)"
        key, value = (x.strip() for x in arg.split("=", 1))
        if key not in SotaConfig.__dataclass_fields__:
            return f"unknown setting {key!r}; see `config`"
        try:
            new = _coerce(value, getattr(self.settings, key))
        except ValueError as e:
            return str(e)
        self.settings = replace(self.settings, **{key: new})
        if key == "auto":
            self.auto = bool(new)
        if key in WRITER_KEYS:
            self._reconfigure = True
            when = "now" if self.current is None else "when the running job finishes (or use `now` to pause it)"
            return f"{key} = {new!r}; models reload {when}"
        if key in ("sources",):
            from aof.refine.sources import build_sources

            self.sources = build_sources(list(new))
        return f"{key} = {new!r}"

    def _models(self) -> str:
        roles = dict.fromkeys([self.settings.writer, self.settings.judge, self.settings.confirm, "small"])
        loaded = getattr(self.backends, "_backends", {})
        lines = ["role        ctx     loaded  model"]
        for role in roles:
            ctx = self.config.role_context.get(role, self.config.model.n_ctx)
            path = Path(resolve_role_path(self.config, role)).name
            state = "yes" if role in loaded else "no"
            uses = [u for u, r in (("writer", self.settings.writer), ("judge", self.settings.judge),
                                   ("confirm", self.settings.confirm), ("hub names", "small")) if r == role]
            lines.append(f"{role:<11} {ctx:<7} {state:<7} {path}  {'+'.join(uses)}")
        plan = getattr(self.backends, "plan", None)
        if plan is not None and getattr(plan, "roles", None):
            lines += ["", plan.render()]
        if self._reconfigure:
            lines.append("(model settings changed; reload pending)")
        return "\n".join(lines)

    def status(self, full: bool = False) -> str:
        lines = []
        if self.current:
            secs = int(time.time() - self._started_at) if getattr(self, "_started_at", None) else 0
            lines.append(f"running: {self.current.label()} ({secs // 60}m{secs % 60:02d}s) — {self._last_progress[:100]}")
        else:
            lines.append("running: (idle)" + (" [paused]" if self.paused else ""))
        q = self.jobs.queued()
        lines.append(f"queued: {len(q)}" + ("" if not q else " — next: " + ", ".join(j.label() for j in q[:4])))
        if self._side:
            lines.append(f"side tasks: {', '.join(t.get_name() for t in self._side)}")
        lines.append(f"autopilot: {'on' if self.auto else 'off'}; focus: {self.focus or '-'}; paused: {self.paused}")
        lines.append(f"writer: {self.writer_name()} ctx {self.writer_ctx()}; check: {self.settings.check}")
        if self.guidance:
            lines.append("guidance: " + " | ".join(g[:60] for g in self.guidance[-4:]))
        if full:
            lines.append("")
            lines += [f"  {j.label()} [{j.origin}, prio {j.priority}]" for j in q]
            lines += ["recent:"] + [f"  {j.label()} -> {j.status} {j.result[:80] or j.error[:80]}" for j in self.jobs.recent()]
        return "\n".join(lines)

    # ------------------------------------------------------------ side tasks (run beside the current job)

    def _side_task(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._side.add(task)

        def done(t: asyncio.Task) -> None:
            self._side.discard(t)
            if not t.cancelled() and t.exception() is not None:
                self.emit(f"{name} failed: {type(t.exception()).__name__}: {t.exception()}")

        task.add_done_callback(done)

    async def _ask(self, question: str) -> str:
        from aof.refine.report import cited_sentences, clean_citations

        graph = await self.graph()
        budget = max(1000, self.writer_ctx() - 1500)
        pack = await graph.retrieve([question], k=self.settings.ask_evidence, token_budget=budget, per_work=3)
        if not pack:
            self.emit(f"ask: the vault has nothing on: {question}")
            return ""
        evidence = "\n".join(f"[{i}] {e.text} — {e.source_label}" for i, e in enumerate(pack, 1))
        system = (
            "Answer the question from the numbered evidence only (quotes from papers in a curated vault). End every "
            "factual sentence with the numbers it rests on, e.g. [2] or [1, 5]. Say plainly what the evidence does "
            "not cover. Expert, direct prose, 120-350 words."
        )
        guide = ("\nGuidance: " + "; ".join(self.guidance)) if self.guidance else ""
        text = await self.generate(system, f"Question: {question}{guide}\n\nEvidence:\n{evidence}", 900)
        text, _ = clean_citations(text, set(range(1, len(pack) + 1)))
        cited = {n for s, nums in cited_sentences(text) for n in nums}
        refs = "\n".join(f"  [{i}] {e.source_label} — {e.text[:140]}" for i, e in enumerate(pack, 1) if i in cited)
        answer = f"Q: {question}\n\n{text}\n\nReferences:\n{refs}"
        self.emit("answer:\n" + answer)
        with open(self.dir / "answers.md", "a", encoding="utf-8") as f:
            f.write(f"\n## {question}\n\n{text}\n\n" + "\n".join(
                f"- [{i}] {e.source_label}: “{e.text}” {e.source_url}" for i, e in enumerate(pack, 1)) + "\n")
        return answer

    async def _map(self, topic: str) -> str:
        if not topic:
            return "usage: map <topic>"
        graph = await self.graph()
        entries = await graph.vault_map([topic], hubs=10, concepts=20)
        hubs = [f"  {e.title} ({e.size} claims, {e.activation:.2f})" for e in entries if e.kind == "hub"]
        concepts = ", ".join(f"{e.title} ({e.size})" for e in entries if e.kind == "concept")
        return f"topic hubs near '{topic}':\n" + "\n".join(hubs) + f"\nconcepts: {concepts}"

    async def _find(self, text: str) -> str:
        if not text:
            return "usage: find <text>"
        graph = await self.graph()
        pack = await graph.retrieve([text], k=8, per_work=2)
        return "\n".join(f"  {e.score:.2f} {e.text[:180]} — {e.source_label[:60]}" for e in pack) or "(nothing)"

    # ------------------------------------------------------------ job runners

    async def _research(self, question: str, *, tags=()) -> str:
        from aof.refine.pipeline import refine_question
        from aof.refine.vault import question_slug

        s = self.settings
        chain = {p.name for p in self.registry.providers_for("classify")}
        judge = f"role:{s.judge}" if f"role:{s.judge}" in chain else None
        confirm = f"role:{s.confirm}" if f"role:{s.confirm}" in chain else None
        report = await refine_question(
            question, store=self.store, registry=self.registry, sources=self.sources, per_source=s.per_source,
            claims_per_doc=s.claims_per_doc, do_curate=s.curate, curate_only=judge, curate_confirm=confirm,
            verify_top=s.verify_top, concurrency=s.concurrency, standalone_check=True, extra_tags=list(tags),
        )
        done_path = self.root / "refined.json"
        done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
        done[question_slug(question)] = report.summary()
        done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        self._research_since_structure += 1
        self._research_since_report += 1
        summary = report.summary()
        return f"{summary.get('documents', 0)} documents, {summary.get('canonical', 0)} claims stored"

    async def _research_for_report(self, focus: str) -> None:
        # The writer refreshes its graph afterwards, so new claims are retrievable by similarity at once; a full
        # rebuild (hub names from a model) would stall the report, so it is left to `structure` / the autopilot.
        self.progress(f"researching thin section: {focus[:80]}")
        await self._research(focus, tags=["report-gap"])

    async def _deepen(self, question: str) -> str:
        from aof.refine import gaps
        from aof.refine.vault import question_slug

        index = gaps.VaultIndex(self.store, self._embed)
        await index.refresh()
        judge = f"role:{self.settings.confirm}"
        report = await gaps.find_gaps(question, index, self.registry, judge_only=judge)
        self.progress(f"gaps: {report.counts()}")

        async def research(sub: str, parent: str) -> None:
            self.progress(f"researching gap: {sub[:90]}")
            await self._research(sub, tags=[f"gap-of:{question_slug(parent)}"])

        await gaps.fill_gaps(report, index, self.registry, research, judge_only=judge)
        done_path = self.root / "gaps_done.json"
        done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
        done[question_slug(question)] = report.counts()
        done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        with open(self.root / "gaps.md", "a", encoding="utf-8") as f:
            f.write("\n" + gaps.render_markdown([report], title=f"Gaps: {question}"))
        return str(report.counts())

    async def _structure(self, quiet: bool = False) -> str:
        from aof.refine.graph import GraphOptions
        from aof.refine.structure import structure_vault

        options = GraphOptions(link_k=self.settings.graph_link_k, max_hub_size=self.settings.graph_max_hub_size)
        say = (lambda m: None) if quiet else self.progress
        result = await structure_vault(self.store, self.registry, options=options, workspace_root=self.root, progress=say)
        self._research_since_structure = 0
        if self._graph is not None:
            await self._graph.refresh()
        return f"{result['hubs']} hubs, {result['concepts']} concepts, {result['sources']} sources, {result['claim_links']} claim links"

    async def _report(self, job: Job) -> str:
        from aof.refine.report import ReportCancelled, ReportOptions, ReportWriter

        s = self.settings
        kind = job.options.get("kind", s.report_kind)
        options = ReportOptions(
            kind=kind, sections=int(job.options.get("sections", s.report_sections)), ctx_tokens=self.writer_ctx(),
            section_words=s.section_words, section_out_tokens=s.section_out_tokens, evidence_max=s.evidence_max,
            per_work=s.per_work, check=job.options.get("check", s.check), research_thin=s.research_thin,
        )

        async def control() -> list[str]:
            if job.options.get("_cancel"):
                raise ReportCancelled()
            return self._fresh_guidance.pop(job.id, [])

        graph = await self.graph()
        writer = ReportWriter(
            self.store, graph, self.generate, judge=self._judge_fn(s.judge), confirm=self._judge_fn(s.confirm),
            research=self._research_for_report, options=options, workspace_root=self.root, progress=self.progress,
            control=control, writer_name=self.writer_name(), judge_concurrency=s.concurrency,
        )
        fresh = bool(job.options.get("fresh"))
        state = await writer.run(job.arg, kind=kind, guidance=list(self.guidance), resume=not fresh)
        self._research_since_report = 0
        path = self.root / "reports" / (Path(writer.dir_for(job.arg)).name + ".md")
        checks = [d.check.get("final", {}) for d in state.drafts]
        supported = sum(c.get("supported", 0) for c in checks)
        cited = sum(c.get("cited", 0) for c in checks)
        return f"{state.title}: {len(state.plan)} sections, {supported}/{cited} cited sentences supported -> {path}"

    async def _verify(self, n: int) -> str:
        from collections import Counter

        from aof.refine.assess import live_claims
        from aof.refine.verify import apply_verification, verify_note

        rank = {"grade:core": 0, "grade:supporting": 1}
        todo = [x for x in await live_claims(self.store) if x.kind == "claim" and "**Corroboration**" not in x.content]
        todo.sort(key=lambda x: min((rank.get(t, 2) for t in x.tags), default=2))
        gate, verdicts = asyncio.Semaphore(self.settings.concurrency), []

        async def one(note):
            async with gate:
                v = await verify_note(note, registry=self.registry, sources=self.sources, embed=self._embed)
                await apply_verification(self.store, note, v)
                verdicts.append(v.verdict)

        await asyncio.gather(*(one(x) for x in todo[:n]))
        return str(dict(Counter(verdicts)))

    async def _execute(self, job: Job) -> str:
        if job.kind == "research":
            return await self._research(job.arg)
        if job.kind == "deepen":
            return await self._deepen(job.arg)
        if job.kind == "structure":
            return await self._structure()
        if job.kind == "report":
            return await self._report(job)
        if job.kind == "verify":
            return await self._verify(int(job.arg or 50))
        if job.kind == "export":
            from aof.refine.export import export_vault

            counts = await export_vault(self.store, self.root / "export", title=f"{self.root.name} vault")
            return f"exported to {self.root / 'export'}: {counts}"
        if job.kind == "metrics":
            from aof.refine.metrics import compute_metrics

            text = (await compute_metrics(self.store, self.registry)).render()
            self.emit("metrics:\n" + text)
            return "metrics computed"
        if job.kind == "assess":
            return await self._assess()
        raise ValueError(f"unknown job kind {job.kind}")

    async def _assess(self) -> str:
        from aof.refine.assess import assess_vault, render_markdown
        from aof.research.queue_file import load_question_set

        if not self.queue_file:
            return "assess needs --queue-file"
        qset = load_question_set(self.queue_file)
        assessments = await assess_vault(list(qset.questions), self.store, self.registry, judge_only=f"role:{self.settings.confirm}")
        out = self.root / "assessment.md"
        out.write_text(render_markdown(assessments, title=f"Vault assessment: {qset.name}"), encoding="utf-8")
        covered = sum(f.covered for a in assessments for f in a.facts)
        return f"rubric coverage {covered}/{sum(len(a.facts) for a in assessments)} -> {out}"

    # ------------------------------------------------------------ autopilot

    async def _plan_auto(self) -> Job | None:
        from aof.refine.vault import question_slug

        s = self.settings
        if self._research_since_structure >= s.auto_structure_every and not self.jobs.has_pending("structure"):
            return self.jobs.add("structure", origin="auto")
        if s.auto_report_every and self._research_since_report >= s.auto_report_every:
            topic = await self._least_reported_area()
            if topic and not self.jobs.has_pending("report", topic):
                return self.jobs.add("report", topic, origin="auto")
        questions = self._queue_questions()
        done = self._done("refined.json")
        todo = [q for q in questions if question_slug(q) not in done]
        if todo:
            return self.jobs.add("research", await self._closest_to_focus(todo), origin="auto")
        deepened = self._done("gaps_done.json")
        todo = [q for q in questions if question_slug(q) not in deepened]
        if todo:
            return self.jobs.add("deepen", await self._closest_to_focus(todo), origin="auto")
        if self._research_since_structure and not self.jobs.has_pending("structure"):
            return self.jobs.add("structure", origin="auto")
        return None

    def _done(self, name: str) -> dict:
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _queue_questions(self) -> list[str]:
        if not self.queue_file:
            return []
        from aof.research.queue_file import load_question_set

        return [q.question for q in load_question_set(self.queue_file).questions]

    async def _closest_to_focus(self, questions: list[str]) -> str:
        if not self.focus or len(questions) == 1 or self._embed is None:
            return questions[0]
        from aof.refine.vectors import cosine_to

        vecs = await self._embed([self.focus, *questions])
        sims = cosine_to(vecs[0], vecs[1:])
        return questions[int(sims.argmax())]

    async def _least_reported_area(self) -> str:
        hubs = await self.store.get_notes_by_tags(["hub-level:0"], limit=100, exclude_tags=["archived"])
        reports = {n.title.lower() for n in await self.store.get_notes_by_tags(["report"], limit=1000)}
        hubs = [h for h in hubs if h.status != "archived" and h.title.lower() not in reports]
        hubs.sort(key=lambda h: -len(h.links))
        return hubs[0].title if hubs else ""

    # ------------------------------------------------------------ scheduler

    def _load_offset(self) -> int:
        p = self.dir / "inbox.offset"
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    async def poll_inbox(self) -> None:
        """Messages sent with `aof sota send` (or appended to sota/inbox.jsonl by any tool)."""
        if not self.inbox_path.exists():
            return
        with open(self.inbox_path, "rb") as f:
            f.seek(self._inbox_offset)
            data = f.read()
        if not data:
            return
        complete = data[: data.rfind(b"\n") + 1]
        if not complete:
            return
        self._inbox_offset += len(complete)
        (self.dir / "inbox.offset").write_text(str(self._inbox_offset), encoding="utf-8")
        for raw in complete.decode("utf-8", errors="replace").splitlines():
            try:
                text = json.loads(raw).get("text", "")
            except json.JSONDecodeError:
                text = raw
            reply = await self.submit(text, origin="inbox")
            if reply:
                self.emit(reply)

    def _should_preempt(self, running: Job) -> Job | None:
        nxt = self.jobs.next()
        if nxt is None or nxt.priority <= running.priority:
            return None
        return nxt  # 'now' jobs outrank everything; user jobs outrank autopilot jobs

    async def _apply_reconfigure(self) -> None:
        self.emit("reloading models with new settings")
        await self._close_models()
        await self._build_models()
        self._reconfigure = False
        self.emit(f"models ready: writer {self.writer_name()} ctx {self.writer_ctx()}")

    async def run_job(self, job: Job) -> None:
        self.current = job
        self._started_at = time.time()
        self._last_progress = ""
        self.jobs.mark(job, "running")
        self.emit(f"start {job.label()}" + (f" [{job.origin}]" if job.origin != "user" else ""))
        self._task = asyncio.create_task(self._execute(job), name=job.label())
        preempted_by = None
        try:
            while not self._task.done():
                await asyncio.wait({self._task}, timeout=1.0)
                await self.poll_inbox()
                if not self._task.done() and not self.quit and not job.options.get("_cancel"):
                    preempted_by = self._should_preempt(job)
                    if preempted_by is not None:
                        self.emit(f"pausing {job.label()} for {preempted_by.label()}")
                        self._task.cancel()
                        break
            try:
                result = await self._task
            except asyncio.CancelledError:
                result = None
            except Exception as e:
                from aof.refine.report import ReportCancelled

                if isinstance(e, ReportCancelled):
                    result = None
                else:
                    logger.exception("job %s failed", job.label())
                    self.jobs.mark(job, "failed", error=f"{type(e).__name__}: {e}")
                    self.emit(f"failed {job.label()}: {type(e).__name__}: {e}")
                    return
            if result is None:  # cancelled, paused for a more urgent job, or harness quitting
                if job.options.pop("_cancel", None):
                    self.jobs.mark(job, "cancelled")
                    self.emit(f"cancelled {job.label()}")
                else:
                    job.preempted += 1
                    job.status = "queued"
                    self.jobs.save()
                    if not self.quit:
                        self.emit(f"paused {job.label()} (it resumes after more urgent work)")
                return
            self.jobs.mark(job, "done", result=str(result))
            self.emit(f"done {job.label()}: {result}")
        finally:
            self.current = None
            self._task = None
            self._fresh_guidance.pop(job.id, None)

    async def run(self, *, until_idle: bool = False) -> None:
        """Scheduler loop. With `until_idle`, return once nothing is queued (one-shot CLI commands)."""
        while not self.quit:
            await self.poll_inbox()
            if self._reconfigure and self.current is None:
                await self._apply_reconfigure()
            job = None if self.paused else self.jobs.next()
            if job is None and self.auto and not self.paused:
                job = await self._plan_auto()
            if job is None:
                if until_idle and not self._side:
                    return
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            await self.run_job(job)


def send_message(workspace: str, text: str, workspaces_root: str | Path | None = None) -> Path:
    """Append a message to a workspace's harness inbox (picked up within a second by a running harness)."""
    from aof.researcher.workspace import ensure_workspace

    ws = ensure_workspace(workspace, Path(workspaces_root) if workspaces_root else None)
    path = ws.root / "sota" / "inbox.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "text": text}) + "\n")
    return path


def settings_from_args(base: SotaConfig, overrides: dict) -> SotaConfig:
    """CLI overrides (None = not given) applied to the [sota] config."""
    clean = {k: v for k, v in overrides.items() if v is not None and k in SotaConfig.__dataclass_fields__}
    if "sources" in clean and isinstance(clean["sources"], str):
        clean["sources"] = tuple(s.strip() for s in clean["sources"].split(",") if s.strip())
    return replace(base, **clean)
