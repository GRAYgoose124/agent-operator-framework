"""SOTA harness: command language, job queue, settings, inbox, and (with a local model) preemption and steering."""

from __future__ import annotations

import asyncio
import os

import pytest

from aof.config import load_config, resolve_role_path
from aof.sota.commands import parse_command
from aof.sota.harness import Harness, effective_config, send_message, settings_from_args
from aof.sota.jobs import JobQueue

CONFIG = load_config("config.toml")


def test_parse_command_routes_free_text():
    assert parse_command("ask what is the TRN?").name == "ask"
    assert parse_command("What does the TRN do?").name == "ask"  # a question is answered
    c = parse_command("focus more on human studies")
    assert c.name == "focus"
    c = parse_command("Prefer human evidence where it exists")
    assert (c.name, c.arg) == ("steer", "Prefer human evidence where it exists")  # plain text steers
    c = parse_command("report sleep spindles and memory -i --sections 5 --fresh")
    assert (c.name, c.arg) == ("report", "sleep spindles and memory")
    assert c.flags == {"kind": "investigation", "sections": 5, "fresh": True}
    assert parse_command("s").name == "status" and parse_command("now report x").arg == "report x"


def test_job_queue_priority_persistence_and_recovery(tmp_path):
    q = JobQueue(tmp_path / "jobs.json")
    auto = q.add("research", "auto question", origin="auto")
    user = q.add("report", "user topic", kind="investigation")
    urgent = q.add("metrics", urgent=True)
    assert [j.id for j in q.queued()] == [urgent.id, user.id, auto.id]
    assert user.options == {"kind": "investigation"}
    q.mark(urgent, "running")
    again = JobQueue(tmp_path / "jobs.json")  # a harness that died mid-job runs it again
    assert again.get(urgent.id).status == "queued"
    assert again.add("structure").id == "4"  # ids keep counting after a restart
    with pytest.raises(ValueError):
        q.add("nonsense")


def test_effective_config_applies_writer_context_slots_and_thinking():
    s = settings_from_args(CONFIG.sota, {"writer": "large", "writer_ctx": 8192, "writer_parallel": 1,
                                         "writer_thinking": True, "judge": None})
    eff = effective_config(CONFIG, s)
    assert eff.role_context.large == 8192
    ls = eff.llama_server.for_role("large")
    assert ls.n_parallel == 1 and ls.enable_thinking is True
    assert settings_from_args(CONFIG.sota, {"sources": "pubmed, web"}).sources == ("pubmed", "web")


@pytest.fixture
async def harness(tmp_path):
    lines: list[str] = []
    h = Harness(CONFIG, "t", emit=lines.append, workspaces_root=tmp_path)
    await h.start()
    h.lines = lines
    yield h
    await h.close()


async def test_console_commands_act_immediately(harness):
    h = harness
    assert "writer" in await h.submit("models")
    reply = await h.submit("set writer_ctx=8192")
    assert "reload" in reply and h._reconfigure  # model settings reload when no job is using them
    assert h.settings.writer_ctx == 8192
    assert "unknown setting" in await h.submit("set nope=1")
    assert "expected on/off" in await h.submit("set curate=maybe")
    assert "check = 'strict'" in await h.submit("set check=strict")

    assert "guidance for every later job" in await h.submit("Prefer primary research over reviews")
    assert h.guidance == ["Prefer primary research over reviews"]
    assert "focus: thalamus" in await h.submit("focus thalamus")

    reply = await h.submit("report the thalamic reticular nucleus -i")
    assert "queued #1 report" in reply
    job = h.jobs.get("1")
    assert job.options["kind"] == "investigation"
    assert "cancelled #1" in await h.submit("cancel 1")
    assert "running: (idle)" in await h.submit("status")
    assert "usage" in await h.submit("research")
    await h._apply_reconfigure()
    assert h.writer_ctx() == 8192 and not h._reconfigure


async def test_inbox_messages_are_picked_up(harness, tmp_path):
    h = harness
    send_message("t", "steer cite human studies first", workspaces_root=tmp_path)
    send_message("t", "research what is the thalamic reticular nucleus", workspaces_root=tmp_path)
    await h.poll_inbox()
    assert h.guidance == ["cite human studies first"]
    assert h.jobs.next().kind == "research"
    await h.poll_inbox()  # already consumed: nothing happens twice
    assert len(h.jobs.jobs) == 1
    assert any("message (inbox)" in line for line in h.lines)


async def test_autopilot_plans_research_then_graph_then_gaps(tmp_path):
    import json

    from aof.refine.vault import question_slug

    queue = tmp_path / "q.toml"
    queue.write_text(
        '[meta]\nname = "t"\n'
        '[[questions]]\nid = "a"\nquestion = "What does the thalamic reticular nucleus do?"\n'
        '[[questions]]\nid = "b"\nquestion = "How do grid cells support navigation?"\n',
        encoding="utf-8",
    )
    h = Harness(CONFIG, "t", emit=lambda m: None, queue_file=str(queue), workspaces_root=tmp_path)
    await h.start()
    try:
        await h.submit("focus grid cells and navigation")
        job = await h._plan_auto()
        assert (job.kind, job.origin) == ("research", "auto")
        assert job.arg == "How do grid cells support navigation?"  # the focus picks the closest question
        root = tmp_path / "t"
        done = {question_slug(q): {} for q in ("What does the thalamic reticular nucleus do?",
                                                "How do grid cells support navigation?")}
        (root / "refined.json").write_text(json.dumps(done), encoding="utf-8")
        h.jobs.mark(job, "done")
        h._research_since_structure = h.settings.auto_structure_every
        assert (await h._plan_auto()).kind == "structure"  # enough new claims: rebuild the graph first
        h.jobs.mark(h.jobs.next(), "done")
        h._research_since_structure = 0
        job = await h._plan_auto()
        assert job.kind == "deepen" and job.arg == "How do grid cells support navigation?"
        assert h._should_preempt(job) is None
        user = h.jobs.add("report", "anything")
        assert h._should_preempt(job) is user  # a user job pauses an autopilot job
    finally:
        await h.close()


# ---------------------------------------------------------------- real model: preemption, steering, ask

needs_small = pytest.mark.skipif(
    not os.path.exists(resolve_role_path(CONFIG, "small")), reason="local small role model not available"
)


@needs_small
async def test_urgent_job_pauses_a_report_which_then_resumes(tmp_path):
    from dataclasses import replace

    from aof.memory.store import MemoryStore

    settings = replace(CONFIG.sota, writer="small", judge="small", confirm="small", report_sections=3,
                       section_words=120, section_out_tokens=400, research_thin=False, check="cheap")
    lines: list[str] = []
    h = Harness(CONFIG, "t", emit=lines.append, settings=settings, workspaces_root=tmp_path)
    await h.start()
    try:
        store: MemoryStore = h.store
        sentences = [
            "The thalamic reticular nucleus is a shell of GABAergic neurons surrounding the dorsal thalamus.",
            "Sleep spindles are generated by interactions between reticular and thalamocortical neurons.",
            "T-type calcium channels in reticular neurons produce burst firing during spindles.",
            "Reticular neurons receive collaterals from corticothalamic axons.",
            "Optogenetic stimulation of the reticular nucleus induces sleep spindles in mice.",
            "The reticular nucleus gates sensory relay during attention.",
        ]
        for i, s in enumerate(sentences):
            url = f"https://doi.org/10.1000/p{i % 3}"
            await store.create_note(s[:60], f'{s}\n\n**Evidence**\n- "{s}" — Paper {i % 3} (2020), {url}',
                                    ["claim"], note_id=f"c{i}", kind="claim", status="refined", sources=[url])
        await h.submit("report the thalamic reticular nucleus in sleep")
        runner = asyncio.create_task(h.run(until_idle=True))
        while not any("section 1/" in ln for ln in lines):  # wait until the report is writing
            await asyncio.sleep(0.2)
        assert "steering #1" in await h.submit("Keep it short and mention mice")
        await h.submit("now metrics")
        await asyncio.wait_for(runner, timeout=900)
        assert any("pausing #1 report" in ln for ln in lines)
        order = [ln for ln in lines if ln.split("] ", 1)[-1].startswith(("start #", "done #"))]
        assert "start #2 metrics" in order[1] and "done #2" in order[2]  # the urgent job ran in between
        report = h.jobs.get("1")
        assert report.status == "done" and report.preempted == 1
        state = (tmp_path / "t" / "reports" / "the-thalamic-reticular-nucleus-in-sleep" / "state.json").read_text()
        assert "Keep it short and mention mice" in state
        await h._ask("What generates sleep spindles?")
        assert any(ln.split("] ", 1)[-1].startswith("answer:") and "[" in ln for ln in lines)
    finally:
        await h.close()
