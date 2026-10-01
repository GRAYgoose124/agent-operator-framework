"""Long-form reports written from the vault: reviews and insight investigations whose every claim cites vault notes.

The framework owns the loop; the large model only writes, one bounded step at a time:

1. **Map**: graph retrieval around the topic gives the writer the hubs and concepts the vault actually covers.
2. **Outline**: the writer plans sections (title, focus question, search queries). Fallback: the top topic hubs.
3. **Evidence**: per section, graph-aware retrieval (`VaultGraph.retrieve`) fills a numbered evidence pack sized to
   the writer's context window. Thin evidence can trigger gap research (a callback) before writing.
4. **Draft**: the writer drafts the section from its pack only, citing `[n]` after every factual sentence.
5. **Check**: citations to numbers outside the pack are removed; each cited sentence is judged against the evidence it
   cites (cheap judge first; a negative is confirmed by the strong judge, as everywhere in `aof.refine`). Each
   unsupported sentence is repaired on its own against its closest evidence (or deleted); a repair must pass the
   judges too, and sentences that cannot be repaired are marked with a dagger.
6. **Synthesis**: cross-cutting insights and open questions are written from the sections' cited key points, then an
   abstract. Numbers are re-assigned by first appearance and a reference list quotes each cited claim.

State is saved after every step (`<workspace>/reports/<slug>/state.json`), so a stopped or crashed run resumes where
it stopped. Guidance can be added while a report is being written: it is read again before every section.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.refine.retrieve import Evidence, VaultGraph, estimate_tokens
from aof.refine.text import split_sentences

logger = logging.getLogger(__name__)

Generate = Callable[[str, str, int], Awaitable[str]]  # (system, user, max_tokens) -> text
Judge = Callable[[str, str], Awaitable[str]]  # (statement, evidence) -> verdict
Research = Callable[[str], Awaitable[None]]  # focus question -> adds claims to the vault

KINDS = ("review", "investigation")
REPORT_AGENT = "report"
DAGGER = "†"


class ReportCancelled(Exception):
    """Raised by a control hook to stop a report between steps (state is saved; the report can resume)."""


@dataclass
class ReportOptions:
    kind: str = "review"
    sections: int = 7
    ctx_tokens: int = 16384  # the writer's context window
    section_words: int = 650
    section_out_tokens: int = 1800
    evidence_max: int = 45  # evidence items per section (also bounded by the context budget)
    per_work: int = 3
    min_evidence: int = 8  # fewer usable items than this counts as thin evidence
    min_top_score: float = 0.45  # best item scoring below this counts as thin evidence
    check: str = "cheap"  # none | cheap (cheap judge, negatives confirmed) | strict (strong judge for all)
    revise: bool = True
    research_thin: bool = False  # run gap research for sections with thin evidence (needs a research callback)
    synthesis: bool = True
    temperature_note: str = ""  # free-form extra instruction appended to every writer prompt


@dataclass
class SectionPlan:
    title: str
    focus: str
    queries: list[str] = field(default_factory=list)


@dataclass
class SectionDraft:
    title: str
    text: str = ""
    evidence: list[str] = field(default_factory=list)  # claim ids in the pack
    key_points: str = ""
    thin: bool = False
    researched: bool = False
    check: dict = field(default_factory=dict)


@dataclass
class ReportState:
    topic: str
    kind: str = "review"
    title: str = ""
    guidance: list[str] = field(default_factory=list)
    plan: list[SectionPlan] = field(default_factory=list)
    drafts: list[SectionDraft] = field(default_factory=list)
    numbering: dict[str, int] = field(default_factory=dict)  # claim id -> working citation number
    evidence: dict[str, dict] = field(default_factory=dict)  # claim id -> Evidence fields
    synthesis: str = ""
    open_questions: str = ""
    abstract: str = ""
    synthesis_check: dict = field(default_factory=dict)
    stage: str = "new"  # new | planned | sections | synthesis | done
    started: str = ""
    finished: str = ""
    writer: str = ""
    ctx_tokens: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ReportState":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["plan"] = [SectionPlan(**p) for p in data.get("plan", [])]
        data["drafts"] = [SectionDraft(**d) for d in data.get("drafts", [])]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def report_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "report"


def report_note_id(topic: str) -> str:
    """Id of the vault note a finished report is stored as."""
    return "report-" + report_slug(topic)[:48].strip("-")


# ---------------------------------------------------------------- prompts

_OUTLINE_SYSTEM = (
    "You plan a comprehensive {kind_name} written from a curated knowledge vault of quotes from scientific papers. "
    "Given the topic and a map of what the vault covers (topic clusters and concepts, with claim counts), design "
    "{n} sections that together answer the topic thoroughly, in a logical order ({order}). Each section needs a "
    "specific title, a one-sentence focus question, and 2-4 short search queries (3-8 words) that would retrieve the "
    "evidence it needs. Prefer subjects the map shows the vault covers well. Sections must not overlap. Do not plan "
    "an introduction, overview, summary, conclusion or synthesis section: those are written separately.\n"
    "Reply in exactly this format, nothing else:\n"
    "TITLE: <title of the whole {kind_name}>\n"
    "SECTION: <section title>\nFOCUS: <focus question>\nQUERIES: <query> | <query> | <query>\n"
    "(one SECTION/FOCUS/QUERIES block per section)"
)
_ORDER = {
    "review": "foundations and definitions, then mechanisms, then evidence across methods and species, then debates, "
              "implications and synthesis",
    "investigation": "the question and competing hypotheses, then the evidence for each, then mechanisms that could "
                     "decide between them, then limits of the evidence",
}
_KIND_NAME = {"review": "review article", "investigation": "insight investigation"}

_SECTION_SYSTEM = {
    "review": (
        "You are an expert scientific reviewer writing one section of a comprehensive review article. "
        "You write only from the numbered evidence provided: quotes from papers, curated in a knowledge vault."
    ),
    "investigation": (
        "You are an expert analyst writing one section of an insight investigation: a rigorous, question-driven "
        "analysis that weighs evidence for competing explanations. You write only from the numbered evidence "
        "provided: quotes from papers, curated in a knowledge vault."
    ),
}
_SECTION_RULES = (
    "\nRules:\n"
    "- End every factual sentence with the numbers of the evidence it rests on, in square brackets: [12] or [3, 17]. "
    "Never cite a number that is not in the evidence list.\n"
    "- Do not state anything no evidence item supports. Where evidence is thin, indirect or conflicting, say so.\n"
    "- Synthesize: connect findings, explain mechanisms, compare studies, methods and species, weigh agreement and "
    "disagreement. Do not walk through the evidence one item per sentence.\n"
    "- Expert prose in Markdown paragraphs; ### subheadings are fine; bullets only for short enumerations.\n"
    "- Stay within this section's focus; the outline shows what other sections cover. No introduction or conclusion "
    "for the whole document.\n"
    "- About {words} words. Start directly with the prose, without a heading."
)
_KEYPOINTS_SYSTEM = (
    "Summarize the section below as 3-5 bullet key points for the author of later sections. Keep the citation "
    "numbers in square brackets exactly as they appear. One line per bullet, no preamble."
)
_FIX_SYSTEM = (
    "You fix one sentence of a scientific text that its cited evidence does not support. Rewrite it so that the "
    "evidence below supports it: correct the citation numbers, or narrow the statement to what the evidence says. "
    "End it with the supporting numbers in square brackets, using only numbers from the list. Reply with the one "
    "sentence only. If no item supports any version of it, reply exactly: DELETE"
)
_SYNTH_SYSTEM = (
    "You write the synthesis of a {kind_name}: the insights that only appear when the sections are read together. "
    "You are given each section's key points with citation numbers. Identify cross-cutting principles, tensions and "
    "contradictions between lines of evidence, candidate unifying models, and what the combined evidence implies. "
    "Every factual sentence ends with citation numbers taken from the key points, e.g. [4, 19]. Do not introduce "
    "facts that are not in the key points. Expert prose, about {words} words, optionally with ### subheadings."
)
_OPEN_SYSTEM = (
    "List the most important open questions this {kind_name} leaves unresolved: 5-8 bullets, each one or two "
    "sentences saying what is unknown and why it matters, and what kind of evidence would settle it. Write about the "
    "science itself; never refer to 'the review', 'the article' or 'the text'. Where the key points show the gap, "
    "cite their numbers in square brackets. No preamble."
)
_ABSTRACT_SYSTEM = (
    "Write the abstract of a {kind_name}: 150-230 words, one paragraph, stating the scope, the main findings and the "
    "central insight, without citations. No heading, no preamble."
)


# ---------------------------------------------------------------- parsing helpers

_CITE_RE = re.compile(r"\s?\[(\d+(?:\s*[,;–\-]\s*\d+)*)\]")


def _numbers(group: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*[,;]\s*", group):
        if re.fullmatch(r"\d+\s*[–\-]\s*\d+", part):
            a, b = (int(x) for x in re.split(r"\s*[–\-]\s*", part))
            if 0 < b - a <= 20:
                out.extend(range(a, b + 1))
                continue
        if part.strip().isdigit():
            out.append(int(part))
    return out


def clean_citations(text: str, allowed: set[int]) -> tuple[str, int]:
    """Drop citation numbers outside `allowed`; returns (text, number of citations removed)."""
    removed = 0

    def sub(m: re.Match) -> str:
        nonlocal removed
        nums = _numbers(m.group(1))
        keep = [x for x in dict.fromkeys(nums) if x in allowed]
        removed += len(nums) - len(keep)
        return f" [{', '.join(map(str, keep))}]" if keep else ""

    return _CITE_RE.sub(sub, text), removed


def cited_sentences(text: str) -> list[tuple[str, list[int]]]:
    """(sentence, cited numbers) for every sentence in the prose (headings and bullets are split too)."""
    out = []
    for block in re.split(r"\n\s*\n|\n(?=#{1,6} |[-*] )", text):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        for sent in split_sentences(block.lstrip("-* ")):
            nums = [x for m in _CITE_RE.finditer(sent) for x in _numbers(m.group(1))]
            out.append((sent, list(dict.fromkeys(nums))))
    return out


def parse_outline(text: str, limit: int) -> tuple[str, list[SectionPlan]]:
    title = ""
    plans: list[SectionPlan] = []
    for raw in text.splitlines():
        line = raw.strip().strip("*").strip()
        m = re.match(r"^(TITLE|SECTION|FOCUS|QUERIES)\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if not m:
            continue
        key, value = m.group(1).upper(), m.group(2).strip().strip('"')
        if key == "TITLE" and not title:
            title = value
        elif key == "SECTION":
            plans.append(SectionPlan(re.sub(r"^\d+[.)]\s*", "", value), ""))
        elif plans and key == "FOCUS":
            plans[-1].focus = value
        elif plans and key == "QUERIES":
            plans[-1].queries = [q.strip().strip('"') for q in re.split(r"\s*[|;]\s*", value) if 3 <= len(q.strip()) <= 120][:4]
    plans = [p for p in plans if p.title and len(p.title) <= 140]
    # the synthesis, abstract and open questions are written separately; a planned one would duplicate them
    content = [p for p in plans if not _FRAME_SECTION.search(p.title)]
    if len(content) >= 3:
        plans = content
    for p in plans:
        p.focus = p.focus or p.title
    return title, plans[:limit]


_FRAME_SECTION = re.compile(r"^(?:\d+\.\s*)?(?:introduction|overview|summary|conclusions?|synthesis|concluding)\b|"
                            r"\b(?:synthesis|conclusions?)\b", re.IGNORECASE)
# First-person self-descriptions and chat preambles ("I am Qwythos, an AI model created by ...", "Sure, here is ...")
_PREAMBLE = re.compile(
    r"^\s*(?:(?:I am|I'm)\b[^.\n]*\.|As an? (?:AI|language model)\b[^.\n]*\.|(?:Sure|Certainly|Of course)\b[^.\n]*[.:!]|"
    r"Here(?: is|'s)\b[^.\n]*[.:])\s*",
    re.IGNORECASE,
)


def _strip_heading(text: str, title: str) -> str:
    lines = text.strip().splitlines()
    while lines and (lines[0].startswith("#") and (title.lower()[:20] in lines[0].lower() or len(lines[0]) < 120)):
        lines = lines[1:]
    return clean_prose("\n".join(lines))


def clean_prose(text: str) -> str:
    """Remove chat preambles / self-identification, repeated sentences and stray paragraph indentation."""
    text = text.strip()
    while True:
        new = _PREAMBLE.sub("", text, count=1)
        if new == text:
            break
        text = new
    seen: set[str] = set()
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para.startswith(("#", "-", "*", "|")) or not para:
            paragraphs.append(para)
            continue
        kept = []
        for sent in split_sentences(para):
            key = re.sub(r"\W+", " ", _CITE_RE.sub("", sent)).strip().lower()
            if key and key in seen:
                continue  # the writer repeated itself verbatim
            seen.add(key)
            kept.append(sent)
        paragraphs.append(" ".join(kept))
    return "\n\n".join(p for p in paragraphs if p).strip()


# ---------------------------------------------------------------- the writer

class ReportWriter:
    def __init__(
        self,
        store: MemoryStore,
        graph: VaultGraph,
        generate: Generate,
        *,
        judge: Judge | None = None,
        confirm: Judge | None = None,
        research: Research | None = None,
        options: ReportOptions | None = None,
        workspace_root: str | Path = ".",
        progress: Callable[[str], None] | None = None,
        control: Callable[[], Awaitable[list[str]]] | None = None,
        writer_name: str = "",
        judge_concurrency: int = 4,
    ) -> None:
        self.store, self.graph, self.generate = store, graph, generate
        self.judge, self.confirm, self.research = judge, confirm, research
        self.opt = options or ReportOptions()
        if self.opt.kind not in KINDS:
            raise ValueError(f"report kind must be one of {KINDS}")
        self.root = Path(workspace_root)
        self.say = progress or (lambda m: logger.info("report: %s", m))
        self.control = control  # returns new guidance lines; raises ReportCancelled to stop
        self.writer_name = writer_name
        self._gate = asyncio.Semaphore(max(1, judge_concurrency))

    # ------------------------------------------------------------ paths & state

    def dir_for(self, topic: str) -> Path:
        return self.root / "reports" / report_slug(topic)

    def _save(self, state: ReportState) -> None:
        state.save(self.dir_for(state.topic) / "state.json")

    async def _checkpoint(self, state: ReportState) -> None:
        self._save(state)
        if self.control is not None:
            for line in await self.control():  # may raise ReportCancelled
                if line and line not in state.guidance:
                    state.guidance.append(line)
                    self.say(f"guidance added: {line[:80]}")

    def _kind_name(self, state: ReportState) -> str:
        return _KIND_NAME[state.kind]

    def _guidance_block(self, state: ReportState) -> str:
        lines = [*state.guidance]
        if self.opt.temperature_note:
            lines.append(self.opt.temperature_note)
        return ("\n\nGuidance from the user (follow it):\n" + "\n".join(f"- {g}" for g in lines)) if lines else ""

    # ------------------------------------------------------------ steps

    async def _plan(self, state: ReportState) -> None:
        topic = state.topic
        vmap = await self.graph.vault_map([topic], hubs=14, concepts=30)
        hubs = [m for m in vmap if m.kind == "hub"]
        concepts = [m for m in vmap if m.kind == "concept"]
        listing = "Topic clusters:\n" + "\n".join(f"- {m.title} ({m.size} claims)" for m in hubs)
        listing += "\n\nConcepts:\n" + ", ".join(f"{m.title} ({m.size})" for m in concepts)
        system = _OUTLINE_SYSTEM.format(
            kind_name=self._kind_name(state), n=self.opt.sections, order=_ORDER[state.kind],
        )
        user = f"Topic: {topic}{self._guidance_block(state)}\n\nWhat the vault covers:\n{listing}\n\nPlan the sections."
        text = await self.generate(system, user, 900)
        title, plans = parse_outline(text, self.opt.sections)
        if len(plans) < 3:  # the writer did not follow the format: fall back to the vault's own topic structure
            self.say("outline unusable; falling back to the vault's topic hubs")
            plans = [
                SectionPlan(m.title, f"What does the evidence show about {m.title.lower()} in relation to: {topic}?",
                            [m.title, f"{m.title} {topic}"[:120]])
                for m in hubs[: self.opt.sections]
            ]
        state.title = title or topic
        state.plan = plans
        state.drafts = [SectionDraft(p.title) for p in plans]
        state.stage = "planned"
        self.say(f"outline: {state.title} | " + " / ".join(p.title for p in plans))

    def _outline_text(self, state: ReportState, current: int) -> str:
        return "\n".join(
            f"{i + 1}. {p.title} — {p.focus}" + ("  <- you are writing this one" if i == current else "")
            for i, p in enumerate(state.plan)
        )

    def _number(self, state: ReportState, ev: Evidence) -> int:
        if ev.note_id not in state.numbering:
            state.numbering[ev.note_id] = len(state.numbering) + 1
            state.evidence[ev.note_id] = asdict(ev)
        return state.numbering[ev.note_id]

    def _evidence_line(self, state: ReportState, ev: Evidence) -> str:
        tag = "; corroborated" if ev.corroborated else ""
        return f"[{self._number(state, ev)}] {ev.text} — {ev.source_label}{tag}"

    def _budget(self, state: ReportState, i: int) -> int:
        """Tokens left for the evidence pack in section `i`'s prompt."""
        fixed = estimate_tokens(_SECTION_SYSTEM[state.kind] + _SECTION_RULES) + 200
        fixed += estimate_tokens(self._outline_text(state, i)) + estimate_tokens(self._guidance_block(state))
        fixed += sum(estimate_tokens(d.key_points) for d in state.drafts[:i])
        return max(800, self.opt.ctx_tokens - self.opt.section_out_tokens - fixed - 300)

    async def _pack(self, state: ReportState, i: int) -> list[Evidence]:
        plan = state.plan[i]
        used = {cid for d in state.drafts[:i] for cid in d.evidence}
        queries = [plan.focus, plan.title, *plan.queries]
        return await self.graph.retrieve(
            queries, k=self.opt.evidence_max, token_budget=self._budget(state, i), per_work=self.opt.per_work,
            used=used,
        )

    async def _section(self, state: ReportState, i: int) -> None:
        plan, draft = state.plan[i], state.drafts[i]
        pack = await self._pack(state, i)
        thin = len(pack) < self.opt.min_evidence or (pack and pack[0].score < self.opt.min_top_score) or not pack
        if thin and self.opt.research_thin and self.research is not None and not draft.researched:
            self.say(f"section {i + 1}: thin evidence ({len(pack)} items); researching: {plan.focus[:80]}")
            await self.research(plan.focus)
            draft.researched = True
            await self.graph.refresh()
            pack = await self._pack(state, i)
            thin = len(pack) < self.opt.min_evidence or (pack and pack[0].score < self.opt.min_top_score) or not pack
        draft.thin = bool(thin)
        draft.evidence = [e.note_id for e in pack]
        evidence_block = "\n".join(self._evidence_line(state, e) for e in pack) or "(no evidence found)"
        earlier = "\n".join(f"{j + 1}. {d.title}\n{d.key_points}" for j, d in enumerate(state.drafts[:i]) if d.key_points)
        system = _SECTION_SYSTEM[state.kind] + _SECTION_RULES.format(words=self.opt.section_words)
        user = (
            f"Document: {state.title}\nOverall topic: {state.topic}\n\nOutline:\n{self._outline_text(state, i)}"
            f"{self._guidance_block(state)}\n\n"
            + (f"What earlier sections established (do not repeat):\n{earlier}\n\n" if earlier else "")
            + f"Section {i + 1}: {plan.title}\nFocus: {plan.focus}\n\nEvidence:\n{evidence_block}\n\n"
            + f"Write section {i + 1} now."
        )
        self.say(f"section {i + 1}/{len(state.plan)}: drafting '{plan.title}' from {len(pack)} evidence items"
                 + (" (thin)" if thin else ""))
        text = _strip_heading(await self.generate(system, user, self.opt.section_out_tokens), plan.title)
        allowed = {state.numbering[e.note_id] for e in pack}
        text, removed = clean_citations(text, allowed)
        text, restated = await self._drop_restatements(state, i, text)
        draft.text = text
        draft.check = {"invalid_citations_removed": removed, "uncited_restatements_removed": restated}
        if self.opt.check != "none" and self.judge is not None:
            await self._check_and_revise(state, draft, evidence_block, allowed)
        kp = await self.generate(_KEYPOINTS_SYSTEM, f"Section: {plan.title}\n\n{draft.text}", 400)
        draft.key_points = clean_citations(kp.strip(), allowed)[0]

    async def _drop_restatements(self, state: ReportState, i: int, text: str, threshold: float = 0.85) -> tuple[str, int]:
        """Remove *uncited* sentences that restate an earlier section (writers like to recap before starting).

        Cited sentences are kept: re-using evidence in a new argument is legitimate; an uncited recap is not."""
        earlier = [s for d in state.drafts[:i] for s, _ in cited_sentences(d.text)]
        sents = cited_sentences(text)
        uncited = [s for s, nums in sents if not nums and len(s) > 40]
        if not earlier or not uncited:
            return text, 0
        try:
            vecs = await self.graph.embed([*uncited, *earlier])
        except Exception:
            return text, 0
        u = np.asarray(vecs[: len(uncited)], dtype=np.float32)
        e = np.asarray(vecs[len(uncited):], dtype=np.float32)
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        drop = [s for s, sim in zip(uncited, (u @ e.T).max(axis=1)) if sim >= threshold]
        for s in drop:
            text = text.replace(s, "", 1)
        return clean_prose(re.sub(r"[ \t]{2,}", " ", text)), len(drop)

    # ------------------------------------------------------------ support check

    def _evidence_for(self, state: ReportState, nums: Sequence[int]) -> str:
        by_num = {n: cid for cid, n in state.numbering.items()}
        return "\n".join(f"- {state.evidence[by_num[n]]['text']}" for n in nums if n in by_num)

    async def _judge_sentence(self, state: ReportState, sentence: str, nums: Sequence[int]) -> str:
        statement = _CITE_RE.sub("", sentence).strip()
        evidence = self._evidence_for(state, nums)
        if not evidence:
            return "unsupported"
        async with self._gate:
            first = self.confirm if (self.opt.check == "strict" and self.confirm) else self.judge
            verdict = await first(statement, evidence)
            if verdict != "supported" and self.confirm is not None and first is not self.confirm:
                verdict = await self.confirm(statement, evidence)  # negatives always get the strong judge
        return verdict

    async def _check(self, state: ReportState, text: str) -> tuple[dict, list[str]]:
        sents = cited_sentences(text)
        cited = [(s, n) for s, n in sents if n]
        verdicts = await asyncio.gather(*(self._judge_sentence(state, s, n) for s, n in cited))
        bad = [s for (s, _), v in zip(cited, verdicts) if v != "supported"]
        stats = {
            "sentences": len(sents), "cited": len(cited), "uncited": len(sents) - len(cited),
            "supported": sum(v == "supported" for v in verdicts), "unsupported": len(bad),
        }
        return stats, bad

    async def _fix_sentence(self, state: ReportState, sentence: str, allowed: set[int]) -> str | None:
        """A supported rewrite of `sentence` (checked by the judges), "" to delete it, or None if it cannot be fixed.

        Sentence-level repair: the writer sees the sentence with its closest evidence (cited items plus the most
        similar items of the section's pack), which is short and reliable; rewriting whole sections was not (the
        writer returned fragments, which had to be discarded).
        """
        by_num = {n: cid for cid, n in state.numbering.items()}
        pool = [n for n in sorted(allowed) if n in by_num]
        cited = [x for m in _CITE_RE.finditer(sentence) for x in _numbers(m.group(1))]
        texts = [state.evidence[by_num[n]]["text"] for n in pool]
        try:
            vecs = await self.graph.embed([_CITE_RE.sub("", sentence), *texts])
            sims = (np.asarray(vecs[1:]) @ np.asarray(vecs[0])).tolist()
            nearest = [n for _, n in sorted(zip(sims, pool), reverse=True)[:8]]
        except Exception:  # no embedder: the cited items and the first few of the pack
            nearest = pool[:8]
        shown = list(dict.fromkeys([*cited, *nearest]))
        evidence = "\n".join(f"[{n}] {state.evidence[by_num[n]]['text']}" for n in shown if n in by_num)
        raw = (await self.generate(_FIX_SYSTEM, f"Evidence:\n{evidence}\n\nSentence: {sentence}", 220)).strip()
        line = raw.splitlines()[0].strip() if raw else ""
        if line.upper().startswith("DELETE"):
            return ""
        fixed, _ = clean_citations(line, set(shown))
        nums = [x for m in _CITE_RE.finditer(fixed) for x in _numbers(m.group(1))]
        if not nums or len(fixed) < 20:
            return None
        return fixed if await self._judge_sentence(state, fixed, nums) == "supported" else None

    async def _check_and_revise(self, state: ReportState, draft: SectionDraft, evidence_block: str, allowed: set[int]) -> None:
        stats, bad = await self._check(state, draft.text)
        draft.check.update({"first_pass": stats})
        if bad and self.opt.revise:
            self.say(f"  support check: {stats['supported']}/{stats['cited']} supported; repairing {len(bad)} sentence(s)")
            fixed = deleted = 0
            still: list[str] = []
            for sentence in bad:
                repl = await self._fix_sentence(state, sentence, allowed)
                if repl is None:
                    still.append(sentence)
                elif repl == "":
                    draft.text = draft.text.replace(sentence, "", 1)
                    deleted += 1
                else:
                    draft.text = draft.text.replace(sentence, repl, 1)
                    fixed += 1
            draft.text = re.sub(r"  +", " ", draft.text).replace(" \n", "\n")
            stats = {**stats, "supported": stats["supported"] + fixed, "unsupported": len(still),
                     "repaired": fixed, "deleted": deleted, "cited": stats["cited"] - deleted}
            bad = still
        draft.text = self._mark(draft.text, bad)
        draft.check["final"] = stats
        self.say(f"  support check: {stats['supported']}/{stats['cited']} cited sentences supported"
                 + (f", {len(bad)} marked {DAGGER}" if bad else ""))

    @staticmethod
    def _mark(text: str, bad: Sequence[str]) -> str:
        for s in bad:
            if s in text and not s.endswith(DAGGER):
                text = text.replace(s, s + DAGGER, 1)
        return text

    # ------------------------------------------------------------ synthesis

    async def _synthesis(self, state: ReportState) -> None:
        points = "\n\n".join(f"{i + 1}. {d.title}\n{d.key_points}" for i, d in enumerate(state.drafts) if d.key_points)
        allowed = set(state.numbering.values())
        kind_name = self._kind_name(state)
        base = f"Document: {state.title}\nTopic: {state.topic}{self._guidance_block(state)}\n\nKey points by section:\n{points}"
        self.say("writing synthesis")
        synth = await self.generate(_SYNTH_SYSTEM.format(kind_name=kind_name, words=500), base, 1500)
        state.synthesis = clean_citations(_strip_heading(synth, "Synthesis"), allowed)[0]
        if self.opt.check != "none" and self.judge is not None:
            stats, bad = await self._check(state, state.synthesis)
            state.synthesis = self._mark(state.synthesis, bad)
            state.synthesis_check = stats
        thin = [d.title for d in state.drafts if d.thin]
        extra = ("\n\nSections the vault could only support thinly: " + "; ".join(thin)) if thin else ""
        self.say("writing open questions")
        state.open_questions = clean_citations(
            _strip_heading(await self.generate(_OPEN_SYSTEM.format(kind_name=kind_name), base + extra, 800), "Open"),
            allowed,
        )[0]
        self.say("writing abstract")
        state.abstract = _CITE_RE.sub("", _strip_heading(await self.generate(
            _ABSTRACT_SYSTEM.format(kind_name=kind_name), base + f"\n\nSynthesis:\n{state.synthesis}", 500,
        ), "Abstract"))

    # ------------------------------------------------------------ run

    async def run(self, topic: str, *, kind: str | None = None, guidance: Sequence[str] = (), resume: bool = True) -> ReportState:
        path = self.dir_for(topic) / "state.json"
        state = ReportState.load(path) if (resume and path.exists()) else ReportState(topic, kind or self.opt.kind)
        if state.stage == "done" and resume:
            self.say("report already finished; re-rendering (use a fresh run to rewrite)")
        state.kind = kind or state.kind
        for g in guidance:
            if g not in state.guidance:
                state.guidance.append(g)
        state.writer, state.ctx_tokens = self.writer_name, self.opt.ctx_tokens
        state.started = state.started or datetime.now(timezone.utc).isoformat(timespec="seconds")
        if state.stage == "new":
            await self._plan(state)
            await self._checkpoint(state)
        for i in range(len(state.plan)):
            if state.drafts[i].text:
                continue
            state.stage = "sections"
            await self._section(state, i)
            await self._checkpoint(state)
        if self.opt.synthesis and state.stage != "done" and not state.abstract:
            state.stage = "synthesis"
            await self._synthesis(state)
        state.stage = "done"
        state.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save(state)
        await self.publish(state)
        return state

    # ------------------------------------------------------------ rendering

    def render(self, state: ReportState, *, wikilinks: bool) -> str:
        """Markdown for the finished report. Citation numbers are re-assigned in order of first appearance."""
        parts = [state.abstract, *(d.text for d in state.drafts), state.synthesis, state.open_questions]
        order: dict[int, int] = {}
        for part in parts:
            for m in _CITE_RE.finditer(part or ""):
                for x in _numbers(m.group(1)):
                    order.setdefault(x, len(order) + 1)

        def renumber(text: str) -> str:
            def sub(m: re.Match) -> str:
                nums = sorted({order[x] for x in _numbers(m.group(1)) if x in order})
                return f" [{', '.join(map(str, nums))}]" if nums else ""
            return _CITE_RE.sub(sub, text or "")

        by_num = {n: cid for cid, n in state.numbering.items()}
        checks = [d.check.get("final", {}) for d in state.drafts if d.check.get("final")]
        if state.synthesis_check:
            checks.append(state.synthesis_check)
        cited = sum(c.get("cited", 0) for c in checks)
        supported = sum(c.get("supported", 0) for c in checks)
        works = {state.evidence[by_num[o]]["source_label"] for o in order if o in by_num}
        date = (state.finished or state.started)[:10]
        kind_name = _KIND_NAME[state.kind].capitalize()
        meta = f"*{kind_name} written from the knowledge vault ({date}): {len(order)} cited claims from {len(works)} works"
        if cited:
            meta += f"; {supported}/{cited} cited sentences confirmed against their evidence"
        meta += f". Writer: {state.writer or 'n/a'}, context {state.ctx_tokens} tokens.*"
        lines = [f"# {state.title}", "", meta, ""]
        if state.abstract:
            lines += ["## Abstract", "", state.abstract.strip(), ""]
        lines += ["## Contents", ""]
        lines += [f"{i + 1}. {p.title}" for i, p in enumerate(state.plan)]
        if state.synthesis:
            lines.append(f"{len(state.plan) + 1}. Synthesis and insights")
        lines.append("")
        for i, (p, d) in enumerate(zip(state.plan, state.drafts)):
            lines += [f"## {i + 1}. {p.title}", ""]
            if d.thin:
                lines += ["> Evidence in the vault for this section is thin; treat it as provisional.", ""]
            lines += [renumber(d.text).strip(), ""]
        if state.synthesis:
            lines += [f"## {len(state.plan) + 1}. Synthesis and insights", "", renumber(state.synthesis).strip(), ""]
        if state.open_questions:
            lines += ["## Open questions", "", renumber(state.open_questions).strip(), ""]
        lines += ["## How this was made", "",
                  f"Sections were planned from the vault's topic hubs and concepts; each was written only from a "
                  f"numbered evidence pack retrieved through the vault graph (similarity plus concept and topic "
                  f"structure, at most {self.opt.per_work} claims per work). Every cited sentence was judged against "
                  f"the claims it cites (a small judge first; every negative confirmed by a stronger one). Failing "
                  f"sentences were rewritten against their closest evidence and re-judged, or removed; any that "
                  f"could not be repaired are marked {DAGGER}.", ""]
        if state.guidance:
            lines += ["Guidance followed: " + "; ".join(state.guidance), ""]
        lines += ["## References", ""]
        for old, new in sorted(order.items(), key=lambda kv: kv[1]):
            cid = by_num.get(old)
            if cid is None:
                continue
            ev = state.evidence[cid]
            link = f" [[{cid}|→ note]]" if wikilinks else ""
            lines.append(f"{new}. “{ev['text']}” — {ev['source_label']}. {ev['source_url']}{link}")
        return "\n".join(lines).rstrip() + "\n"

    async def publish(self, state: ReportState) -> Path:
        """Write the report file and store it as a `report` note linked to every claim it cites."""
        out = self.root / "reports" / f"{report_slug(state.topic)}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(state, wikilinks=False), encoding="utf-8")
        content = self.render(state, wikilinks=True)
        body = content.split("\n", 1)[1].strip()  # the note's title carries the H1
        cited = [cid for cid in state.numbering if f"[[{cid}|" in content]
        note_id = report_note_id(state.topic)
        existing = await self.store.get_note(note_id)
        tags = ["report", f"report-kind:{state.kind}"]
        sources = list(dict.fromkeys(state.evidence[c]["source_url"] for c in cited))
        if existing is None:
            await self.store.create_note(state.title, body, tags, links=cited, agent_id=REPORT_AGENT, note_id=note_id,
                                         kind="report", status="canonical", sources=sources)
        else:
            existing.title, existing.content, existing.links, existing.sources = state.title, body, cited, sources
            existing.tags, existing.status, existing.kind = tags, "canonical", "report"
            await self.store.update_note(existing)
        self.say(f"report written: {out}")
        return out
