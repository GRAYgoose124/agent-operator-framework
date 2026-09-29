"""`aof refine ...`: build a curated vault from a question set, and assess it against the set's rubric."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aof.config import AppConfig, config_with_workspace
from aof.refine.sources import SOURCE_CLASSES

logger = logging.getLogger(__name__)


def add_refine_parser(subparsers, common: argparse.ArgumentParser) -> None:
    refine_p = subparsers.add_parser(
        "refine", help="Evidence-first vault refinement (see docs/VAULT_OVERHAUL.md)", parents=[common]
    )
    sub = refine_p.add_subparsers(dest="refine_command")

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workspace", required=True, help="Workspace name (data/workspaces/<name>)")
        p.add_argument("--queue-file", required=True, help="TOML question set (see examples/queues/)")
        p.add_argument("--ids", default="", help="Comma-separated question ids to include (default: all)")

    run_p = sub.add_parser("run", help="Gather, extract, merge, curate and verify claims for each question")
    shared(run_p)
    run_p.add_argument("--sources", default="pubmed,openalex,wikipedia",
                       help=f"Comma-separated evidence sources from: {', '.join(sorted(SOURCE_CLASSES))}")
    run_p.add_argument("--per-source", type=int, default=5, help="Documents requested per source per query")
    run_p.add_argument("--claims-per-doc", type=int, default=6)
    run_p.add_argument("--topics", default="", help="Optional comma-separated topic labels to tag claims with")
    run_p.add_argument("--curate", action="store_true", help="Grade claims and archive irrelevant ones")
    run_p.add_argument("--curate-with", default="role:small", help="First-pass curation provider")
    run_p.add_argument("--confirm-with", default="role:large",
                       help="Provider that confirms every 'irrelevant' before archiving ('' to disable)")
    run_p.add_argument("--verify-top", type=int, default=0, help="Verify the N most relevant claims per question")
    run_p.add_argument("--concurrency", type=int, default=1,
                       help="Concurrent model calls for curation/classification (match [llama_server] n_parallel)")
    run_p.add_argument("--force", action="store_true", help="Re-run questions that already have notes")

    def workspace_only(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workspace", required=True, help="Workspace name (data/workspaces/<name>)")

    structure_p = sub.add_parser("structure", help="Link related claims and build hub notes")
    workspace_only(structure_p)
    structure_p.add_argument("--link-min-sim", type=float, default=0.55)
    structure_p.add_argument("--hub-threshold", type=float, default=0.5)
    structure_p.add_argument("--max-hub-size", type=int, default=60, help="Split hubs larger than this into subtopics")

    repair_p = sub.add_parser("repair", help="Fix known defects in vaults written by earlier versions")
    workspace_only(repair_p)

    verify_p = sub.add_parser("verify", help="Look up unverified claims in independent works (core-graded first)")
    workspace_only(verify_p)
    verify_p.add_argument("--top", type=int, default=100, help="How many unverified claims to verify")
    verify_p.add_argument("--concurrency", type=int, default=4)
    verify_p.add_argument("--sources", default="pubmed,openalex,wikipedia")

    metrics_p = sub.add_parser("metrics", help="Vault quality metrics (duplicates, orphans, sourcing, verification)")
    workspace_only(metrics_p)

    export_p = sub.add_parser("export", help="Export the vault as an Obsidian-style markdown folder")
    workspace_only(export_p)
    export_p.add_argument("--out", default="", help="Output folder (default: <workspace>/export)")
    export_p.add_argument("--no-archive", action="store_true", help="Omit archived (merged / curated-out) claims")

    assess_p = sub.add_parser("assess", help="Score the vault against the question set's rubric")
    shared(assess_p)
    assess_p.add_argument("--judge-with", default="role:large", help="Provider used to judge coverage ('' = chain)")
    assess_p.add_argument("--out", default="", help="Markdown report path (default: <workspace>/assessment.md)")


def _select(qset, ids: str):
    wanted = {i.strip() for i in ids.split(",") if i.strip()}
    if not wanted:
        return list(qset.questions)
    unknown = wanted - {q.id for q in qset.questions}
    if unknown:
        raise SystemExit(f"unknown question id(s): {', '.join(sorted(unknown))}")
    return [q for q in qset.questions if q.id in wanted]


async def run_refine_command(args: argparse.Namespace, config: AppConfig) -> None:
    from aof.memory.store import MemoryStore
    from aof.refine.assess import assess_vault, render_markdown
    from aof.refine.pipeline import refine_question
    from aof.refine.sources import build_sources
    from aof.refine.vault import question_slug
    from aof.research.queue_file import load_question_set
    from aof.researcher.workspace import ensure_workspace
    from aof.specialists import build_registry
    from aof.specialists.backends import RoleBackends

    if args.refine_command not in ("run", "assess", "structure", "metrics", "export", "repair", "verify"):
        print("Use: aof refine run|assess|structure|metrics|export|repair|verify --workspace NAME [--queue-file PATH]")
        return
    workspace = ensure_workspace(args.workspace)
    config = config_with_workspace(config, workspace.root)
    qset = specs = None
    if hasattr(args, "queue_file"):
        qset = load_question_set(args.queue_file)
        specs = _select(qset, args.ids)

    store = MemoryStore(config.memory)
    await store.initialize()
    backends = RoleBackends(config)
    registry = build_registry(config, backends.get)
    try:
        if args.refine_command == "run":
            sources = build_sources([s.strip() for s in args.sources.split(",") if s.strip()])
            topics = [t.strip() for t in args.topics.split(",") if t.strip()]
            done_path = workspace.root / "refined.json"  # completion marker per question (resumable runs)
            done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
            for i, spec in enumerate(specs, 1):
                slug = question_slug(spec.question)
                if not args.force and slug in done:
                    print(f"[{i}/{len(specs)}] {spec.id}: already refined (use --force to redo)")
                    continue
                report = await refine_question(
                    spec.question, store=store, registry=registry, sources=sources, topics=topics,
                    per_source=args.per_source, claims_per_doc=args.claims_per_doc,
                    do_curate=args.curate, curate_only=args.curate_with or None,
                    curate_confirm=args.confirm_with or None, verify_top=args.verify_top,
                    concurrency=args.concurrency,
                )
                done[slug] = report.summary()
                done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
                print(f"[{i}/{len(specs)}] {spec.id}: {report.summary()}", flush=True)
            print("Specialist usage:", registry.summary())
        elif args.refine_command == "structure":
            from aof.refine.structure import structure_vault

            print("Structured:", await structure_vault(
                store, registry, link_min_sim=args.link_min_sim, hub_threshold=args.hub_threshold,
                max_hub_size=args.max_hub_size,
            ))
        elif args.refine_command == "repair":
            from aof.refine.pipeline import registry_embedder
            from aof.refine.repair import repair_false_corroboration

            examined, repaired = await repair_false_corroboration(store)
            print(f"Repair: {repaired} of {examined} corroborated claims had self-corroboration removed.")
            embed = registry_embedder(registry)
            if embed is not None:
                import numpy as np

                from aof.refine.repair import repair_weak_conflicts
                from aof.refine.vectors import cosine

                cache: dict[str, list[float]] = {}

                async def prime(texts: list[str]) -> None:
                    todo = [t for t in dict.fromkeys(texts) if t not in cache]
                    if todo:
                        cache.update(zip(todo, await embed(todo)))

                # similarity() is synchronous, so embed every sentence involved up front
                from aof.refine.assess import live_claims
                from aof.refine.repair import _LINE, _split_sections

                texts = []
                for n in await live_claims(store):
                    if "conflict" in n.tags:
                        head, sections = _split_sections(n.content)
                        texts.append(head)
                        texts += [m.group(1) for ln in sections.get("Conflicting evidence", "").splitlines()
                                  if (m := _LINE.match(ln.strip()))]
                await prime(texts)
                flagged, removed = await repair_weak_conflicts(store, lambda a, b: cosine(cache[a], cache[b]))
                print(f"Repair: {removed} of {flagged} conflict flags removed (evidence not about the claim).")
        elif args.refine_command == "verify":
            import asyncio

            from aof.refine.assess import live_claims
            from aof.refine.pipeline import registry_embedder
            from aof.refine.verify import apply_verification, verify_note

            rank = {"grade:core": 0, "grade:supporting": 1}
            todo = [n for n in await live_claims(store) if n.kind == "claim" and "**Corroboration**" not in n.content]
            todo.sort(key=lambda n: min((rank.get(t, 2) for t in n.tags), default=2))
            batch, embed = todo[: args.top], registry_embedder(registry)
            sources = build_sources([s.strip() for s in args.sources.split(",") if s.strip()])
            gate, verdicts = asyncio.Semaphore(args.concurrency), []

            async def one(note):
                async with gate:
                    v = await verify_note(note, registry=registry, sources=sources, embed=embed)
                    await apply_verification(store, note, v)
                    verdicts.append(v.verdict)

            await asyncio.gather(*(one(n) for n in batch))
            from collections import Counter

            print(f"Verified {len(batch)} of {len(todo)} unverified claims:", dict(Counter(verdicts)))
        elif args.refine_command == "metrics":
            from aof.refine.metrics import compute_metrics

            print((await compute_metrics(store, registry)).render())
        elif args.refine_command == "export":
            from aof.refine.export import export_vault

            out = Path(args.out) if args.out else workspace.root / "export"
            counts = await export_vault(store, out, title=f"{args.workspace} vault", include_archive=not args.no_archive)
            print(f"Exported to {out}: {counts}")
        else:
            assessments = await assess_vault(specs, store, registry, judge_only=args.judge_with or None)
            out = Path(args.out) if args.out else workspace.root / "assessment.md"
            out.write_text(render_markdown(assessments, title=f"Vault assessment: {qset.name}"), encoding="utf-8")
            covered = sum(f.covered for a in assessments for f in a.facts)
            total = sum(len(a.facts) for a in assessments)
            print(f"Rubric coverage: {covered}/{total} key facts. Report: {out}")
    finally:
        await registry.close()
        await backends.close()
        await store.close()
