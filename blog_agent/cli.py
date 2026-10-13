"""Command line: python -m blog_agent "topic" [--kind analysis] [--words 800] ..."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys

from .api import write_blog
from .config import Settings


def _summary(result) -> str:
    lines = [
        "=" * 100,
        f"KIND: {result.kind} | MODE: {result.mode} | EVIDENCE: {len(result.evidence)}",
        f"TITLE: {result.title}",
        f"DESCRIPTION: {result.description}",
        f"TAGS: {result.tags}",
    ]
    if result.editor_issues:
        lines.append(f"EDITOR_ISSUES: {len(result.editor_issues)}")
        lines += [f"  - {i}" for i in result.editor_issues[:8]]
    if result.section_issues:
        lines.append(f"SECTION_ISSUES: {len(result.section_issues)}")
        lines += [f"  - {i}" for i in result.section_issues]
    if result.warnings:
        lines.append("WARNINGS:")
        lines += [f"  - {w}" for w in result.warnings]
    tokens = {m: (u.get("input_tokens", 0), u.get("output_tokens", 0)) for m, u in result.usage.items()}
    lines.append(f"TOKENS (in, out): {tokens}")
    if result.estimated_cost_usd is not None:
        lines.append(f"EST_COST_USD: {result.estimated_cost_usd}")
    lines += [f"SAVED_TO: {result.output_path}", f"QA_SNAPSHOTS: {result.qa_dir}", "=" * 100]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="blog_agent", description="Research and write a blog post.")
    p.add_argument("topic", nargs="?", help="The blog request. Omit when using --resume.")
    p.add_argument("--kind", choices=["auto", "technical", "analysis"], default="auto")
    p.add_argument("--words", type=int, help="Target word count (analysis pieces).")
    p.add_argument("--as-of", help="As-of date, YYYY-MM-DD (default: today).")
    p.add_argument("--no-save", action="store_true", help="Don't write the post to the output folder.")
    p.add_argument("--no-qa", action="store_true", help="Don't write QA snapshots.")
    p.add_argument("--no-grounding", action="store_true", help="Skip the technical grounding check.")
    p.add_argument("--checkpoint-db", help="SQLite file for checkpoints, so a failed run can be resumed.")
    p.add_argument("--resume", metavar="RUN_ID", help="Continue a failed run (needs --checkpoint-db).")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    p.add_argument("-v", "--verbose", action="store_true", help="Log each node as it finishes.")
    args = p.parse_args(argv)

    if not args.topic and not args.resume:
        p.error("a topic is required unless --resume is given")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")

    settings = Settings.from_env()
    if args.no_grounding:
        settings = settings.with_overrides(grounding_check=False)

    with contextlib.ExitStack() as stack:
        checkpointer = None
        if args.checkpoint_db:
            from .checkpoint import sqlite_checkpointer
            checkpointer = stack.enter_context(sqlite_checkpointer(args.checkpoint_db))
        result = write_blog(
            args.topic or "",
            args.kind,
            target_words=args.words,
            as_of=args.as_of,
            save=not args.no_save,
            qa=not args.no_qa,
            settings=settings,
            checkpointer=checkpointer,
            resume_run_id=args.resume,
        )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
