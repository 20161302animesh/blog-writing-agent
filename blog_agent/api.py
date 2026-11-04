"""Public entry points: write_blog() and as_tool()."""
from __future__ import annotations

import logging
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.callbacks import UsageMetadataCallbackHandler

from .config import Settings, estimate_cost
from .graph import build_graph
from .qa import RunRecorder
from .schemas import BlogResult
from .stages import StageLedger, stage_table
from .state import SearchFn
from .tracing import finish_trace, run_trace
from .utils import slugify

log = logging.getLogger("blog_agent")


def _usage_summary(handler: UsageMetadataCallbackHandler):
    usage = {
        model: {k: int(v) for k, v in u.items() if isinstance(v, (int, float))}
        for model, u in handler.usage_metadata.items()
    }
    costs = [estimate_cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0)) for model, u in usage.items()]
    return usage, (round(sum(costs), 4) if None not in costs else None)


def write_blog(
    topic: str,
    kind: Literal["auto", "technical", "analysis"] = "auto",
    *,
    target_words: Optional[int] = None,
    as_of: Optional[str] = None,
    save: bool = True,
    qa: bool = True,
    settings: Optional[Settings] = None,
    checkpointer=None,
    resume_run_id: Optional[str] = None,
    models=None,
    search: Optional[SearchFn] = None,
) -> BlogResult:
    """Research, plan, write, edit and tag one blog post.

    kind:          "technical" (outline + parallel section writers), "analysis" (thesis + one writer + editor),
                   or "auto" to let a classifier pick.
    target_words:  analysis only; the planner decides otherwise.
    save:          write the post with front matter to settings.output_dir.
    qa:            write per-node snapshots, versions and diffs to settings.qa_dir/<run_id>/.
    checkpointer:  a LangGraph checkpointer (e.g. SqliteSaver). With one, a failed run can be continued
                   with resume_run_id=<run_id>.
    models/search: test doubles; leave as None for real runs.
    """
    settings = settings or Settings.from_env()
    if models is None or search is None:
        settings.check_credentials()
    if resume_run_id and checkpointer is None:
        raise ValueError("resume_run_id needs a checkpointer")

    app = build_graph(settings, models=models, search=search, checkpointer=checkpointer)

    if resume_run_id:
        run_id = resume_run_id
        graph_input = None   # continue from the last checkpoint
    else:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(topic)}"
        graph_input = {
            "topic": topic,
            "kind": kind,
            "target_words": target_words,
            "as_of": as_of or date.today().isoformat(),
            "run_id": run_id,
            "save": save,
            "sections": [],
            "section_issues": [],
            "warnings": [],
        }

    usage_handler = UsageMetadataCallbackHandler()
    config = {
        "configurable": {"thread_id": run_id},
        "max_concurrency": settings.max_concurrency,
        "callbacks": [usage_handler],
        "run_name": f"blog_agent:{run_id}",
    }
    recorder = RunRecorder(Path(settings.qa_dir) / run_id) if qa else None
    ledger = StageLedger(recorder.run_dir / "stages.jsonl" if recorder else None)

    # LangSmith: one trace per call, findable by blog_run_id (a resumed run gets a second trace)
    trace_meta = {
        "blog_run_id": run_id,
        "kind": kind,
        "resumed": bool(resume_run_id),
        "models": {role: spec.model for role, spec in settings.models.items()},
    }
    trace_inputs = {"topic": topic, "kind": kind, "target_words": target_words, "as_of": as_of,
                    "resume_run_id": resume_run_id}

    state: Optional[dict] = None
    pending: List[tuple] = []
    with run_trace(trace_inputs, trace_meta, tags=["blog_agent"]) as rt, ledger.active():
        try:
            for mode, chunk in app.stream(graph_input, config=config, stream_mode=["updates", "values"]):
                if mode == "updates":
                    for node, update in chunk.items():
                        log.info("finished node %s", node)
                        pending.append((node, update or {}))
                    continue
                state = chunk
                if recorder:
                    recorder.record_step(pending, chunk)
                pending = []
        except Exception:
            if recorder:
                (recorder.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise

        usage, cost = _usage_summary(usage_handler)
        if rt is not None:
            rt.add_outputs({"title": (state or {}).get("title"), "usage": usage, "estimated_cost_usd": cost,
                            "stages": stage_table(ledger.ordered())})
    trace_url = finish_trace(rt)

    if state is None or not state.get("final"):
        raise RuntimeError(f"Run {run_id} ended without a final post (already finished, or nothing to resume?)")
    if recorder:
        recorder.finish(state)

    return BlogResult(
        run_id=run_id,
        kind=state["kind"],
        mode=state.get("mode", ""),
        title=state["title"],
        description=state.get("description", ""),
        markdown=state["final"],
        tags=state.get("tags", []),
        output_path=state.get("output_path") or None,
        qa_dir=str(recorder.run_dir) if recorder else None,
        evidence=state.get("evidence", []),
        editor_issues=state.get("editor_issues", []),
        section_issues=[f"task {i}: {p}" for i, p in state.get("section_issues", [])],
        warnings=state.get("warnings", []),
        usage=usage,
        estimated_cost_usd=cost,
        stages=ledger.ordered(),
        trace_id=str(rt.trace_id) if rt is not None else None,
        trace_url=trace_url,
        plan=state["plan"].model_dump() if state.get("plan") is not None else {},
    )


def as_tool(settings: Optional[Settings] = None):
    """Expose write_blog as a LangChain tool, so another agent can call it."""
    from langchain_core.tools import StructuredTool

    def write_blog_post(topic: str, kind: Literal["auto", "technical", "analysis"] = "auto",
                        target_words: Optional[int] = None) -> dict:
        result = write_blog(topic, kind, target_words=target_words, settings=settings)
        return {"title": result.title, "output_path": result.output_path, "tags": result.tags,
                "kind": result.kind, "description": result.description}

    return StructuredTool.from_function(
        write_blog_post,
        name="write_blog_post",
        description=(
            "Research and write a complete blog post on a topic and save it as Markdown. "
            "kind: 'technical' for engineering/AI/dev topics, 'analysis' for media/culture/business "
            "opinion pieces, or 'auto'. Returns the title, saved file path and tags. Takes a few minutes."
        ),
    )
