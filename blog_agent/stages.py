"""Per-stage records: what each step of the workflow did, how long it took, and what it cost.

Every graph node is decorated with `@stage`. It times the node, collects the model calls made
while it runs (Models reports them through `record_llm_call`), pulls the node's decisions out of
its state update, and catches its error. The finished StageRecord goes to:

- the StageLedger for this write_blog() call (BlogResult.stages, and runs/<run_id>/stages.jsonl
  as each stage finishes, so a failed run still shows how far it got);
- the node's LangSmith run, as `stage` metadata, when tracing is on.

Nodes can add details their state update doesn't carry with `note(key=value)`.
"""
from __future__ import annotations

import functools
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator, List, Optional, Union

from langsmith import get_current_run_tree

from .config import estimate_cost
from .schemas import AnalysisPlan, LLMCall, Plan, StageRecord

_current: ContextVar[Optional["_OpenStage"]] = ContextVar("blog_agent_stage", default=None)
_ledger: ContextVar[Optional["StageLedger"]] = ContextVar("blog_agent_ledger", default=None)


class StageLedger:
    """Collects the StageRecords of one write_blog() call. Thread-safe (section workers run in parallel)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path   # stages.jsonl; appended to, so a resumed run adds to the same file
        self.records: List[StageRecord] = []
        self._seq = 0
        self._lock = Lock()

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def add(self, record: StageRecord) -> None:
        with self._lock:
            self.records.append(record)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")

    def ordered(self) -> List[StageRecord]:
        return sorted(self.records, key=lambda r: r.seq or 0)

    @contextmanager
    def active(self) -> Iterator["StageLedger"]:
        token = _ledger.set(self)
        try:
            yield self
        finally:
            _ledger.reset(token)


class _OpenStage:
    def __init__(self, name: str, seq: Optional[int]):
        self.name = name
        self.seq = seq
        self.llm_calls: List[LLMCall] = []
        self.notes: dict = {}


def note(**decisions) -> None:
    """Add decisions to the running stage's record (no-op outside a stage)."""
    st = _current.get()
    if st is not None:
        st.notes.update(decisions)


def record_llm_call(role: str, model: str, message=None, started: Optional[float] = None,
                    error: Optional[str] = None) -> None:
    """Called by Models after each request (message is the AIMessage, or None if the request failed)."""
    st = _current.get()
    if st is None:
        return
    usage = getattr(message, "usage_metadata", None) or {}
    meta = getattr(message, "response_metadata", None) or {}
    model = meta.get("model_name") or meta.get("model") or model
    tokens_in, tokens_out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    cost = estimate_cost(model, tokens_in, tokens_out)
    st.llm_calls.append(LLMCall(
        role=role, model=model, input_tokens=tokens_in, output_tokens=tokens_out,
        stop_reason=meta.get("stop_reason"),
        duration_s=round(time.perf_counter() - started, 3) if started else 0.0,
        cost_usd=round(cost, 6) if cost is not None else None,
        error=error,
    ))


def stage(fn_or_name: Union[Callable, str]):
    """Decorator: each run of the node leaves a StageRecord.

    `@stage` names the stage after the function without its `_node` suffix (router_node -> "router");
    `@stage("name")` names it explicitly. Extra arguments (deps, config) pass through, and
    functools.wraps keeps the node's signature visible to LangGraph.
    """
    if isinstance(fn_or_name, str):
        return lambda fn: _wrap(fn_or_name, fn)
    return _wrap(fn_or_name.__name__.removesuffix("_node"), fn_or_name)


def _wrap(name: str, fn: Callable) -> Callable:
    @functools.wraps(fn)
    def run(state: dict, *args, **kwargs) -> dict:
        ledger = _ledger.get()
        st = _OpenStage(name, ledger.next_seq() if ledger else None)
        token = _current.set(st)
        started_at, t0 = datetime.now(timezone.utc), time.perf_counter()
        update, error = None, None
        try:
            update = fn(state, *args, **kwargs)
            return update
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _current.reset(token)
            record = _finish(st, state, update, error, started_at, time.perf_counter() - t0)
            rt = get_current_run_tree()   # this node's LangSmith run, when tracing is on
            if rt is not None:
                record.langsmith_run_id = str(rt.id)
                rt.add_metadata({"stage": record.model_dump(mode="json", exclude={"langsmith_run_id"})})
            if ledger:
                ledger.add(record)

    return run


def _finish(st: _OpenStage, state: dict, update: Optional[dict], error: Optional[str],
            started_at: datetime, duration: float) -> StageRecord:
    calls = st.llm_calls
    costs = [c.cost_usd for c in calls]
    decisions = {}
    if update is not None:
        try:
            decisions = summarize(st.name, state, update)
        except Exception as e:   # a summary bug must never fail the node
            decisions = {"summary_error": str(e)}
    decisions.update(st.notes)
    return StageRecord(
        stage=st.name,
        seq=st.seq,
        started_at=started_at.isoformat(timespec="milliseconds"),
        ended_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        duration_s=round(duration, 3),
        status="error" if error else "ok",
        llm_calls=calls,
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        cost_usd=round(sum(costs), 6) if None not in costs else None,
        decisions=decisions,
        warnings=list((update or {}).get("warnings", [])),
        error=error,
    )


def summarize(name: str, state: dict, update: dict) -> dict:
    """The decisions a node made, from its input and state update (small values only, no full text)."""
    from .utils import prose_word_count   # utils imports llm, which imports this module

    if name == "classify":
        return {"kind": update.get("kind")}
    if name == "router":
        return {k: update.get(k) for k in ("mode", "needs_research", "queries", "recency_days")}
    if name == "research":
        return {"evidence_kept": len(update.get("evidence", [])),
                "evidence_urls": [e.url for e in update.get("evidence", [])]}
    if name == "orchestrator":
        plan: Plan = update["plan"]
        return {"title": plan.blog_title, "blog_kind": plan.blog_kind,
                "tasks": [{"id": t.id, "title": t.title, "target_words": t.target_words,
                           "requires_code": t.requires_code, "evidence_ids": t.evidence_ids} for t in plan.tasks]}
    if name == "worker":
        task = state.get("task", {})
        (_, section), = update.get("sections", [(None, "")])
        return {"task_id": task.get("id"), "title": task.get("title"), "target_words": task.get("target_words"),
                "words": prose_word_count(section), "issues": [p for _, p in update.get("section_issues", [])]}
    if name == "reducer":
        return {"sections": len({i for i, _ in state.get("sections", [])}),
                "words": prose_word_count(update.get("draft", ""))}
    if name == "grounding":
        issues = update.get("editor_issues", [])
        rejected = [i for i in issues if i.startswith("Not applied")]
        return {"ran": "editor_revision" in update, "fixes_applied": len(issues) - len(rejected),
                "fixes_rejected": len(rejected), "issues": issues}
    if name == "planner":
        plan: AnalysisPlan = update["plan"]
        return {"title": plan.title, "thesis": plan.thesis, "piece_kind": plan.piece_kind,
                "target_words": plan.target_words, "beats": len(plan.beats)}
    if name == "writer":
        return {"words": prose_word_count(update.get("draft", ""))}
    if name == "editor":
        return {"title": update.get("title"), "title_changed": update.get("title") != state.get("title"),
                "issues": update.get("editor_issues", []), "words": prose_word_count(update.get("final", ""))}
    if name == "tagger":
        return {"tags": update.get("tags", [])}
    if name == "save":
        return {"output_path": update.get("output_path") or None}
    return {}


def stage_table(records: List[StageRecord]) -> List[dict]:
    """One compact row per stage, for the LangSmith root run and the CLI."""
    rows = []
    for r in records:
        label = r.stage if r.stage != "worker" else f"worker#{r.decisions.get('task_id')}"
        rows.append({"seq": r.seq, "stage": label, "status": r.status, "duration_s": r.duration_s,
                     "llm_calls": len(r.llm_calls), "input_tokens": r.input_tokens,
                     "output_tokens": r.output_tokens, "cost_usd": r.cost_usd})
    return rows


def load_stages(path: Path) -> List[StageRecord]:
    """Read a run's stages.jsonl (all attempts, if it was resumed)."""
    return [StageRecord(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
