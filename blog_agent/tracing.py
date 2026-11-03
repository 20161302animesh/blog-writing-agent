"""Optional LangSmith tracing.

Off unless the standard LangSmith variables are set (LANGSMITH_TRACING=true, LANGSMITH_API_KEY,
optionally LANGSMITH_PROJECT). When on, each write_blog() call becomes one trace: a root run
named "write_blog" with the graph under it, every model call tagged with its role, and the
run's token usage and cost estimate as the root's outputs.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

import langsmith
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import utils as ls_utils
from langsmith.run_trees import RunTree

log = logging.getLogger("blog_agent")


def tracing_enabled() -> bool:
    return bool(ls_utils.tracing_is_enabled())


@contextmanager
def run_trace(inputs: dict, metadata: dict, tags: list) -> Iterator[Optional[RunTree]]:
    """Open the root run for one write_blog() call, or yield None when tracing is off."""
    if not tracing_enabled():
        yield None
        return
    with langsmith.trace("write_blog", run_type="chain", inputs=inputs, metadata=metadata, tags=tags) as rt:
        yield rt


def finish_trace(rt: Optional[RunTree]) -> Optional[str]:
    """Send pending runs and return the trace's URL (None if tracing is off or the lookup fails)."""
    if rt is None:
        return None
    wait_for_all_tracers()
    try:
        return rt.get_url()
    except Exception as e:   # a missing link shouldn't fail a finished post
        log.warning("could not get the LangSmith trace URL: %s", e)
        return None
