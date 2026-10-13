"""Tavily search and the research node (shared by both pipelines)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

from .llm import LLMError
from .profiles import PROFILES
from .schemas import EvidencePack
from .state import Deps, State
from .utils import is_blocked, iso_to_date

TIME_RANGE = {"open_book": "week", "hybrid": None, "closed_book": None}


def tavily_search(query: str, max_results: int = 5, topic: str = "general",
                  time_range: Optional[str] = None) -> List[dict]:
    kwargs = {"max_results": max_results, "topic": topic}
    if time_range:
        kwargs["time_range"] = time_range
    response = TavilySearch(**kwargs).invoke({"query": query})
    if not isinstance(response, dict):
        raise TypeError(f"Expected a Tavily response dict, got {type(response).__name__}")
    if "error" in response:
        raise RuntimeError(f"Tavily search failed: {response['error']}")

    normalized: List[dict] = []
    for r in response.get("results", []) or []:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
            }
        )
    return normalized


def research_node(state: State, *, deps: Deps) -> dict:
    profile = PROFILES[state["kind"]]
    queries = (state.get("queries") or [])[: profile.max_queries]
    mode = state.get("mode", "hybrid")
    topic = "news" if mode == "open_book" else "general"   # "news" returns published dates
    time_range = TIME_RANGE.get(mode)
    warnings: List[str] = []

    def run_query(q: str) -> List[dict]:
        try:
            return deps.search(q, profile.results_per_query, topic, time_range)
        except Exception as e:   # one failed query shouldn't sink the run
            warnings.append(f"search failed for {q!r}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max(1, deps.settings.search_concurrency)) as pool:
        raw_results = [r for batch in pool.map(run_query, queries) for r in batch]

    # Enforce source rules in code, before the LLM sees anything
    raw_results = [r for r in raw_results if r.get("url") and not is_blocked(r["url"])]
    if not raw_results:
        if queries:
            warnings.append("research returned no usable results; writing without evidence")
        return {"evidence": [], "raw_sources": {}, "warnings": warnings}
    raw_by_url = {r["url"]: r["snippet"] for r in raw_results}

    try:
        pack = deps.models.structured(
            "research",
            EvidencePack,
            [
                SystemMessage(content=profile.research_prompt),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"As-of date: {state['as_of']}\n\n"
                        f"Raw results:\n{raw_results}"
                    )
                ),
            ],
        )
    except LLMError as e:
        warnings.append(f"evidence extraction failed ({e}); writing without evidence")
        return {"evidence": [], "raw_sources": {}, "warnings": warnings}

    # Only keep URLs that search actually returned (the extractor can't invent sources)
    dedup = {e.url: e for e in pack.evidence if e.url in raw_by_url and not is_blocked(e.url)}
    evidence = list(dedup.values())

    # Drop items known to be too old; keep undated ones (time_range already filtered at search)
    if mode == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := iso_to_date(e.published_at)) is None or d >= cutoff]

    evidence = evidence[: deps.settings.max_evidence]
    return {
        "evidence": evidence,
        "raw_sources": {e.url: raw_by_url.get(e.url, "") for e in evidence},
        "warnings": warnings,
    }
