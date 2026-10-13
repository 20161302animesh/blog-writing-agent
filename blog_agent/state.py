"""Graph state shared by both pipelines, plus the dependencies the nodes need."""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Callable, Dict, List, Optional, Tuple, TypedDict, Union

from .config import Settings
from .llm import Models
from .schemas import AnalysisPlan, EvidenceItem, Plan

# search(query, max_results, topic, time_range) -> [{"title", "url", "snippet", "published_at"}]
SearchFn = Callable[[str, int, str, Optional[str]], List[dict]]


@dataclass
class Deps:
    settings: Settings
    models: Models          # anything with .structured() and .text() works (tests use a fake)
    search: SearchFn


class State(TypedDict, total=False):
    # input
    topic: str
    kind: str                       # "auto" | "technical" | "analysis"; classify resolves "auto"
    target_words: Optional[int]     # analysis only; planner decides otherwise
    as_of: str
    run_id: str
    save: bool

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    recency_days: int
    evidence: List[EvidenceItem]
    raw_sources: Dict[str, str]     # url -> Tavily's original text, for grounding checks

    # planning
    plan: Optional[Union[Plan, AnalysisPlan]]

    # technical workers
    sections: Annotated[List[Tuple[int, str]], operator.add]          # (task_id, section_md)
    section_issues: Annotated[List[Tuple[int, str]], operator.add]    # (task_id, problem) left after retry

    # drafts (both kinds): draft -> editor_revision -> final
    draft: str
    editor_issues: List[str]
    editor_revision: str
    final: str                      # Markdown with H1 title
    title: str
    description: str

    # output
    tags: List[str]
    output_path: str
    warnings: Annotated[List[str], operator.add]
