"""Offline doubles for Models and Tavily search."""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from blog_agent.llm import LLMError
from blog_agent.schemas import (
    AnalysisPlan, Beat, BlogTags, EditResult, EvidenceItem, EvidencePack, GroundingFix, GroundingReport,
    KindDecision, Plan, RouterDecision, Task,
)

URL_A = "https://example.com/report-a"
URL_B = "https://news.example.org/story-b"


def fake_search(query: str, max_results: int, topic: str, time_range: Optional[str]) -> List[dict]:
    return [
        {"title": "Report A", "url": URL_A, "snippet": "Report A says the market grew 12% in 2025.",
         "published_at": "2026-09-01"},
        {"title": "Story B", "url": URL_B, "snippet": "Story B quotes a producer on budgets.", "published_at": None},
        {"title": "Wiki", "url": "https://en.wikipedia.org/wiki/Thing", "snippet": "blocked", "published_at": None},
    ]


def default_structured() -> Dict[str, Callable]:
    return {
        "KindDecision": lambda: KindDecision(kind="analysis", reason="opinion question"),
        "RouterDecision": lambda: RouterDecision(needs_research=True, mode="hybrid", queries=["q1", "q2"]),
        "EvidencePack": lambda: EvidencePack(evidence=[
            EvidenceItem(title="Report A", url=URL_A, published_at="2026-09-01", snippet="Market grew 12% in 2025."),
            EvidenceItem(title="Story B", url=URL_B, snippet="A producer on budgets."),
            EvidenceItem(title="Invented", url="https://invented.example.net/x", snippet="not from search"),
        ]),
        "Plan": lambda: Plan(
            blog_title="Tech Title", description="A technical post.", audience="devs", tone="crisp",
            blog_kind="news_roundup",   # wrong for hybrid; the orchestrator must correct it
            tasks=[
                Task(id=2, title="Second", goal="g", bullets=["b"], target_words=150, evidence_ids=[1, 9]),
                Task(id=1, title="First", goal="g", bullets=["b"], target_words=150, requires_code=True),
            ],
        ),
        "GroundingReport": lambda: GroundingReport(fixes=[
            GroundingFix(quote="The market grew 50%.", replacement="The market grew 12%.", reason="Fixed figure"),
            GroundingFix(quote="not in the post", replacement="x", reason="bad quote"),
            GroundingFix(quote="Section two body.", replacement="See [x](https://evil.example/)", reason="new url"),
        ]),
        "AnalysisPlan": lambda: AnalysisPlan(
            title="Analysis Title", thesis="Yes, but slowly.", audience="readers", tone="wry", target_words=20,
            beats=[Beat(id=1, point="p", support="s", evidence_ids=[1, 5])], counterpoint="c",
        ),
        "EditResult": lambda: EditResult(
            issues=["Tightened intro"], revised_title="Better Title",
            revised_markdown="Edited body with exactly enough words to land inside the target range ok.",
        ),
        "BlogTags": lambda: BlogTags(tags=["Indian OTT", "opinion", "indian-ott", "Streaming Economics"]),
    }


class FakeModels:
    def __init__(self, structured: Optional[Dict[str, Callable]] = None, text: Optional[Dict[str, Callable]] = None):
        self._structured = {**default_structured(), **(structured or {})}
        self._text = text or {}
        self.calls: List[tuple] = []
        self._lock = threading.Lock()

    def structured(self, role, schema, messages):
        with self._lock:
            self.calls.append((role, schema.__name__, messages))
        value = self._structured[schema.__name__]()
        if isinstance(value, Exception):
            raise value
        return value

    def text(self, role, messages):
        with self._lock:
            self.calls.append((role, "text", messages))
        prompt = messages[-1].content
        if role in self._text:
            return self._text[role](prompt)
        if "Section title: First" in prompt:
            if "YOUR PREVIOUS ATTEMPT" in prompt:
                return "## First\n\n```python\nprint('ok')\n```", {"stop_reason": "end_turn"}
            return "## First\n\n```python\ndef broken(:\n```", {"stop_reason": "end_turn"}
        if "Section title: Second" in prompt:
            return "## Second\n\nThe market grew 50%. Section two body.", {"stop_reason": "end_turn"}
        if "DRAFT" in prompt or "Original request" in prompt:
            return "Draft body.", {"stop_reason": "end_turn"}
        return "Adjusted body.", {"stop_reason": "end_turn"}


def raising(msg: str):
    return lambda: LLMError(msg)
