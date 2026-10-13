"""Per-kind settings for the shared nodes (router, research, tagger).

The shared nodes are identical for both kinds; only the prompts and a few numbers differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet

MIN_TAGS, MAX_TAGS = 5, 8


@dataclass(frozen=True)
class Profile:
    kind: str
    router_prompt: str
    research_prompt: str
    tagger_prompt: str
    generic_tags: FrozenSet[str]
    max_queries: int
    results_per_query: int
    # Recency window given to the planner as context. Only open_book filters evidence by date.
    recency_days: Dict[str, int] = field(default_factory=dict)


TECHNICAL = Profile(
    kind="technical",
    router_prompt="""
You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
- closed_book (needs_research=false):
    Evergreen topics where correctness does not depend on recent facts. (concepts, fundamentals)
- hybrid (needs_research=true):
    Mostly evergreen but needs up-to-date examples/tools/models to be useful
- open_book (needs_research=true):
    Mostly volatile: weekly round-ups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week" or "this week" or "latest", reflect that constraint IN THE QUERIES.
""",
    research_prompt="""You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- Drop results that are not specifically about the topic (company homepages, generic
  encyclopedia pages, unrelated news) even if they come from an authoritative domain.
- Snippets must reflect what the source actually says; never add claims not in the raw result.
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short (1-3 sentences) but keep the key factual claims.
- Deduplicate by URL.
""",
    tagger_prompt=f"""You assign tags to a finished technical blog post.

Tags:
- {MIN_TAGS}-{MAX_TAGS} tags, lowercase, hyphen-separated (e.g. "graph-rag", "langgraph", "vector-search").
- Always include the post's main subject as a tag.
- Tags must reflect what the post actually covers in depth, not things mentioned in passing.
- Prefer specific technologies, techniques and concepts. Avoid generic tags like "ai", "technology", "blog", "guide", "2026".
- No near-duplicates (e.g. not both "rag" and "retrieval-augmented-generation"; pick the common short form).
""",
    generic_tags=frozenset({"ai", "technology", "tech", "blog", "guide", "tutorial", "news"}),
    max_queries=10,
    results_per_query=6,
    recency_days={"open_book": 7, "hybrid": 45, "closed_book": 3650},
)


ANALYSIS = Profile(
    kind="analysis",
    router_prompt="""
You are a routing module for an analysis and opinion blog (media, culture, business, society).
Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
    Purely conceptual or reflective pieces where no specific facts, examples or numbers are needed.
- hybrid (needs_research=true):
    The argument is the writer's, but it needs real, current examples, data and context to be credible.
    Most analysis pieces belong here.
- open_book (needs_research=true):
    The piece is about recent events: "this week", "latest", breaking developments, new releases.

If needs_research=true:
- Output 4-8 specific queries that would surface evidence for AND against the likely answer.
- Split queries across: (a) the craft or creative side (critics, creators, interviews),
  (b) concrete examples (named shows, companies, people), and (c) data (ratings, subscribers,
  market reports) only where the argument needs numbers. Don't let market data crowd out
  the substance of the question.
- Keep queries in the topic's own context (e.g. Indian market, Indian platforms) where the topic is local.
- If the user asked for "this week" or "latest", reflect that constraint in the queries.
""",
    research_prompt="""You are a research assistant for an analysis and opinion blog.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url that are specifically about the topic.
- Drop homepages, listing/category pages, generic encyclopedia pages, SEO filler and unrelated news.
- Prefer reporting, data and informed commentary: established publications, trade press, industry reports,
  interviews, critics. Keep a mix of sources that support and challenge the likely answer.
- Snippets: 1-3 sentences that keep the concrete facts (names, numbers, dates, quotes) the source actually states.
  Never add anything that is not in the raw result.
- If a published date is explicitly present, keep it as YYYY-MM-DD. Otherwise published_at=null. Do NOT guess.
- Deduplicate by URL.
""",
    tagger_prompt=f"""You assign tags to a finished analysis or opinion piece.

- {MIN_TAGS}-{MAX_TAGS} tags, lowercase, hyphen-separated (e.g. "indian-ott", "television-comedy", "streaming-economics").
- Always include the piece's main subject as a tag.
- Tags must reflect what the piece actually discusses in depth, not things mentioned in passing.
- Prefer specific subjects, places, industries and named works over generic words like "opinion", "analysis",
  "entertainment", "culture", "blog".
- No near-duplicates.
""",
    generic_tags=frozenset({"opinion", "analysis", "blog", "culture", "entertainment", "news", "essay"}),
    max_queries=8,
    results_per_query=5,
    recency_days={"open_book": 7, "hybrid": 365, "closed_book": 3650},
)


PROFILES: Dict[str, Profile] = {"technical": TECHNICAL, "analysis": ANALYSIS}
