"""Pydantic schemas: shared research/tagging models, the two plan shapes, and the public result.

List fields carry no length limits in the schema, because native structured outputs reject most
array-length constraints; the ranges live in the prompts.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["technical", "analysis"]


# ---------- shared ----------

class KindDecision(BaseModel):
    kind: Kind
    reason: str = Field(..., description="One short sentence explaining the choice.")


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class BlogTags(BaseModel):
    tags: List[str] = Field(..., description="5-8 specific tags, lowercase and hyphen-separated.")


# ---------- technical pipeline ----------

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(..., description="3 to 7 concrete, non-overlapping subpoints to cover in this section.")
    target_words: int = Field(..., description="Target word count for this section (120-550).")
    tags: List[str] = Field(default_factory=list)
    evidence_ids: List[int] = Field(
        default_factory=list,
        description="Numbers of evidence items (from the numbered list) this section relies on. Empty if none.",
    )
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    description: str = Field(..., description="One-sentence summary of the post, for previews and SEO.")
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class GroundingFix(BaseModel):
    quote: str = Field(..., description="Exact text from the post to replace, copied character for character.")
    replacement: str = Field(..., description="Replacement text. Empty string removes the quote.")
    reason: str


class GroundingReport(BaseModel):
    fixes: List[GroundingFix] = Field(default_factory=list)


# ---------- analysis pipeline ----------

class Beat(BaseModel):
    id: int
    point: str = Field(..., description="The claim this part of the piece makes, in one sentence.")
    support: str = Field(..., description="How the point will be supported: examples, data, reasoning.")
    evidence_ids: List[int] = Field(
        default_factory=list,
        description="Numbers of evidence items (from the numbered list) this beat relies on. Empty if pure reasoning.",
    )


class AnalysisPlan(BaseModel):
    title: str = Field(..., description="Headline. Specific and honest, not clickbait.")
    thesis: str = Field(..., description="One sentence: the piece's answer to the question.")
    piece_kind: Literal["analysis", "opinion", "explainer", "trend_piece", "review"] = "analysis"
    audience: str
    tone: str = Field(..., description="e.g. sharp and conversational, measured, wry.")
    target_words: int = Field(..., description="Total word count for the piece.")
    beats: List[Beat] = Field(..., description="3-6 beats, in the order the piece will make them.")
    counterpoint: str = Field(..., description="The strongest objection to the thesis, stated fairly.")
    constraints: List[str] = Field(default_factory=list)


class EditResult(BaseModel):
    issues: List[str] = Field(..., description="Problems found in the draft and what was done about each.")
    revised_title: Optional[str] = Field(
        None, description="A better headline only if the revision changed the piece's angle; otherwise null."
    )
    revised_markdown: str = Field(..., description="The full revised piece in Markdown, without the H1 title.")


# ---------- public result ----------

class BlogResult(BaseModel):
    run_id: str
    kind: Kind
    mode: str
    title: str
    description: str
    markdown: str = Field(..., description="Final post with H1 title, without front matter.")
    tags: List[str]
    output_path: Optional[str] = None
    qa_dir: Optional[str] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    editor_issues: List[str] = Field(default_factory=list)
    section_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    usage: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Token usage per model.")
    estimated_cost_usd: Optional[float] = None
    plan: dict = Field(default_factory=dict)
