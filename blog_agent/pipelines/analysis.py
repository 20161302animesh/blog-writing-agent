"""Analysis pipeline: planner (thesis + beats) -> one writer -> editor (critique + revise)."""
from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import LLMError, stop_problem
from ..schemas import AnalysisPlan, EditResult, EvidenceItem
from ..state import Deps, State
from ..utils import evidence_block, prose_word_count, valid_ids

# ---------- PLANNER ----------

PLANNER_SYSTEM = """You are the commissioning editor of a smart, independent publication on media, culture,
business and society. Plan ONE analysis or opinion piece.

The plan must be argument-shaped:
- thesis: a clear, arguable one-sentence answer to the question. Not a fence-sit ("it depends").
  A qualified answer is fine ("Yes, but not from the platforms everyone expects").
- beats: 3-6 steps that build the argument in order. Each beat makes ONE point and says how it will be supported.
  Where a beat makes factual claims, list the evidence numbers it relies on.
- counterpoint: the strongest objection to the thesis, stated fairly. The piece will address it.

Length:
- "short": 450-650 words.
- Unspecified: 700-1000 words. Longer only if the request asks for it.
- If a target word count is given, use it.

Grounding:
- Only reference evidence numbers that exist in the provided list.
- Do not plan beats around specific facts (ratings, numbers, events) that the evidence does not contain.
  If evidence is thin, lean the beats on reasoning and well-known context, and keep claims modest.

Avoid:
- Listicle structure, "5 reasons why" framing, or a beat per example.
- A beat that only summarizes the topic. Context belongs inside the first beat, briefly.
"""


def planner_node(state: State, *, deps: Deps) -> dict:
    target = state.get("target_words")
    evidence = state.get("evidence", [])
    plan = deps.models.structured(
        "planner", AnalysisPlan,
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=(
                    f"Request: {state['topic']}\n"
                    f"Mode: {state['mode']}\n"
                    f"As-of: {state['as_of']}\n"
                    f"Target words: {target or 'decide from the request'}\n\n"
                    f"Evidence:\n{evidence_block(evidence)}"
                )
            ),
        ],
    )
    if target:
        plan.target_words = target
    for b in plan.beats:
        b.evidence_ids = valid_ids(b.evidence_ids, len(evidence))
    return {"plan": plan}


# ---------- WRITER ----------

WRITER_SYSTEM = """You write sharp, readable analysis for an independent publication on media, culture,
business and society. Write the piece described by the plan, in Markdown.

Structure:
- Do NOT include the H1 title; it is added later.
- Open with a concrete hook (a scene, a fact, a contrast), then state the thesis within the first two paragraphs.
- Follow the beats in order. Use at most 2-3 short subheadings (##), or none if the piece flows better without.
- Address the counterpoint honestly before the ending, then close on the thesis with something sharper than a summary.
- Stay within ±10% of the target word count.

Grounding:
- Specific facts (numbers, ratings, dates, quotes, deals, who-said-what) must come from the evidence snippets.
- Cite inline, journalism style: link the phrase the fact appears in, e.g. "...[topped the platform's charts](URL)...".
  Use only URLs from the evidence list, and only for claims that URL's snippet actually supports.
- If you need a fact the evidence doesn't contain, make the point without the specific figure, or leave it out.
- Analysis, interpretation and opinion are yours to make; they don't need citations.
- Evidence marked date:unknown may be old. Don't present its quotes or figures as current
  ("still", "now", "today") unless the snippet says when.

Style:
- Concrete over abstract: name shows, people, platforms, moments.
- Vary sentence length. Short paragraphs.
- No clichés: "in today's fast-paced world", "game-changer", "it remains to be seen", "only time will tell",
  "in conclusion", "a testament to", "delve".
- No hedging every sentence. Commit to the thesis.
"""


def plan_text(plan: AnalysisPlan, evidence: List[EvidenceItem],
              raw_sources: Optional[Dict[str, str]] = None) -> str:
    beats = "\n".join(
        f"{b.id}. {b.point}\n   Support: {b.support}\n"
        f"   Evidence: {', '.join(f'[{i}]' for i in b.evidence_ids) or 'reasoning only'}"
        for b in plan.beats
    )
    return (
        f"Title: {plan.title}\n"
        f"Thesis: {plan.thesis}\n"
        f"Kind: {plan.piece_kind}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Target words: {plan.target_words}\n"
        f"Constraints: {plan.constraints}\n\n"
        f"Beats:\n{beats}\n\n"
        f"Counterpoint to address: {plan.counterpoint}\n\n"
        f"Evidence:\n{evidence_block(evidence, raw_sources)}"
    )


def writer_node(state: State, *, deps: Deps) -> dict:
    plan: AnalysisPlan = state["plan"]
    draft, meta = deps.models.text(
        "writer",
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=f"Original request: {state['topic']}\n\n{plan_text(plan, state.get('evidence', []))}"),
        ],
    )
    problem = stop_problem(meta)
    if problem:
        raise LLMError(f"writer: {problem}")
    return {"draft": draft, "title": plan.title, "description": plan.thesis}


# ---------- EDITOR ----------

EDITOR_SYSTEM = """You are a demanding but fair senior editor. Review the draft against the plan and evidence,
then return the revised piece.

Check, in order:
1. Argument: is the thesis stated early and actually argued? Does each section earn its place?
   Is the counterpoint addressed honestly rather than waved away?
2. Grounding: every specific fact (number, date, rating, quote, deal) must be supported by the
   Source text of the evidence item it links to (use the Snippet only if there is no Source text).
   Remove or soften any claim that isn't. Never add new facts or new URLs.
3. Reasoning: does each inference actually follow from the fact it cites? Flag statistics used to
   prove more than they show (e.g. a genre share that includes the very category being argued for).
4. Prose: cut clichés, filler, repetition and throat-clearing. Tighten flabby sentences. Keep the writer's voice;
   don't flatten it into corporate prose.
5. Length: within ±10% of the target word count.

Output:
- issues: a short list of what you found and what you did about each (e.g. "Removed unsupported subscriber
  figure in para 4"). Empty list only if the draft genuinely needed nothing.
- revised_markdown: the complete revised piece, without the H1 title.
- revised_title: a better headline only if your revision changed the piece's angle; otherwise null.
"""

LENGTH_SYSTEM = ("Adjust this piece's length. Keep the argument, voice, facts and links unchanged. "
                 "Return only the Markdown body, no title.")


def editor_node(state: State, *, deps: Deps) -> dict:
    plan: AnalysisPlan = state["plan"]
    warnings: List[str] = []
    try:
        result = deps.models.structured(
            "editor", EditResult,
            [
                SystemMessage(content=EDITOR_SYSTEM),
                HumanMessage(
                    content=(
                        f"{plan_text(plan, state.get('evidence', []), state.get('raw_sources'))}\n\n"
                        f"---\nDRAFT:\n\n{state['draft']}"
                    )
                ),
            ],
        )
        body = result.revised_markdown.strip()
        issues = list(result.issues)
        if result.revised_title:
            plan = plan.model_copy(update={"title": result.revised_title})
    except LLMError as e:
        # Keep the unedited draft rather than failing the whole run
        body, issues = state["draft"], []
        warnings.append(f"editor failed, publishing the unedited draft: {e}")
    editor_revision = body

    # Verify length in code; one corrective pass if the editor missed
    words = prose_word_count(body)
    lo, hi = int(plan.target_words * 0.9), int(plan.target_words * 1.1)
    if not lo <= words <= hi:
        adjusted, meta = deps.models.text(
            "editor",
            [
                SystemMessage(content=LENGTH_SYSTEM),
                HumanMessage(content=f"Current: {words} words (excluding links). "
                                     f"Target: {plan.target_words} (±10%).\n\n{body}"),
            ],
        )
        problem = stop_problem(meta)
        if problem:
            warnings.append(f"length pass discarded: {problem}")
        else:
            body = adjusted
            issues.append(f"Length pass: {words} → {prose_word_count(body)} words.")

    return {
        "plan": plan,
        "title": plan.title,
        "final": f"# {plan.title}\n\n{body}\n",
        "editor_issues": issues,
        "editor_revision": editor_revision,
        "warnings": warnings,
    }
