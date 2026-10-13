"""Shared nodes: classify, router, tagger, save."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMError
from .profiles import MAX_TAGS, PROFILES
from .schemas import AnalysisPlan, BlogTags, KindDecision, Plan, RouterDecision
from .state import Deps, State
from .utils import clean_tags, slugify, yaml_str

# ---------- CLASSIFY ----------

CLASSIFY_SYSTEM = """Decide which pipeline should write this blog request.

- technical: software, data, AI/ML, infrastructure, developer tools, engineering how-tos, tutorials,
  system design, technical comparisons or tech news round-ups. Readers are practitioners.
- analysis: media, culture, business, society, strategy or opinion. The piece argues a point
  (a thesis) rather than teaching a technique, even if it touches technology.

Pick the one that fits the request best.
"""


def classify_node(state: State, *, deps: Deps) -> dict:
    """Resolve kind="auto" and fill input defaults (so the graph also runs from LangGraph Studio)."""
    update: dict = {}
    if not state.get("as_of"):
        update["as_of"] = date.today().isoformat()
    if not state.get("run_id"):
        update["run_id"] = f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(state['topic'])}"

    kind = state.get("kind") or "auto"
    if kind not in PROFILES:
        decision = deps.models.structured(
            "classifier", KindDecision,
            [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=f"Request: {state['topic']}")],
        )
        kind = decision.kind
    update["kind"] = kind
    return update


# ---------- ROUTER ----------

def router_node(state: State, *, deps: Deps) -> dict:
    profile = PROFILES[state["kind"]]
    decision = deps.models.structured(
        "router", RouterDecision,
        [
            SystemMessage(content=profile.router_prompt),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
    )
    return {
        "needs_research": decision.needs_research and bool(decision.queries),
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": profile.recency_days[decision.mode],
    }


def to_pipeline(state: State) -> str:
    return "orchestrator" if state["kind"] == "technical" else "planner"


def route_after_router(state: State) -> str:
    return "research" if state.get("needs_research") else to_pipeline(state)


# ---------- TAGGER ----------

def _tagger_context(state: State) -> str:
    plan = state["plan"]
    if isinstance(plan, Plan):
        outline = "\n".join(f"- {t.title} (planner tags: {', '.join(t.tags) or 'none'})" for t in plan.tasks)
        return f"Blog kind: {plan.blog_kind}\nAudience: {plan.audience}\n\nOutline:\n{outline}"
    if isinstance(plan, AnalysisPlan):
        return f"Thesis: {plan.thesis}\nPiece kind: {plan.piece_kind}"
    return ""


def tagger_node(state: State, *, deps: Deps) -> dict:
    profile = PROFILES[state["kind"]]
    try:
        result = deps.models.structured(
            "tagger", BlogTags,
            [
                SystemMessage(content=profile.tagger_prompt),
                HumanMessage(
                    content=(
                        f"Title: {state['title']}\n"
                        f"Request: {state['topic']}\n"
                        f"{_tagger_context(state)}\n\n"
                        f"Post (may be truncated):\n{state['final'][:12000]}"
                    )
                ),
            ],
        )
    except LLMError as e:
        return {"tags": [], "warnings": [f"tagging failed: {e}"]}
    return {"tags": clean_tags(result.tags, profile.generic_tags, MAX_TAGS)}


# ---------- SAVE ----------

def front_matter(state: State) -> str:
    plan = state["plan"]
    lines = [
        "---",
        f"title: {yaml_str(state['title'])}",
        f"date: {state['as_of']}",
        f"kind: {state['kind']}",
    ]
    if isinstance(plan, Plan):
        lines.append(f"blog_kind: {plan.blog_kind}")
    elif isinstance(plan, AnalysisPlan):
        lines.append(f"piece_kind: {plan.piece_kind}")
    if state.get("description"):
        lines.append(f"description: {yaml_str(state['description'])}")
    lines.append("tags: [" + ", ".join(state.get("tags", [])) + "]")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def save_node(state: State, *, deps: Deps) -> dict:
    if state.get("save") is False:
        return {"output_path": ""}
    output_dir = Path(deps.settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{state['run_id']}.md"
    output_path.write_text(front_matter(state) + state["final"], encoding="utf-8")
    return {"output_path": str(output_path)}
