"""Technical pipeline: orchestrator -> parallel section workers -> reducer -> grounding check."""
from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from ..llm import LLMError
from ..schemas import EvidenceItem, GroundingReport, Plan, Task
from ..state import Deps, State
from ..utils import check_markdown, evidence_block, urls_in, valid_ids

# ---------- ORCHESTRATOR (planning) ----------

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
    1) goal (1 sentence)
    2) 3-6 bullets that are concrete, specific, and non-overlapping
    3) target word count (120-550)
- description: one sentence summarizing the post for previews.

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
    * minimal code sketch / MWE (set requires_code=True for that section)
    * edge cases / failure modes
    * performance/cost considerations
    * security/privacy considerations (if relevant)
    * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
    - Use evidence for up-to-date examples (models/tools/releases) in bullets.
    - Mark sections using fresh info as requires_research=True and requires_citations=True,
      and list the evidence numbers they rely on in evidence_ids.
- Only reference evidence numbers that exist in the provided list.
- Only use blog_kind = "news_roundup" in mode open_book. Never choose it for closed_book or hybrid.
- Mode open_book:
    - Set blog_kind = "news_roundup".
    - Every section is about summarizing events + implications.
    - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
    - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources" and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State, *, deps: Deps) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = deps.models.structured(
        "planner", Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state.get('recency_days')})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n{evidence_block(evidence)}\n\n"
                    f"Instruction: If mode=open_book, your plan must NOT drift into a tutorial."
                )
            ),
        ],
    )

    # open_book always gets news_roundup; other modes never do
    if forced_kind:
        plan.blog_kind = "news_roundup"
    elif plan.blog_kind == "news_roundup":
        plan.blog_kind = "explainer"
        plan.constraints = [c for c in plan.constraints if "tutorial" not in c.lower()]

    for t in plan.tasks:
        t.evidence_ids = valid_ids(t.evidence_ids, len(evidence))

    return {"plan": plan}


# ---------- FANOUT ----------

def fanout(state: State) -> List[Send]:
    plan: Plan = state["plan"]
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state.get("recency_days"),
                "plan": plan.model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in plan.tasks
    ]


# ---------- WORKER ----------

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard (prevents mid-blog topic drift):
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book (weekly news):
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true (hybrid sections):
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning (concepts, intuition) is OK without citations unless requires_citations is true.
- Only cite a URL for a claim its Snippet actually supports. If no snippet supports a specific
  outside-world claim (a product feature, integration, report, number), leave the claim out
  rather than citing a loosely related URL.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.
- Every code block must be complete and syntactically valid, and must end with a closing fence.
- Keep each snippet under ~40 lines. If the full example would be longer, show the core logic
  and describe the rest in prose rather than starting something you can't finish.
- Helpers the snippet calls but doesn't define must appear as clearly marked stubs,
  e.g. `def query_tsdb(params): ...  # your historian client`.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict, *, deps: Deps) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    relevant = ", ".join(f"[{i}]" for i in task.evidence_ids) or "none listed"

    prompt = (
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Constraints: {plan.constraints}\n"
        f"Topic: {payload['topic']}\n"
        f"Mode: {payload.get('mode', 'closed_book')}\n"
        f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"requires_research: {task.requires_research}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"requires_code: {task.requires_code}\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence most relevant to this section: {relevant}\n"
        f"Evidence (ONLY use these URLs when citing; each Snippet shows what that source supports):\n"
        f"{evidence_block(evidence)}\n"
    )

    # Write, check, and retry with the problems spelled out
    feedback = ""
    problems: List[str] = []
    section_md = ""
    for _ in range(1 + deps.settings.worker_retries):
        section_md, meta = deps.models.text(
            "writer", [SystemMessage(content=WORKER_SYSTEM), HumanMessage(content=prompt + feedback)]
        )
        problems = check_markdown(section_md, task.requires_code, meta)
        if not problems:
            break
        feedback = (
            "\n\nYOUR PREVIOUS ATTEMPT HAD THESE PROBLEMS. Rewrite the whole section and fix them:\n- "
            + "\n- ".join(problems)
            + "\nKeep code blocks short and complete."
        )

    # Still broken after retries: close any open fence so the rest of the post renders, and record it
    if section_md.count("```") % 2:
        section_md += "\n```"

    return {
        "sections": [(task.id, section_md)],
        "section_issues": [(task.id, p) for p in problems],
    }


# ---------- REDUCER ----------

def reducer_node(state: State) -> dict:
    plan: Plan = state["plan"]
    # Keep only the latest section per task id (a resumed run can append a task twice)
    by_id = dict(sorted(state["sections"], key=lambda x: x[0]))
    body = "\n\n".join(by_id[k] for k in sorted(by_id)).strip()
    return {
        "draft": body,
        "editor_revision": body,
        "title": plan.blog_title,
        "description": plan.description,
        "final": f"# {plan.blog_title}\n\n{body}\n",
    }


# ---------- GROUNDING CHECK ----------

GROUNDING_SYSTEM = """You are a fact-checking editor for a technical blog. You do not rewrite the post;
you return a short list of precise edits.

Check every claim that cites an evidence URL (a Markdown link to one of the evidence URLs), and every
specific outside-world claim (product features, releases, numbers, benchmarks, pricing, dates, quotes):
- A cited claim must be supported by the Source text of the evidence item it links to
  (use the Snippet only if there is no Source text).
- If a claim is not supported, soften it so it is, move the link to an evidence item that does
  support it, or remove the claim.
- Never add new facts or new URLs. Never edit code blocks. Don't touch evergreen explanations.

Each fix:
- quote: exact text from the post, copied character for character, long enough to be unique
  (usually one full sentence).
- replacement: the corrected text (empty string to remove the quote).
- reason: one short line.

Return an empty list if every claim checks out.
"""


def grounding_node(state: State, *, deps: Deps) -> dict:
    evidence = state.get("evidence", [])
    if not deps.settings.grounding_check or not evidence:
        return {"editor_issues": []}

    body = state["draft"]
    try:
        report = deps.models.structured(
            "editor", GroundingReport,
            [
                SystemMessage(content=GROUNDING_SYSTEM),
                HumanMessage(
                    content=(
                        f"Evidence:\n{evidence_block(evidence, state.get('raw_sources'))}\n\n"
                        f"---\nPOST:\n\n{body}"
                    )
                ),
            ],
        )
    except LLMError as e:
        return {"editor_issues": [], "warnings": [f"grounding check skipped: {e}"]}

    allowed = {e.url for e in evidence}
    issues: List[str] = []
    for fix in report.fixes:
        if body.count(fix.quote) != 1:
            issues.append(f"Not applied (quote not found exactly once): {fix.reason}")
            continue
        new_urls = [u for u in urls_in(fix.replacement) if u not in allowed]
        if new_urls:
            issues.append(f"Not applied (adds URL outside evidence {new_urls[0]}): {fix.reason}")
            continue
        body = body.replace(fix.quote, fix.replacement)
        issues.append(fix.reason)

    plan: Plan = state["plan"]
    return {
        "editor_issues": issues,
        "editor_revision": body,
        "final": f"# {plan.blog_title}\n\n{body}\n",
    }
