"""Extract recorded runs from runs/ into docs/demo-data.js for the replay page (docs/demo.html).

runs/ stays local (it is not in the repo), so this script only works on a machine that has the runs.
Run from the project root:
    python docs/build_demo_data.py                 # the default runs below
    python docs/build_demo_data.py RUN_ID [...]    # specific runs
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUTPUT = ROOT / "docs" / "demo-data.js"

DEFAULT_RUNS = [
    "20261014-141835_on_demand_skills_for_software_engineering_in_the",   # analysis
    "20261014-142246_on_demand_skills_for_software_engineering_in_the",   # technical, same topic
    "20261013-172152_short_piece_why_do_indian_streaming_platforms",      # analysis
    "20261013-172341_explainer_how_langgraph_checkpointers_work_sqlite",  # technical
]

MODELS = {
    "classify": "Haiku", "router": "Haiku", "research": "Haiku", "tagger": "Haiku",
    "orchestrator": "Sonnet", "worker": "Sonnet", "planner": "Sonnet", "writer": "Sonnet",
    "editor": "Opus", "grounding": "Opus", "reducer": "code", "save": "code",
}


def words(md: str) -> int:
    md = re.sub(r"^# .*$", "", md or "", count=1, flags=re.M)
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    return len(re.findall(r"\b\w+\b", md))


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def sentences(md: str) -> list:
    out = []
    for para in (md or "").split("\n"):
        out.extend(s for s in re.split(r"(?<=[.!?])\s+", para.strip()) if s)
    return out


def changes(before: str, after: str, limit: int = 4) -> list:
    """Up to `limit` before/after sentence pairs the editor changed."""
    a, b = sentences(before), sentences(after)
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if old.startswith("```") or new.startswith("```"):
            continue
        pairs.append({"before": clip(old, 320), "after": clip(new, 320)})
        if len(pairs) == limit:
            break
    return pairs


def load_run(run_id: str) -> dict:
    d = RUNS / run_id
    files = sorted(d.glob("[0-9][0-9]_*.json"))
    t0 = os.path.getmtime(files[0])
    final = json.loads((d / "final_state.json").read_text(encoding="utf-8"))
    first = json.loads(files[0].read_text(encoding="utf-8"))["state"]

    steps, workers = [], []
    prev_end = 0.0
    for f in files[1:]:
        snap = json.loads(f.read_text(encoding="utf-8"))
        node, up, end = snap["node"], snap.get("update") or {}, round(os.path.getmtime(f) - t0, 1)
        if node == "worker":
            sid, md = up["sections"][0]
            heading = next((l[3:].strip() for l in md.splitlines() if l.startswith("## ")), f"Section {sid}")
            workers.append({
                "id": sid, "title": heading, "words": words(md), "code": "```" in md,
                "issues": [p for _, p in up.get("section_issues", [])],
                "text": re.sub(r"^## .*\n+", "", md).strip(), "end": end,   # full text, streamed by the page
            })
            continue
        steps.append({"node": node, "update": up, "end": end})

    # Steps run one after another; each starts when the previous step finished
    nodes = []
    worker_start = None
    for s in steps:
        node, up, end = s["node"], s["update"], s["end"]
        if node == "reducer" and workers:
            worker_start = prev_end
            prev_end = max(w["end"] for w in workers)
        detail = {}
        if node == "classify":
            detail = {"kind": up.get("kind"), "given": first.get("kind") in ("technical", "analysis")}
        elif node == "router":
            detail = {k: up.get(k) for k in ("mode", "needs_research", "queries", "recency_days")}
        elif node == "research":
            detail = {"evidence": [{"title": e["title"], "url": e["url"], "date": e.get("published_at"),
                                    "snippet": clip(e.get("snippet"), 220)} for e in up.get("evidence", [])],
                      "warnings": up.get("warnings", [])}
        elif node == "orchestrator":
            p = up["plan"]
            detail = {"title": p["blog_title"], "description": p["description"], "blog_kind": p["blog_kind"],
                      "tasks": [{"id": t["id"], "title": t["title"], "goal": t["goal"], "words": t["target_words"],
                                 "code": t["requires_code"], "evidence": t["evidence_ids"]} for t in p["tasks"]]}
        elif node == "planner":
            p = up["plan"]
            detail = {"title": p["title"], "thesis": p["thesis"], "piece_kind": p["piece_kind"],
                      "target_words": p["target_words"], "counterpoint": p["counterpoint"],
                      "beats": [{"point": b["point"], "evidence": b["evidence_ids"]} for b in p["beats"]]}
        elif node == "writer":
            detail = {"words": words(up["draft"]), "text": up["draft"].strip()}   # full draft, streamed by the page
        elif node == "reducer":
            detail = {"words": words(up["draft"]), "sections": len(workers)}
        elif node in ("editor", "grounding"):
            before = final.get("draft", "")
            detail = {"issues": up.get("editor_issues", []),
                      "changes": changes(before, up.get("editor_revision", "")),
                      "title": up.get("title"), "words": words(up.get("final", ""))}
        elif node == "tagger":
            detail = {"tags": up.get("tags", [])}
        elif node == "save":
            detail = {"file": Path(up.get("output_path") or "").name, "excerpt": clip(final["final"], 900)}
        nodes.append({"key": node, "model": MODELS.get(node, ""), "start": prev_end, "end": end, "detail": detail})
        prev_end = end

    plan = final["plan"]
    return {
        "id": run_id,
        "topic": final["topic"],
        "kind": final["kind"],
        "mode": final.get("mode"),
        "title": final.get("title"),
        "sources": len(final.get("evidence", [])),
        "duration": nodes[-1]["end"],
        "nodes": nodes,
        "workers": [dict(w, start=worker_start) for w in sorted(workers, key=lambda w: w["id"])],
        "final_words": words(final.get("final", "")),
        "tags": final.get("tags", []),
        "piece": plan.get("piece_kind") or plan.get("blog_kind"),
    }


if __name__ == "__main__":
    ids = sys.argv[1:] or DEFAULT_RUNS
    runs = [load_run(i) for i in ids]
    js = ("// Generated by docs/build_demo_data.py from recorded runs. Do not edit by hand.\n"
          "window.DEMO_RUNS = " + json.dumps(runs, ensure_ascii=False, indent=1) + ";\n")
    OUTPUT.write_text(js, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")
    for r in runs:
        print(f"  {r['kind']:<9} {r['duration']:>6.1f}s  {len(r['nodes'])} steps, {len(r['workers'])} workers  {r['title']}")
