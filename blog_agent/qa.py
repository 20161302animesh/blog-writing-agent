"""QA snapshots for a run: one JSON file per node, the final state, draft versions and diffs.

Layout of runs/<run_id>/:
    00_input.json, NN_<node>.json (update + state_after), final_state.json
    v1_draft.md, v2_editor_revision.md, v3_final.md
    diff_1_draft_to_editor.diff, diff_2_editor_to_final.diff
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel

from .utils import strip_h1


def to_jsonable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def sentences(md: str) -> List[str]:
    """One sentence per line, so diffs show the sentence that changed, not the whole paragraph."""
    lines: List[str] = []
    for para in md.strip().split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(s for s in re.split(r"(?<=[.!?])\s+", para.strip()) if s)
    return lines


def write_diff(path: Path, before: str, after: str, before_name: str, after_name: str) -> None:
    diff = difflib.unified_diff(sentences(before), sentences(after),
                                fromfile=before_name, tofile=after_name, lineterm="")
    text = "\n".join(diff)
    path.write_text(text + "\n" if text else "(no changes)\n", encoding="utf-8")


def snapshot_name(node: str, update: dict) -> str:
    # Parallel workers all share the node name "worker"; tell them apart by task id
    if node == "worker" and update.get("sections"):
        return f"worker_task{update['sections'][0][0]:02d}"
    return node


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # A resumed run continues numbering after the existing snapshots
        self.step = len(list(self.run_dir.glob("[0-9][0-9]_*.json")))

    def record_step(self, updates: list, state: dict) -> None:
        if not updates:
            save_json(self.run_dir / f"{self.step:02d}_input.json", {"step": self.step, "node": "input", "state": state})
        for node, update in updates:
            save_json(
                self.run_dir / f"{self.step:02d}_{snapshot_name(node, update)}.json",
                {"step": self.step, "node": node, "update": update, "state_after": state},
            )
        self.step += 1

    def finish(self, state: dict) -> None:
        save_json(self.run_dir / "final_state.json", state)
        draft = state.get("draft", "")
        revision = state.get("editor_revision", "") or draft
        final_body = strip_h1(state.get("final", ""))
        versions = {"v1_draft.md": draft, "v2_editor_revision.md": revision, "v3_final.md": final_body}
        for name, text in versions.items():
            (self.run_dir / name).write_text(text, encoding="utf-8")
        write_diff(self.run_dir / "diff_1_draft_to_editor.diff", draft, revision, "draft", "editor_revision")
        write_diff(self.run_dir / "diff_2_editor_to_final.diff", revision, final_body, "editor_revision", "final")
