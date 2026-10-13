"""Pure helpers: no model calls, no I/O. All covered by tests/test_utils.py."""
from __future__ import annotations

import ast
import re
from datetime import date
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

from .llm import stop_problem
from .schemas import EvidenceItem

BLOCKED_DOMAINS = {
    "wikipedia.org", "instagram.com", "facebook.com", "x.com", "twitter.com",
    "youtube.com", "reddit.com", "quora.com", "pinterest.com",
}


def to_snake_case(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text)
    return "_".join(word.lower() for word in words)


def slugify(text: str, max_len: int = 50) -> str:
    slug = to_snake_case(text)
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("_", 1)[0]   # cut at the last full word
    return slug.strip("_") or "untitled"


def iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prose_word_count(md: str) -> int:
    """Word count without link URLs or the H1 title."""
    md = re.sub(r"^# .*$", "", md, count=1, flags=re.M)
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    return len(re.findall(r"\b\w+\b", md))


def strip_h1(md: str) -> str:
    return re.sub(r"^# .*\n+", "", md.lstrip(), count=1)


def is_blocked(url: str, blocked: Iterable[str] = BLOCKED_DOMAINS) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in blocked)


def normalize_tag(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def clean_tags(raw: Iterable[str], generic: Iterable[str], max_tags: int) -> List[str]:
    """Normalize, drop generic tags, dedupe (order-preserving), cap."""
    generic = set(generic)
    tags: List[str] = []
    for t in raw:
        n = normalize_tag(t)
        if n and n not in generic and n not in tags:
            tags.append(n)
    return tags[:max_tags]


def valid_ids(ids: Iterable[int], n: int) -> List[int]:
    """Keep only evidence numbers that exist (1..n), without duplicates."""
    out: List[int] = []
    for i in ids:
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out


def evidence_block(evidence: List[EvidenceItem], raw_sources: Optional[Dict[str, str]] = None,
                   source_chars: int = 1500) -> str:
    """Numbered evidence list. With raw_sources, also includes Tavily's original text for checking."""
    if not evidence:
        return "(no evidence gathered)"
    blocks = []
    for i, e in enumerate(evidence, 1):
        block = (f"[{i}] {e.title} | {e.url} | {e.published_at or 'date:unknown'}\n"
                 f"    Snippet: {e.snippet or '(none)'}")
        if raw_sources and raw_sources.get(e.url):
            block += f"\n    Source text: {raw_sources[e.url][:source_chars]}"
        blocks.append(block)
    return "\n".join(blocks)


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def urls_in(text: str) -> List[str]:
    return [u.rstrip(".,;:") for u in URL_RE.findall(text)]


def check_markdown(md: str, requires_code: bool, response_metadata: Optional[dict]) -> List[str]:
    """Problems with a generated Markdown section; empty means it passed."""
    problems: List[str] = []

    problem = stop_problem(response_metadata)
    if problem:
        problems.append(problem)

    if md.count("```") % 2:
        problems.append("a code block is not closed (odd number of ``` fences)")

    blocks = re.findall(r"```([\w+-]*)[^\n]*\n(.*?)```", md, flags=re.S)
    if requires_code and not blocks:
        problems.append("requires_code is true but the section has no code block")

    for lang, body in blocks:
        if lang.lower() in ("python", "py"):
            try:
                ast.parse(body)
            except SyntaxError as e:
                problems.append(f"python code block does not parse: {e.msg} (line {e.lineno})")

    return problems
