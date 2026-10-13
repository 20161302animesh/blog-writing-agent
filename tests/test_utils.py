from blog_agent.llm import stop_problem
from blog_agent.schemas import EvidenceItem
from blog_agent.utils import (
    check_markdown, clean_tags, evidence_block, is_blocked, prose_word_count, slugify, strip_h1, urls_in,
    valid_ids, yaml_str,
)


def test_slugify_cuts_at_word_boundary():
    slug = slugify("Will Indian TV and OTT ever have their Ted Lasso moment, and what would it take?")
    assert len(slug) <= 50 and not slug.endswith("_")
    assert slug.startswith("will_indian_tv")
    assert slugify("!!!") == "untitled"


def test_is_blocked_matches_subdomains_only():
    assert is_blocked("https://en.wikipedia.org/wiki/X")
    assert is_blocked("https://x.com/user")
    assert not is_blocked("https://notx.com/page")
    assert not is_blocked("https://example.com/youtube.com")


def test_prose_word_count_ignores_title_and_urls():
    md = "# A long title here\n\nOne two [three four](https://example.com/a/b/c) five."
    assert prose_word_count(md) == 5


def test_strip_h1():
    assert strip_h1("# Title\n\nBody") == "Body"
    assert strip_h1("Body only") == "Body only"


def test_yaml_str_escapes_quotes():
    assert yaml_str('He said "hi" \\o/') == '"He said \\"hi\\" \\\\o/"'


def test_clean_tags_normalizes_dedupes_and_drops_generic():
    tags = clean_tags(["Graph RAG", "graph-rag", "AI", "LangGraph", "  "], {"ai"}, 8)
    assert tags == ["graph-rag", "langgraph"]
    assert len(clean_tags([f"t{i}" for i in range(20)], set(), 8)) == 8


def test_valid_ids():
    assert valid_ids([0, 1, 2, 2, 7], 3) == [1, 2]


def test_evidence_block_numbers_and_sources():
    ev = [EvidenceItem(title="T", url="https://a.example", snippet="S")]
    assert evidence_block([]) == "(no evidence gathered)"
    block = evidence_block(ev, {"https://a.example": "x" * 3000})
    assert block.startswith("[1] T | https://a.example | date:unknown")
    assert "Source text: " + "x" * 1500 in block and "x" * 1501 not in block


def test_urls_in_strips_trailing_punctuation():
    assert urls_in("see [x](https://a.example/p). and https://b.example/q,") == [
        "https://a.example/p", "https://b.example/q"]


def test_check_markdown():
    ok = {"stop_reason": "end_turn"}
    assert check_markdown("## S\n\n```python\nx = 1\n```", True, ok) == []
    assert "no code block" in check_markdown("## S\n\ntext", True, ok)[0]
    assert "not closed" in check_markdown("```python\nx = 1\n", False, ok)[0]
    assert "does not parse" in check_markdown("```python\ndef f(:\n```", False, ok)[0]
    assert "max_tokens" in check_markdown("## S", False, {"stop_reason": "max_tokens"})[0]


def test_stop_problem():
    assert stop_problem({"stop_reason": "end_turn"}) is None
    assert stop_problem(None) is None
    assert "refused" in stop_problem({"stop_reason": "refusal", "stop_details": {"category": "cyber"}})
