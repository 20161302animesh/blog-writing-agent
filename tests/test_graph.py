"""End-to-end graph runs with fake models and search (no network, no cost)."""
import json

import pytest

from blog_agent import Settings, write_blog
from blog_agent.schemas import RouterDecision

from fakes import URL_A, URL_B, FakeModels, fake_search, raising


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=tmp_path / "posts", qa_dir=tmp_path / "runs")


def run(settings, models, **kw):
    return write_blog(kw.pop("topic", "Some topic"), models=models, search=fake_search, settings=settings, **kw)


def test_analysis_pipeline(settings):
    models = FakeModels()
    result = run(settings, models, kind="auto", target_words=20)

    assert result.kind == "analysis"                       # decided by the classifier
    assert [e.url for e in result.evidence] == [URL_A, URL_B]   # blocked + invented URLs dropped
    assert result.title == "Better Title"                  # editor's revised title wins
    assert result.description == "Yes, but slowly."
    assert result.markdown.startswith("# Better Title\n\n")
    assert result.tags == ["indian-ott", "streaming-economics"]
    assert any(i.startswith("Length pass") for i in result.editor_issues)
    assert result.plan["beats"][0]["evidence_ids"] == [1]  # out-of-range id removed

    text = open(result.output_path, encoding="utf-8").read()
    assert text.startswith('---\ntitle: "Better Title"\n')
    assert "kind: analysis\npiece_kind: analysis\n" in text
    assert "tags: [indian-ott, streaming-economics]" in text

    qa = settings.qa_dir / result.run_id
    for name in ["final_state.json", "v1_draft.md", "v2_editor_revision.md", "v3_final.md",
                 "diff_1_draft_to_editor.diff", "diff_2_editor_to_final.diff"]:
        assert (qa / name).exists(), name
    assert any(p.name.endswith("_editor.json") for p in qa.iterdir())


def test_technical_pipeline(settings):
    models = FakeModels()
    result = run(settings, models, kind="technical")

    assert result.kind == "technical"
    assert not any(c[1] == "KindDecision" for c in models.calls)   # explicit kind skips the classifier
    assert result.plan["blog_kind"] == "explainer"          # news_roundup corrected outside open_book
    assert result.plan["tasks"][0]["evidence_ids"] == [1]
    # Sections in task-id order; the broken code block was retried and fixed
    body = result.markdown
    assert body.index("## First") < body.index("## Second")
    assert "print('ok')" in body and result.section_issues == []
    # Grounding: valid fix applied, bad quote and new-URL fixes rejected
    assert "The market grew 12%." in body and "50%" not in body
    assert "evil.example" not in body
    assert sum(i.startswith("Not applied") for i in result.editor_issues) == 2
    assert result.description == "A technical post."
    qa = settings.qa_dir / result.run_id
    assert (qa / "diff_1_draft_to_editor.diff").read_text(encoding="utf-8") != "(no changes)\n"
    assert any(p.name.endswith("_worker_task01.json") for p in qa.iterdir())


def test_closed_book_skips_research(settings):
    models = FakeModels({"RouterDecision": lambda: RouterDecision(needs_research=False, mode="closed_book")})
    result = run(settings, models, kind="technical", save=False, qa=False)
    assert result.evidence == [] and result.output_path is None and result.qa_dir is None
    assert not any(c[1] == "EvidencePack" for c in models.calls)
    assert not any(c[1] == "GroundingReport" for c in models.calls)   # nothing to check against


def test_search_failure_degrades_gracefully(settings):
    def broken_search(*a):
        raise RuntimeError("tavily down")
    result = write_blog("t", "analysis", models=FakeModels(), search=broken_search, settings=settings, qa=False)
    assert result.evidence == []
    assert any("tavily down" in w for w in result.warnings)
    assert any("no usable results" in w for w in result.warnings)


def test_editor_failure_keeps_draft(settings):
    models = FakeModels({"EditResult": raising("editor: response stopped with stop_reason='max_tokens'")})
    result = run(settings, models, kind="analysis", target_words=2, qa=False)
    assert result.title == "Analysis Title"
    assert result.markdown == "# Analysis Title\n\nDraft body.\n"
    assert any("unedited draft" in w for w in result.warnings)


def test_open_book_drops_old_evidence(settings):
    from blog_agent.schemas import EvidenceItem, EvidencePack
    models = FakeModels({
        "RouterDecision": lambda: RouterDecision(needs_research=True, mode="open_book", queries=["q"]),
        "EvidencePack": lambda: EvidencePack(evidence=[
            EvidenceItem(title="old", url=URL_A, published_at="2020-01-01"),
            EvidenceItem(title="undated", url=URL_B),
        ]),
    })
    result = run(settings, models, kind="technical", as_of="2026-10-13", qa=False, save=False)
    assert [e.url for e in result.evidence] == [URL_B]
    assert result.plan["blog_kind"] == "news_roundup"


def test_resume_after_failure(settings, tmp_path, caplog):
    from blog_agent import sqlite_checkpointer

    calls = {"n": 0}

    def flaky_tags():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("network blip")
        from blog_agent.schemas import BlogTags
        return BlogTags(tags=["resumed-tag", "another-tag"])

    models = FakeModels({"BlogTags": flaky_tags})
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as cp:
        with pytest.raises(RuntimeError, match="network blip"):
            run(settings, models, kind="analysis", checkpointer=cp)
        run_id = next(settings.qa_dir.iterdir()).name
        assert (settings.qa_dir / run_id / "error.txt").exists()

        planner_calls = sum(c[1] == "AnalysisPlan" for c in models.calls)
        result = write_blog("", models=models, search=fake_search, settings=settings,
                            checkpointer=cp, resume_run_id=run_id)

    assert result.run_id == run_id and result.tags == ["resumed-tag", "another-tag"]
    assert "unregistered type" not in caplog.text   # state types are allowlisted for the checkpointer
    assert sum(c[1] == "AnalysisPlan" for c in models.calls) == planner_calls   # earlier nodes not re-run
    final = json.loads((settings.qa_dir / run_id / "final_state.json").read_text(encoding="utf-8"))
    assert final["plan"]["title"] == "Better Title"
