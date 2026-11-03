"""LangSmith tracing: one trace per run, captured with a patched client (no network)."""
from unittest import mock

import langsmith
import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context

from blog_agent import Settings, write_blog
from blog_agent.llm import Models

from fakes import FakeModels, fake_search


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=tmp_path / "posts", qa_dir=tmp_path / "runs")


def test_tracing_off_by_default(settings):
    result = write_blog("Some topic", "analysis", models=FakeModels(), search=fake_search, settings=settings)
    assert result.trace_id is None and result.trace_url is None


def test_run_is_one_trace(settings):
    created, updated = [], []
    client = Client(api_url="http://langsmith.test", api_key="test", auto_batch_tracing=False)
    search = langsmith.traceable(name="search")(fake_search)
    with mock.patch.object(Client, "create_run", lambda self, name, inputs, run_type, **kw:
                           created.append(dict(name=name, inputs=inputs, **kw))), \
         mock.patch.object(Client, "update_run", lambda self, run_id, **kw: updated.append(dict(id=run_id, **kw))), \
         mock.patch("langsmith.run_trees.RunTree.get_url", lambda self: "http://langsmith.test/r/1"):
        with tracing_context(enabled=True, client=client, project_name="tests"):
            result = write_blog("Some topic", "analysis", models=FakeModels(), search=search, settings=settings)

    assert result.trace_url == "http://langsmith.test/r/1"
    assert {str(r["trace_id"]) for r in created} == {result.trace_id}

    root = next(r for r in created if r["name"] == "write_blog")
    assert root["extra"]["metadata"]["blog_run_id"] == result.run_id
    assert root["extra"]["metadata"]["models"]["editor"] == settings.models["editor"].model

    # Searches run in a thread pool but still nest under the research node
    research = next(r for r in created if r["name"] == "research")
    searches = [r for r in created if r["name"] == "search"]
    assert searches and all(r["parent_run_id"] == research["id"] for r in searches)

    root_end = next(u for u in updated if str(u["id"]) == str(root["id"]))
    assert "usage" in root_end["outputs"] and "estimated_cost_usd" in root_end["outputs"]


def test_models_are_tagged_with_their_role(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = Models(Settings()).get("writer")
    assert llm.tags == ["role:writer"] and llm.metadata["role"] == "writer"
