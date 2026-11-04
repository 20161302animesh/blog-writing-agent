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
    assert [row["stage"] for row in root_end["outputs"]["stages"]][:2] == ["classify", "router"]

    # Each node's run carries its stage record, and the record points back at the run
    router_end = next(u for u in updated if u["name"] == "router")
    stage_meta = router_end["extra"]["metadata"]["stage"]
    assert stage_meta["decisions"]["mode"] == "hybrid" and stage_meta["input_tokens"] == 100
    router_record = next(s for s in result.stages if s.stage == "router")
    assert router_record.langsmith_run_id == str(router_end["id"])


def test_models_are_tagged_with_their_role(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = Models(Settings()).get("writer")
    assert llm.tags == ["role:writer"] and llm.metadata["role"] == "writer"


def test_models_report_calls_to_the_running_stage(monkeypatch):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    from blog_agent.stages import StageLedger, stage

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    models = Models(Settings())
    reply = AIMessage("Body.", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
                      response_metadata={"model_name": "claude-opus-5-5", "stop_reason": "end_turn"})
    models._cache["writer"] = GenericFakeChatModel(messages=iter([reply]))

    ledger = StageLedger()
    with ledger.active():
        stage("writer")(lambda state: {"draft": models.text("writer", [HumanMessage("x")])[0]})({})

    (call,) = ledger.records[0].llm_calls
    # The answering model (here a fallback to Opus) is recorded and priced, not the configured Sonnet
    assert (call.role, call.model, call.input_tokens, call.output_tokens) == ("writer", "claude-opus-5-5", 1000, 200)
    assert call.stop_reason == "end_turn" and call.cost_usd == pytest.approx(0.008)
    assert ledger.records[0].decisions == {"words": 1}
