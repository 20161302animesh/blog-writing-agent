"""Builds the combined graph.

START -> classify -> router -> (research) -+- technical: orchestrator -> [worker x N] -> reducer -> grounding -+
                                           +- analysis:  planner -> writer -> editor --------------------------+
                                                                                                tagger -> save -> END
"""
from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import Models
from .nodes import classify_node, route_after_router, router_node, save_node, tagger_node, to_pipeline
from .pipelines import analysis, technical
from .research import research_node, tavily_search
from .state import Deps, SearchFn, State


def build_graph(settings: Optional[Settings] = None, *, models=None, search: Optional[SearchFn] = None,
                checkpointer=None):
    """Compile the graph. `models` and `search` can be swapped for fakes in tests."""
    settings = settings or Settings.from_env()
    deps = Deps(settings=settings, models=models or Models(settings), search=search or tavily_search)

    def node(fn):
        return partial(fn, deps=deps)

    g = StateGraph(State)
    g.add_node("classify", node(classify_node))
    g.add_node("router", node(router_node))
    g.add_node("research", node(research_node))
    # technical
    g.add_node("orchestrator", node(technical.orchestrator_node))
    g.add_node("worker", node(technical.worker_node))
    g.add_node("reducer", technical.reducer_node)
    g.add_node("grounding", node(technical.grounding_node))
    # analysis
    g.add_node("planner", node(analysis.planner_node))
    g.add_node("writer", node(analysis.writer_node))
    g.add_node("editor", node(analysis.editor_node))
    # shared tail
    g.add_node("tagger", node(tagger_node))
    g.add_node("save", node(save_node))

    g.add_edge(START, "classify")
    g.add_edge("classify", "router")
    g.add_conditional_edges("router", route_after_router, ["research", "orchestrator", "planner"])
    g.add_conditional_edges("research", to_pipeline, ["orchestrator", "planner"])

    g.add_conditional_edges("orchestrator", technical.fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", "grounding")
    g.add_edge("grounding", "tagger")

    g.add_edge("planner", "writer")
    g.add_edge("writer", "editor")
    g.add_edge("editor", "tagger")

    g.add_edge("tagger", "save")
    g.add_edge("save", END)

    return g.compile(checkpointer=checkpointer)
