"""Graph entry point for LangGraph Studio / `langgraph dev` (see langgraph.json).

Input in Studio: {"topic": "...", "kind": "auto"}. Other fields get defaults in the classify node.
"""
from .config import Settings
from .graph import build_graph

graph = build_graph(Settings.from_env())
