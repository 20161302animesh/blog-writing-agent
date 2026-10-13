"""Checkpointer setup, so a failed run can be resumed.

Checkpoints store the graph state, which holds our Pydantic types (evidence, plans). LangGraph's
default serializer loads unregistered types with a warning and will block them in a future
version, so the serializer here allowlists them explicitly.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from .schemas import AnalysisPlan, Beat, EvidenceItem, Plan, Task

# Pydantic types stored in graph state
CHECKPOINT_TYPES = [EvidenceItem, Plan, Task, AnalysisPlan, Beat]


def make_serializer() -> JsonPlusSerializer:
    """Serializer for any LangGraph checkpointer (e.g. PostgresSaver(conn, serde=make_serializer()))."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@contextmanager
def sqlite_checkpointer(path: Union[str, Path]) -> Iterator[SqliteSaver]:
    """SQLite checkpointer for write_blog(checkpointer=...). Creates the file if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)   # as SqliteSaver.from_conn_string does
    try:
        yield SqliteSaver(conn, serde=make_serializer())
    finally:
        conn.close()
