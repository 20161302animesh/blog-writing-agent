"""Model access for the nodes.

Nodes never build ChatAnthropic themselves; they call `Models.structured()` or `Models.text()`
with a role name. That keeps model choice in Settings, adds stop_reason checks in one place,
and lets tests swap in a fake with the same two methods.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from .config import FALLBACK_MODELS, Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A model call finished but its output can't be used (truncated, refused, unparseable)."""


def stop_problem(response_metadata: Optional[dict]) -> Optional[str]:
    """Describe a bad stop_reason, or None if the response finished normally."""
    meta = response_metadata or {}
    stop = meta.get("stop_reason")
    if stop in (None, "end_turn", "stop_sequence"):
        return None
    if stop == "refusal":
        details = meta.get("stop_details") or {}
        return f"model refused (category: {details.get('category') if isinstance(details, dict) else details})"
    return f"response stopped with stop_reason={stop!r} (usage: {meta.get('usage', {})})"


class Models:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: Dict[str, ChatAnthropic] = {}

    def get(self, role: str) -> ChatAnthropic:
        if role not in self._cache:
            spec = self.settings.models[role]
            kwargs = dict(
                model=spec.model,
                max_tokens=spec.max_tokens,
                max_retries=self.settings.max_retries,
                timeout=self.settings.request_timeout,
            )
            if spec.effort:
                kwargs["effort"] = spec.effort
            if self.settings.refusal_fallbacks and spec.model in FALLBACK_MODELS:
                # On a policy refusal the API re-runs the request on a suitable fallback model
                kwargs["betas"] = ["server-side-fallback-2026-07-01"]
                kwargs["model_kwargs"] = {"fallbacks": "default"}
            self._cache[role] = ChatAnthropic(**kwargs)
        return self._cache[role]

    def structured(self, role: str, schema: Type[T], messages: List[BaseMessage]) -> T:
        runnable = self.get(role).with_structured_output(schema, method="json_schema", include_raw=True)
        out = runnable.invoke(messages)
        problem = stop_problem(getattr(out.get("raw"), "response_metadata", None))
        if problem:
            raise LLMError(f"{role} ({schema.__name__}): {problem}")
        if out.get("parsing_error") or out.get("parsed") is None:
            raise LLMError(f"{role} ({schema.__name__}): could not parse output: {out.get('parsing_error')}")
        return out["parsed"]

    def text(self, role: str, messages: List[BaseMessage]) -> Tuple[str, dict]:
        """Plain text response plus its response_metadata (callers decide how to treat stop_reason)."""
        resp = self.get(role).invoke(messages)
        return resp.text.strip(), resp.response_metadata
