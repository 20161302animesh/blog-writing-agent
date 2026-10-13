"""Settings: models per role, output folders, concurrency and retries.

Everything can be overridden from the environment (or a .env file), e.g.
    BLOG_AGENT_EDITOR_MODEL=claude-opus-5-5
    BLOG_AGENT_WRITER_MAX_TOKENS=16000
    BLOG_AGENT_ROUTER_EFFORT=low
    BLOG_AGENT_OUTPUT_DIR=D:/blog/posts
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelSpec:
    model: str
    max_tokens: int
    effort: Optional[str] = None   # None = the model's default effort


# Haiku for cheap structured steps, Sonnet for planning and writing, Opus for editing.
# max_tokens includes thinking tokens (thinking is on by default on these models), so leave headroom.
DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "classifier": ModelSpec("claude-haiku-5-5", 2048),
    "router":     ModelSpec("claude-haiku-5-5", 2048),
    "research":   ModelSpec("claude-haiku-5-5", 8192),
    "planner":    ModelSpec("claude-sonnet-5-5", 8192),
    "writer":     ModelSpec("claude-sonnet-5-5", 16000),
    "editor":     ModelSpec("claude-opus-5-5", 16000),
    "tagger":     ModelSpec("claude-haiku-5-5", 2048),
}

# USD per 1M tokens (input, output), used only for the cost estimate in BlogResult.
# Prices as of 2026-10; update when pricing changes.
PRICES: Dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Models that accept the server-side refusal fallback ("fallbacks": "default")
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class MissingCredentialsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    models: Dict[str, ModelSpec] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    output_dir: Path = field(default_factory=lambda: Path("blog_posts").resolve())
    qa_dir: Path = field(default_factory=lambda: Path("runs").resolve())

    max_concurrency: int = 3          # parallel graph nodes (section workers); keeps under rate limits
    search_concurrency: int = 4       # parallel Tavily queries
    max_retries: int = 3              # Anthropic SDK retries (429, 5xx, connection errors)
    request_timeout: float = 600.0    # seconds per Anthropic request
    refusal_fallbacks: bool = True    # server-side fallback on refusals (Sonnet/Opus 5.5 only)

    worker_retries: int = 1           # technical: rewrites of a section that fails its checks
    grounding_check: bool = True      # technical: Opus checks cited claims against sources
    max_evidence: int = 20            # evidence items kept after research

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, **overrides) -> "Settings":
        load_dotenv(env_file)
        models = {}
        for role, spec in DEFAULT_MODELS.items():
            prefix = f"BLOG_AGENT_{role.upper()}_"
            models[role] = ModelSpec(
                model=os.getenv(prefix + "MODEL", spec.model),
                max_tokens=int(os.getenv(prefix + "MAX_TOKENS", spec.max_tokens)),
                effort=os.getenv(prefix + "EFFORT") or spec.effort,
            )

        values = {"models": models}
        if os.getenv("BLOG_AGENT_OUTPUT_DIR"):
            values["output_dir"] = Path(os.environ["BLOG_AGENT_OUTPUT_DIR"]).resolve()
        if os.getenv("BLOG_AGENT_QA_DIR"):
            values["qa_dir"] = Path(os.environ["BLOG_AGENT_QA_DIR"]).resolve()
        if os.getenv("BLOG_AGENT_MAX_CONCURRENCY"):
            values["max_concurrency"] = int(os.environ["BLOG_AGENT_MAX_CONCURRENCY"])
        if os.getenv("BLOG_AGENT_GROUNDING_CHECK"):
            values["grounding_check"] = os.environ["BLOG_AGENT_GROUNDING_CHECK"].lower() in ("1", "true", "yes")
        if os.getenv("BLOG_AGENT_REFUSAL_FALLBACKS"):
            values["refusal_fallbacks"] = os.environ["BLOG_AGENT_REFUSAL_FALLBACKS"].lower() in ("1", "true", "yes")

        values.update(overrides)
        return cls(**values)

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **changes)

    def check_credentials(self) -> None:
        missing = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
        if missing:
            raise MissingCredentialsError(
                f"Missing environment variables: {', '.join(missing)}. Set them in your shell or .env file."
            )
