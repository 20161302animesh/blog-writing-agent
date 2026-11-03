# blog_agent

One LangGraph agent that writes two kinds of posts:

- **technical**: outline, parallel section writers with code checks, then an Opus grounding check
- **analysis**: thesis and beats, one writer, then an Opus editor

```
classify -> router -> (research) -+- orchestrator -> [worker x N] -> reducer -> grounding -+- tagger -> save
                                  +- planner -> writer -> editor --------------------------+
```

## Use it

```python
from blog_agent import write_blog

result = write_blog("Will Indian OTT ever have its Ted Lasso moment?", kind="analysis", target_words=800)
print(result.output_path, result.tags, result.estimated_cost_usd)
```

```bash
python -m blog_agent "How LangGraph checkpointers work" --kind technical -v
python -m blog_agent "..." --checkpoint-db runs/checkpoints.sqlite          # resumable
python -m blog_agent --resume <run_id> --checkpoint-db runs/checkpoints.sqlite
python -m blog_agent --help
```

- **Resumable runs from Python:** `with sqlite_checkpointer("runs/checkpoints.sqlite") as cp: write_blog(..., checkpointer=cp)`, then `write_blog("", checkpointer=cp, resume_run_id=run_id)`. For another checkpointer such as Postgres, pass `serde=blog_agent.checkpoint.make_serializer()`.
- **From another agent:** `blog_agent.as_tool()` returns a LangChain tool.
- **From LangGraph Studio:** `langgraph dev` uses `langgraph.json` at the repo root.

## Configure

`Settings.from_env()` reads `.env`. It needs `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`. Optional overrides:

| Variable | Example |
|---|---|
| `BLOG_AGENT_<ROLE>_MODEL` / `_MAX_TOKENS` / `_EFFORT` | roles: classifier, router, research, planner, writer, editor, tagger |
| `BLOG_AGENT_OUTPUT_DIR`, `BLOG_AGENT_QA_DIR` | default `./blog_posts`, `./runs` |
| `BLOG_AGENT_MAX_CONCURRENCY` | parallel section workers (default 3) |
| `BLOG_AGENT_GROUNDING_CHECK` | `false` to skip the technical grounding check |
| `BLOG_AGENT_REFUSAL_FALLBACKS` | `false` to turn off server-side refusal fallback on Sonnet/Opus |

## Trace with LangSmith

Tracing is off by default. To send each run to LangSmith, set:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=blog-agent   # optional; LangSmith uses "default" otherwise
```

Each `write_blog()` call becomes one trace. It has a root run named `write_blog`, the graph nodes under it, and every model call and Tavily search inside its node.

- **Token usage:** LangSmith records input and output tokens on every LLM run. Each model call is tagged `role:<role>` (with matching `role` metadata), so you can filter or group usage by planner, writer, editor and so on.
- **Run totals:** the root run's outputs hold the per-model usage and the package's cost estimate. That estimate uses `PRICES` in `config.py`. If LangSmith shows no cost for a model, add the model's price in LangSmith's model pricing settings.
- **Finding a run:** the root run's metadata includes `blog_run_id` (the QA folder name), `kind`, `resumed` and the model for each role. `BlogResult.trace_id` and `BlogResult.trace_url` point to the trace, and the CLI prints the link.
- **Resumed runs:** a resumed run gets a second trace with the same `blog_run_id`.

## Layout

| File | What it holds |
|---|---|
| `config.py` | Settings, model per role, prices |
| `llm.py` | `Models` wrapper: stop_reason checks, retries, fallbacks |
| `schemas.py` | Pydantic plans, evidence, `BlogResult` |
| `profiles.py` | Per-kind router/research/tagger prompts |
| `research.py` | Tavily search (parallel) and the research node |
| `nodes.py` | classify, router, tagger, save |
| `pipelines/technical.py`, `pipelines/analysis.py` | The two writing branches |
| `graph.py` | `build_graph()` |
| `qa.py` | Per-node snapshots, versions, diffs |
| `checkpoint.py` | `sqlite_checkpointer()` for resumable runs |
| `tracing.py` | Optional LangSmith root run per call |
| `api.py` | `write_blog()`, `as_tool()` |
| `cli.py` | Command line |

## Test

```bash
python -m pytest
```

The tests use fake models and fake search, so they run offline and cost nothing.

## Credits

The technical pipeline (router, research, orchestrator, parallel workers, reducer) started from the [blog-writing-agent](https://github.com/campusx-official/blog-writing-agent) exercise by CampusX (Nitish Singh). See the [project README](../README.md#credits).
