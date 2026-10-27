# blog-writing-agent

A LangGraph agent, built on Claude and Tavily, that researches, plans, writes, edits and tags blog posts. It writes two kinds of piece:

| Kind | Pipeline |
|---|---|
| **technical** (engineering, AI, dev tools) | Outline of 5–9 sections → parallel section writers, with code checks and a retry → Opus grounding check of cited claims |
| **analysis** (media, culture, business, opinion) | Thesis plus argument beats → one writer → Opus editor that critiques and revises |

```
classify -> router -> (research) -+- orchestrator -> [worker x N] -> reducer -> grounding -+- tagger -> save
                                  +- planner -> writer -> editor --------------------------+
```

Each run saves the post as Markdown with front matter. It also writes a QA folder with per-node snapshots, draft versions, sentence-level diffs, token usage and an estimated cost.

## Quick start

```bash
pip install -e .            # Python 3.11+
cp .env.example .env        # add ANTHROPIC_API_KEY and TAVILY_API_KEY
python -m blog_agent "Will Indian OTT ever have its Ted Lasso moment?" --kind analysis --words 800 -v
```

```python
from blog_agent import write_blog

result = write_blog("How LangGraph checkpointers work", kind="technical")
print(result.title, result.output_path, result.estimated_cost_usd)
```

`kind="auto"` lets a classifier choose. A typical run costs $0.20–0.40.

## Learn more

Project site: https://20161302animesh.github.io/blog-writing-agent/ (landing page, [README](https://20161302animesh.github.io/blog-writing-agent/readme.html), [how it works](https://20161302animesh.github.io/blog-writing-agent/docs/how-it-works.html), and the [walkthrough](https://20161302animesh.github.io/blog-writing-agent/docs/walkthrough.html)).

After re-running `walkthrough.ipynb`, rebuild its web page with `python docs/build_walkthrough.py`.

- [docs/how-it-works.html](docs/how-it-works.html): a one-page visual explainer of the pipeline, the two writing branches and the quality checks. Download it and open it in a browser.
- [walkthrough.ipynb](walkthrough.ipynb): a guided tour that runs the analysis pipeline one node at a time, then the technical pipeline end to end. It includes the QA output, resuming a failed run, and how to extend the package. Set `LIVE = False` to run it offline for free.
- [blog_agent/README.md](blog_agent/README.md): configuration, module layout and entry points (Python, CLI, LangChain tool, LangGraph Studio).
- The two notebooks the package was built from:
  - [blog_writing_agent_v3_research_finetuned_claude_mod.ipynb](blog_writing_agent_v3_research_finetuned_claude_mod.ipynb) (technical)
  - [blog_analysis_agent_v1.ipynb](blog_analysis_agent_v1.ipynb) (analysis)

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests run both pipelines end to end with fake models and fake search, so they need no API keys and cost nothing.
