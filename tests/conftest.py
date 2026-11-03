import os
import sys
from pathlib import Path

# Make tests/fakes.py and the blog_agent package importable without installing
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep a LANGSMITH_TRACING=true in the shell or .env from sending test runs to LangSmith
# (load_dotenv doesn't override variables that are already set)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
