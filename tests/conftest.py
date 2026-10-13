import sys
from pathlib import Path

# Make tests/fakes.py and the blog_agent package importable without installing
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
