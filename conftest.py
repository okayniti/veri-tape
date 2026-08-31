"""Ensures the repo root is on sys.path for `pytest` (not just `python -m
pytest`), so `import loan_intelligence` and `import demo` resolve the same
way they do for every `python -m loan_intelligence.<pkg>.<module>` command
documented in the README."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
