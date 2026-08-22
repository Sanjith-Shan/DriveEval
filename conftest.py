"""Make the package importable without an install step.

Several agents' test files bootstrapped sys.path themselves; this removes the
need and means `pytest tests/` works from a clean checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))
