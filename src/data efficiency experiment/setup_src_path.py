"""Ensure the parent ``src`` directory is importable when running from this folder."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> Path:
    """Insert ``.../src`` at the front of ``sys.path`` and return that path."""
    src_dir = Path(__file__).resolve().parent.parent
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_dir


# Auto-run on import so notebooks can simply: import setup_src_path
ensure_src_on_path()
