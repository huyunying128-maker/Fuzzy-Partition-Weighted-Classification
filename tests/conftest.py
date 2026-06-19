"""Pytest configuration for source-tree imports."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))
