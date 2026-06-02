"""Pytest configuration for source-checkout imports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "Lib"

if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))