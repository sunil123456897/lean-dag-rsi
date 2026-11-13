"""Core package for the Lean-DAG-RSI & SLEEP-DAG-RSI v8 engine modules."""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.append(_PKG_DIR)
