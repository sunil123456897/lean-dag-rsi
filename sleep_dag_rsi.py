"""
Backward-compatible thin facade for `sleep_dag_rsi`.

The canonical implementation lives at `dream_rsi_scripts/sleep_dag_rsi.py`.
This root shim preserves existing `from sleep_dag_rsi import ...` callers
without duplicating code.
"""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dream_rsi_scripts"
)
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dream_rsi_scripts import sleep_dag_rsi as _impl  # noqa: E402

sys.modules[__name__] = _impl
