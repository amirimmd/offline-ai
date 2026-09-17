"""
Offline AI package.

Public entry point:
    from offline_ai import LocalAI
    ai = LocalAI("./workspace")
"""

from __future__ import annotations

__version__ = "0.1.0"

from offline_ai.core.engine import LocalAI

__all__ = ["LocalAI", "__version__"]
