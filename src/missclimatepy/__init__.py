# SPDX-License-Identifier: MIT
"""
Top-level exports for missclimatepy.

We expose:
- MissClimateImputer (from .api)
- RFParams, evaluate_all_stations_fast (from .evaluate)
- Optional: run_quickstart, QuickstartConfig (from .quickstart if present)
"""

from __future__ import annotations

# Core API
from .api import MissClimateImputer  # <- only this from api

# Evaluation utilities
from .evaluate import RFParams, evaluate_all_stations_fast

# Optional quickstart: re-export only if quickstart.py exists
try:
    from .quickstart import run_quickstart, QuickstartConfig  # type: ignore
except Exception:
    def run_quickstart(*args, **kwargs):  # type: ignore
        raise ImportError(
            "missclimatepy.quickstart is not available. "
            "Include quickstart.py or import MissClimateImputer / "
            "evaluate_all_stations_fast directly."
        )

    class QuickstartConfig:  # type: ignore
        """Stub to avoid import errors when quickstart is absent."""
        pass

__all__ = [
    "MissClimateImputer",
    "RFParams",
    "evaluate_all_stations_fast",
    "run_quickstart",
    "QuickstartConfig",
]
