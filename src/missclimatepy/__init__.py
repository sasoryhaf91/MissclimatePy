# SPDX-License-Identifier: MIT
"""
missclimatepy
=============

Top-level package exports for **missclimatepy**.

We keep imports minimal and stable to avoid circular-import issues during test
collection. Public objects are re-exported here for a convenient API:

- ``MissClimateImputer``: a lightweight, test-friendly imputer wrapper.
- ``evaluate_all_stations_fast``: station-wise evaluator with optional
  K-neighborhood training and controlled inclusion of target rows.
- ``RFParams``: a small dataclass for RandomForest hyperparameters.

Version is obtained from package metadata when available.
"""

from __future__ import annotations

# ---- Version -----------------------------------------------------------------
try:
    # Python 3.8+: read version from installed package metadata
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

__version__: str
try:
    __version__ = version("missclimatepy") if version else "0.0.0"
except PackageNotFoundError:  # during local dev / editable installs
    __version__ = "0.0.0"

# ---- Public API re-exports ---------------------------------------------------
from .api import MissClimateImputer  # noqa: E402
from .evaluate import evaluate_all_stations_fast, RFParams  # noqa: E402

__all__ = [
    "MissClimateImputer",
    "evaluate_all_stations_fast",
    "RFParams",
    "__version__",
]
