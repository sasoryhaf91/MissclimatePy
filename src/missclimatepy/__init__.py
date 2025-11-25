# SPDX-License-Identifier: MIT
"""
missclimatepy
=============

Spatial–temporal imputation tools for daily climate station records.

Durante la refactorización interna, este módulo expone únicamente la
versión del paquete. Los puntos de entrada de alto nivel (`impute`,
`evaluate`, `ClimateImputer`) viven en los submódulos correspondientes
y pueden importarse directamente, por ejemplo:

    from missclimatepy.impute import impute_dataset
    from missclimatepy.evaluate import evaluate_stations
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------

try:  # pragma: no cover - durante desarrollo el paquete puede no estar instalado
    __version__ = version("missclimatepy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
