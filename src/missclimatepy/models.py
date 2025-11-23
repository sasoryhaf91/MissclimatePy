# SPDX-License-Identifier: MIT
"""
missclimatepy.models
====================

Model backends for MissClimatePy.

This module centralizes the creation of scikit-learn regressors so that
`evaluate` and `impute` can switch between different algorithms by
passing a simple string name plus a parameter dictionary.

Supported backends
------------------

- "rf"      : RandomForestRegressor (default)
- "knn"     : KNeighborsRegressor
- "linear"  : LinearRegression
- "ann"     : MLPRegressor (alias: "mlp")
- "svd"     : TruncatedSVD + LinearRegression pipeline

Notes
-----

- The Mean Climatology Model (MCM) is *not* implemented here as a
  scikit-learn estimator, because it is deterministic and already
  handled explicitly in `missclimatepy.metrics`. For evaluation we
  keep MCM as a baseline; for imputation there will be a dedicated
  branch in the low-level imputer (no ML model needed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


def create_regressor(
    backend: str = "rf",
    params: Optional[Dict[str, Any]] = None,
) -> RegressorMixin:
    """
    Create a scikit-learn regressor for the requested backend.

    Parameters
    ----------
    backend : {"rf", "knn", "linear", "ann", "mlp", "svd"}
        Name of the model family to use.
    params : dict, optional
        Keyword arguments passed to the underlying scikit-learn class.
        For "svd", some parameters are interpreted specially (see below).

    Backend details
    ---------------
    - "rf":
        RandomForestRegressor(**params)

    - "knn":
        KNeighborsRegressor(**params)

    - "linear":
        LinearRegression(**params)

    - "ann" / "mlp":
        MLPRegressor(**params)

    - "svd":
        A Pipeline consisting of:
            TruncatedSVD(n_components=..., random_state=...)
            LinearRegression(**regressor_params)

        Recognized `params` keys:
        - "n_components"      : int, default 10
        - "random_state"      : int, default 42
        - "regressor_params"  : dict of kwargs for LinearRegression

    Returns
    -------
    sklearn.base.RegressorMixin
        Unfitted estimator ready to be trained.

    Raises
    ------
    ValueError
        If the backend name is not recognized.
    """
    backend = str(backend).lower().strip()
    params = dict(params or {})

    if backend == "rf":
        return RandomForestRegressor(**params)

    if backend == "knn":
        return KNeighborsRegressor(**params)

    if backend == "linear":
        return LinearRegression(**params)

    if backend in {"ann", "mlp"}:
        return MLPRegressor(**params)

    if backend == "svd":
        n_components = int(params.pop("n_components", 10))
        random_state = params.pop("random_state", 42)
        regressor_params = params.pop("regressor_params", {})
        return Pipeline(
            steps=[
                ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
                ("regressor", LinearRegression(**regressor_params)),
            ]
        )

    raise ValueError(
        f"Unknown backend={backend!r}. "
        "Supported backends are: 'rf', 'knn', 'linear', 'ann', 'mlp', 'svd'."
    )


__all__ = ["create_regressor"]
