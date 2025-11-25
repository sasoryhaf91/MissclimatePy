# SPDX-License-Identifier: MIT
"""
Model factory for MissClimatePy.

This module centralizes all regression models that can be trained on XYZT-style
features:

    X = (x, y, z, t) = (longitude, latitude, elevation, calendar/aux features)

and a single scalar target (e.g. precipitation, tmin, tmax, evap).

Design goals
------------
- One unified entry point: :func:`make_model`.
- All models configurable via a simple ``model_params`` dict.
- Sensible defaults for each model kind.
- Only CPU-friendly, tabular regressors from scikit-learn (plus optional XGBoost).

Supported model kinds
---------------------
"rf"       : RandomForestRegressor
"etr"      : ExtraTreesRegressor
"gbrt"     : GradientBoostingRegressor
"hgbt"     : HistGradientBoostingRegressor
"linreg"   : LinearRegression
"ridge"    : Ridge
"lasso"    : Lasso
"knn"      : KNeighborsRegressor
"svr"      : SVR (RBF kernel)
"mlp"      : MLPRegressor
"xgb"      : XGBRegressor (if xgboost is installed)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# xgboost is optional
try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore[import]

    _HAS_XGB = True
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Registry of supported models
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, Tuple[str, Any]] = {
    "rf": ("RandomForestRegressor", RandomForestRegressor),
    "etr": ("ExtraTreesRegressor", ExtraTreesRegressor),
    "gbrt": ("GradientBoostingRegressor", GradientBoostingRegressor),
    "hgbt": ("HistGradientBoostingRegressor", HistGradientBoostingRegressor),
    "linreg": ("LinearRegression", LinearRegression),
    "ridge": ("Ridge", Ridge),
    "lasso": ("Lasso", Lasso),
    "knn": ("KNeighborsRegressor", KNeighborsRegressor),
    "svr": ("SVR", SVR),
    "mlp": ("MLPRegressor", MLPRegressor),
    "xgb": ("XGBRegressor", XGBRegressor),  # may be None at runtime
}


# ---------------------------------------------------------------------------
# Default hyperparameters per model kind
# ---------------------------------------------------------------------------


def _default_params(kind: str) -> Dict[str, Any]:
    """
    Return a dict of default hyperparameters for a given model kind.

    These defaults are intentionally conservative and CPU-friendly, while
    being reasonable starting points for climate XYZT regression.
    """
    if kind == "rf":
        return {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 42,
        }
    if kind == "etr":
        return {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "n_jobs": -1,
            "random_state": 42,
        }
    if kind == "gbrt":
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 1.0,
            "random_state": 42,
        }
    if kind == "hgbt":
        return {
            "max_depth": None,
            "learning_rate": 0.05,
            "max_iter": 300,
            "l2_regularization": 0.0,
            "random_state": 42,
        }
    if kind == "linreg":
        return {}  # LinearRegression has good defaults
    if kind == "ridge":
        return {"alpha": 1.0, "random_state": None}
    if kind == "lasso":
        return {"alpha": 0.001, "max_iter": 10000, "random_state": None}
    if kind == "knn":
        return {"n_neighbors": 10, "weights": "distance", "n_jobs": -1}
    if kind == "svr":
        return {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
    if kind == "mlp":
        return {
            "hidden_layer_sizes": (64, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "max_iter": 300,
            "random_state": 42,
        }
    if kind == "xgb":
        # Only used if xgboost is installed
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }

    # Fallback (should not happen if SUPPORTED_MODELS is used consistently)
    return {}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_model(
    model_kind: str = "rf",
    model_params: Dict[str, Any] | None = None,
):
    """
    Create a configured scikit-learn compatible regressor for XYZT modelling.

    Parameters
    ----------
    model_kind : str, default "rf"
        Identifier of the model to construct. See ``SUPPORTED_MODELS`` for
        the full list of options. Comparison is case-insensitive.
    model_params : dict or None
        Optional dictionary of hyperparameters. Any keys here override
        the model-specific defaults returned by :func:`_default_params`.

    Returns
    -------
    estimator
        An unfitted regressor object implementing ``fit(X, y)`` and
        ``predict(X)``.

    Raises
    ------
    ValueError
        If the requested ``model_kind`` is not supported.
    ImportError
        If ``model_kind="xgb"`` but the ``xgboost`` package is not installed.
    """
    kind = (model_kind or "rf").lower()

    if kind not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_kind '{model_kind}'. "
            f"Supported kinds are: {sorted(SUPPORTED_MODELS.keys())}."
        )

    name, ctor = SUPPORTED_MODELS[kind]

    if kind == "xgb":
        if not _HAS_XGB or ctor is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "model_kind='xgb' requires the 'xgboost' package. "
                "Install it or choose a different model_kind."
            )

    # Start from defaults, then override with user-provided params
    params = _default_params(kind)
    if model_params:
        # Explicitly cast to dict to tolerate mappings like ConfigDict, etc.
        params.update(dict(model_params))

    return ctor(**params)


__all__ = [
    "SUPPORTED_MODELS",
    "make_model",
]
