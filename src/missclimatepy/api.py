# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

Lightweight, generic imputer API + utilities for quick end-to-end runs.

- `MissClimateImputer`: fit/transform imputer using a RandomForest per target.
- Column names are fully generic (user-provided): id/date/lat/lon/alt/target.
- Altitude can be passed as 'altitude' or 'elevation' (automatic fallback).
- RF hyperparameters can be provided via:
    * top-level kwargs (e.g., n_estimators=200, max_depth=20, n_jobs=-1)
    * `rf_params` dict
  Values in `rf_params` override top-level kwargs.

This module intentionally stays minimal: no dependencies on neighbors/evaluate
so tests that only import the imputer remain fast and self-contained.

Example
-------
>>> from missclimatepy import MissClimateImputer
>>> imp = MissClimateImputer(model="rf", target="tmin", n_estimators=50, n_jobs=-1)
>>> filled = imp.fit_transform(df)   # df must include coords/time/alt+target
>>> rep = imp.report(filled)         # {"MAE":..., "RMSE":..., "R2":...}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as _np
import pandas as _pd

# --- Metrics (explicit, version-safe aliases) ---
from sklearn.metrics import mean_absolute_error as _MAE
from sklearn.metrics import mean_squared_error as _MSE
from sklearn.metrics import r2_score as _R2
from sklearn.ensemble import RandomForestRegressor as _RF


# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #
from pandas.api.types import DatetimeTZDtype as _DatetimeTZDtype

def _is_tz(s: _pd.Series) -> bool:
    """True if dtype is timezone-aware datetime (pandas >=2.0 safe)."""
    return isinstance(s.dtype, _DatetimeTZDtype)


def _ensure_datetime_naive(s: _pd.Series) -> _pd.Series:
    """Parse to datetime, drop tz if present."""
    s = _pd.to_datetime(s, errors="coerce")
    if _is_tz(s):  # type: ignore[arg-type]
        s = s.dt.tz_localize(None)
    return s


def _coalesce_alt_col(df: _pd.DataFrame, user_alt_col: str) -> str:
    """
    Return the altitude column name to use:
    - If `user_alt_col` exists, use it.
    - Else if 'elevation' exists, use that.
    - Else raise.
    """
    if user_alt_col in df.columns:
        return user_alt_col
    if "elevation" in df.columns:  # compatibility with older datasets/tests
        return "elevation"
    raise KeyError(
        f"Altitude column '{user_alt_col}' not found and fallback 'elevation' not present. "
        f"Available columns: {list(df.columns)[:10]}..."
    )


def _merge_rf_params(top_level: Dict[str, Any], rf_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge top-level RF kwargs with an rf_params dict (rf_params wins).
    Apply safe defaults compatible with scikit-learn>=1.0.
    """
    base = dict(
        n_estimators=top_level.get("n_estimators", 100),
        max_depth=top_level.get("max_depth", None),
        min_samples_split=top_level.get("min_samples_split", 2),
        min_samples_leaf=top_level.get("min_samples_leaf", 1),
        max_features=top_level.get("max_features", "sqrt"),  # 'auto' deprecated for regressors
        bootstrap=top_level.get("bootstrap", True),
        n_jobs=top_level.get("n_jobs", -1),
        random_state=top_level.get("random_state", 42),
    )
    if rf_params:
        base.update(rf_params)
    return base


def _make_time_features(df: _pd.DataFrame, date_col: str) -> _pd.DataFrame:
    """Add year/month/doy; do not add cyclics here to keep it minimal."""
    out = df.copy()
    out[date_col] = _ensure_datetime_naive(out[date_col])
    out = out.dropna(subset=[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    return out


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
class MissClimateImputer:
    """
    Simple, fast RF-based imputer.

    Parameters
    ----------
    model : {"rf"}, optional
        Currently only "rf" is supported. Kept for forward-compatibility.
    engine : {"rf"}, optional
        Synonym of `model`.
    target : str
        Name of the target variable to impute (e.g., "tmin", "prec").
    id_col, date_col, lat_col, lon_col, alt_col : str
        Generic schema column names. `alt_col` will fallback to 'elevation'
        automatically if 'altitude' is not found.
    rf_params : dict, optional
        Dict with RandomForest hyperparameters. Keys override any top-level
        kwargs with the same name (e.g., n_estimators, max_depth).
    **kwargs :
        Passed as top-level RF default overrides (e.g., n_estimators=200).
        Also allows ignoring unknown args used in tests (e.g., min_obs_per_station).

    Notes
    -----
    - `fit()` learns from rows where target is present (not NA).
    - `transform()` predicts on rows where target is NA; other rows are left as-is.
    - Coordinates + (year, month, doy) are used as features by default.
    """

    # ---- constructor ----
    def __init__(
        self,
        *,
        model: str = "rf",
        engine: Optional[str] = None,
        target: str,
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
        rf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # accept model/engine synonyms; only rf for now
        self._engine = (engine or model or "rf").lower()
        if self._engine != "rf":
            raise ValueError(f"Only 'rf' engine is available for now, got: {self._engine}")

        # schema
        self._target = target
        self._id_col = id_col
        self._date_col = date_col
        self._lat_col = lat_col
        self._lon_col = lon_col
        self._alt_col_user = alt_col  # may fallback to 'elevation' later

        # RF params (merge top-level kwargs with rf_params dict)
        self._rf_kwargs = _merge_rf_params(kwargs, rf_params)

        # placeholders set in fit()
        self._model: Optional[_RF] = None
        self._features_: Optional[list[str]] = None
        self._alt_col_resolved_: Optional[str] = None

        # keep original input for reporting
        self._last_input_df_: Optional[_pd.DataFrame] = None

    # ---- fit ----
    def fit(self, df: _pd.DataFrame, *, start: Optional[str] = None, end: Optional[str] = None) -> "MissClimateImputer":
        """
        Train the RF on rows with observed target.

        Parameters
        ----------
        df : DataFrame
            Must contain id/date/lat/lon/alt/target columns under the names configured.
        start, end : str or None
            Optional inclusive window to clip training by date.

        Returns
        -------
        self
        """
        for c in [self._id_col, self._date_col, self._lat_col, self._lon_col, self._target]:
            if c not in df.columns:
                raise KeyError(f"Column '{c}' not found. Available: {list(df.columns)[:10]}...")

        # alt/elev resolution
        alt_col = _coalesce_alt_col(df, self._alt_col_user)
        self._alt_col_resolved_ = alt_col

        # Preserve original input for report()
        self._last_input_df_ = df.copy()

        # pre-process (time features + optional date clipping)
        work = _make_time_features(
            df[[self._id_col, self._date_col, self._lat_col, self._lon_col, alt_col, self._target]].copy(),
            self._date_col,
        )

        if start or end:
            lo = _pd.to_datetime(start) if start else work[self._date_col].min()
            hi = _pd.to_datetime(end) if end else work[self._date_col].max()
            work = work[(work[self._date_col] >= lo) & (work[self._date_col] <= hi)]

        # training mask: rows with valid features & observed target
        feats = [self._lat_col, self._lon_col, alt_col, "year", "month", "doy"]
        self._features_ = feats

        valid_X = ~work[feats].isna().any(axis=1)
        valid_y = work[self._target].notna()
        mask = valid_X & valid_y

        train = work.loc[mask]
        if train.empty:
            raise ValueError("No valid rows to train on (features or target missing).")

        X = train[feats].to_numpy(copy=False)
        y = train[self._target].to_numpy(copy=False)

        # Fit RF
        self._model = _RF(**self._rf_kwargs)
        self._model.fit(X, y)

        return self

    # ---- transform ----
    def transform(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Impute missing target rows using the fitted RF. Returns a copy.

        Parameters
        ----------
        df : DataFrame
            Same schema as used in `fit()`.

        Returns
        -------
        DataFrame
            Copy of input with target column imputed on NA rows (when features are valid).
        """
        if self._model is None or self._features_ is None or self._alt_col_resolved_ is None:
            raise ValueError("Call fit() before transform().")

        out = df.copy()
        alt_col = _coalesce_alt_col(out, self._alt_col_resolved_)

        # prepare features on the fly (same ops as fit)
        work = _make_time_features(
            out[[self._id_col, self._date_col, self._lat_col, self._lon_col, alt_col, self._target]].copy(),
            self._date_col,
        )

        feats = list(self._features_)
        # Identify rows to impute: target is NA and features are present
        need = work[self._target].isna() & (~work[feats].isna().any(axis=1))

        if need.any():
            X = work.loc[need, feats].to_numpy(copy=False)
            yhat = self._model.predict(X)
            # align back to original index
            out.loc[need.index[need], self._target] = yhat

        return out

    # ---- convenience: fit + transform ----
    def fit_transform(self, df: _pd.DataFrame, **fit_kwargs: Any) -> _pd.DataFrame:
        """Shortcut for `fit(df, **fit_kwargs).transform(df)`."""
        return self.fit(df, **fit_kwargs).transform(df)

    # ---- simple report over last fit vs given (imputed) df ----
    def report(self, df: _pd.DataFrame) -> dict:
        """
        Compute MAE / RMSE / R2 comparing the imputed `df` (after transform)
        against the original observed values captured during `fit()`.

        Parameters
        ----------
        df : DataFrame
            The imputed dataframe returned by `transform()` / `fit_transform()`.

        Returns
        -------
        dict
            {"MAE": float, "RMSE": float, "R2": float}
        """
        if getattr(self, "_last_input_df_", None) is None:
            raise ValueError("report() requires a prior fit(); original data not found.")

        idc = self._id_col
        datec = self._date_col
        y = self._target

        prev = self._last_input_df_[[idc, datec, y]].copy()
        now = df[[idc, datec, y]].copy()

        # normalize dates to naive datetime
        prev[datec] = _ensure_datetime_naive(prev[datec])
        now[datec] = _ensure_datetime_naive(now[datec])

        # Align and keep rows where the original had observed target
        m = prev.merge(now, on=[idc, datec], suffixes=("_true", "_pred"))
        m = m[m[f"{y}_true"].notna()]

        y_true = m[f"{y}_true"].to_numpy()
        y_pred = m[f"{y}_pred"].to_numpy()

        if y_true.size == 0:
            return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

        mae = float(_MAE(y_true, y_pred))

        # Use version-safe path (no 'squared' kw): RMSE = sqrt(MSE)
        mse = float(_MSE(y_true, y_pred))
        rmse = float(mse ** 0.5)

        # R2 only when >1 sample and non-degenerate variance
        r2 = float(_R2(y_true, y_pred)) if y_true.size >= 2 and float(_np.var(y_true)) > 0.0 else float("nan")

        return {"MAE": mae, "RMSE": rmse, "R2": r2}
