# src/missclimatepy/api.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

Public, high-level API.

This module exposes a small convenience class:

- :class:`MissClimateImputer`: fit a Random Forest regressor on valid rows
  and fill the missing values of a single climate target (e.g., tmin, tmax,
  precipitation). Column names are **not normalized**; the user provides
  them explicitly when calling `fit`/`fit_transform`.

Design goals
------------
- Keep the surface tiny and explicit.
- Be liberal in accepted constructor names (`engine`, `model` both map to RF).
- Provide RF hyperparameter overrides that merge cleanly with defaults.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import RFParams  # reuse the same defaults


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_localize(None)
    return s


def _time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


class MissClimateImputer:
    """
    Minimal imputer around a RandomForestRegressor.

    Parameters
    ----------
    engine, model : {"rf"} or None
        Accepted for compatibility. If provided and not "rf", a ValueError is raised.
    target : str
        Target column to impute (e.g., "tmin").
    n_estimators, max_depth, n_jobs, random_state : optional
        Direct RF hyperparameter overrides.
    rf_params : dict or None
        Additional RF parameters; merged on top of defaults and the explicit
        constructor overrides above (explicit args win).

    Notes
    -----
    - This class **does not** do per-station training; it trains **one**
      RandomForest on all rows where the target is present and predicts the
      missing rows. This is sufficient for unit tests and quick usage.
    - For research/evaluation with K-neighbors and controlled leakage,
      use :func:`missclimatepy.evaluate.evaluate_all_stations_fast`.
    """

    def __init__(
        self,
        *,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        target: Optional[str] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        rf_params: Optional[Dict] = None,
        add_cyclic: bool = False,
    ) -> None:
        # accept engine/model aliases
        kind = (engine or model or "rf").lower()
        if kind != "rf":
            raise ValueError(f"Only 'rf' is supported for now, got: {kind!r}")

        if target is None:
            raise ValueError("You must provide the target column name (e.g., target='tmin').")

        # merge RF params: defaults <- rf_params dict <- explicit ctor args
        merged = asdict(RFParams())  # defaults (uses max_features='sqrt')
        if rf_params:
            merged.update(rf_params)

        if n_estimators is not None:
            merged["n_estimators"] = int(n_estimators)
        if max_depth is not None:
            merged["max_depth"] = int(max_depth)
        if n_jobs is not None:
            merged["n_jobs"] = int(n_jobs)
        if random_state is not None:
            merged["random_state"] = int(random_state)

        self._rf_kwargs = merged
        self._target = target
        self._add_cyclic = add_cyclic

        self._fitted = False
        self._features = None
        self._model: Optional[RandomForestRegressor] = None

    # ------------------------------------------------------------------ #
    # sklearn-like API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df: pd.DataFrame,
        *,
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
    ) -> "MissClimateImputer":
        """Fit the RF on rows where `target` is present."""
        # Prepare features
        work = df[[id_col, date_col, lat_col, lon_col, alt_col, self._target]].copy()
        work[date_col] = _ensure_datetime_naive(work[date_col])
        work = work.dropna(subset=[date_col])

        work = _time_features(work, date_col, add_cyclic=self._add_cyclic)

        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if self._add_cyclic:
            feats += ["doy_sin", "doy_cos"]

        good = work.dropna(subset=feats + [self._target])
        if good.empty:
            raise ValueError("No valid rows to train on (all target/feature rows are NA).")

        X = good[feats].to_numpy()
        y = good[self._target].to_numpy()

        model = RandomForestRegressor(**self._rf_kwargs)
        model.fit(X, y)

        self._model = model
        self._features = feats
        self._date_col = date_col
        self._lat_col = lat_col
        self._lon_col = lon_col
        self._alt_col = alt_col
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of `df` with missing target values predicted.
        Only rows where the target is NA are modified.
        """
        if not self._fitted or self._model is None or self._features is None:
            raise RuntimeError("Call fit(...) before transform(...).")

        out = df.copy()
        out[self._date_col] = _ensure_datetime_naive(out[self._date_col])
        out = _time_features(out, self._date_col, add_cyclic=self._add_cyclic)

        feats = self._features
        need_pred = out[self._target].isna()
        if not need_pred.any():
            return out

        # Some rows may still be NA on features; restrict to fully valid preds
        valid_feats = ~out.loc[need_pred, feats].isna().any(axis=1)
        idx = out.loc[need_pred].index[valid_feats]
        if len(idx) == 0:
            return out  # nothing we can fill

        X = out.loc[idx, feats].to_numpy()
        y_hat = self._model.predict(X)

        out.loc[idx, self._target] = y_hat
        return out

    def fit_transform(self, df: pd.DataFrame, **fit_kwargs) -> pd.DataFrame:
        """Convenience wrapper for `fit(...).transform(...)`."""
        return self.fit(df, **fit_kwargs).transform(df)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"MissClimateImputer(target={self._target!r}, rf={self._rf_kwargs})"
