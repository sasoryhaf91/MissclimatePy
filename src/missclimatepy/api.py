# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

High-level public API for **missclimatepy**.

This module exposes a lightweight imputer class that is intentionally
simple and test-friendly:

- ``MissClimateImputer``: a Random Forest based imputer that learns a global
  mapping from coordinates + calendar features to a target climate variable
  and fills missing values. It supports:
    * Custom column names for station, date, latitude, longitude,
      and altitude (with graceful fallback to 'elevation').
    * Optional cyclic (sin/cos) features for day-of-year seasonality.
    * Minimum number of observed rows per station (filtering).
    * RandomForest hyperparameters provided either as explicit kwargs
      or via ``rf_params`` dict (merged with precedence rules).

Notes
-----
This class focuses on being stable for unit tests and everyday usage.
The *per-station / K-neighbors* evaluators and more advanced tooling live
in :mod:`missclimatepy.evaluate` and :mod:`missclimatepy.neighbors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    """Parse to datetime and drop tz-info if present."""
    s = pd.to_datetime(s, errors="coerce")
    # Future-proof: pandas deprecates is_datetime64tz_dtype; keep robust.
    try:
        import pandas as _pd  # local alias to avoid mypy noise
        if _pd.api.types.is_datetime64tz_dtype(s):
            s = s.dt.tz_localize(None)
    except Exception:
        # Fallback for older/newer pandas versions:
        try:
            # If dtype has .tz or is a DatetimeTZDtype, normalize:
            if getattr(s.dtype, "tz", None) is not None:
                s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool) -> pd.DataFrame:
    """Append year/month/day-of-year (+ optional sine/cosine) to a copy of df."""
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def _resolve_altitude_column(df: pd.DataFrame, alt_col: str) -> str:
    """
    Return the name of the altitude column to use. If `alt_col` is not present
    but 'elevation' is, use 'elevation'. Raise a clear error otherwise.
    """
    if alt_col in df.columns:
        return alt_col
    if "elevation" in df.columns:
        return "elevation"
    raise KeyError(
        f"Altitude column '{alt_col}' not found and fallback 'elevation' is also missing. "
        f"Available columns include: {list(df.columns)[:10]}..."
    )


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise a helpful error if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. "
                       f"Available={list(df.columns)[:10]}...")


# --------------------------------------------------------------------------- #
# Public Imputer
# --------------------------------------------------------------------------- #

@dataclass
class _ImputerConfig:
    """
    Internal container for imputer configuration. We keep defaults explicit and
    simple; merging with user kwargs happens in MissClimateImputer.__init__.
    """
    id_col: str = "station"
    date_col: str = "date"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    alt_col: str = "altitude"        # will fallback to 'elevation' if absent

    target: str = "tmin"             # climate variable to impute
    add_cyclic: bool = False         # add sin/cos(doy)
    min_obs_per_station: int = 0     # station-level filtering (observed rows)

    # RandomForest defaults (merged with rf_params and explicit kwargs)
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, int, float]] = None  # None|'sqrt'|'log2'|int|float
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: Optional[int] = None


class MissClimateImputer:
    """
    Random-Forest-based imputer for daily climate series.

    Parameters
    ----------
    model : str, optional
        Model engine name; accepted values: {"rf"}. (Alias: ``engine``.)
    engine : str, optional
        Same as ``model``. If both given, ``model`` takes precedence.
    target : str
        Name of the target column to impute (e.g., "tmin", "prec", ...).
    id_col, date_col, lat_col, lon_col, alt_col : str
        Column names in the input DataFrame. If ``alt_col`` is not found,
        the class will try fallback to "elevation".
    add_cyclic : bool, default False
        Add sine/cosine features of day-of-year.
    min_obs_per_station : int, default 0
        Require at least this many observed (non-missing) target rows per station
        for the station to be used for *training*. This does not drop stations
        from the *output*; it only controls what goes into the model.
    rf_params : dict, optional
        RandomForest hyperparameters to merge with defaults. Explicit kwargs
        like ``n_estimators`` or ``max_depth`` override both defaults and
        this dict.
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    max_features, bootstrap, n_jobs, random_state :
        Top-level RF kwargs for convenience and backwards compatibility.

    Notes
    -----
    * The model is trained **globally** on all rows with a valid target using
      features: [lat, lon, altitude/elevation, year, month, doy, (doy_sin, doy_cos)*].
    * Only rows with missing target are *predicted/filled* at transform time.
    * Existing (observed) target values are preserved.
    """

    def __init__(
        self,
        *,
        model: str = "rf",
        engine: Optional[str] = None,
        # columns
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
        # target & features
        target: str = "tmin",
        add_cyclic: bool = False,
        min_obs_per_station: int = 0,
        # RF params (dict) + explicit convenience kwargs
        rf_params: Optional[Dict[str, Any]] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        max_features: Optional[Union[str, int, float]] = None,
        bootstrap: Optional[bool] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        # Accept aliases: model/engine (tests may send either)
        engine_normalized = (model or engine or "rf").lower()
        if engine_normalized not in {"rf"}:
            raise ValueError(f"Unsupported engine/model '{engine_normalized}'. Only 'rf' is supported.")

        self._engine = engine_normalized

        # Build config with defaults then merge dict + explicit kwargs (explicit wins)
        cfg = _ImputerConfig(
            id_col=id_col,
            date_col=date_col,
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            target=target,
            add_cyclic=add_cyclic,
            min_obs_per_station=int(min_obs_per_station) if min_obs_per_station is not None else 0,
        )

        # Start with defaults as a dict
        merged: Dict[str, Any] = {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_split": cfg.min_samples_split,
            "min_samples_leaf": cfg.min_samples_leaf,
            "max_features": cfg.max_features,  # None by default (valid in sklearn>=1.1)
            "bootstrap": cfg.bootstrap,
            "n_jobs": cfg.n_jobs,
            "random_state": cfg.random_state,
        }
        # Merge rf_params (if provided)
        if rf_params:
            merged.update({k: v for k, v in rf_params.items() if v is not None})
        # Merge explicit kwargs (highest precedence)
        for k, v in [
            ("n_estimators", n_estimators),
            ("max_depth", max_depth),
            ("min_samples_split", min_samples_split),
            ("min_samples_leaf", min_samples_leaf),
            ("max_features", max_features),
            ("bootstrap", bootstrap),
            ("n_jobs", n_jobs),
            ("random_state", random_state),
        ]:
            if v is not None:
                merged[k] = v

        # Store user-facing config & RF kwargs
        self._cfg = cfg
        self._rf_kwargs = merged

        # Learned objects (set in fit)
        self._rf: Optional[RandomForestRegressor] = None
        self._alt_col_resolved: Optional[str] = None
        self._feature_cols: Optional[Sequence[str]] = None

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "MissClimateImputer":
        """
        Fit the underlying model on rows with a valid target.

        Parameters
        ----------
        df : DataFrame
            Must contain the configured columns for station, date, lat, lon and
            altitude/elevation and the target variable.

        Returns
        -------
        self
        """
        cfg = self._cfg

        # Resolve altitude column (altitude|elevation)
        alt_col = _resolve_altitude_column(df, cfg.alt_col)
        self._alt_col_resolved = alt_col

        _require_columns(df, [cfg.id_col, cfg.date_col, cfg.lat_col, cfg.lon_col, alt_col, cfg.target])

        work = df[[cfg.id_col, cfg.date_col, cfg.lat_col, cfg.lon_col, alt_col, cfg.target]].copy()
        work[cfg.date_col] = _ensure_datetime_naive(work[cfg.date_col])
        work = work.dropna(subset=[cfg.date_col])

        # Optional station filtering for training set based on observed target rows
        if cfg.min_obs_per_station and cfg.min_obs_per_station > 0:
            obs_counts = (
                work[work[cfg.target].notna()]
                .groupby(cfg.id_col)[cfg.target]
                .size()
                .rename("n_obs")
            )
            keep_ids = set(obs_counts[obs_counts >= int(cfg.min_obs_per_station)].index.tolist())
            if keep_ids:
                work = work[work[cfg.id_col].isin(keep_ids)]

        # Add time features
        work = _add_time_features(work, cfg.date_col, cfg.add_cyclic)

        # Define features
        feats = [cfg.lat_col, cfg.lon_col, alt_col, "year", "month", "doy"]
        if cfg.add_cyclic:
            feats += ["doy_sin", "doy_cos"]
        self._feature_cols = feats

        # Training rows = target present + all required features present
        train = work.dropna(subset=feats + [cfg.target])
        if train.empty:
            # Nothing to learn from; keep a dummy regressor to avoid surprises
            self._rf = RandomForestRegressor(n_estimators=1, random_state=cfg.random_state)
            self._rf.fit(np.zeros((1, len(feats))), np.array([0.0]))
            return self

        X = train[feats].to_numpy(copy=False)
        y = train[cfg.target].to_numpy(copy=False)

        # Build RF with merged kwargs
        rf = RandomForestRegressor(**self._rf_kwargs)
        self._rf = rf.fit(X, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of ``df`` with missing target values filled by the model.

        Only rows with ``NaN`` in the target column are modified. Existing
        observed values are preserved.

        Parameters
        ----------
        df : DataFrame

        Returns
        -------
        DataFrame
            Same columns as input with the target column imputed.
        """
        if self._rf is None or self._feature_cols is None:
            raise RuntimeError("MissClimateImputer must be fitted before calling transform().")

        cfg = self._cfg
        alt_col = self._alt_col_resolved or _resolve_altitude_column(df, cfg.alt_col)
        _require_columns(df, [cfg.id_col, cfg.date_col, cfg.lat_col, cfg.lon_col, alt_col, cfg.target])

        out = df.copy()
        out[cfg.date_col] = _ensure_datetime_naive(out[cfg.date_col])
        out = _add_time_features(out, cfg.date_col, cfg.add_cyclic)

        feats = list(self._feature_cols)

        # Predict only where target is missing and features are available
        need = out[cfg.target].isna()
        if not need.any():
            return out  # nothing to do

        # Ensure all features exist (gracefully if user removed them)
        missing_feats = [c for c in feats if c not in out.columns]
        if missing_feats:
            raise KeyError(f"Missing required feature columns at transform-time: {missing_feats}")

        mask = need & (~out[feats].isna().any(axis=1))
        if not mask.any():
            # No rows with target missing AND full features present; return unchanged
            return out

        Xmiss = out.loc[mask, feats].to_numpy(copy=False)
        yhat = self._rf.predict(Xmiss)

        out.loc[mask, cfg.target] = yhat
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``fit(df).transform(df)``."""
        return self.fit(df).transform(df)


# Backwards-compatible alias (if external code expects this name)
__all__ = [
    "MissClimateImputer",
]
