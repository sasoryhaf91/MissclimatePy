# SPDX-License-Identifier: MIT
"""
High-level API for MissClimatePy
================================

This module exposes a compact, user-facing interface for the core
functionality of MissClimatePy:

- ``ClimateImputer``  : scikit-learn–style class for full-series imputation.
- ``evaluate``        : high-level wrapper around station-wise evaluation.
- ``impute``          : high-level wrapper around dataset imputation.

Design principles
-----------------

* Minimal inputs: station id, coordinates, date, and a single target variable.
* Predictors restricted to space–time only: (latitude, longitude, altitude)
  plus calendar features (year, month, day-of-year, optional harmonics).
* Multiple interchangeable backends sharing the same feature space:

  - "rf"     : RandomForestRegressor
  - "knn"    : KNeighborsRegressor
  - "linear" : LinearRegression
  - "mlp"    : MLPRegressor (ANN)
  - "svd"    : TruncatedSVD + LinearRegression
  - "mcm"    : Mean Climatology Model (pure temporal baseline)

* Full access to all options offered by :mod:`missclimatepy.evaluate`
  and :mod:`missclimatepy.impute`, while keeping a clean, concise API
  for end users and for the JOSS example.

Typical usage
-------------

Full-series imputation for a single variable::

    from missclimatepy.api import ClimateImputer

    imp = ClimateImputer(
        target_col="tmin",
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        start="1981-01-01",
        end="2023-12-31",
        k_neighbors=20,
        include_target_pct=0.0,
        model_kind="rf",   # or "mcm", "knn", "mlp", ...
    )

    imp.fit(df)
    df_imputed = imp.transform()   # or imp.fit_transform(df)

Evaluation of interpolation performance::

    from missclimatepy.api import evaluate

    metrics_df = evaluate(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="1981-01-01",
        end="2020-12-31",
        k_neighbors=20,
        include_target_pct=0.0,
        model_kind="rf",
        min_station_rows=365,
        show_progress=True,
    )

One-shot imputation (no class)::

    from missclimatepy.api import impute

    full = impute(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmax",
        start="1981-01-01",
        end="2023-12-31",
        k_neighbors=20,
        include_target_pct=0.0,
        model_kind="rf",
        show_progress=True,
    )
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from .evaluate import RFParams, evaluate_stations
from .impute import impute_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_rf_params(rf_params: Optional[Union[RFParams, Mapping[str, Any]]]) -> RFParams:
    """
    Normalize RF hyperparameters into an RFParams instance.

    Parameters
    ----------
    rf_params :
        - None        → default RFParams()
        - RFParams    → returned as-is
        - dict/mapping→ merged into default RFParams

    Returns
    -------
    RFParams
    """
    if rf_params is None:
        return RFParams()
    if isinstance(rf_params, RFParams):
        return rf_params
    base = asdict(RFParams())
    base.update(dict(rf_params))
    return RFParams(**base)


# ---------------------------------------------------------------------------
# High-level class: ClimateImputer
# ---------------------------------------------------------------------------


class ClimateImputer:
    """
    High-level, scikit-learn–style imputer for daily climate station records.

    This class is a thin wrapper around :func:`missclimatepy.impute.impute_dataset`,
    exposing all relevant configuration options in a clean interface while
    keeping the implementation fully driven by the low-level engine.

    Notes
    -----
    * The imputer works **per target variable** (single-target).
    * It uses only spatial coordinates (lat, lon, alt) and calendar features
      (year, month, day-of-year, optional harmonic terms) as predictors.
    * Several interchangeable backends are available via ``model_kind``:

      - "rf"     : RandomForestRegressor
      - "knn"    : KNeighborsRegressor
      - "linear" : LinearRegression
      - "mlp"    : MLPRegressor (ANN)
      - "svd"    : TruncatedSVD + LinearRegression
      - "mcm"    : Mean Climatology Model (pure temporal baseline)

    Parameters
    ----------
    target_col : str
        Name of the column to impute (e.g. ``"prec"``, ``"tmin"``).
    id_col, date_col, lat_col, lon_col, alt_col : str, default:
        - ``id_col="station"``
        - ``date_col="date"``
        - ``lat_col="latitude"``
        - ``lon_col="longitude"``
        - ``alt_col="altitude"``
        Column names in the input DataFrame.
    start, end : str or None, optional
        Inclusive date window for imputation. If None, inferred from data.
    add_cyclic : bool, default False
        If True, add sin/cos of day-of-year as predictors.
    feature_cols : sequence of str or None, optional
        Custom feature set. If None, defaults to:
        ``[lat_col, lon_col, alt_col, "year", "month", "doy"]`` (+ cyclic if requested).
    station_ids : iterable or None, optional
        Explicit list of station ids to impute. If None and no other filter is
        given, all stations are imputed.
    prefix : iterable[str] or str or None, optional
        One or more prefixes; any station whose string id starts with any of
        these prefixes will be selected for imputation.
    regex : str or None, optional
        Regular expression applied to the string representation of station ids.
    custom_filter : callable or None, optional
        Callable taking a station id and returning True if it should be imputed.
    k_neighbors : int or None, default 20
        If not None and ``neighbor_map`` is None, build a K-NN haversine
        neighbor map over station centroids, and train each station using only
        its K neighbors (excluding itself). If None, train using all other
        stations.
    neighbor_map : dict or None, optional
        Custom neighbor map, overriding ``k_neighbors``. Dict of
        ``{station_id -> list_of_neighbor_ids}``.
    include_target_pct : float, default 0.0
        Fraction (%) of valid target rows of the **target station** to include
        in the training set for that station. A value of 0.0 enforces strict
        interpolation (pure spatial–temporal model).
    include_target_seed : int, default 42
        Random seed used when sampling target rows for leakage.
    min_station_rows : int or None, optional
        Minimum number of original rows **in the time window** for a station
        to be eligible for imputation.
    model_kind : {"rf", "knn", "linear", "mlp", "svd", "mcm"}, default "rf"
        Backend used for imputation. All backends operate on the same feature
        space. If "mcm", a Mean Climatology Model is used and no neighbor
        information is required.
    rf_params : RFParams or dict or None, optional
        Random Forest hyperparameters (only relevant when
        ``model_kind="rf"``). If None, reasonable defaults are used.
    model_params : mapping or None, optional
        Extra parameters specific to the chosen backend.
    mcm_mode : {"doy", "month", "global"}, default "doy"
        Temporal grouping used by the MCM imputer and fallback.
    mcm_min_samples : int, default 1
        Minimum number of observations required per group to compute a local
        MCM mean. Groups with fewer samples fall back to the global mean.
    show_progress : bool, default False
        If True, print a compact status line per station during imputation.
    """

    def __init__(
        self,
        *,
        target_col: str,
        # schema
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
        # window
        start: Optional[str] = None,
        end: Optional[str] = None,
        # features
        add_cyclic: bool = False,
        feature_cols: Optional[Sequence[str]] = None,
        # station selection
        station_ids: Optional[Iterable[Union[str, int]]] = None,
        prefix: Optional[Iterable[str] | str] = None,
        regex: Optional[str] = None,
        custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
        # neighbors / leakage / MDR
        k_neighbors: Optional[int] = 20,
        neighbor_map: Optional[Dict[Hashable, Iterable[Hashable]]] = None,
        include_target_pct: float = 0.0,
        include_target_seed: int = 42,
        min_station_rows: Optional[int] = None,
        # model
        model_kind: str = "rf",
        rf_params: Optional[Union[RFParams, Mapping[str, Any]]] = None,
        model_params: Optional[Mapping[str, Any]] = None,
        mcm_mode: str = "doy",
        mcm_min_samples: int = 1,
        # logging
        show_progress: bool = False,
    ) -> None:
        if target_col is None or str(target_col).strip() == "":
            raise ValueError("`target_col` must be a non-empty string.")

        # Basic schema
        self.target_col = str(target_col)
        self.id_col = id_col
        self.date_col = date_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.alt_col = alt_col

        # Time window
        self.start = start
        self.end = end

        # Features
        self.add_cyclic = bool(add_cyclic)
        self.feature_cols = list(feature_cols) if feature_cols is not None else None

        # Station selection
        if isinstance(prefix, str):
            prefix = [prefix]
        self.station_ids = list(station_ids) if station_ids is not None else None
        self.prefix = list(prefix) if prefix is not None else None
        self.regex = regex
        self.custom_filter = custom_filter

        # Neighbors / leakage / MDR
        self.k_neighbors = k_neighbors
        self.neighbor_map = neighbor_map
        self.include_target_pct = float(include_target_pct)
        self.include_target_seed = int(include_target_seed)
        self.min_station_rows = min_station_rows

        # Model config
        self.model_kind = str(model_kind).lower()
        self.rf_params_ = _normalize_rf_params(rf_params)
        self.model_params = dict(model_params) if model_params is not None else None
        self.mcm_mode = mcm_mode
        self.mcm_min_samples = int(mcm_min_samples)

        # Logging
        self.show_progress = bool(show_progress)

        # Fitted state
        self.fitted_: bool = False
        self.data_: Optional[pd.DataFrame] = None

        # Store a lightweight config dict (useful for debugging and reproducibility)
        self.config_: Dict[str, Any] = {
            "target_col": self.target_col,
            "id_col": self.id_col,
            "date_col": self.date_col,
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "alt_col": self.alt_col,
            "start": self.start,
            "end": self.end,
            "add_cyclic": self.add_cyclic,
            "feature_cols": self.feature_cols,
            "station_ids": self.station_ids,
            "prefix": self.prefix,
            "regex": self.regex,
            "k_neighbors": self.k_neighbors,
            "min_station_rows": self.min_station_rows,
            "include_target_pct": self.include_target_pct,
            "include_target_seed": self.include_target_seed,
            "model_kind": self.model_kind,
            "mcm_mode": self.mcm_mode,
            "mcm_min_samples": self.mcm_min_samples,
            "show_progress": self.show_progress,
        }

    # ------------------------------------------------------------------ #
    # scikit-learn style API: fit / transform / fit_transform
    # ------------------------------------------------------------------ #

    def fit(self, data: pd.DataFrame) -> "ClimateImputer":
        """
        Store the training data and mark the imputer as fitted.

        This method does not perform the imputation itself; it simply validates
        and stores the DataFrame so that :meth:`transform` (or
        :meth:`fit_transform`) can call the low-level imputation engine.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame.")
        if data.empty:
            raise ValueError("`data` is empty; nothing to fit.")

        self.data_ = data
        self.fitted_ = True
        return self

    def transform(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform full-series imputation using the configuration defined in
        the constructor and the data passed to :meth:`fit`.

        Parameters
        ----------
        data : pandas.DataFrame or None, optional
            If provided, this DataFrame will be used instead of the one passed
            to :meth:`fit`. If None (default), the data stored during
            :meth:`fit` is used.

        Returns
        -------
        pandas.DataFrame
            Minimal, long-format table with columns::

                [id_col, date_col, lat_col, lon_col, alt_col,
                 target_col, "source"]

            where ``source`` is either ``"observed"`` or ``"imputed"``.
        """
        if data is None:
            if not self.fitted_ or self.data_ is None:
                raise RuntimeError(
                    "ClimateImputer has not been fitted yet. "
                    "Call `fit(data)` or pass `data` explicitly to `transform`."
                )
            df = self.data_
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("`data` must be a pandas.DataFrame.")
            df = data

        return impute_dataset(
            df,
            id_col=self.id_col,
            date_col=self.date_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            alt_col=self.alt_col,
            target_col=self.target_col,
            start=self.start,
            end=self.end,
            add_cyclic=self.add_cyclic,
            feature_cols=self.feature_cols,
            station_ids=self.station_ids,
            prefix=self.prefix,
            regex=self.regex,
            custom_filter=self.custom_filter,
            k_neighbors=self.k_neighbors,
            neighbor_map=self.neighbor_map,
            include_target_pct=self.include_target_pct,
            include_target_seed=self.include_target_seed,
            min_station_rows=self.min_station_rows,
            rf_params=self.rf_params_,
            model_kind=self.model_kind,
            model_params=self.model_params,
            mcm_mode=self.mcm_mode,
            mcm_min_samples=self.mcm_min_samples,
            show_progress=self.show_progress,
        )

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method equivalent to calling :meth:`fit` followed by
        :meth:`transform` on the same DataFrame.
        """
        return self.fit(data).transform()


# ---------------------------------------------------------------------------
# High-level function: impute
# ---------------------------------------------------------------------------


def impute(
    data: pd.DataFrame,
    *,
    # schema
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str,
    # window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors / leakage / MDR
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Hashable, Iterable[Hashable]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    # model
    model_kind: str = "rf",
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    mcm_mode: str = "doy",
    mcm_min_samples: int = 1,
    # logging
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    One-shot helper to perform full-series imputation without explicitly
    instantiating :class:`ClimateImputer`.

    This is equivalent to::

        ClimateImputer(...).fit_transform(data)

    See :class:`ClimateImputer` for detailed parameter descriptions.
    """
    imputer = ClimateImputer(
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        start=start,
        end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        station_ids=station_ids,
        prefix=prefix,
        regex=regex,
        custom_filter=custom_filter,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
        min_station_rows=min_station_rows,
        model_kind=model_kind,
        rf_params=rf_params,
        model_params=model_params,
        mcm_mode=mcm_mode,
        mcm_min_samples=mcm_min_samples,
        show_progress=show_progress,
    )
    return imputer.fit_transform(data)


# ---------------------------------------------------------------------------
# High-level function: evaluate
# ---------------------------------------------------------------------------


def evaluate(
    data: pd.DataFrame,
    *,
    # schema
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    # model configuration
    model_kind: str = "rf",
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[Hashable, Iterable[Hashable]]] = None,
    # leakage and MDR
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    # MCM / metrics configuration
    mcm_mode: str = "doy",
    mcm_min_samples: int = 1,
    agg_for_metrics: str = "sum",
    # ordering / I/O / logging
    order_by: Optional[tuple[str, bool]] = ("RMSE_d", True),
    save_table_path: Optional[str] = None,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    High-level wrapper around :func:`missclimatepy.evaluate.evaluate_stations`,
    providing a single entry point for Leave-One-Station-Out (or similar)
    evaluation of interpolation performance.

    The heavy lifting is implemented in :mod:`missclimatepy.evaluate`; this
    function simply normalizes user inputs and calls the low-level engine.
    """
    norm_rf = _normalize_rf_params(rf_params)

    result = evaluate_stations(
        data=data,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        station_ids=station_ids,
        prefix=prefix,
        regex=regex,
        custom_filter=custom_filter,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
        min_station_rows=min_station_rows,
        rf_params=norm_rf,
        model_kind=model_kind,
        model_params=model_params,
        mcm_mode=mcm_mode,
        mcm_min_samples=mcm_min_samples,
        agg_for_metrics=agg_for_metrics,
        order_by=order_by,
        save_table_path=save_table_path,
        log_csv=log_csv,
        flush_every=flush_every,
        show_progress=show_progress,
    )
    return result


__all__ = [
    "ClimateImputer",
    "impute",
    "evaluate",
    "RFParams",
]

