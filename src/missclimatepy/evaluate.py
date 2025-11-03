# src/missclimatepy/evaluate.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .api import MissClimateImputer
from .neighbors import neighbor_distances  # builds nearest-neighbor tables


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _safe_var(x: np.ndarray) -> float:
    """Return variance as float, safe for empty arrays."""
    if x.size == 0:
        return 0.0
    return float(np.var(x))


def _as_str_list(x: Iterable[Any]) -> str:
    """Serialize an iterable of scalars as a comma-separated string."""
    return ",".join(str(i) for i in x)


# ---------------------------------------------------------------------
# Masking for evaluation-only stations
# ---------------------------------------------------------------------
def mask_by_inclusion(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    min_obs: int,
    random_state: int = 42,
    only_stations: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a masked copy of `df` where we hide the target values ONLY for the
    stations we want to evaluate. All other stations remain intact and can be
    fully used for training. This lets us train on the *full* network while
    evaluating just a subset.

    Parameters
    ----------
    df : DataFrame
        Must contain at least ["station", target].
    target : str
        Target column to mask.
    train_frac : float in [0, 1]
        Percentage of valid observations kept for *training* inside each
        evaluated station (the rest are masked and used for validation).
        - 0.0 == strict LOSO (no rows from target station kept in training).
        - 0.8 == keep 80% for training, mask 20% for validation.
    min_obs : int
        Minimum valid rows in a station to consider it for masking/evaluation.
    random_state : int
        RNG seed for sampling.
    only_stations : iterable of str or None
        If given, only those stations are masked/evaluated. Otherwise, mask all.

    Returns
    -------
    (masked_df, info_df)
      - masked_df : DataFrame with masked `target` on evaluated stations.
      - info_df   : Per-station counts of total/train/valid used during masking.
    """
    assert 0.0 <= train_frac <= 1.0
    rng = np.random.default_rng(random_state)

    masked = df.copy()
    masked["station"] = masked["station"].astype(str)

    eval_set: Optional[set] = None
    if only_stations is not None:
        eval_set = set(str(s) for s in only_stations)

    info_rows: List[Dict[str, Any]] = []
    for st, sub in masked.groupby("station", sort=False):
        idx_obs = sub[sub[target].notna()].index.values
        if len(idx_obs) < int(min_obs):
            continue

        # Mask only when station is in the evaluation set (or if no filter provided).
        do_mask = (eval_set is None) or (st in eval_set)

        if do_mask:
            n_train = int(len(idx_obs) * train_frac)
            if n_train > 0:
                train_idx = rng.choice(idx_obs, size=n_train, replace=False)
                valid_idx = np.setdiff1d(idx_obs, train_idx)
            else:
                train_idx = np.array([], dtype=int)
                valid_idx = idx_obs

            masked.loc[valid_idx, target] = np.nan
            info_rows.append(
                {
                    "station": st,
                    "n_total": int(len(idx_obs)),
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                }
            )
        # else: leave this station untouched; it will contribute fully to training.

    return masked, pd.DataFrame(info_rows)


# ---------------------------------------------------------------------
# Neighbor correlation/overlap (for metadata columns)
# ---------------------------------------------------------------------
def _neighbor_stats(
    df_long: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    val_col: str,
    station: str,
    neighbor_ids: List[str],
    min_overlap: int = 60,
) -> Tuple[List[str], List[float], List[int], List[str], List[float]]:
    """
    Compute per-neighbor Pearson correlations and overlap counts (days in common)
    between a target station and each neighbor.

    Returns tuple of lists all aligned to `neighbor_ids` order:
      (neighbors_ids, neighbors_corr, neighbors_overlap, top5_ids, top5_corr)

    Notes
    -----
    - Correlation is set to NaN if overlap < `min_overlap` or variance is zero.
    - Date alignment is done with an inner join on [date_col].
    """
    if not neighbor_ids:
        return [], [], [], [], []

    # Subset only the rows we need, pivot to a (date x station) wide matrix
    df = df_long[[id_col, date_col, val_col]].copy()
    df = df[df[id_col].isin([station] + neighbor_ids)]

    wide = df.pivot_table(index=date_col, columns=id_col, values=val_col, aggfunc="mean")
    wide = wide.sort_index()

    if station not in wide.columns:
        return neighbor_ids, [np.nan] * len(neighbor_ids), [0] * len(neighbor_ids), [], []

    s_main = wide[station]
    corr_list: List[float] = []
    olap_list: List[int] = []

    for nid in neighbor_ids:
        if nid not in wide.columns:
            corr_list.append(np.nan)
            olap_list.append(0)
            continue
        pair = pd.concat([s_main, wide[nid]], axis=1, join="inner").dropna()
        n = int(len(pair))
        if n < int(min_overlap):
            corr_list.append(np.nan)
            olap_list.append(n)
            continue
        a = pair.iloc[:, 0].to_numpy()
        b = pair.iloc[:, 1].to_numpy()
        if _safe_var(a) == 0.0 or _safe_var(b) == 0.0:
            corr_list.append(np.nan)
            olap_list.append(n)
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        corr_list.append(corr)
        olap_list.append(n)

    # Top-5 neighbors by correlation (descending), ignoring NaNs
    order = (
        pd.Series(corr_list, index=neighbor_ids)
        .sort_values(ascending=False, na_position="last")
        .index.tolist()
    )
    top_ids = [nid for nid in order if pd.notna(pd.Series(corr_list, index=neighbor_ids).get(nid))][:5]
    top_corr = [float(pd.Series(corr_list, index=neighbor_ids).get(nid)) for nid in top_ids]

    return neighbor_ids, corr_list, olap_list, top_ids, top_corr


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def evaluate_per_station(
    df_src: pd.DataFrame,
    target: str,
    train_frac: float = 0.7,
    min_obs: int = 60,
    engine: str = "rf",
    k_neighbors: Optional[int] = None,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    eval_stations: Optional[Iterable[str]] = None,
    neighbor_map: Optional[Dict[str, List[str]]] = None,
    corr_min_overlap: int = 60,
) -> pd.DataFrame:
    """
    Train on the full (period-filtered) universe and evaluate ONLY the stations
    in `eval_stations` (if provided). Masking is applied *only* to those stations.

    The returned DataFrame includes metrics and rich metadata:

        ["station","n_valid","MAE","RMSE","R2",
         "rows_train","rows_test","used_k_neighbors",
         "neighbors_ids","neighbors_corr","neighbors_overlap",
         "neighbors_top5","neighbors_top5_corr"]

    Parameters
    ----------
    df_src : DataFrame
        Long-format table with at least:
          ["station","latitude","longitude","date", target]
    target : str
        Target column to impute/predict.
    train_frac : float
        0.0 = strict LOSO; 0.8 = keep 80% for training inside evaluated stations.
    min_obs : int
        Minimum valid rows for a station to be even considered for evaluation.
    engine : str
        Model engine name passed to `MissClimateImputer` (e.g., "rf").
    k_neighbors : int or None
        If given, the imputer may restrict training/prediction to local neighbors
        when making per-station predictions.
    model_params : dict or None
        Forwarded to the imputer (e.g., {"rf_params": {...}}).
    random_state : int
        RNG seed for masking and underlying models.
    n_jobs : int
        Parallelism for the underlying model (when supported).
    eval_stations : iterable of str or None
        Subset of stations to evaluate; if None, evaluate all.
    neighbor_map : dict or None
        Optional precomputed neighbor map {station -> [neighbor_ids]} used both
        by the imputer and for correlation metadata. If None and k_neighbors is
        provided, it is built here from coordinates.
    corr_min_overlap : int
        Minimum overlapping days when computing neighbor correlations.

    Returns
    -------
    DataFrame with metrics and metadata per evaluated station.
    """
    # Normalize station as string
    df_src = df_src.copy()
    df_src["station"] = df_src["station"].astype(str)

    # 1) Decide which stations are evaluated
    if eval_stations is None:
        eval_set = set(df_src["station"].unique().tolist())
    else:
        eval_set = set(str(s) for s in eval_stations)

    # 2) Mask ONLY evaluated stations; keep others intact for training
    masked, info_df = mask_by_inclusion(
        df_src, target, train_frac, min_obs,
        random_state=random_state, only_stations=eval_set
    )

    # 3) Build neighbor map if requested and not provided
    if k_neighbors is not None and neighbor_map is None:
        nn = neighbor_distances(
            df_src[["station", "latitude", "longitude"]].drop_duplicates("station"),
            k=int(k_neighbors) + 1  # self + k neighbors
        )
        nmap: Dict[str, List[str]] = {}
        for st, rows in nn.groupby("station"):
            # sort by distance and take first k that are not itself
            ids = rows.sort_values("distance_km")["neighbor"].astype(str).tolist()
            nmap[str(st)] = [i for i in ids if i != str(st)][: int(k_neighbors)]
        neighbor_map = nmap

    # 4) Global imputer (it can still use local neighbors internally)
    imp = MissClimateImputer(
        engine=engine,
        target=target,
        k_neighbors=k_neighbors,
        min_obs_per_station=max(0, int(min_obs * train_frac)),  # 0 for strict LOSO
        n_jobs=n_jobs,
        random_state=random_state,
        model_params=model_params,
    )

    # Fit & transform across the whole dataset (masked only on eval stations)
    imp_df = imp.fit_transform(masked)

    # 5) Collect metrics per evaluated station + metadata
    rows: List[Dict[str, Any]] = []

    # Preselect columns for correlation stats (avoid copying inside loop)
    has_date = "date" in df_src.columns
    if not has_date:
        # For correlation metadata we need a date column; if not present, we skip it.
        pass

    for st, sub in df_src.groupby("station", sort=False):
        if st not in eval_set:
            continue

        # indices where that station was masked (validation set)
        msub = masked[masked["station"] == st]
        idx = msub[msub[target].isna() & sub[target].notna()].index
        if len(idx) == 0:
            # nothing to score (either not enough obs or train_frac ~ 1.0)
            continue

        y_true = sub.loc[idx, target].to_numpy()
        y_pred = imp_df.loc[idx, target].to_numpy()
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        if ok.sum() == 0:
            continue

        # ---- Neighbor metadata ----
        used_k = int(k_neighbors or 0)
        neigh_ids: List[str] = []
        neigh_corr: List[float] = []
        neigh_olap: List[int] = []
        top_ids: List[str] = []
        top_corr: List[float] = []

        if used_k > 0 and neighbor_map is not None:
            neigh_ids = neighbor_map.get(st, [])[:used_k]
            if has_date and len(neigh_ids) > 0:
                ids, corr, olap, top5_ids, top5_corr = _neighbor_stats(
                    df_src,
                    id_col="station",
                    date_col="date",
                    val_col=target,
                    station=st,
                    neighbor_ids=neigh_ids,
                    min_overlap=int(corr_min_overlap),
                )
                neigh_ids, neigh_corr, neigh_olap = ids, corr, olap
                top_ids, top_corr = top5_ids, top5_corr

        rows.append(
            {
                "station": st,
                "n_valid": int(ok.sum()),
                "MAE": mean_absolute_error(y_true[ok], y_pred[ok]),
                "RMSE": mean_squared_error(y_true[ok], y_pred[ok], squared=False),
                "R2": r2_score(y_true[ok], y_pred[ok]),
                # training / test sizes
                "rows_train": int(masked[target].notna().sum()),
                "rows_test": int(len(idx)),
                # neighbors metadata
                "used_k_neighbors": used_k,
                "neighbors_ids": _as_str_list(neigh_ids),
                "neighbors_corr": _as_str_list(f"{c:.6f}" if pd.notna(c) else "nan" for c in neigh_corr),
                "neighbors_overlap": _as_str_list(neigh_olap),
                "neighbors_top5": _as_str_list(top_ids),
                "neighbors_top5_corr": _as_str_list(f"{c:.6f}" if pd.notna(c) else "nan" for c in top_corr),
            }
        )

    cols = [
        "station", "n_valid", "MAE", "RMSE", "R2",
        "rows_train", "rows_test",
        "used_k_neighbors",
        "neighbors_ids", "neighbors_corr", "neighbors_overlap",
        "neighbors_top5", "neighbors_top5_corr",
    ]
    return (
        pd.DataFrame(rows, columns=cols).sort_values("RMSE").reset_index(drop=True)
        if rows else
        pd.DataFrame(columns=cols)
    )
