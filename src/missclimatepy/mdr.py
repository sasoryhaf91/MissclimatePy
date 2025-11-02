from __future__ import annotations
import pandas as pd
from typing import Iterable, Dict
from .evaluate import evaluate_per_station

def mdr_grid_search(
    df: pd.DataFrame, target: str,
    missing_fracs: Iterable[float],         # e.g. [0.1, 0.3]
    grid_K: Iterable[int],                  # e.g. [5, 8, 12]
    grid_min_obs: Iterable[int],            # e.g. [30, 60, 120]
    grid_trees: Iterable[int],              # e.g. [200, 300]
    metric_thresholds: Dict[str, float] = {"RMSE": 2.0, "R2": 0.4},
    random_state: int = 42,
):
    rows = []
    for miss in missing_fracs:
        train_frac = 1.0 - float(miss)
        for K in grid_K:
            for mobs in grid_min_obs:
                for trees in grid_trees:
                    res = evaluate_per_station(
                        df_src=df, target=target,
                        train_frac=train_frac, min_obs=mobs,
                        k_neighbors=K, n_estimators=trees,
                        random_state=random_state,
                    )
                    total = int(len(res))
                    passed = int(((res["RMSE"] <= metric_thresholds.get("RMSE", 1e9)) &
                                  (res["R2"]   >= metric_thresholds.get("R2", -1e9))).sum()) if total else 0
                    rows.append({
                        "missing_frac": miss, "train_frac": train_frac,
                        "k_neighbors": K, "min_obs": mobs, "n_estimators": trees,
                        "stations_evaluated": total, "stations_passed": passed,
                        "pass_rate": (passed/total) if total else 0.0
                    })
    return pd.DataFrame(rows).sort_values(["missing_frac","pass_rate"], ascending=[True, False])
