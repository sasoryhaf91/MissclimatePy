# src/missclimatepy/viz.py
from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Optional

def _ensure_dir(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

def plot_inclusion_aggregate(sweep_df: pd.DataFrame, metric: str="RMSE",
                             out_png: Optional[str]=None,
                             quantiles: Iterable[float]=(0.1,0.5,0.9),
                             title: str="Aggregate inclusion curves"):
    if metric not in sweep_df.columns: raise ValueError(f"'{metric}' not in sweep_df")
    grp = sweep_df.groupby("include_pct")[metric].quantile(quantiles).unstack()
    plt.figure(figsize=(7,4))
    for q in grp.columns:
        plt.plot(grp.index, grp[q], marker="o", label=f"q{int(float(q)*100)}")
    plt.xlabel("include_pct"); plt.ylabel(metric); plt.title(title); plt.legend()
    plt.tight_layout()
    if out_png: _ensure_dir(out_png); plt.savefig(out_png, dpi=150); plt.close()
    else: plt.show()

def plot_inclusion_curves(sweep_df: pd.DataFrame, station: str,
                          metrics: Iterable[str]=("RMSE","R2"),
                          out_png: Optional[str]=None, title_extra: str=""):
    g = sweep_df[sweep_df["station"].astype(str)==str(station)]
    if g.empty: return
    plt.figure(figsize=(8,4))
    for m in metrics:
        if m in g.columns:
            plt.plot(g["include_pct"], g[m], marker="o", label=m)
    plt.xlabel("include_pct"); plt.title(f"Station {station} {title_extra}"); plt.legend()
    plt.tight_layout()
    if out_png: _ensure_dir(out_png); plt.savefig(out_png, dpi=150); plt.close()
    else: plt.show()
