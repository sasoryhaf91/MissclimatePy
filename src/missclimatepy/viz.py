from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Optional

def _ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def plot_station_series(
    df_orig: pd.DataFrame,
    df_imp: pd.DataFrame,
    station: str,
    target: str,
    out_png: Optional[str] = None,
    title_extra: str = "",
):
    """
    Dibuja la serie original (gris), la imputada (línea continua),
    y resalta los puntos imputados (marcadores) para una estación.
    """
    st = str(station)
    o = df_orig[df_orig["station"].astype(str) == st].sort_values("date")
    h = df_imp[df_imp["station"].astype(str) == st].sort_values("date")

    if o.empty or h.empty:
        raise ValueError(f"No data for station {st}")

    # puntos imputados: donde original era NaN y ahora hay valor
    was_nan = o[target].isna()
    x = o["date"].values
    y_imp = h[target].values

    plt.figure(figsize=(10, 4))
    # original
    plt.plot(o["date"], o[target], linewidth=1, alpha=0.35)
    # imputada
    plt.plot(h["date"], h[target], linewidth=1)
    # puntos imputados
    plt.scatter(o.loc[was_nan, "date"], h.loc[was_nan, target], s=12)

    plt.title(f"Station {st} · {target} {title_extra}")
    plt.xlabel("date"); plt.ylabel(target)
    plt.tight_layout()
    if out_png:
        _ensure_dir(out_png)
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_metrics_distribution(
    metrics_df: pd.DataFrame,
    out_png: Optional[str] = None,
    title: str = "Per-station metrics",
    columns: Iterable[str] = ("MAE", "RMSE", "R2"),
):
    """
    Histograma simple para MAE, RMSE y R2.
    """
    pdf = metrics_df.copy()
    n = len(columns)
    for col in columns:
        if col not in pdf.columns:
            raise ValueError(f"Column '{col}' not in metrics_df")

    for col in columns:
        plt.figure(figsize=(6, 3.5))
        plt.hist(pdf[col].dropna(), bins=30)
        plt.title(f"{title}: {col}")
        plt.xlabel(col); plt.ylabel("count")
        plt.tight_layout()
        if out_png:
            base = out_png.replace(".png", f"_{col}.png")
            _ensure_dir(base)
            plt.savefig(base, dpi=150)
            plt.close()
        else:
            plt.show()
