# SPDX-License-Identifier: MIT
"""
missclimatepy.viz
=================

Lightweight matplotlib plots for:
- missingness heatmaps
- metric distributions
- predicted vs observed scatter

(We avoid seaborn to keep runtime deps minimal.)
"""
from __future__ import annotations
from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_missing_heatmap(missing_matrix: pd.DataFrame, *, figsize=(10, 4)):
    """Heatmap from masking.missing_matrix (1=missing, 0=observed)."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(missing_matrix.T, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(missing_matrix.shape[1]))
    ax.set_yticklabels(missing_matrix.columns.astype(str))
    ax.set_xlabel("Date index")
    ax.set_ylabel("Station")
    ax.set_title("Missingness (1=missing, 0=observed)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_metric_boxplots(
    report: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("MAE_d", "RMSE_d", "R2_d"),
    group_by: Optional[str] = "include_target_pct",
    figsize=(10, 4),
):
    """
    Boxplots of metrics grouped by a column (e.g., include_target_pct or used_k_neighbors).
    """
    groups = [None] if group_by is None else sorted(report[group_by].dropna().unique())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, figsize[1]))
    if n == 1:
        axes = [axes]
    for i, m in enumerate(metrics):
        ax = axes[i]
        if group_by is None:
            ax.boxplot([report[m].dropna().values], labels=[m])
        else:
            data = [report.loc[report[group_by] == g, m].dropna().values for g in groups]
            ax.boxplot(data, labels=[str(g) for g in groups])
            ax.set_title(m)
            ax.set_xlabel(group_by)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_pred_vs_obs(
    preds: pd.DataFrame,
    *,
    sample: int = 20000,
    figsize=(5, 5),
):
    """
    Scatter of y_obs vs y_mod from evaluate_all_stations_fast(return_predictions=True).
    """
    df = preds[["y_obs", "y_mod"]].dropna()
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df["y_obs"].values, df["y_mod"].values, s=8, alpha=0.4)
    lims = [np.nanmin(df.values), np.nanmax(df.values)]
    ax.plot(lims, lims, lw=2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Observed vs Predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def compare_reports_long(
    named_reports: dict[str, pd.DataFrame],
    *,
    take: Iterable[str] = ("MAE_d", "RMSE_d", "R2_d"),
) -> pd.DataFrame:
    """
    Convert {method: report_df} into a long tidy table for comparison.
    """
    rows = []
    for name, rep in named_reports.items():
        for m in take:
            if m in rep:
                r = rep[["station", m]].copy()
                r["metric"] = m
                r["value"] = r[m].astype(float)
                r["method"] = name
                rows.append(r[["station", "method", "metric", "value"]])
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
