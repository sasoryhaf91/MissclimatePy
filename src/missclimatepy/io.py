# src/missclimatepy/io.py
# SPDX-License-Identifier: MIT
"""
Lightweight I/O helpers for missclimatepy.

- read_csv: thin wrapper around pandas.read_csv with sane defaults.
- write_csv: convenience writer that creates parent dirs if needed.
"""

from __future__ import annotations  # ← Debe ser la primera sentencia (tras el docstring)

import os
from typing import Optional, Dict, Any

import pandas as pd


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_csv(
    path: str,
    *,
    parse_dates: Optional[list[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read CSV with safe defaults.

    Parameters
    ----------
    path : str
        File path.
    parse_dates : list[str] | None
        Columns to parse as dates (optional).
    dtype : dict | None
        Dtype mapping for columns (optional).
    low_memory : bool
        Keep False to avoid mixed dtypes across chunks.
    **kwargs : Any
        Extra arguments forwarded to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(
        path,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=low_memory,
        **kwargs,
    )


def write_csv(df: pd.DataFrame, path: str, *, index: bool = False, **kwargs: Any) -> str:
    """
    Write DataFrame to CSV, creating parent directory if needed.

    Returns
    -------
    str
        The path written.
    """
    _ensure_parent_dir(path)
    df.to_csv(path, index=index, **kwargs)
    return path
