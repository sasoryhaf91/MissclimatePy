"""
example_mdr.py

Minimum Data Requirement (MDR) experiment using MissClimatePy.

This script:
  1. Downloads the public SMN daily dataset (1991–2020) from Zenodo.
  2. Filters stations from the State of Mexico (prefix "15").
  3. Runs station-wise evaluation for a range of include_target_pct values.
  4. Saves a combined MDR report to CSV.

WARNING:
  - This example may take several minutes depending on your hardware.
  - For quick testing, reduce N_STATIONS_MAX or the MDR_PCTS list.

Reference dataset:
  Antonio-Fernández, H., Vaquera-Huerta, H., Rosengaus-Monshinsky, M. M., 
  Pérez-Rodríguez, P., & Crossa, J. (2025). Daily Meteorological Records for 
  Operational SMN Stations in Mexico (1991–2020) with NASA POWER Co-located Variables 
  (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17636066
"""

from __future__ import annotations

import pathlib
from typing import Iterable, List

import pandas as pd

from missclimatepy.evaluate import evaluate_stations

# ---------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------

# Zenodo CSV with daily SMN records, 1991–2020
ZENODO_URL = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"

# Temporal window for the experiment
START = "1991-01-01"
END = "2020-12-31"

# Station selection: Estado de México (prefix "15")
STATION_PREFIXES: List[str] = ["15"]

# Minimum number of observed rows per station (MDR baseline)
MIN_STATION_ROWS = 365 * 10  # ~10 years of data

# Maximum number of stations to keep (for speed)
N_STATIONS_MAX = 40

# MDR percentages to test (fraction of target station used in training)
MDR_PCTS: Iterable[float] = (0, 4, 8, 16, 20, 40, 60, 80)

# Target variable
TARGET_COL = "tmin"

# Output path
OUT_CSV = pathlib.Path("mdr_report_tmin_mex_state.csv")


# ---------------------------------------------------------------------
# Helper: load and pre-filter data
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load daily SMN data from Zenodo and return a pre-filtered DataFrame."""
    print(f"[load] Reading data from {ZENODO_URL!r} ...")
    df = pd.read_csv(ZENODO_URL, parse_dates=["date"])

    # Basic schema checks (will raise if missing)
    required = {"station", "date", "latitude", "longitude", "altitude", TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # Temporal filter
    mask_time = (df["date"] >= START) & (df["date"] <= END)
    df = df.loc[mask_time].copy()

    # Prefix-based station filter (Estado de México)
    if STATION_PREFIXES:
        station_str = df["station"].astype(str)
        mask_prefix = station_str.str.startswith(tuple(STATION_PREFIXES))
        df = df.loc[mask_prefix].copy()

    # For speed: limit number of stations
    unique_st = df["station"].unique()
    if len(unique_st) > N_STATIONS_MAX:
        keep = unique_st[:N_STATIONS_MAX]
        df = df[df["station"].isin(keep)].copy()

    print(
        f"[load] Using {df['station'].nunique()} stations and "
        f"{len(df):,} daily records in {START}–{END}."
    )
    return df


# ---------------------------------------------------------------------
# Main MDR loop
# ---------------------------------------------------------------------
def run_mdr_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run station-wise evaluation for various include_target_pct values.

    Returns
    -------
    pd.DataFrame
        Concatenated station-level reports with an extra column
        'include_target_pct' indicating the MDR scenario.
    """
    reports = []

    for pct in MDR_PCTS:
        print(f"[MDR] Evaluating include_target_pct = {pct}% ...")
        report, _preds = evaluate_stations(
            data=df,
            id_col="station",
            date_col="date",
            lat_col="latitude",
            lon_col="longitude",
            alt_col="altitude",
            target_col=TARGET_COL,
            start=START,
            end=END,
            prefix=STATION_PREFIXES,
            min_station_rows=MIN_STATION_ROWS,
            k_neighbors=20,
            include_target_pct=float(pct),
            include_target_seed=42,
            model_kind="rf",
            model_params={
                "n_estimators": 200,
                "max_depth": 30,
                "random_state": 42,
                "n_jobs": -1,
            },
            agg_for_metrics="sum" if TARGET_COL == "prec" else "mean",
            show_progress=True,
        )
        report["include_target_pct"] = pct
        reports.append(report)

    full_report = pd.concat(reports, ignore_index=True)
    return full_report


def main() -> None:
    df = load_data()
    full_report = run_mdr_experiment(df)

    print(f"[save] Writing MDR report to {OUT_CSV} ...")
    full_report.to_csv(OUT_CSV, index=False)
    print(
        "[done] MDR experiment finished.\n"
        f"       Output: {OUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
