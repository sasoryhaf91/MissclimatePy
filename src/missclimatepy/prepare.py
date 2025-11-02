# src/missclimatepy/prepare.py
from __future__ import annotations
import pandas as pd

# Columns we require once standardized
REQUIRED = ["station", "latitude", "longitude", "elevation", "date"]

# Accept common aliases and normalize to our schema
ALIASES = {
    "station": ["station", "id", "estacion"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "x"],
    "elevation": ["elevation", "altitude", "alt", "z"],
    "date": ["date", "fecha", "time", "datetime"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower = {c.lower(): c for c in df.columns}
    for std, cands in ALIASES.items():
        for cand in cands:
            if cand in lower:
                mapping[lower[cand]] = std
                break
    # Keep other columns (targets/vars) intact; only rename what we know
    return df.rename(columns=mapping)

def enforce_schema(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Ensure a standard schema:
      - column names are normalized to: station, latitude, longitude, elevation, date
      - target column must exist
      - dtypes are coerced sensibly
    Accepts common aliases like 'altitude' for 'elevation'.
    """
    out = _normalize_columns(df.copy())

    need = set(REQUIRED + [target])
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Types
    out["station"] = out["station"].astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["elevation"] = pd.to_numeric(out["elevation"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Target stays as-is; just ensure exists
    return out

def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].copy()

def missing_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    grp = df.groupby("station")[target].apply(lambda s: s.isna().mean())
    return grp.reset_index(name="missing_rate")

def select_stations(df: pd.DataFrame, target: str, stations=None, min_obs: int = 60) -> pd.DataFrame:
    if stations is not None:
        df = df[df["station"].astype(str).isin([str(s) for s in stations])]
    # keep stations with at least min_obs non-missing rows
    ok = (
        df.assign(_ok=df[target].notna())
          .groupby("station")["_ok"].sum()
          .reset_index(name="n_valid")
    )
    keep = set(ok.loc[ok["n_valid"] >= int(min_obs), "station"].astype(str))
    return df[df["station"].astype(str).isin(keep)].copy()
