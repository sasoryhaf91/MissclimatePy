from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from .impute import impute_local_station

@dataclass
class MissClimateImputer:
    # Accept both names; 'model' is kept for backward-compat and mapped to 'engine'
    engine: str = "rf"
    model: str | None = None

    target: str = "tmin"
    k_neighbors: int | None = 15
    radius_km: float | None = None
    min_obs_per_station: int = 50
    n_estimators: int = 300
    n_jobs: int = -1
    random_state: int = 42

    _fitted: bool = field(init=False, default=False)

    def __post_init__(self):
        # map legacy 'model' param to 'engine'
        if self.model is not None:
            self.engine = self.model
        # validate only allowed engines
        if self.engine not in {"rf"}:
            raise ValueError(f"Unsupported engine '{self.engine}'. Supported: 'rf'.")

    def fit(self, df: pd.DataFrame):
        req = {"station","date","latitude","longitude","elevation", self.target}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        self._fitted = True
        return self

    def impute_station(self, df_all: pd.DataFrame, target_station: str, train_frac: float = 1.0) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before impute_station()")
        out, _ = impute_local_station(
            df_all=df_all,
            target_station=target_station,
            target_var=self.target,
            k_neighbors=self.k_neighbors,
            radius_km=self.radius_km,
            min_obs_per_station=self.min_obs_per_station,
            train_frac=train_frac,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        outs = []
        for st in df["station"].unique():
            st_out = self.impute_station(df_all=df, target_station=st, train_frac=1.0)
            outs.append(st_out if st_out is not None else df[df.station==st].copy())
        return pd.concat(outs, ignore_index=True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

