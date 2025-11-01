
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from .features import add_calendar_features
from .models.rf import RFRegressor
from .models.idw import IDWRegressor
from .metrics import regression_report

SUPPORTED_MODELS = {
    "rf": RFRegressor,
    "idw": IDWRegressor,
}

@dataclass
class MissClimateImputer:
    model: str = "rf"           # "rf" or "idw"
    target: str = "tmin"        # variable to impute
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    elev_col: str = "elevation"
    date_col: str = "date"
    station_col: str = "station"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1

    _model_obj: object = field(init=False, default=None)
    _fitted: bool = field(init=False, default=False)
    _feature_cols: list[str] = field(init=False, default_factory=list)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        if not np.issubdtype(df2[self.date_col].dtype, np.datetime64):
            df2[self.date_col] = pd.to_datetime(df2[self.date_col])
        df2 = add_calendar_features(df2, date_col=self.date_col)
        self._feature_cols = [self.lat_col, self.lon_col, self.elev_col,
                              "doy_sin", "doy_cos", "year"]
        return df2

    def fit(self, df: pd.DataFrame) -> "MissClimateImputer":
        if self.target not in df.columns:
            raise ValueError(f"target '{self.target}' not found in columns")
        df_feat = self._build_features(df)
        train = df_feat[df_feat[self.target].notna()]

        X = train[self._feature_cols].to_numpy()
        y = train[self.target].to_numpy()

        if self.model == "rf":
            self._model_obj = RFRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
        elif self.model == "idw":
            self._model_obj = IDWRegressor(power=2.0)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        self._model_obj.fit(X, y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        df_feat = self._build_features(df)
        out = df_feat.copy()

        mask_missing = out[self.target].isna()
        if mask_missing.any():
            Xmiss = out.loc[mask_missing, self._feature_cols].to_numpy()
            yhat = self._model_obj.predict(Xmiss)
            out.loc[mask_missing, self.target] = yhat
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df_valid: pd.DataFrame) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before report()")
        df_feat = self._build_features(df_valid)
        valid = df_feat[df_feat[self.target].notna()].copy()
        X = valid[self._feature_cols].to_numpy()
        y = valid[self.target].to_numpy()
        yhat = self._model_obj.predict(X)
        return regression_report(y, yhat)
