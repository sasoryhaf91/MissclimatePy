import numpy as np
import pandas as pd


from missclimatepy.impute import impute_dataset


def _make_simple_df():
    """
    Helper: crea un DataFrame sencillo con 2 estaciones, 5 días y algunos NaN.
    """
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    data = []
    for st in [1, 2]:
        for d in dates:
            data.append(
                {
                    "station": st,
                    "date": d,
                    "lat": 10.0 + st,
                    "lon": -99.0 - st,
                    "alt": 2000.0 + 10 * st,
                    "target": np.nan,
                }
            )
    df = pd.DataFrame(data)

    # Ponemos algunos valores observados
    df.loc[(df["station"] == 1) & (df["date"] == dates[0]), "target"] = 1.0
    df.loc[(df["station"] == 1) & (df["date"] == dates[1]), "target"] = 2.0
    df.loc[(df["station"] == 2) & (df["date"] == dates[0]), "target"] = 5.0
    return df


# -------------------------------------------------------------------
# Tests básicos de estructura
# -------------------------------------------------------------------


def test_impute_dataset_returns_expected_columns():
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        model_kind="rf",
        rf_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    expected_cols = ["station", "date", "lat", "lon", "alt", "target", "source"]
    assert list(out.columns) == expected_cols

    # Debe tener el mismo número de filas que el intervalo filtrado
    assert len(out) == len(df)


def test_impute_dataset_marks_observed_and_imputed():
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        model_kind="rf",
        rf_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # Filas con valor original (no NaN en df) deben ser "observed"
    merged = df.merge(
        out,
        on=["station", "date", "lat", "lon", "alt"],
        suffixes=("_orig", "_imp"),
        how="left",
    )

    obs_mask = merged["target_orig"].notna()
    imp_mask = merged["target_orig"].isna()

    assert set(merged.loc[obs_mask, "source"]) == {"observed"}
    # Para filas originalmente NaN, deben existir algunas "imputed"
    assert "imputed" in set(merged.loc[imp_mask, "source"])


def test_impute_dataset_does_not_change_observed_values():
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        model_kind="rf",
        rf_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    merged = df.merge(
        out,
        on=["station", "date", "lat", "lon", "alt"],
        suffixes=("_orig", "_imp"),
        how="left",
    )

    obs_mask = merged["target_orig"].notna()
    # Los valores observados no deben cambiar
    assert np.allclose(
        merged.loc[obs_mask, "target_orig"],
        merged.loc[obs_mask, "target_imp"],
        equal_nan=False,
    )


# -------------------------------------------------------------------
# Tests del backend MCM
# -------------------------------------------------------------------


def test_impute_dataset_mcm_backend_fills_missing():
    df = _make_simple_df()

    # Usamos sólo una estación para que la climatología sea trivial
    df1 = df[df["station"] == 1].copy()

    out = impute_dataset(
        df1,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        model_kind="mcm",
        mcm_mode="global",
        mcm_min_samples=1,
        show_progress=False,
    )

    # No debe quedar ningún NaN en target
    assert out["target"].notna().all()

    # Los días que tenían datos observados deben conservarse
    merged = df1.merge(
        out,
        on=["station", "date", "lat", "lon", "alt"],
        suffixes=("_orig", "_imp"),
        how="left",
    )

    obs_mask = merged["target_orig"].notna()
    assert np.allclose(
        merged.loc[obs_mask, "target_orig"],
        merged.loc[obs_mask, "target_imp"],
        equal_nan=False,
    )


# -------------------------------------------------------------------
# Tests de selección de estaciones
# -------------------------------------------------------------------


def test_impute_dataset_respects_station_ids_filter():
    df = _make_simple_df()

    # Sólo imputamos la estación 1
    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        station_ids=[1],
        model_kind="mcm",
        mcm_mode="global",
        mcm_min_samples=1,
        show_progress=False,
    )

    # Sólo debe aparecer la estación 1 en el resultado
    assert set(out["station"].unique()) == {1}


def test_impute_dataset_with_min_station_rows_skips_small_stations():
    df = _make_simple_df()

    # Recortamos la estación 2 para que tenga muy pocas filas
    df = df[~((df["station"] == 2) & (df["date"] > df["date"].min()))].copy()
    # Ahora station=2 solo tiene 1 fila, station=1 tiene 5

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="target",
        min_station_rows=3,  # station 2 debe quedar fuera
        model_kind="mcm",
        mcm_mode="global",
        mcm_min_samples=1,
        show_progress=False,
    )

    assert set(out["station"].unique()) == {1}
