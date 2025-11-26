# tests/test_impute.py

import numpy as np
import pandas as pd

from missclimatepy.impute import impute_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_station_df():
    """
    Dos estaciones (A,B) con 3 días.

    A: [1.0, NaN, NaN]
    B: [2.0, 2.0, 2.0]
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    df = pd.DataFrame(
        {
            "station": ["A"] * 3 + ["B"] * 3,
            "date": list(dates) * 2,
            "lat": [10.0] * 3 + [11.0] * 3,
            "lon": [-100.0] * 3 + [-101.0] * 3,
            "alt": [2000.0] * 6,
            "value": [1.0, np.nan, np.nan, 2.0, 2.0, 2.0],
        }
    )
    return df


def _make_min_rows_df():
    """
    Dos estaciones (S1,S2) con 4 días.

    S1: 1 observado, 3 NaN  -> por debajo de min_station_rows=2
    S2: 4 observados        -> por encima del umbral
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 4 + ["S2"] * 4,
            "date": list(dates) * 2,
            "lat": [10.0] * 4 + [11.0] * 4,
            "lon": [-100.0] * 4 + [-101.0] * 4,
            "alt": [2000.0] * 8,
            "value": [1.0, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_impute_dataset_fills_missing_and_marks_source():
    """
    Caso básico: A tiene huecos que deben imputarse usando B como vecina.

    - La salida debe contener la serie completa (3 días para A y 3 para B).
    - En A, las observaciones originales se mantienen como 'observed'.
    - En A, los días sin observación se rellenan y marcan como 'imputed'.
    """
    df = _make_two_station_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        k_neighbors=1,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # Serie completa para ambas estaciones: 2 * 3 = 6 filas
    assert out.shape[0] == 6

    a = out[out["station"] == "A"].copy()
    b = out[out["station"] == "B"].copy()

    # A: 3 fechas
    assert a.shape[0] == 3
    # Todos los valores de A deben estar definidos (1 observed + 2 imputed)
    assert a["value"].isna().sum() == 0
    assert set(a["source"]) == {"observed", "imputed"}
    # Solo una observación original
    assert (a["source"] == "observed").sum() == 1

    # B: completamente observada, sin imputaciones
    assert b["value"].isna().sum() == 0
    assert set(b["source"]) == {"observed"}


def test_impute_dataset_respects_min_station_rows():
    """
    Estaciones por debajo de `min_station_rows` no deben imputarse:

    - Se devuelve la serie completa,
    - Las fechas observadas permanecen con su valor y `source='observed'`,
    - Las fechas sin observación quedan NaN y `source='missing'`.
    """
    df = _make_min_rows_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        min_station_rows=2,  # S1 no cumple
        k_neighbors=1,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    s1 = out[out["station"] == "S1"].copy()
    s2 = out[out["station"] == "S2"].copy()

    # S1 -> serie completa (4 días)
    assert s1.shape[0] == 4
    # Solo 1 valor observado, 3 NaN
    assert s1["value"].isna().sum() == 3
    assert (s1["source"] == "observed").sum() == 1
    # Los NaN deben marcarse como 'missing'
    assert set(s1.loc[s1["value"].isna(), "source"]) == {"missing"}

    # S2 -> todo observado, sin imputaciones
    assert s2["value"].isna().sum() == 0
    assert set(s2["source"]) == {"observed"}


def test_impute_dataset_neighbor_map_controls_training():
    """
    Un `neighbor_map` puede impedir o permitir el entrenamiento:

    Caso 1: A no tiene vecinos y `include_target_pct=0.0` -> no hay pool de
            entrenamiento, sus valores permanecen NaN/'missing'.
    Caso 2: A usa B como vecina -> se entrena con B y A puede imputarse.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    df = pd.DataFrame(
        {
            "station": ["A"] * 3 + ["B"] * 3,
            "date": list(dates) * 2,
            "lat": [10.0] * 3 + [11.0] * 3,
            "lon": [-100.0] * 3 + [-101.0] * 3,
            "alt": [2000.0] * 6,
            # A: todo NaN, B: completamente observada
            "value": [np.nan, np.nan, np.nan, 5.0, 5.0, 5.0],
        }
    )

    # Caso 1: sin vecinos
    out_empty = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        neighbor_map={"A": [], "B": []},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    a_empty = out_empty[out_empty["station"] == "A"]
    assert a_empty["value"].isna().all()
    assert set(a_empty["source"]) == {"missing"}

    # Caso 2: A usa B como vecina
    out_nb = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        neighbor_map={"A": ["B"], "B": ["A"]},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    a_nb = out_nb[out_nb["station"] == "A"]
    # Ahora A debe estar completamente imputada (sin NaNs)
    assert a_nb["value"].isna().sum() == 0
    assert set(a_nb["source"]) == {"imputed"}


def test_impute_dataset_respects_start_end_window():
    """
    Cuando se proporcionan `start` y `end`, la salida solo debe contener
    fechas dentro de esa ventana. La serie sigue siendo completa dentro
    del intervalo.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 5,
            "date": dates,
            "lat": [10.0] * 5,
            "lon": [-100.0] * 5,
            "alt": [2000.0] * 5,
            "value": [1.0, np.nan, 1.5, np.nan, 2.0],
        }
    )

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        start="2020-01-02",
        end="2020-01-04",
        # aquí no necesitamos necesariamente imputar, solo probar la ventana
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # Solo fechas 2..4
    assert out["date"].min() == pd.Timestamp("2020-01-02")
    assert out["date"].max() == pd.Timestamp("2020-01-04")
    # Una sola estación, 3 días
    assert out.shape[0] == 3
