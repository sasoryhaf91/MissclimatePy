# tests/test_viz.py
import pandas as pd
import matplotlib.pyplot as plt
from missclimatepy import viz

def _toy_pred_df():
    return pd.DataFrame({
        "station": ["S1"]*5,
        "date": pd.date_range("2000-01-01", periods=5),
        "latitude": [19.0]*5,
        "longitude": [-99.0]*5,
        "y_obs": [1, 2, 3, 4, 5],
        "y_mod": [1.1, 1.9, 3.2, 3.8, 4.9],
        "RMSE_d": [0.5]*5
    })

def test_all_viz_functions_run():
    df = _toy_pred_df()

    ax1 = viz.plot_parity_scatter(df)
    ax2 = viz.plot_time_series_overlay(df, station_id="S1", id_col="station",
                                       date_col="date", y_true_col="y_obs", y_pred_col="y_mod")
    ax3 = viz.plot_spatial_scatter(df, lat_col="latitude", lon_col="longitude", value_col="RMSE_d")

    # Each function should return an Axes instance
    for ax in (ax1, ax2, ax3):
        assert hasattr(ax, "plot")

    plt.close("all")
