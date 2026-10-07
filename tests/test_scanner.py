"""Pruebas unitarias para el motor de escaneo y reglas de señales técnicas."""

import pytest
import numpy as np
import pandas as pd
from iwannaberich.scanner import scan_asset_signals, scan_market


def test_scan_asset_signals_dip_and_momentum():
    dates = pd.date_range("2023-01-01", periods=250, freq="B")

    # Activo en tendencia alcista fuerte
    bull_prices = np.linspace(100, 300, 250)
    s_bull = pd.Series(bull_prices, index=dates)

    res_bull = scan_asset_signals(s_bull, "BULL.SN")
    assert res_bull is not None
    assert res_bull["Tendencia (SMA200)"] == "ALCISTA"
    assert res_bull["Ticker"] == "BULL.SN"

    # Activo en colapso fuerte
    crash_prices = np.linspace(300, 50, 250)
    s_crash = pd.Series(crash_prices, index=dates)

    res_crash = scan_asset_signals(s_crash, "CRASH.SN")
    assert res_crash is not None
    assert res_crash["Tendencia (SMA200)"] == "BAJISTA"
    assert res_crash["Señal Técnica"] in ("SOBREVENTA EXTREMA", "NEUTRAL")


def test_scan_market_dataframe_input():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "T1.SN": np.linspace(50, 100, 100),
            "T2.SN": np.linspace(100, 60, 100),
        },
        index=dates,
    )
    res = scan_market(df, output_csv=None)
    assert len(res) == 2
    assert "Señal Técnica" in res.columns
