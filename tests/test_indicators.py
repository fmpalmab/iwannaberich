"""Pruebas unitarias para indicadores técnicos."""

import pytest
import numpy as np
import pandas as pd
from iwannaberich.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_bollinger_bands,
    calculate_drawdown_series,
    calculate_volatility,
)


def test_calculate_rsi_bounds():
    # Serie creciente sostenida -> RSI alto
    series_up = pd.Series(np.linspace(10, 100, 50))
    rsi_up = calculate_rsi(series_up, period=14).dropna()
    assert not rsi_up.empty
    assert (rsi_up >= 0.0).all()
    assert (rsi_up <= 100.0).all()
    assert rsi_up.iloc[-1] > 70.0

    # Serie decreciente sostenida -> RSI bajo
    series_down = pd.Series(np.linspace(100, 10, 50))
    rsi_down = calculate_rsi(series_down, period=14).dropna()
    assert rsi_down.iloc[-1] < 30.0


def test_calculate_sma():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sma3 = calculate_sma(series, window=3)
    # Último valor es (3+4+5)/3 = 4.0
    assert sma3.iloc[-1] == pytest.approx(4.0)


def test_calculate_bollinger_bands():
    series = pd.Series([10.0] * 30)
    sma, upper, lower = calculate_bollinger_bands(series, window=10)
    # Varianza cero -> bandas colapsadas en la media
    assert upper.iloc[-1] == pytest.approx(10.0)
    assert lower.iloc[-1] == pytest.approx(10.0)


def test_calculate_drawdown_series():
    series = pd.Series([100.0, 120.0, 60.0, 90.0])
    dd = calculate_drawdown_series(series)
    # Peak fue 120. Con 60 cayó a -50%
    assert dd.iloc[2] == pytest.approx(-50.0)


def test_calculate_volatility_positive():
    series = pd.Series([100, 105, 98, 102, 97, 103, 108])
    vol = calculate_volatility(series)
    assert vol > 0.0
