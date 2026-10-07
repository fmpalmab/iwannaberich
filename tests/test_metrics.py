"""Pruebas unitarias para métricas de riesgo y ratios financieros."""

import pytest
import numpy as np
import pandas as pd
from iwannaberich.metrics import compute_ticker_metrics, compute_metrics_matrix


def test_compute_ticker_metrics_keys_and_values():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    # Camino aleatorio con drift positivo
    rets = rng.normal(0.001, 0.015, size=100)
    prices = 1000.0 * np.exp(np.cumsum(rets))
    series = pd.Series(prices, index=dates)

    metrics = compute_ticker_metrics(series, risk_free_rate=0.04)

    expected_keys = [
        "Retorno_Anualizado",
        "Prob_Subida",
        "Volatilidad_Anual",
        "Desv_Bajista",
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Skewness",
        "Kurtosis",
        "VaR_95_Diario",
        "CVaR_95_Diario",
        "Max_Drawdown",
        "Calmar_Ratio",
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)

    # El CVaR diario es la cola izquierda, debe ser menor o igual al VaR
    assert metrics["CVaR_95_Diario"] <= metrics["VaR_95_Diario"]
    # Max drawdown debe ser <= 0
    assert metrics["Max_Drawdown"] <= 0.0


def test_compute_metrics_matrix():
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    df = pd.DataFrame(
        {
            "ACTIVO_A": np.linspace(100, 150, 80),
            "ACTIVO_B": np.linspace(200, 180, 80),
        },
        index=dates,
    )
    matrix = compute_metrics_matrix(df, min_history_days=50)
    assert len(matrix) == 2
    assert "Sharpe_Ratio" in matrix.columns
