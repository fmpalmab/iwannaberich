import numpy as np
import pandas as pd
import pytest

from iwannaberich.portfolio import optimize_portfolio, plot_efficient_frontier


def test_optimize_portfolio_weights_and_performance():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    data = {
        "ASSET_A": 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, size=150))),
        "ASSET_B": 50.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=150))),
        "ASSET_C": 200.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, size=150))),
    }
    df = pd.DataFrame(data, index=dates)

    res = optimize_portfolio(df, risk_free_rate=0.04)

    assert len(res.tickers) == 3
    # Comprobar suma de ponderaciones = 1.0
    sum_w_sharpe = sum(res.max_sharpe_portfolio.weights.values())
    sum_w_minvar = sum(res.min_variance_portfolio.weights.values())
    assert sum_w_sharpe == pytest.approx(1.0, abs=1e-3)
    assert sum_w_minvar == pytest.approx(1.0, abs=1e-3)

    # GMV debe tener menor o igual volatilidad que Max Sharpe
    assert res.min_variance_portfolio.volatility <= res.max_sharpe_portfolio.volatility + 1e-4

    # Ponderaciones no negativas (long-only)
    for w in res.max_sharpe_portfolio.weights.values():
        assert w >= -1e-5
    for w in res.min_variance_portfolio.weights.values():
        assert w >= -1e-5


def test_plot_efficient_frontier(tmp_path):
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {
            "TICKER1": 100 + np.random.randn(60).cumsum(),
            "TICKER2": 80 + np.random.randn(60).cumsum() * 1.5,
        },
        index=dates,
    )
    res = optimize_portfolio(df)
    out_file = tmp_path / "frontier.png"
    fig = plot_efficient_frontier(res, output_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
