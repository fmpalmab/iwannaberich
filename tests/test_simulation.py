import numpy as np
import pandas as pd
import pytest

from iwannaberich.simulation import simulate_monte_carlo, plot_monte_carlo


def test_simulate_monte_carlo_gbm_properties():
    # Serie sintetizada
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=100))), index=dates)

    res = simulate_monte_carlo(prices, ticker="TEST.SN", n_simulations=500, horizon_days=60, model="gbm", seed=123)

    assert res.ticker == "TEST.SN"
    assert res.start_price == pytest.approx(float(prices.iloc[-1]), rel=1e-4)
    assert res.paths.shape == (500, 61)
    assert len(res.dates) == 61

    # Percentiles deben guardar orden monotónico: p5 <= p25 <= p50 <= p75 <= p95
    assert np.all(res.percentiles["p5"] <= res.percentiles["p25"])
    assert np.all(res.percentiles["p25"] <= res.percentiles["p50"])
    assert np.all(res.percentiles["p50"] <= res.percentiles["p75"])
    assert np.all(res.percentiles["p75"] <= res.percentiles["p95"])

    # VaR y CVaR deben ser coherentes
    assert res.var_95_pct >= res.cvar_95_pct  # Al ser negativos, cvar es pérdida más severa (más negativo o igual)
    assert 0.0 <= res.prob_loss <= 1.0


def test_simulate_monte_carlo_jump_diffusion():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    prices = pd.Series(5000.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, size=50))), index=dates)

    res = simulate_monte_carlo(prices, ticker="JUMP.SN", n_simulations=200, horizon_days=30, model="jump_diffusion", seed=7)
    assert res.paths.shape == (200, 31)
    assert res.median_terminal_price > 0


def test_plot_monte_carlo(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(1000.0 + np.arange(30) * 5.0, index=dates)
    res = simulate_monte_carlo(prices, ticker="PLOT.SN", n_simulations=50, horizon_days=20, seed=1)

    out_file = tmp_path / "mc_test.png"
    fig = plot_monte_carlo(res, output_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
