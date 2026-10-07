"""
Módulo de simulaciones estocásticas avanzadas (Monte Carlo, Jump-Diffusion, VaR/CVaR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iwannaberich.config import TRADING_DAYS_PER_YEAR


@dataclass
class MonteCarloResult:
    ticker: str
    start_price: float
    horizon_days: int
    n_simulations: int
    paths: np.ndarray  # shape: (n_simulations, horizon_days + 1)
    dates: pd.DatetimeIndex
    mean_terminal_price: float
    median_terminal_price: float
    p5_terminal_price: float
    p95_terminal_price: float
    var_95_pct: float     # Pérdida máxima al 95% de confianza (ej. -0.15 = 15% pérdida)
    cvar_95_pct: float    # Expected Shortfall al 95%
    var_99_pct: float     # Pérdida máxima al 99% de confianza
    cvar_99_pct: float    # Expected Shortfall al 99%
    prob_loss: float       # Probabilidad de terminar por debajo del precio inicial
    annualized_drift: float
    annualized_volatility: float
    percentiles: Dict[str, np.ndarray]


def simulate_monte_carlo(
    prices: pd.Series,
    ticker: str = "ASSET",
    n_simulations: int = 1000,
    horizon_days: int = 252,
    model: str = "gbm",  # 'gbm' o 'jump_diffusion'
    jump_intensity: float = 0.05,
    jump_mean: float = -0.02,
    jump_std: float = 0.06,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Ejecuta simulaciones estocásticas de Monte Carlo basadas en Movimiento Browniano
    Geométrico (GBM) o Salto-Difusión de Merton.
    """
    clean_series = prices.dropna().astype(float)
    if len(clean_series) < 10:
        raise ValueError(f"Serie de precios insuficiente para simulación (mínimo 10 observaciones, recibido {len(clean_series)}).")

    log_returns = np.diff(np.log(clean_series.values))
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Estimación de parámetros
    mu_daily = float(np.mean(log_returns))
    sigma_daily = float(np.std(log_returns, ddof=1))
    if sigma_daily < 1e-6:
        sigma_daily = 0.01

    ann_drift = mu_daily * TRADING_DAYS_PER_YEAR + 0.5 * (sigma_daily ** 2) * TRADING_DAYS_PER_YEAR
    ann_vol = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    rng = np.random.default_rng(seed)
    s0 = float(clean_series.iloc[-1])

    # Generación de shocks normales estándar
    # dW ~ N(0, dt)
    z = rng.standard_normal((n_simulations, horizon_days))
    drift_term = (mu_daily - 0.5 * (sigma_daily ** 2)) * 1.0
    diffusion_term = sigma_daily * z

    if model.lower() == "jump_diffusion":
        # Merton Jump-Diffusion
        # Saltos Poisson en dt (aquí escala diaria)
        poisson_jumps = rng.poisson(lam=jump_intensity, size=(n_simulations, horizon_days))
        jump_magnitudes = rng.normal(loc=jump_mean, scale=jump_std, size=(n_simulations, horizon_days))
        jump_term = poisson_jumps * jump_magnitudes
        step_returns = drift_term + diffusion_term + jump_term
    else:
        step_returns = drift_term + diffusion_term

    # Matriz acumulativa de precios
    cumulative_log = np.zeros((n_simulations, horizon_days + 1))
    cumulative_log[:, 1:] = np.cumsum(step_returns, axis=1)
    paths = s0 * np.exp(cumulative_log)

    # Fechas proyectadas
    last_date = clean_series.index[-1] if isinstance(clean_series.index, pd.DatetimeIndex) else pd.Timestamp.now()
    proj_dates = pd.date_range(start=last_date, periods=horizon_days + 1, freq="B")

    # Terminal statistics
    terminal_prices = paths[:, -1]
    terminal_returns = (terminal_prices - s0) / s0

    mean_term = float(np.mean(terminal_prices))
    median_term = float(np.median(terminal_prices))
    p5_term = float(np.percentile(terminal_prices, 5))
    p95_term = float(np.percentile(terminal_prices, 95))

    # VaR y CVaR (medidos como retornos negativos)
    var_95 = float(np.percentile(terminal_returns, 5))
    tail_95 = terminal_returns[terminal_returns <= var_95]
    cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

    var_99 = float(np.percentile(terminal_returns, 1))
    tail_99 = terminal_returns[terminal_returns <= var_99]
    cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

    prob_loss = float(np.mean(terminal_prices < s0))

    # Conos de confianza temporales
    percentiles = {
        "p5": np.percentile(paths, 5, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }

    return MonteCarloResult(
        ticker=ticker,
        start_price=s0,
        horizon_days=horizon_days,
        n_simulations=n_simulations,
        paths=paths,
        dates=proj_dates,
        mean_terminal_price=mean_term,
        median_terminal_price=median_term,
        p5_terminal_price=p5_term,
        p95_terminal_price=p95_term,
        var_95_pct=var_95,
        cvar_95_pct=cvar_95,
        var_99_pct=var_99,
        cvar_99_pct=cvar_99,
        prob_loss=prob_loss,
        annualized_drift=ann_drift,
        annualized_volatility=ann_vol,
        percentiles=percentiles,
    )


def plot_monte_carlo(
    res: MonteCarloResult,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Genera visualización profesional de las trayectorias estocásticas y distribución terminal.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    x = np.arange(res.horizon_days + 1)

    # Gráfico de trayectorias y bandas
    # Muestra de 40 trayectorias aleatorias
    n_sample = min(40, res.n_simulations)
    for i in range(n_sample):
        ax1.plot(x, res.paths[i], color="#1f77b4", alpha=0.10, lw=1)

    # Bandas de confianza
    ax1.fill_between(x, res.percentiles["p5"], res.percentiles["p95"], color="#1f77b4", alpha=0.18, label="IC 90% (P5-P95)")
    ax1.fill_between(x, res.percentiles["p25"], res.percentiles["p75"], color="#1f77b4", alpha=0.30, label="IC 50% (P25-P75)")
    ax1.plot(x, res.percentiles["p50"], color="#08306b", lw=2.2, label=f"Mediana (Final: {res.median_terminal_price:,.0f})")
    ax1.axhline(res.start_price, color="#d62728", linestyle="--", lw=1.5, label=f"Precio Inicial ({res.start_price:,.0f})")

    ax1.set_title(f"Simulación Monte Carlo ({res.ticker}) - {res.n_simulations:,} Escenarios a {res.horizon_days} días", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Días Hábiles Proyectados", fontsize=10)
    ax1.set_ylabel("Precio (CLP)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Histograma de distribución final
    term = res.paths[:, -1]
    ax2.hist(term, bins=40, orientation="horizontal", color="#4682b4", alpha=0.7, edgecolor="white", density=True)
    ax2.axhline(res.start_price, color="#d62728", linestyle="--", lw=1.5)
    ax2.axhline(res.p5_terminal_price, color="#e6550d", linestyle=":", lw=1.8, label=f"VaR 95% ({res.p5_terminal_price:,.0f})")
    ax2.axhline(res.median_terminal_price, color="#08306b", lw=2)

    ax2.set_title("Distribución Terminal", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Densidad", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
