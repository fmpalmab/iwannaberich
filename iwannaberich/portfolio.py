"""
Módulo de optimización moderna de portafolios (Markowitz Mean-Variance, Frontera Eficiente, Max Sharpe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from iwannaberich.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


@dataclass
class PortfolioAllocation:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass
class EfficientFrontierResult:
    tickers: List[str]
    max_sharpe_portfolio: PortfolioAllocation
    min_variance_portfolio: PortfolioAllocation
    frontier_returns: np.ndarray
    frontier_volatilities: np.ndarray
    cov_matrix: pd.DataFrame
    expected_returns: pd.Series
    risk_free_rate: float


def calculate_portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Tuple[float, float, float]:
    """Calcula el retorno esperado, volatilidad y ratio de Sharpe de un vector de pesos."""
    port_return = float(np.dot(weights, expected_returns))
    port_variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    port_volatility = float(np.sqrt(max(1e-10, port_variance)))
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 1e-6 else 0.0
    return port_return, port_volatility, sharpe


def optimize_portfolio(
    prices_df: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight_per_asset: float = 1.0,
    min_weight_per_asset: float = 0.0,
) -> EfficientFrontierResult:
    """
    Optimiza una cartera multiactivos mediante Teoría Moderna de Portafolios (Markowitz).
    Calcula la Cartera de Máximo Sharpe y la de Mínima Varianza Global (GMV).
    """
    clean_df = prices_df.dropna(how="all").ffill().bfill()
    # Filtrar columnas con varianza cero o constantes
    valid_cols = [c for c in clean_df.columns if clean_df[c].nunique() > 5]
    if len(valid_cols) < 2:
        raise ValueError("Se requieren al menos 2 activos válidos para optimizar portafolio.")

    df_subset = clean_df[valid_cols]
    log_returns = np.log(df_subset / df_subset.shift(1)).dropna()

    mean_daily = log_returns.mean()
    cov_daily = log_returns.cov()

    # Anualización
    ann_returns = mean_daily * TRADING_DAYS_PER_YEAR
    ann_cov = cov_daily * TRADING_DAYS_PER_YEAR

    n_assets = len(valid_cols)
    tickers = list(valid_cols)
    mu = ann_returns.values
    cov = ann_cov.values

    # Restricciones base: suma de pesos = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = tuple((min_weight_per_asset, max_weight_per_asset) for _ in range(n_assets))
    init_weights = np.ones(n_assets) / n_assets

    # 1. Cartera de Máximo Sharpe (minimizar -Sharpe)
    def neg_sharpe(w: np.ndarray) -> float:
        _, _, s = calculate_portfolio_performance(w, mu, cov, risk_free_rate)
        return -s

    res_sharpe = minimize(
        neg_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_sharpe = res_sharpe.x if res_sharpe.success else init_weights
    w_sharpe = np.clip(w_sharpe, 0.0, 1.0)
    w_sharpe /= np.sum(w_sharpe)
    ret_sh, vol_sh, s_sh = calculate_portfolio_performance(w_sharpe, mu, cov, risk_free_rate)
    max_sharpe_alloc = PortfolioAllocation(
        weights={t: float(w) for t, w in zip(tickers, w_sharpe)},
        expected_return=ret_sh,
        volatility=vol_sh,
        sharpe_ratio=s_sh,
    )

    # 2. Cartera de Mínima Varianza Global (GMV)
    def port_vol(w: np.ndarray) -> float:
        _, v, _ = calculate_portfolio_performance(w, mu, cov, risk_free_rate)
        return v

    res_minvar = minimize(
        port_vol,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_minvar = res_minvar.x if res_minvar.success else init_weights
    w_minvar = np.clip(w_minvar, 0.0, 1.0)
    w_minvar /= np.sum(w_minvar)
    ret_mv, vol_mv, s_mv = calculate_portfolio_performance(w_minvar, mu, cov, risk_free_rate)
    min_var_alloc = PortfolioAllocation(
        weights={t: float(w) for t, w in zip(tickers, w_minvar)},
        expected_return=ret_mv,
        volatility=vol_mv,
        sharpe_ratio=s_mv,
    )

    # 3. Trazado de la Frontera Eficiente
    target_returns = np.linspace(ret_mv, max(np.max(mu), ret_sh * 1.15), 40)
    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t},
        ]
        res_t = minimize(port_vol, init_weights, method="SLSQP", bounds=bounds, constraints=cons)
        if res_t.success:
            frontier_vols.append(res_t.fun)
            frontier_rets.append(target)

    return EfficientFrontierResult(
        tickers=tickers,
        max_sharpe_portfolio=max_sharpe_alloc,
        min_variance_portfolio=min_var_alloc,
        frontier_returns=np.array(frontier_rets),
        frontier_volatilities=np.array(frontier_vols),
        cov_matrix=ann_cov,
        expected_returns=ann_returns,
        risk_free_rate=risk_free_rate,
    )


def plot_efficient_frontier(
    result: EfficientFrontierResult,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Grafica la Frontera Eficiente de Markowitz, la Línea de Asignación de Capital (CAL)
    y el desglose de ponderaciones del portafolio óptimo.
    """
    fig, (ax_curve, ax_weights) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]})

    # 1. Curva de Frontera Eficiente
    if len(result.frontier_volatilities) > 0:
        ax_curve.plot(
            result.frontier_volatilities,
            result.frontier_returns,
            color="#2b5c8f",
            lw=2.5,
            label="Frontera Eficiente (Markowitz)",
        )

    # Activos individuales
    for ticker in result.tickers:
        asset_vol = np.sqrt(result.cov_matrix.loc[ticker, ticker])
        asset_ret = result.expected_returns[ticker]
        ax_curve.scatter(asset_vol, asset_ret, color="#888888", alpha=0.7, s=40)
        ax_curve.annotate(
            ticker.replace(".SN", ""),
            (asset_vol, asset_ret),
            textcoords="offset points",
            xytext=(5, 2),
            fontsize=8,
            alpha=0.8,
        )

    # Puntos óptimos
    sh = result.max_sharpe_portfolio
    mv = result.min_variance_portfolio

    ax_curve.scatter(
        sh.volatility,
        sh.expected_return,
        color="#d62728",
        marker="*",
        s=200,
        zorder=5,
        label=f"Máximo Sharpe ({sh.sharpe_ratio:.2f}) [Ret: {sh.expected_return*100:.1f}%, Vol: {sh.volatility*100:.1f}%]",
    )

    ax_curve.scatter(
        mv.volatility,
        mv.expected_return,
        color="#2ca02c",
        marker="o",
        s=120,
        zorder=5,
        label=f"Mínima Varianza [Ret: {mv.expected_return*100:.1f}%, Vol: {mv.volatility*100:.1f}%]",
    )

    # Línea de Asignación de Capital (CAL)
    cal_x = np.linspace(0, max(sh.volatility * 1.4, 0.4), 20)
    cal_y = result.risk_free_rate + sh.sharpe_ratio * cal_x
    ax_curve.plot(cal_x, cal_y, "--", color="#ff7f0e", lw=1.5, label=f"Capital Allocation Line (Rf={result.risk_free_rate*100:.1f}%)")

    ax_curve.set_title("Frontera Eficiente de Markowitz & Asignación Óptima", fontsize=12, fontweight="bold")
    ax_curve.set_xlabel("Volatilidad Anualizada (Riesgo)", fontsize=10)
    ax_curve.set_ylabel("Retorno Esperado Anualizado", fontsize=10)
    ax_curve.grid(True, linestyle=":", alpha=0.6)
    ax_curve.legend(loc="upper left", fontsize=8.5)
    ax_curve.set_xlim(left=0)

    # 2. Desglose de pesos del portafolio Máximo Sharpe
    weights_series = pd.Series(sh.weights).sort_values(ascending=True)
    # Filtrar solo ponderaciones > 0.5%
    weights_series = weights_series[weights_series >= 0.005]
    if weights_series.empty:
        weights_series = pd.Series(sh.weights).tail(5)

    y_pos = np.arange(len(weights_series))
    ax_weights.barh(y_pos, weights_series.values * 100, color="#1f77b4", edgecolor="#08306b", alpha=0.85)
    ax_weights.set_yticks(y_pos)
    ax_weights.set_yticklabels([t.replace(".SN", "") for t in weights_series.index], fontsize=9)
    ax_weights.set_xlabel("Ponderación (%)", fontsize=10)
    ax_weights.set_title(f"Ponderaciones Óptimas - Cartera Tangente (Max Sharpe)", fontsize=11, fontweight="bold")
    ax_weights.grid(True, linestyle=":", alpha=0.5, axis="x")

    for i, v in enumerate(weights_series.values * 100):
        ax_weights.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
