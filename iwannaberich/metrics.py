"""
Métricas estocásticas, de riesgo y ratios de eficiencia financiera.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from iwannaberich.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def compute_ticker_metrics(
    series: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Calcula todas las métricas de rendimiento y riesgo para una serie de precios de un ticker.
    """
    clean_prices = series.dropna()
    if len(clean_prices) < 20:
        return {}

    # Retornos logarítmicos
    log_returns = np.diff(np.log(clean_prices))
    n_days = len(log_returns)

    mean_ret = float(np.mean(log_returns))
    std_ret = float(np.std(log_returns, ddof=1)) if n_days > 1 else 0.0

    ret_anual = mean_ret * TRADING_DAYS_PER_YEAR
    vol_anual = std_ret * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Probabilidad de día alcista
    prob_subida = float(np.mean(log_returns > 0)) if n_days > 0 else 0.0

    # Desviación a la baja (downside deviation)
    neg_returns = log_returns[log_returns < 0]
    if len(neg_returns) > 1:
        downside_std = float(np.std(neg_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        downside_std = 1e-6

    # Sharpe & Sortino ratios
    sharpe = (ret_anual - risk_free_rate) / vol_anual if vol_anual > 1e-6 else 0.0
    sortino = (ret_anual - risk_free_rate) / downside_std if downside_std > 1e-6 else 0.0

    # Momentos de la distribución
    skew_val = float(skew(log_returns)) if n_days > 3 else 0.0
    kurt_val = float(kurtosis(log_returns, fisher=False)) if n_days > 3 else 3.0

    # Riesgo de cola: VaR y CVaR (95%)
    var_95 = float(np.percentile(log_returns, 5)) if n_days >= 20 else 0.0
    tail_losses = log_returns[log_returns <= var_95]
    cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

    # Caída máxima (Max Drawdown) usando retornos simples
    peak = clean_prices.cummax()
    dd_series = (clean_prices - peak) / peak
    max_dd = float(dd_series.min())

    # Calmar ratio
    calmar = ret_anual / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    return {
        "Retorno_Anualizado": ret_anual,
        "Prob_Subida": prob_subida,
        "Volatilidad_Anual": vol_anual,
        "Desv_Bajista": downside_std,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "VaR_95_Diario": var_95,
        "CVaR_95_Diario": cvar_95,
        "Max_Drawdown": max_dd,
        "Calmar_Ratio": calmar,
    }


def compute_metrics_matrix(
    prices_df: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    min_history_days: int = 60,
) -> pd.DataFrame:
    """Calcula la matriz estocástica completa para todas las columnas del DataFrame de precios."""
    results = {}
    for col in prices_df.columns:
        series = prices_df[col].dropna()
        if len(series) >= min_history_days:
            metrics = compute_ticker_metrics(series, risk_free_rate=risk_free_rate)
            if metrics:
                results[col] = metrics

    df = pd.DataFrame.from_dict(results, orient="index")
    return df
