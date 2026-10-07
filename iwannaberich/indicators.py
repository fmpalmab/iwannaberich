"""
Cálculo de indicadores técnicos y señales de momento sobre series temporales de precios.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el Relative Strength Index (RSI) con suavizado de Wilder o media móvil simple.
    Garantiza valores en el rango [0, 100].
    """
    series_clean = series.dropna()
    if len(series_clean) < period + 1:
        return pd.Series(np.nan, index=series.index)

    delta = series_clean.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Suavizado exponencial (estilo Wilder)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Prevenir divisiones por cero
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Reindexar con el índice original completo
    return rsi.reindex(series.index)


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calcula la media móvil simple (SMA)."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calcula la media móvil exponencial (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula las bandas de Bollinger (Media, Superior, Inferior)."""
    sma = calculate_sma(series, window)
    rstd = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    upper = sma + (rstd * num_std)
    lower = sma - (rstd * num_std)
    return sma, upper, lower


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula MACD, Línea de Señal e Histograma."""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    """Calcula la serie de caída porcentual desde el máximo acumulado (drawdown)."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    peak = clean.cummax()
    dd = (clean - peak) / peak.replace(0, np.nan)
    return dd.reindex(series.index) * 100.0


def calculate_volatility(series: pd.Series, annualize_factor: float = np.sqrt(252)) -> float:
    """Calcula la volatilidad anualizada a partir de retornos logarítmicos."""
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    returns = np.diff(np.log(clean))
    return float(np.std(returns, ddof=1) * annualize_factor * 100.0)
