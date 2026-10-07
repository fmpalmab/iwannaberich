"""
Motor de escaneo y detección de señales técnicas cuantitativas.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from iwannaberich.config import LIQUID_PRICES_FILE, OPPORTUNITIES_REPORT_CSV
from iwannaberich.indicators import calculate_rsi, calculate_sma, calculate_volatility


def scan_asset_signals(series: pd.Series, ticker: str) -> Optional[Dict[str, Any]]:
    """Evalúa indicadores y reglas de trading sobre una serie de precios."""
    clean = series.dropna()
    if len(clean) < 60:
        return None

    current_price = float(clean.iloc[-1])

    # Medias móviles
    sma_50_series = calculate_sma(clean, 50)
    sma_200_series = calculate_sma(clean, 200)

    sma_50 = float(sma_50_series.dropna().iloc[-1]) if not sma_50_series.dropna().empty else current_price
    sma_200 = float(sma_200_series.dropna().iloc[-1]) if not sma_200_series.dropna().empty else current_price

    # RSI
    rsi_series = calculate_rsi(clean, period=14).dropna()
    rsi_current = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    # Drawdown anual (últimos 252 días hábiles o disponible)
    window_52w = clean.tail(min(252, len(clean)))
    max_52w = float(window_52w.max())
    drawdown_pct = ((current_price - max_52w) / max_52w) * 100.0 if max_52w > 0 else 0.0

    # Volatilidad
    volatility = calculate_volatility(clean)

    # Clasificación de tendencia
    trend = "ALCISTA" if current_price > sma_200 else "BAJISTA"

    # Clasificación de señal algorítmica
    signal = "NEUTRAL"
    if trend == "ALCISTA" and rsi_current < 40.0:
        signal = "OPORTUNIDAD (DIP)"
    elif rsi_current < 30.0:
        signal = "SOBREVENTA EXTREMA"
    elif current_price > sma_50 and current_price > sma_200 and rsi_current > 60.0:
        if drawdown_pct > -3.0:
            signal = "RUPTURA (BREAKOUT 52W)"
        else:
            signal = "FUERZA (MOMENTUM)"

    return {
        "Ticker": ticker,
        "Precio": round(current_price, 2),
        "Tendencia (SMA200)": trend,
        "RSI (14)": round(rsi_current, 2),
        "Distancia Max 52W (%)": round(drawdown_pct, 2),
        "Volatilidad Anual (%)": round(volatility, 2),
        "Señal Técnica": signal,
    }


def scan_market(
    prices_input: Union[str, Path, pd.DataFrame] = LIQUID_PRICES_FILE,
    output_csv: Optional[Union[str, Path]] = OPPORTUNITIES_REPORT_CSV,
) -> pd.DataFrame:
    """Escanea todos los activos en el archivo o DataFrame de precios."""
    if isinstance(prices_input, (str, Path)):
        p = Path(prices_input)
        if not p.is_file():
            raise FileNotFoundError(f"No existe el archivo de precios: {p}")
        df_prices = pd.read_parquet(p)
    else:
        df_prices = prices_input

    results: List[Dict[str, Any]] = []
    for ticker in df_prices.columns:
        res = scan_asset_signals(df_prices[ticker], ticker)
        if res:
            results.append(res)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by=["Señal Técnica", "RSI (14)"], ascending=[True, True])

    if output_csv and not df_results.empty:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_p, index=False)

    return df_results
