"""
Consolidación de matrices de mercado, alineación temporal y filtros de liquidez.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
import pandas as pd

from iwannaberich.config import (
    HISTORICAL_PRICES_DIR,
    LIQUID_PRICES_FILE,
    LIQUIDITY_REPORT_FILE,
    DEFAULT_MIN_DAILY_VOLUME_CLP,
)


def load_and_consolidate(input_dir: Union[str, Path] = HISTORICAL_PRICES_DIR) -> pd.DataFrame:
    """Carga todos los archivos .parquet del directorio y consolida en una matriz de Close y Volume."""
    p_in = Path(input_dir)
    files = list(p_in.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    data_frames = []
    for f in files:
        ticker = f.stem
        try:
            df = pd.read_parquet(f)
            if "Close" in df.columns and "Volume" in df.columns:
                sub_df = df[["Close", "Volume"]].copy()
                sub_df.columns = [f"{ticker}_Close", f"{ticker}_Volume"]
                data_frames.append(sub_df)
        except Exception as e:
            print(f"[Error leyendo {ticker}]: {e}")

    if not data_frames:
        return pd.DataFrame()

    market_matrix = pd.concat(data_frames, axis=1).sort_index()
    return market_matrix


def filter_by_liquidity(
    market_matrix: pd.DataFrame,
    min_daily_volume_clp: float = DEFAULT_MIN_DAILY_VOLUME_CLP,
    recent_days: int = 90,
    output_report_file: Optional[Union[str, Path]] = LIQUIDITY_REPORT_FILE,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filtra los activos por volumen transado diario promedio (Precio Cierre * Volumen)
    durante los últimos N días hábiles.
    """
    tickers = [c.replace("_Close", "") for c in market_matrix.columns if c.endswith("_Close")]
    valid_tickers: List[str] = []
    liquidity_stats = []

    recent_data = market_matrix.tail(recent_days)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        vol_col = f"{ticker}_Volume"
        if close_col not in recent_data.columns or vol_col not in recent_data.columns:
            continue

        traded_value = recent_data[close_col] * recent_data[vol_col]
        avg_traded = float(traded_value.mean()) if not traded_value.empty else 0.0
        last_price = float(recent_data[close_col].dropna().iloc[-1]) if not recent_data[close_col].dropna().empty else 0.0

        liquidity_stats.append({
            "Ticker": ticker,
            "Avg_Traded_CLP": avg_traded,
            "Last_Price": last_price,
        })

        if avg_traded >= min_daily_volume_clp:
            valid_tickers.append(ticker)

    df_stats = pd.DataFrame(liquidity_stats).sort_values("Avg_Traded_CLP", ascending=False)

    if output_report_file:
        out_p = Path(output_report_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_stats.to_csv(out_p, index=False)

    return valid_tickers, df_stats


def process_market_pipeline(
    input_dir: Union[str, Path] = HISTORICAL_PRICES_DIR,
    output_file: Union[str, Path] = LIQUID_PRICES_FILE,
    min_daily_volume_clp: float = DEFAULT_MIN_DAILY_VOLUME_CLP,
) -> pd.DataFrame:
    """Ejecuta el pipeline completo: consolidar, filtrar liquidez y guardar parquet final."""
    full_matrix = load_and_consolidate(input_dir)
    if full_matrix.empty:
        raise ValueError(f"No se pudieron cargar datos desde {input_dir}")

    liquid_tickers, _ = filter_by_liquidity(full_matrix, min_daily_volume_clp=min_daily_volume_clp)
    if not liquid_tickers:
        # Fallback: si el filtro es muy estricto, tomar los 10 con mayor liquidez
        print("[Aviso] Ningún ticker superó el umbral estricto; usando los más líquidos disponibles.")
        all_tickers = [c.replace("_Close", "") for c in full_matrix.columns if c.endswith("_Close")]
        liquid_tickers = all_tickers[:15]

    close_cols = [f"{t}_Close" for t in liquid_tickers if f"{t}_Close" in full_matrix.columns]
    clean_prices = full_matrix[close_cols].copy()
    clean_prices.columns = [c.replace("_Close", "") for c in close_cols]
    clean_prices = clean_prices.ffill().dropna(how="all")

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    clean_prices.to_parquet(out_p)

    return clean_prices
