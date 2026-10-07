"""
Cargador de datos de mercado, soporte para descargas vía yfinance y generador sintético.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from iwannaberich.config import HISTORICAL_PRICES_DIR, MASTER_TICKERS_FILE


def generate_synthetic_stock_data(
    tickers: Optional[List[str]] = None,
    n_days: int = 500,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera series de precios sintéticas realistas usando Movimiento Browniano Geométrico
    con saltos (Jump-Diffusion) para pruebas reproducibles sin requerir internet.
    """
    if tickers is None:
        tickers = ["BCI.SN", "BSANTANDER.SN", "CHILE.SN", "COPEC.SN", "SQM-B.SN", "FALABELLA.SN", "LTM.SN"]

    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="B")

    data_dict = {}

    for ticker in tickers:
        initial_price = rng.uniform(100.0, 50_000.0)
        mu = rng.uniform(0.05, 0.18) / 252.0  # Drift diario
        sigma = rng.uniform(0.15, 0.35) / np.sqrt(252.0)  # Volatilidad diaria

        # Shock estocástico
        daily_returns = rng.normal(mu, sigma, size=n_days)
        # Saltos ocasionales (eventos de cola)
        jumps = rng.choice([0, -0.04, 0.04], size=n_days, p=[0.97, 0.015, 0.015])
        total_returns = daily_returns + jumps

        price_path = initial_price * np.exp(np.cumsum(total_returns))
        volume = rng.lognormal(mean=12.0, sigma=1.2, size=n_days)

        df_t = pd.DataFrame(
            {
                "Close": price_path,
                "Volume": volume,
            },
            index=dates,
        )

        data_dict[ticker] = df_t

        if output_dir:
            out_path = Path(output_dir) / f"{ticker}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_t.to_parquet(out_path)

    # DataFrame consolidado de precios de cierre
    consolidated = pd.DataFrame({t: data_dict[t]["Close"] for t in tickers}, index=dates)
    return consolidated


def download_tickers_yfinance(
    tickers: List[str],
    period: str = "5y",
    output_dir: Union[str, Path] = HISTORICAL_PRICES_DIR,
) -> int:
    """Descarga cotizaciones históricas desde Yahoo Finance a archivos Parquet."""
    try:
        import yfinance as yf
    except ImportError:
        print("[Aviso] yfinance no está instalado. Omite la descarga externa.")
        return 0

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"Descargando datos históricos para {len(tickers)} activos...")
    data = yf.download(tickers, period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True)

    success = 0
    for ticker in tickers:
        try:
            df_t = data[ticker].copy() if len(tickers) > 1 else data.copy()
            if df_t.empty or df_t["Close"].isna().all():
                continue
            filename = out_p / f"{ticker}.parquet"
            df_t.to_parquet(filename)
            success += 1
        except Exception:
            pass

    return success
