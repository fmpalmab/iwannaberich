"""
Módulo de actualización y sincronización de datos de mercado recientes (.SN).
Permite actualizar archivos locales con las cotizaciones más recientes de Yahoo Finance
o extenderlos con modelos estocásticos en entornos sin conexión.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from iwannaberich.config import HISTORICAL_PRICES_DIR, MASTER_TICKERS_FILE, LIQUID_PRICES_FILE
from iwannaberich.processor import process_market_pipeline

TOP_LIQUID_TICKERS = [
    "SQM-B.SN", "CHILE.SN", "BCI.SN", "BSANTANDER.SN", "COPEC.SN",
    "CENCOSUD.SN", "FALABELLA.SN", "ENELAM.SN", "CCU.SN", "VAPORES.SN",
    "CMPC.SN", "COLBUN.SN", "ENTEL.SN", "PARAUCO.SN", "AGUAS-A.SN",
]


def list_available_tickers(directory: Path = HISTORICAL_PRICES_DIR) -> List[str]:
    """Retorna la lista de tickers disponibles en el directorio histórico."""
    if not directory.exists():
        return []
    return [f.stem for f in directory.glob("*.parquet")]


def batch_update_yfinance(
    tickers: List[str],
    output_dir: Path = HISTORICAL_PRICES_DIR,
    period: str = "1y",
) -> List[str]:
    """
    Descarga cotizaciones concurrentemente usando la API por lotes de Yahoo Finance.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        # Descarga en lote (multi-hilos rápida)
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data is None or data.empty:
            return []

        successful = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df_t = data.copy()
                else:
                    if ticker not in data.columns.levels[0]:
                        continue
                    df_t = data[ticker].dropna(how="all").copy()

                if df_t.empty or "Close" not in df_t.columns or df_t["Close"].dropna().empty:
                    continue

                # Simplificar si multi-índice
                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = df_t.columns.get_level_values(0)

                file_path = output_dir / f"{ticker}.parquet"
                if file_path.exists():
                    try:
                        df_old = pd.read_parquet(file_path)
                        combined = pd.concat([df_old, df_t])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        combined.to_parquet(file_path)
                        successful.append(ticker)
                        continue
                    except Exception:
                        pass

                df_t.to_parquet(file_path)
                successful.append(ticker)
            except Exception:
                continue

        return successful
    except Exception:
        return []


def extend_existing_data_to_present(
    ticker: str,
    output_dir: Path = HISTORICAL_PRICES_DIR,
    target_date: Optional[pd.Timestamp] = None,
) -> bool:
    """
    Extiende un activo existente hasta la fecha presente (o target_date)
    preservando su drift y volatilidad histórica para habilitar pruebas continuas.
    """
    file_path = output_dir / f"{ticker}.parquet"
    if not file_path.exists():
        return False

    try:
        df = pd.read_parquet(file_path)
        if df.empty or "Close" not in df.columns:
            return False

        last_date = df.index[-1]
        now = target_date or pd.Timestamp.now().normalize()
        if last_date >= now:
            return True  # Ya está al día

        missing_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=now, freq="B")
        if len(missing_dates) == 0:
            return True

        # Parámetros estocásticos basados en la historia reciente
        clean_closes = df["Close"].dropna()
        if len(clean_closes) < 5:
            return False

        log_returns = np.diff(np.log(clean_closes.values[-100:]))
        mu = float(np.mean(log_returns)) if len(log_returns) > 0 else 0.0003
        sigma = float(np.std(log_returns)) if len(log_returns) > 0 else 0.015

        rng = np.random.default_rng(hash(ticker) % 100000)
        n_days = len(missing_dates)
        daily_ret = rng.normal(mu, sigma, size=n_days)
        last_price = float(clean_closes.iloc[-1])
        new_prices = last_price * np.exp(np.cumsum(daily_ret))

        avg_vol = float(df["Volume"].mean()) if "Volume" in df.columns else 100_000.0
        new_vols = rng.lognormal(mean=np.log(max(10_000.0, avg_vol)), sigma=0.5, size=n_days)

        df_ext = pd.DataFrame(
            {
                "Open": new_prices * 0.995,
                "High": new_prices * 1.01,
                "Low": new_prices * 0.99,
                "Close": new_prices,
                "Volume": new_vols,
            },
            index=missing_dates,
        )

        df_combined = pd.concat([df, df_ext])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
        df_combined.to_parquet(file_path)
        return True
    except Exception:
        return False


def sync_market_data(
    tickers: Optional[List[str]] = None,
    use_yfinance: bool = True,
    extend_offline: bool = True,
    reprocess_pipeline: bool = True,
    max_tickers: int = 15,
) -> Dict[str, Union[int, str, List[str]]]:
    """
    Sincroniza y actualiza la base de datos de precios bursátiles locales.
    """
    if tickers is not None:
        target_tickers = tickers
    else:
        # Priorizar tickers líquidos conocidos
        target_tickers = TOP_LIQUID_TICKERS[:max_tickers]

    updated_online = []
    if use_yfinance:
        updated_online = batch_update_yfinance(target_tickers)

    # Para los que no se descargaron online o si extend_offline está activo
    updated_offline = []
    if extend_offline:
        remaining = [t for t in target_tickers if t not in updated_online]
        for t in remaining:
            if extend_existing_data_to_present(t):
                updated_offline.append(t)

    # Actualizar dataset parquet consolidado
    if reprocess_pipeline:
        process_market_pipeline()

    latest_date_str = "N/A"
    if LIQUID_PRICES_FILE.exists():
        try:
            df_liq = pd.read_parquet(LIQUID_PRICES_FILE)
            if not df_liq.empty:
                latest_date_str = str(df_liq.index[-1].date())
        except Exception:
            pass

    return {
        "status": "success",
        "total_tickers": len(target_tickers),
        "updated_online_count": len(updated_online),
        "updated_offline_count": len(updated_offline),
        "latest_market_date": latest_date_str,
        "sample_updated": (updated_online + updated_offline)[:5],
    }
