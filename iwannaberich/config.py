"""
Configuración centralizada para iwannaberich.
Rutas por defecto, parámetros financieros y constantes estadísticas.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directorio raíz del repositorio
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# Rutas de datos
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_PRICES_DIR = RAW_DIR / "precios_historicos"
RAW_TICKERS_FILE = RAW_DIR / "raw_tickers.txt"
MASTER_TICKERS_FILE = RAW_DIR / "maestro_tickers.csv"
LIQUID_PRICES_FILE = PROCESSED_DIR / "precios_liquidos.parquet"
LIQUIDITY_REPORT_FILE = PROCESSED_DIR / "reporte_liquidez.csv"
STATS_MATRIX_FILE = PROCESSED_DIR / "matriz_estadistica_completa.csv"
OPPORTUNITIES_REPORT_CSV = DATA_DIR / "reporte_oportunidades.csv"
HIGHLIGHTS_REPORT_TXT = DATA_DIR / "reporte_destacados.txt"

# Asegurar directorios requeridos
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_PRICES_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros financieros por defecto
DEFAULT_RISK_FREE_RATE: float = 0.045       # Bono soberano / depósito a plazo referencial (4.5% anual)
DEFAULT_MIN_DAILY_VOLUME_CLP: float = 5_000_000.0  # Mínimo de 5M CLP diarios para liquidez
TRADING_DAYS_PER_YEAR: int = 252            # Días hábiles bursátiles
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_FAST_SMA: int = 50
DEFAULT_SLOW_SMA: int = 200
DEFAULT_VAR_PERCENTILE: float = 5.0         # 95% de confianza (percentil 5)
