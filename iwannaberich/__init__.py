"""
iwannaberich - Suite cuantitativa de análisis bursátil, filtrado estocástico y escaneo de señales para la Bolsa de Santiago (.SN).
"""

from iwannaberich.config import *
from iwannaberich.parser import parse_tickers, clean_single_ticker, classify_ticker
from iwannaberich.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_volatility,
    calculate_drawdown_series,
)
from iwannaberich.metrics import compute_ticker_metrics, compute_metrics_matrix
from iwannaberich.loader import generate_synthetic_stock_data, download_tickers_yfinance
from iwannaberich.processor import load_and_consolidate, filter_by_liquidity, process_market_pipeline
from iwannaberich.scanner import scan_market, scan_asset_signals
from iwannaberich.visualizer import plot_technical_analysis
from iwannaberich.report import generate_market_report
from iwannaberich.pipeline import run_full_pipeline

__version__ = "2.0.0"
