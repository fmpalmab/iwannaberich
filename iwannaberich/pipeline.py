"""
Pipeline orquestador para ejecutar el flujo completo cuantitativo de iwannaberich.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from iwannaberich.config import (
    RAW_TICKERS_FILE,
    MASTER_TICKERS_FILE,
    HISTORICAL_PRICES_DIR,
    LIQUID_PRICES_FILE,
    STATS_MATRIX_FILE,
    OPPORTUNITIES_REPORT_CSV,
    HIGHLIGHTS_REPORT_TXT,
)
from iwannaberich.parser import parse_tickers
from iwannaberich.loader import generate_synthetic_stock_data
from iwannaberich.processor import process_market_pipeline
from iwannaberich.metrics import compute_metrics_matrix
from iwannaberich.scanner import scan_market
from iwannaberich.report import generate_market_report


def run_full_pipeline(
    use_synthetic_if_missing: bool = True,
    min_daily_volume_clp: float = 5_000_000.0,
) -> Dict[str, Any]:
    """Ejecuta el flujo integral de punta a punta."""
    print("=== INICIANDO PIPELINE CUANTITATIVO I WANNABE RICH ===")

    # 1. Parsing de tickers
    if RAW_TICKERS_FILE.is_file():
        print("[1/5] Parseando tickers desde archivo crudo...")
        df_tickers = parse_tickers(RAW_TICKERS_FILE, MASTER_TICKERS_FILE)
        print(f"      Total tickers parseados: {len(df_tickers)}")
    else:
        print("[1/5] Archivo raw_tickers no encontrado, omitiendo paso de parsing.")

    # 2. Verificar datos históricos o sintetizar
    existing_parquets = list(HISTORICAL_PRICES_DIR.glob("*.parquet"))
    if not existing_parquets and use_synthetic_if_missing:
        print("[2/5] No se encontraron archivos de precios. Generando datos sintéticos de alta fidelidad...")
        generate_synthetic_stock_data(output_dir=HISTORICAL_PRICES_DIR, n_days=500)
        existing_parquets = list(HISTORICAL_PRICES_DIR.glob("*.parquet"))
    print(f"[2/5] Archivos históricos disponibles: {len(existing_parquets)}")

    # 3. Consolidación y filtro de liquidez
    print("[3/5] Consolidando datos y aplicando filtro de liquidez bursátil...")
    df_prices = process_market_pipeline(
        input_dir=HISTORICAL_PRICES_DIR,
        output_file=LIQUID_PRICES_FILE,
        min_daily_volume_clp=min_daily_volume_clp,
    )
    print(f"      Activos líquidos consolidados: {len(df_prices.columns)}")

    # 4. Matriz estadística y estocástica
    print("[4/5] Computando matriz estocástica completa...")
    df_metrics = compute_metrics_matrix(df_prices)
    df_metrics.to_csv(STATS_MATRIX_FILE)
    print(f"      Métricas calculadas para {len(df_metrics)} activos -> {STATS_MATRIX_FILE.name}")

    # 5. Escáner técnico y reporte ejecutivo
    print("[5/5] Escaneando oportunidades y generando reporte ejecutivo...")
    df_opportunities = scan_market(df_prices, output_csv=OPPORTUNITIES_REPORT_CSV)
    report_text = generate_market_report(df_metrics, output_report=HIGHLIGHTS_REPORT_TXT)
    print(f"      Oportunidades encontradas: {len(df_opportunities)}")

    print("\n=== PIPELINE FINALIZADO CON ÉXITO ===")
    return {
        "prices": df_prices,
        "metrics": df_metrics,
        "opportunities": df_opportunities,
        "report": report_text,
    }
