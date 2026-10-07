"""
Interfaz de línea de comandos (CLI) unificada para iwannaberich.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from iwannaberich.config import STATS_MATRIX_FILE, LIQUID_PRICES_FILE, DEFAULT_MIN_DAILY_VOLUME_CLP
from iwannaberich.pipeline import run_full_pipeline
from iwannaberich.scanner import scan_market
from iwannaberich.report import generate_market_report
from iwannaberich.metrics import compute_metrics_matrix
from iwannaberich.processor import process_market_pipeline
from iwannaberich.visualizer import plot_technical_analysis
from iwannaberich.loader import generate_synthetic_stock_data


def configure_console_output() -> None:
    """Configura la salida estándar de Windows para evitar errores de codificación cp1252."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    configure_console_output()

    parser = argparse.ArgumentParser(
        prog="iwannaberich",
        description="Suite Cuantitativa de Análisis y Escaneo Bursátil para la Bolsa de Santiago (.SN)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pipeline", action="store_true", help="Ejecutar el pipeline cuantitativo completo")
    group.add_argument("--scan", action="store_true", help="Escanear señales técnicas y oportunidades de mercado")
    group.add_argument("--matrix", action="store_true", help="Calcular matriz estocástica y ratios de riesgo")
    group.add_argument("--report", action="store_true", help="Generar reporte ejecutivo de destacados")
    group.add_argument("--synth", action="store_true", help="Generar dataset sintético para pruebas")

    parser.add_argument("--plot", type=str, metavar="TICKER", help="Generar gráfico técnico de un ticker (ej: SQM-B.SN)")
    parser.add_argument("--output", type=str, help="Ruta de archivo de salida para gráficos o reportes")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_DAILY_VOLUME_CLP, help="Volumen mínimo diario CLP")

    args = parser.parse_args(argv)

    if args.synth:
        print("Generando datos sintéticos para pruebas locales...")
        generate_synthetic_stock_data(n_days=500, output_dir=None)
        print("Datos generados con éxito.")
        return 0

    if args.plot:
        ticker = args.plot.upper()
        if not ticker.endswith(".SN"):
            ticker = f"{ticker}.SN"
        out = args.output or f"analisis_{ticker.replace('.SN', '')}.png"
        plot_technical_analysis(ticker, output_path=out)
        return 0

    if args.matrix:
        print("Calculando matriz estocástica...")
        if not LIQUID_PRICES_FILE.is_file():
            process_market_pipeline(min_daily_volume_clp=args.min_volume)
        import pandas as pd
        df_p = pd.read_parquet(LIQUID_PRICES_FILE)
        df_m = compute_metrics_matrix(df_p)
        df_m.to_csv(STATS_MATRIX_FILE)
        print(f"Matriz guardada en: {STATS_MATRIX_FILE}")
        return 0

    if args.scan:
        print("Escaneando mercado...")
        df_opp = scan_market()
        if not df_opp.empty:
            print("\n" + df_opp.to_string())
        else:
            print("No se encontraron candidatos.")
        return 0

    if args.report:
        print("Generando reporte de inteligencia de mercado...")
        text = generate_market_report()
        print(text)
        return 0

    # Por defecto o con --pipeline
    res = run_full_pipeline(min_daily_volume_clp=args.min_volume)
    print("\n--- RESUMEN DE OPORTUNIDADES ---")
    if not res["opportunities"].empty:
        print(res["opportunities"][["Ticker", "Precio", "Señal Técnica", "RSI (14)"]].head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
