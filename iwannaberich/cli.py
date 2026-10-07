"""
Interfaz de línea de comandos (CLI) unificada para iwannaberich.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from iwannaberich.config import STATS_MATRIX_FILE, LIQUID_PRICES_FILE, DEFAULT_MIN_DAILY_VOLUME_CLP, DEFAULT_RISK_FREE_RATE
from iwannaberich.pipeline import run_full_pipeline
from iwannaberich.scanner import scan_market
from iwannaberich.report import generate_market_report
from iwannaberich.metrics import compute_metrics_matrix
from iwannaberich.processor import process_market_pipeline
from iwannaberich.visualizer import plot_technical_analysis
from iwannaberich.loader import generate_synthetic_stock_data
from iwannaberich.simulation import simulate_monte_carlo, plot_monte_carlo
from iwannaberich.portfolio import optimize_portfolio, plot_efficient_frontier
from iwannaberich.data_updater import sync_market_data


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
        description="Suite Cuantitativa de Análisis, Simulación y Portafolios para la Bolsa de Santiago (.SN)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pipeline", action="store_true", help="Ejecutar el pipeline cuantitativo completo")
    group.add_argument("--scan", action="store_true", help="Escanear señales técnicas y oportunidades de mercado")
    group.add_argument("--matrix", action="store_true", help="Calcular matriz estocástica y ratios de riesgo")
    group.add_argument("--report", action="store_true", help="Generar reporte ejecutivo de destacados")
    group.add_argument("--synth", action="store_true", help="Generar dataset sintético para pruebas")
    group.add_argument("--update-data", action="store_true", help="Sincronizar y actualizar cotizaciones a la fecha más reciente")
    group.add_argument("--portfolio-opt", action="store_true", help="Optimizar portafolio Markowitz y Frontera Eficiente")

    parser.add_argument("--monte-carlo", type=str, metavar="TICKER", help="Ejecutar simulación Monte Carlo para un ticker (ej: SQM-B.SN)")
    parser.add_argument("--plot", type=str, metavar="TICKER", help="Generar gráfico técnico de un ticker (ej: SQM-B.SN)")
    parser.add_argument("--output", type=str, help="Ruta de archivo de salida para gráficos o reportes")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_DAILY_VOLUME_CLP, help="Volumen mínimo diario CLP")
    parser.add_argument("--sims", type=int, default=1000, help="Cantidad de trayectorias en Monte Carlo (default: 1000)")
    parser.add_argument("--horizon", type=int, default=252, help="Horizonte de días hábiles para proyección (default: 252)")
    parser.add_argument("--risk-free", type=float, default=DEFAULT_RISK_FREE_RATE, help="Tasa libre de riesgo anual (default: 0.045)")

    args = parser.parse_args(argv)

    if args.synth:
        print("Generando datos sintéticos para pruebas locales...")
        generate_synthetic_stock_data(n_days=500, output_dir=None)
        print("Datos generados con éxito.")
        return 0

    if args.update_data:
        print("Sincronizando y actualizando datos bursátiles locales (.SN)...")
        res = sync_market_data()
        print(f"Estado: {res['status']}")
        print(f"Tickers actualizados (online): {res['updated_online_count']}")
        print(f"Tickers extendidos (modelo): {res['updated_offline_count']}")
        print(f"Última fecha de mercado registrada: {res['latest_market_date']}")
        return 0

    if args.monte_carlo:
        ticker = args.monte_carlo.upper()
        if not ticker.endswith(".SN"):
            ticker = f"{ticker}.SN"

        if not LIQUID_PRICES_FILE.is_file():
            process_market_pipeline(min_daily_volume_clp=args.min_volume)
        import pandas as pd
        df_p = pd.read_parquet(LIQUID_PRICES_FILE)
        if ticker not in df_p.columns:
            print(f"[Error] Ticker {ticker} no encontrado en la base líquida. Tickers disponibles: {list(df_p.columns[:10])}")
            return 1

        print(f"Ejecutando Simulación Monte Carlo ({args.sims} escenarios, {args.horizon} días) para {ticker}...")
        mc_res = simulate_monte_carlo(
            df_p[ticker],
            ticker=ticker,
            n_simulations=args.sims,
            horizon_days=args.horizon,
        )

        out_img = args.output or f"monte_carlo_{ticker.replace('.SN', '')}.png"
        plot_monte_carlo(mc_res, output_path=out_img)

        print("\n=== RESULTADOS SIMULACIÓN MONTE CARLO ===")
        print(f"Ticker: {mc_res.ticker}")
        print(f"Precio Inicial: ${mc_res.start_price:,.2f} CLP")
        print(f"Precio Mediano Proyectado: ${mc_res.median_terminal_price:,.2f} CLP ({(mc_res.median_terminal_price/mc_res.start_price - 1)*100:+.2f}%)")
        print(f"Rango 90% Confianza: [${mc_res.p5_terminal_price:,.2f} - ${mc_res.p95_terminal_price:,.2f}] CLP")
        print(f"Value at Risk (VaR 95%): {mc_res.var_95_pct*100:.2f}%")
        print(f"Expected Shortfall (CVaR 95%): {mc_res.cvar_95_pct*100:.2f}%")
        print(f"Probabilidad de Pérdida a {mc_res.horizon_days} días: {mc_res.prob_loss*100:.1f}%")
        print(f"Gráfico guardado en: {out_img}")
        return 0

    if args.portfolio_opt:
        if not LIQUID_PRICES_FILE.is_file():
            process_market_pipeline(min_daily_volume_clp=args.min_volume)
        import pandas as pd
        df_p = pd.read_parquet(LIQUID_PRICES_FILE)

        print(f"Optimizando portafolio Markowitz para {len(df_p.columns)} activos líquidos...")
        port_res = optimize_portfolio(df_p, risk_free_rate=args.risk_free)

        out_img = args.output or "frontera_eficiente_markowitz.png"
        plot_efficient_frontier(port_res, output_path=out_img)

        print("\n=== CARTERA ÓPTIMA (MÁXIMO RATIO DE SHARPE) ===")
        print(f"Retorno Esperado Anualizado: {port_res.max_sharpe_portfolio.expected_return*100:.2f}%")
        print(f"Volatilidad Anualizada: {port_res.max_sharpe_portfolio.volatility*100:.2f}%")
        print(f"Ratio de Sharpe: {port_res.max_sharpe_portfolio.sharpe_ratio:.2f}")
        print("\nPonderaciones:")
        for t, w in sorted(port_res.max_sharpe_portfolio.weights.items(), key=lambda x: x[1], reverse=True):
            if w > 0.005:
                print(f"  - {t:<12}: {w*100:5.1f}%")

        print("\n=== CARTERA DE MÍNIMA VARIANZA GLOBAL (GMV) ===")
        print(f"Retorno Esperado: {port_res.min_variance_portfolio.expected_return*100:.2f}%")
        print(f"Volatilidad: {port_res.min_variance_portfolio.volatility*100:.2f}%")
        print(f"Ratio de Sharpe: {port_res.min_variance_portfolio.sharpe_ratio:.2f}")
        print(f"Gráfico guardado en: {out_img}")
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
