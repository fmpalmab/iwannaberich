"""
Visualización técnica de alta calidad de series bursátiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import pandas as pd

from iwannaberich.config import LIQUID_PRICES_FILE
from iwannaberich.indicators import calculate_rsi, calculate_sma, calculate_bollinger_bands


def plot_technical_analysis(
    ticker: str,
    prices_df: Optional[pd.DataFrame] = None,
    prices_file: Union[str, Path] = LIQUID_PRICES_FILE,
    window: int = 252,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Genera gráfico técnico con Price, SMA, Bandas de Bollinger y RSI."""
    if prices_df is None:
        p = Path(prices_file)
        if not p.is_file():
            print(f"[Error] Archivo de precios no encontrado: {p}")
            return None
        prices_df = pd.read_parquet(p)

    if ticker not in prices_df.columns:
        print(f"[Error] Ticker '{ticker}' no está en el conjunto de datos.")
        return None

    full_series = prices_df[ticker].dropna()
    if len(full_series) < 30:
        print(f"[Error] Muy poca historia para graficar '{ticker}'.")
        return None

    # Cortar a la ventana requerida manteniendo historia previa para medias móviles
    data = full_series.tail(window)
    sma_50 = calculate_sma(full_series, 50).tail(window)
    sma_200 = calculate_sma(full_series, 200).tail(window)
    _b_mid, b_upper, b_lower = calculate_bollinger_bands(full_series, 20)
    b_upper = b_upper.tail(window)
    b_lower = b_lower.tail(window)

    rsi = calculate_rsi(full_series, 14).tail(window)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Panel 1: Precio y tendencias
    ax1.plot(data.index, data, label=f"Precio {ticker}", color="#1f1f1f", lw=1.5)
    ax1.plot(sma_50.index, sma_50, label="SMA 50", color="#ff7f0e", lw=1.2, ls="--", alpha=0.8)
    ax1.plot(sma_200.index, sma_200, label="SMA 200", color="#1f77b4", lw=1.3, ls="-.", alpha=0.8)
    ax1.fill_between(
        data.index, b_lower, b_upper, color="#7f7f7f", alpha=0.12, label="Bandas Bollinger (20, 2)"
    )

    ax1.set_title(f"Análisis Técnico Cuantitativo: {ticker}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Precio (CLP)", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: RSI
    ax2.plot(rsi.index, rsi, label="RSI (14)", color="#9467bd", lw=1.4)
    ax2.axhline(70, color="#d62728", linestyle=":", lw=1.2, alpha=0.7, label="Sobrecompra (70)")
    ax2.axhline(30, color="#2ca02c", linestyle=":", lw=1.2, alpha=0.7, label="Sobreventa (30)")
    ax2.fill_between(rsi.index, rsi, 30, where=(rsi < 30), color="#2ca02c", alpha=0.3)
    ax2.fill_between(rsi.index, rsi, 70, where=(rsi > 70), color="#d62728", alpha=0.3)

    ax2.set_ylabel("RSI", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=130)
        print(f"Gráfico técnico guardado en: {out_p}")

    if show:
        plt.show()

    plt.close(fig)
    return fig
