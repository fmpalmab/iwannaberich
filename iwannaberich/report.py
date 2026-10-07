"""
Generador de reportes ejecutivos e inteligencia de mercado cuantitativa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from iwannaberich.config import STATS_MATRIX_FILE, HIGHLIGHTS_REPORT_TXT


def generate_market_report(
    matrix_input: Union[str, Path, pd.DataFrame] = STATS_MATRIX_FILE,
    output_report: Optional[Union[str, Path]] = HIGHLIGHTS_REPORT_TXT,
) -> str:
    """Genera reporte ejecutivo en texto y markdown a partir de la matriz estadística."""
    if isinstance(matrix_input, (str, Path)):
        p = Path(matrix_input)
        if not p.is_file():
            raise FileNotFoundError(f"No existe la matriz estadística: {p}")
        df = pd.read_csv(p, index_col=0)
    else:
        df = matrix_input

    if df.empty:
        return "Matriz vacía; no se puede generar reporte."

    lines = []
    lines.append("=" * 65)
    lines.append("📊 REPORTE DE INTELIGENCIA DE MERCADO BURSÁTIL (.SN)")
    lines.append(f"Total Activos Analizados: {len(df)}")
    lines.append("=" * 65 + "\n")

    # 1. Reyes de la eficiencia (Sharpe Ratio)
    top_sharpe = df.sort_values("Sharpe_Ratio", ascending=False).head(10)
    lines.append("🏆 TOP 10 EFICIENCIA (Sharpe Ratio - Mayor retorno por riesgo)")
    cols_sharpe = ["Sharpe_Ratio", "Retorno_Anualizado", "Volatilidad_Anual"]
    lines.append(top_sharpe[[c for c in cols_sharpe if c in df.columns]].to_string())
    lines.append("\n" + "-" * 50 + "\n")

    # 2. Defensivos (Menor volatilidad)
    low_vol = df.sort_values("Volatilidad_Anual", ascending=True).head(10)
    lines.append("🛡️ TOP 10 DEFENSA (Menor Volatilidad Anualizada)")
    cols_vol = ["Volatilidad_Anual", "Max_Drawdown", "VaR_95_Diario"]
    lines.append(low_vol[[c for c in cols_vol if c in df.columns]].to_string())
    lines.append("\n" + "-" * 50 + "\n")

    # 3. Potencial Rebote / Deep Value (Drawdown con retorno positivo)
    value_candidates = df[df["Retorno_Anualizado"] > 0].sort_values("Max_Drawdown", ascending=True).head(10)
    lines.append("📉 TOP 10 CAÍDAS FUERTES CON RETORNO HISTÓRICO POSITIVO (Rebote)")
    cols_value = ["Max_Drawdown", "Retorno_Anualizado", "Calmar_Ratio"]
    lines.append(value_candidates[[c for c in cols_value if c in df.columns]].to_string())
    lines.append("\n" + "-" * 50 + "\n")

    # 4. Riesgo de cola (Kurtosis alta)
    fat_tails = df.sort_values("Kurtosis", ascending=False).head(10)
    lines.append("⚠️ TOP 10 ALERTA DE COLA (Kurtosis Alta - Posibles Cisnes Negros)")
    cols_tails = ["Kurtosis", "Skewness", "CVaR_95_Diario"]
    lines.append(fat_tails[[c for c in cols_tails if c in df.columns]].to_string())
    lines.append("\n" + "-" * 50 + "\n")

    # 5. Asimetría positiva (Skewness positiva)
    pos_skew = df.sort_values("Skewness", ascending=False).head(10)
    lines.append("🚀 TOP 10 ASIMETRÍA POSITIVA (Skewness > 0 - Subidas más explosivas)")
    cols_skew = ["Skewness", "Retorno_Anualizado", "Prob_Subida"]
    lines.append(pos_skew[[c for c in cols_skew if c in df.columns]].to_string())
    lines.append("\n")

    full_text = "\n".join(lines)

    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Reporte guardado exitosamente en: {out_p}")

    return full_text
