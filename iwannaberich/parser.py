"""
Módulo de parsing y clasificación de tickers bursátiles chilenos (.SN).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from iwannaberich.config import RAW_TICKERS_FILE, MASTER_TICKERS_FILE


def clean_single_ticker(raw_match: str) -> Optional[str]:
    """Limpia un ticker individual eliminando números pegados y duplicados de letras."""
    # Etapa 1: Eliminar dígitos iniciales pegados
    no_digits = re.sub(r"^\d+", "", raw_match.strip())
    if not no_digits:
        return None

    # Etapa 2: Corregir duplicación de primera letra común en fuentes raspadas
    final_ticker = no_digits
    if final_ticker.startswith("CCU"):
        pass
    elif len(final_ticker) > 2 and final_ticker[0] == final_ticker[1]:
        final_ticker = final_ticker[1:]
        # Caso especial triple letra (ej. AAAPL.SN -> AAPL.SN)
        if len(final_ticker) > 2 and final_ticker[0] == final_ticker[1]:
            final_ticker = final_ticker[1:]

    # Filtro de cabeceras o basura
    if "Símbolo" in final_ticker or not final_ticker.endswith(".SN"):
        return None

    return final_ticker


def classify_ticker(ticker: str) -> str:
    """Clasifica el activo según su tipo."""
    if "CFI" in ticker or "-E.SN" in ticker:
        return "FONDO_INVERSION"
    if ticker in {"AAPL.SN", "MSFT.SN", "TSLA.SN", "GOOGL.SN", "NVDA.SN", "AMZN.SN", "AMZNCL.SN"}:
        return "US_TECH_ENERGY_CLP"
    return "CHILE_EQUITY"


def parse_tickers(
    input_file: Union[str, Path] = RAW_TICKERS_FILE,
    output_file: Optional[Union[str, Path]] = MASTER_TICKERS_FILE,
) -> pd.DataFrame:
    """
    Parsea la lista sucia de tickers desde input_file, elimina duplicados,
    clasifica los instrumentos y opcionalmente guarda el maestro CSV.
    """
    path_in = Path(input_file)
    if not path_in.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo de tickers: {path_in}")

    with open(path_in, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    raw_matches = re.findall(r"(\d*[A-Z0-9-]+\.SN)", content)
    clean_tickers: List[str] = []

    for item in raw_matches:
        cleaned = clean_single_ticker(item)
        if cleaned:
            clean_tickers.append(cleaned)

    unique_tickers = sorted(list(set(clean_tickers)))

    rows = [{"Ticker": t, "Category": classify_ticker(t)} for t in unique_tickers]
    df = pd.DataFrame(rows)

    if output_file:
        path_out = Path(output_file)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_out, index=False)

    return df
