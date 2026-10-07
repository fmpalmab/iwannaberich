"""
Compatibility wrapper for 06matrix.py.
Forwards to iwannaberich.metrics.
"""

import pandas as pd
from iwannaberich.config import LIQUID_PRICES_FILE, STATS_MATRIX_FILE
from iwannaberich.metrics import compute_metrics_matrix

def main():
    if not LIQUID_PRICES_FILE.is_file():
        from iwannaberich.processor import process_market_pipeline
        process_market_pipeline()

    prices = pd.read_parquet(LIQUID_PRICES_FILE)
    df_metrics = compute_metrics_matrix(prices)
    df_metrics.to_csv(STATS_MATRIX_FILE)
    print(f"Matriz guardada en: {STATS_MATRIX_FILE}")

if __name__ == "__main__":
    main()