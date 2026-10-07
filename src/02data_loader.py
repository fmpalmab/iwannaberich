"""
Compatibility wrapper for 02data_loader.py.
Forwards to iwannaberich.loader.
"""

import pandas as pd
from iwannaberich.config import MASTER_TICKERS_FILE, HISTORICAL_PRICES_DIR
from iwannaberich.loader import download_tickers_yfinance, generate_synthetic_stock_data

def main():
    if MASTER_TICKERS_FILE.is_file():
        df_master = pd.read_csv(MASTER_TICKERS_FILE)
        tickers = df_master["Ticker"].tolist()
        download_tickers_yfinance(tickers, output_dir=HISTORICAL_PRICES_DIR)
    else:
        print("Generando datos sintéticos de respaldo...")
        generate_synthetic_stock_data(output_dir=HISTORICAL_PRICES_DIR)

if __name__ == "__main__":
    main()