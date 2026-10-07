"""
Compatibility wrapper for 01ticker_parser.py.
Forwards to iwannaberich.parser.
"""

from iwannaberich.parser import parse_tickers

def main():
    df = parse_tickers()
    print(f"Limpieza completada. {len(df)} tickers recuperados.")

if __name__ == "__main__":
    main()