"""
Compatibility wrapper for 03market_processor.py.
Forwards to iwannaberich.processor.
"""

from iwannaberich.processor import process_market_pipeline

def main():
    df = process_market_pipeline()
    print(f"Archivo maestro guardado con {len(df.columns)} activos.")

if __name__ == "__main__":
    main()