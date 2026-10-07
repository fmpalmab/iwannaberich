"""
Compatibility wrapper for 04scanner.py.
Forwards to iwannaberich.scanner.
"""

from iwannaberich.scanner import scan_market

def main():
    df = scan_market()
    print(f"Análisis completado. {len(df)} activos analizados.")

if __name__ == "__main__":
    main()