"""
Compatibility wrapper for 06report.py.
Forwards to iwannaberich.report.
"""

from iwannaberich.report import generate_market_report

def main():
    generate_market_report()

if __name__ == "__main__":
    main()