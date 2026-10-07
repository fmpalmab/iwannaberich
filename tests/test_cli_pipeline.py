"""Pruebas de integración para generador sintético y CLI de iwannaberich."""

import pytest
from iwannaberich.loader import generate_synthetic_stock_data
from iwannaberich.cli import main


def test_synthetic_data_generation():
    df = generate_synthetic_stock_data(tickers=["TEST1.SN", "TEST2.SN"], n_days=100)
    assert df.shape == (100, 2)
    assert not df.isna().any().any()
    assert (df > 0).all().all()


def test_cli_synth():
    exit_code = main(["--synth"])
    assert exit_code == 0


def test_cli_scan():
    exit_code = main(["--scan"])
    assert exit_code == 0


def test_cli_report():
    exit_code = main(["--report"])
    assert exit_code == 0
