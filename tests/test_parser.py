"""Pruebas unitarias para el parser y clasificador de tickers."""

import pytest
from iwannaberich.parser import clean_single_ticker, classify_ticker


def test_clean_single_ticker_normal():
    assert clean_single_ticker("CHILE.SN") == "CHILE.SN"
    assert clean_single_ticker("BCI.SN") == "BCI.SN"


def test_clean_single_ticker_with_digits():
    # Eliminar dígitos pegados
    assert clean_single_ticker("275LLTM.SN") == "LTM.SN"
    assert clean_single_ticker("10BSANTANDER.SN") == "BSANTANDER.SN"


def test_clean_single_ticker_triple_letter_exception():
    # CCU.SN no debe cortarse
    assert clean_single_ticker("CCU.SN") == "CCU.SN"


def test_classify_ticker():
    assert classify_ticker("CFI-INDEX.SN") == "FONDO_INVERSION"
    assert classify_ticker("AAPL.SN") == "US_TECH_ENERGY_CLP"
    assert classify_ticker("COPEC.SN") == "CHILE_EQUITY"
