import pandas as pd
import pytest

from iwannaberich.data_updater import extend_existing_data_to_present, list_available_tickers, sync_market_data


def test_extend_existing_data_to_present(tmp_path):
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 102.0,
            "Low": 99.0,
            "Close": 101.0,
            "Volume": 50000.0,
        },
        index=dates,
    )
    p_file = tmp_path / "TEST.SN.parquet"
    df.to_parquet(p_file)

    target_date = pd.Timestamp("2024-03-15")
    success = extend_existing_data_to_present("TEST.SN", output_dir=tmp_path, target_date=target_date)

    assert success is True
    df_updated = pd.read_parquet(p_file)
    assert len(df_updated) > len(dates)
    assert df_updated.index[-1] == target_date


def test_sync_market_data_integration():
    res = sync_market_data(tickers=["SQM-B.SN"], use_yfinance=False, extend_offline=True, reprocess_pipeline=False)
    assert res["status"] == "success"
    assert res["total_tickers"] == 1
