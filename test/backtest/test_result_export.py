"""`Result.to_csv` and `Result.to_excel`: every table of `to_dataframes()` written out,
its IST times as plain IST date-times (a spreadsheet cell holds no time zone).

The run is the intraday round trip of `accounting_runs`: bought at the 09:15 bar's
close, filled at the 09:30 IST open on 2026-11-04, and sold at the 09:45 open.
"""

import sys

import pandas as pd
import pytest
from accounting_runs import round_trip

from openalgo.backtest.errors import MissingDependencyError


@pytest.fixture(scope="module")
def result():
    return round_trip({"qty": 10}, costs=None, slippage=0)


def test_to_csv_writes_one_file_per_table(result, tmp_path):
    written = result.to_csv(tmp_path / "run")
    names = {path.stem for path in written}
    assert names == set(result.to_dataframes())
    assert all(path.parent == tmp_path / "run" and path.suffix == ".csv" for path in written)


def test_to_csv_writes_ist_times_without_a_zone(result, tmp_path):
    result.to_csv(tmp_path)
    fills = pd.read_csv(tmp_path / "fills.csv")
    assert list(fills["time"]) == ["2026-11-04 09:30:00", "2026-11-04 09:45:00"]


def test_to_csv_writes_the_equity_times_as_a_column(result, tmp_path):
    result.to_csv(tmp_path)
    equity = pd.read_csv(tmp_path / "equity.csv")
    assert equity.columns[0] == "time"
    assert equity["time"].iloc[-1] == "2026-11-04 15:30:00"


def test_to_excel_without_openpyxl_says_how_to_install_it(result, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(MissingDependencyError, match="pip install openpyxl"):
        result.to_excel(tmp_path / "run.xlsx")


def test_to_excel_writes_one_sheet_per_table_with_ist_times(result, tmp_path):
    pytest.importorskip("openpyxl")
    path = result.to_excel(tmp_path / "run.xlsx")
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == set(result.to_dataframes())
    expected = [pd.Timestamp("2026-11-04 09:30:00"), pd.Timestamp("2026-11-04 09:45:00")]
    assert list(sheets["fills"]["time"]) == expected
