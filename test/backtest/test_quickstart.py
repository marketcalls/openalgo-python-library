"""The quick-start example runs end to end on a user DataFrame with no server, and on
OpenAlgo data through a client (09-roadmap, Phase 1 exit criteria).

The example is `examples/backtest_quickstart.py`: an EMA crossover (5 and 20 sessions)
on SBIN delivery with synthetic daily bars from 2024-07-01, a run from 2026-04-01 to
2026-11-06 and Rs 1,00,000. The bars before 2026-04-01 are history: the run keeps the
256 sessions its 20-session EMA needs, the EMAs are computed over them, so they are
ready when the run starts and the strategy decides from its first session. The
OpenAlgo run serves the same bars from an in-process fake client as Historify
(`source="db"`) rows, so it opens no socket. The expected trade dates are worked from
the closes with an EMA written here (alpha 2 / (n + 1), started at the first close of
the bars, 2024-07-01), not by the engine: by 256 sessions the two starts agree within
the warm-up's tolerance, so the crossings are the same. A long opens at the open of the
session after the fast EMA crosses above the slow one at a close of the run, and closes
at the open after it crosses below.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from data_fixtures.fakes import FakeClient, historify_shape

import openalgo.backtest as bt

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "backtest_quickstart.py"
SECONDS_PER_DAY = 86_400


def load_example():
    """The example script as a module (its `main` does not run on import)."""
    spec = importlib.util.spec_from_file_location("backtest_quickstart", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUICKSTART = load_example()


def historify_rows(bars: pd.DataFrame) -> list[dict]:
    """The bars as Historify serves daily rows: stamped 00:00 UTC of the IST date."""
    epoch = pd.Timestamp("1970-01-01")
    return historify_shape(
        [
            {
                "timestamp": (stamp.tz_localize(None).normalize() - epoch).days * SECONDS_PER_DAY,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": int(row.volume),
            }
            for stamp, row in bars.iterrows()
        ]
    )


def expected_round_trips(bars: pd.DataFrame, fast: int, slow: int) -> list[tuple]:
    """(entry date, exit date) of every closed trade, worked from the closes: the EMAs
    over every bar, the decisions at the closes from START on."""
    close = bars["close"]
    fast_line = close.ewm(span=fast, adjust=False).mean().to_numpy()
    slow_line = close.ewm(span=slow, adjust=False).mean().to_numpy()
    dates = [stamp.date() for stamp in bars.index]
    first = dates.index(pd.Timestamp(QUICKSTART.START).date())
    trips, entry = [], None
    for i in range(first, len(close) - 1):
        above_now, above_before = fast_line[i] > slow_line[i], fast_line[i - 1] > slow_line[i - 1]
        below_now, below_before = fast_line[i] < slow_line[i], fast_line[i - 1] < slow_line[i - 1]
        if entry is None and above_now and not above_before:
            entry = dates[i + 1]
        elif entry is not None and below_now and not below_before:
            trips.append((entry, dates[i + 1]))
            entry = None
    return trips


@pytest.fixture(scope="module")
def own_data_result() -> bt.Result:
    return bt.backtest(
        QUICKSTART.EmaCross,
        data=QUICKSTART.daily_bars(),
        start=QUICKSTART.START,
        capital=QUICKSTART.CAPITAL,
    )


@pytest.fixture(scope="module")
def openalgo_run(tmp_path_factory):
    """The quick start on OpenAlgo data: (result, the fake client it read from)."""
    cache = tmp_path_factory.mktemp("bars-cache")
    rows = historify_rows(QUICKSTART.daily_bars())
    client = FakeClient(db={("SBIN", "NSE", "D"): rows})
    with pytest.MonkeyPatch.context() as patch:
        # The local bars cache goes to a temporary folder, not the user's home.
        patch.setenv("OPENALGO_BT_CACHE", str(cache))
        result = bt.backtest(
            QUICKSTART.EmaCross,
            client,
            start=QUICKSTART.START,
            end=QUICKSTART.END,
            capital=QUICKSTART.CAPITAL,
        )
    return result, client


@pytest.fixture(scope="module")
def example_run(tmp_path_factory) -> subprocess.CompletedProcess:
    """The example run as a reader runs it: a script in a fresh interpreter, from an
    empty folder."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path_factory.mktemp("example"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=300,
    )


# ----- on a user DataFrame, no server -----------------------------------------------


def test_the_example_script_runs_and_prints_the_result_summary(example_run):
    assert example_run.returncode == 0, example_run.stderr
    lines = example_run.stdout.splitlines()
    assert lines[0] == "Result(SBIN)"
    labels = [line.split(":")[0].strip() for line in lines[-5:]]
    assert labels == ["Net profit", "Total return", "Max drawdown", "Trades", "Charges"]


def test_the_example_prints_the_figures_of_its_run(example_run, own_data_result):
    metrics = own_data_result.metrics
    assert f"Net profit   : Rs {metrics['net_profit']:,.2f}" in example_run.stdout
    assert f"Trades       : {metrics['trades']} " in example_run.stdout


def test_on_own_data_the_trades_open_and_close_the_session_after_each_cross(own_data_result):
    bars = QUICKSTART.daily_bars()
    trades = own_data_result.trades
    found = [(row.entry_time.date(), row.exit_time.date()) for row in trades.itertuples()]
    assert found == expected_round_trips(bars, QUICKSTART.EmaCross.fast, QUICKSTART.EmaCross.slow)
    assert len(found) > 0


# ----- on OpenAlgo data, through a client --------------------------------------------


def test_on_openalgo_data_the_daily_bars_of_the_whole_range_are_read_from_historify(
    openalgo_run,
):
    # A daily strategy runs on daily bars. "auto" first asks Historify for the daily
    # bars of the range (the coverage probe); it holds every session, so the bars are
    # then read from it. Then the history before the range, from the same source, for
    # what the 20-session EMA needs, 256 sessions (the longer of the two):
    # 256 x 7/5 x 1.1 = 394.24, so 395 calendar days and a week more, 402, before
    # 2026-04-01 is Monday 2025-02-24; 2026-03-31 is an NSE holiday, so the last session
    # asked for is 2026-03-30. Only the read-only history endpoint is called.
    _, client = openalgo_run
    whole_range = ("db", "D", QUICKSTART.START, QUICKSTART.END)
    history = ("db", "D", "2025-02-24", "2026-03-30")
    assert client.history_calls() == [whole_range, whole_range, history]
    assert client.endpoints() == {"history"}
    assert client.keys_seen == {FakeClient.api_key}


def test_on_openalgo_data_the_trades_are_those_worked_from_the_closes(openalgo_run):
    result, _ = openalgo_run
    bars = QUICKSTART.daily_bars()
    found = [(row.entry_time.date(), row.exit_time.date()) for row in result.trades.itertuples()]
    assert found == expected_round_trips(bars, QUICKSTART.EmaCross.fast, QUICKSTART.EmaCross.slow)


def test_on_openalgo_data_the_result_equals_the_run_on_the_same_bars_given_as_data(
    openalgo_run, own_data_result
):
    result, _ = openalgo_run
    pd.testing.assert_frame_equal(result.trades, own_data_result.trades, check_exact=True)
    pd.testing.assert_frame_equal(result.equity, own_data_result.equity, check_exact=True)
    pd.testing.assert_frame_equal(result.fills, own_data_result.fills, check_exact=True)


def test_on_openalgo_data_the_metrics_add_up_from_the_trades_and_the_equity(openalgo_run):
    result, _ = openalgo_run
    metrics, trades = result.metrics, result.trades
    final_equity = result.equity["equity_rupees"].iloc[-1]
    assert metrics["trades"] == len(trades)
    assert metrics["net_profit"] == pytest.approx(final_equity - QUICKSTART.CAPITAL, abs=1e-6)
    assert metrics["total_return"] == pytest.approx(final_equity / QUICKSTART.CAPITAL - 1.0)
    assert metrics["total_costs"] == pytest.approx(trades["costs_rupees"].sum(), abs=1e-9)
    wins = int((trades["net_pnl_rupees"] > 0).sum())
    assert metrics["win_rate"] == pytest.approx(wins / len(trades))


def test_on_openalgo_data_the_manifest_records_the_client_but_not_its_key(openalgo_run):
    result, _ = openalgo_run
    assert result.manifest["arguments"]["client"] == "given"
    assert FakeClient.api_key not in str(result.manifest)
