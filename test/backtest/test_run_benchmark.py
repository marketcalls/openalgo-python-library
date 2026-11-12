"""`bt.backtest(benchmark=...)`: the benchmark of a run (FR-REP-05; `_benchmark`).

The runs are those of `benchmark_runs`: `Hold` buys 100 SBIN at the first of five daily
closes and holds them, and NIFTY on NSE_INDEX moves by BENCH_RETURNS. Expected figures
are computed here from those inputs with the standard library: the strategy's session
returns from the equity the module docstring derives, the benchmark's from its levels.
"""

import datetime as dt
import json
import math
import statistics

import numpy as np
import pandas as pd
import pytest
from accounting_runs import DAY, round_trip
from benchmark_runs import (
    BENCH_RETURNS,
    CAPITAL,
    CLOSES,
    DAYS,
    ENTRY_PRICE,
    NIFTY_LEVELS,
    OPENING_DAY,
    QTY,
    QTY_RANGE,
    Hold,
    client_run,
    fake_client,
    index_bars,
    index_requests,
    nifty_closes,
    own_run,
    session_close,
    share_bars,
)

import openalgo.backtest as bt
from openalgo.backtest._benchmark import benchmark_request
from openalgo.backtest.data.cache import CACHE_ENV
from openalgo.backtest.errors import ConfigError

ONE_DAY = dt.timedelta(days=1)


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Keep the raw rows the fake Historify serves out of the user's cache."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))


def strategy_returns(qty: int = QTY) -> list[float]:
    """Session returns of `Hold`: flat on the first session (its order fills at the
    second open, 1,000.00), then the equity 1,00,000 + qty x (close - 1,000)."""
    equity = [CAPITAL] + [CAPITAL + qty * (close - ENTRY_PRICE) for close in CLOSES[1:]]
    returns = [equity[0] / CAPITAL - 1]
    returns += [later / earlier - 1 for earlier, later in zip(equity, equity[1:], strict=False)]
    return returns


def beta(strategy: list[float], bench: list[float]) -> float:
    return statistics.covariance(strategy, bench) / statistics.variance(bench)


def assert_figures_against(result: bt.Result, bench: list[float]) -> None:
    """Beta, correlation, the benchmark return and its session count, by hand, on the
    sessions whose benchmark return is known (NaN in `bench` marks one that is not)."""
    pairs = [(s, b) for s, b in zip(strategy_returns(), bench, strict=True) if not math.isnan(b)]
    strategy, known = [s for s, _ in pairs], [b for _, b in pairs]
    figures = result.metrics
    assert figures["beta"] == pytest.approx(beta(strategy, known))
    assert figures["correlation"] == pytest.approx(statistics.correlation(strategy, known))
    assert figures["benchmark_total_return"] == pytest.approx(math.prod(1 + b for b in known) - 1)
    assert figures["benchmark_sessions"] == len(known)


# ----- an index symbol through OpenAlgo ----------------------------------------------


def test_an_index_symbol_is_read_from_the_index_exchange_from_the_session_before_the_run():
    client = fake_client()
    client_run(client, benchmark="NIFTY")
    requests = index_requests(client)
    assert {(symbol, exchange) for symbol, exchange, _, _ in requests} == {("NIFTY", "NSE_INDEX")}
    assert min(first for _, _, first, _ in requests) == OPENING_DAY.isoformat()
    assert max(last for _, _, _, last in requests) == DAYS[-1].isoformat()


def test_an_index_symbol_gives_the_benchmark_figures_expected_by_hand():
    assert_figures_against(client_run(fake_client(), benchmark="NIFTY"), BENCH_RETURNS)


def test_alpha_and_the_information_ratio_follow_their_definitions():
    figures = client_run(fake_client(), benchmark="NIFTY").metrics
    strategy = strategy_returns()
    # Annualised by the sessions per year of the NSE calendar (test_metrics_ratios).
    per_year = figures["sessions_per_year"]
    slope = beta(strategy, BENCH_RETURNS)
    alpha = (statistics.mean(strategy) - slope * statistics.mean(BENCH_RETURNS)) * per_year
    active = [s - b for s, b in zip(strategy, BENCH_RETURNS, strict=True)]
    ratio = statistics.mean(active) / statistics.stdev(active) * math.sqrt(per_year)
    assert figures["alpha"] == pytest.approx(alpha)
    assert figures["information_ratio"] == pytest.approx(ratio)


def test_the_assumptions_name_the_benchmark_and_where_its_closes_came_from():
    stated = client_run(fake_client(), benchmark="NIFTY").assumptions["benchmark"]
    assert stated == (
        "NIFTY on NSE_INDEX: 6 daily closes from Historify (source='db'), 2026-10-30 to 2026-11-06"
    )


def test_without_a_benchmark_no_index_is_read_and_no_benchmark_figure_appears():
    client = fake_client()
    result = client_run(client)
    assert index_requests(client) == []
    assert "beta" not in result.metrics
    assert result.assumptions["benchmark"] is None


def test_an_index_the_source_does_not_hold_is_a_warning_and_the_run_has_no_benchmark():
    result = client_run(fake_client(with_index=False), benchmark="NIFTY")
    assert any(
        text.startswith("benchmark NIFTY on NSE_INDEX: no closes") for text in result.warnings
    )
    assert "beta" not in result.metrics
    assert result.assumptions["benchmark"].startswith("NIFTY on NSE_INDEX: no closes")


def test_the_benchmark_keeps_only_closes_made_by_now():
    # At 20:00 on 2026-11-05 the session of 2026-11-06 has not opened.
    result = client_run(fake_client(), benchmark="NIFTY", now="2026-11-05 20:00")
    stated = result.assumptions["benchmark"]
    assert stated.endswith("5 daily closes from Historify (source='db'), 2026-10-30 to 2026-11-05")
    assert result.performance.benchmark_equity.index[-1] == session_close(DAYS[3])


# ----- given closes -------------------------------------------------------------------


@pytest.mark.parametrize(
    "closes",
    [
        nifty_closes(),
        pd.Series(NIFTY_LEVELS, index=[session_close(day) for day in [OPENING_DAY, *DAYS]]),
        pd.DataFrame(
            {
                "ts_close": [session_close(day).value for day in [OPENING_DAY, *DAYS]],
                "close": NIFTY_LEVELS,
            }
        ),
    ],
    ids=["dates", "times at the session close", "bars with ts_close"],
)
def test_given_closes_give_the_benchmark_figures_expected_by_hand(closes):
    result = own_run(benchmark=closes)
    assert_figures_against(result, BENCH_RETURNS)
    assert not any(text.startswith("benchmark") for text in result.warnings)


def test_own_data_holding_the_index_supplies_its_closes():
    result = own_run(data={"SBIN": share_bars(), "NIFTY": index_bars()}, benchmark="NIFTY")
    assert_figures_against(result, BENCH_RETURNS)
    assert "6 daily closes from own data (data=)" in result.assumptions["benchmark"]


def test_own_data_without_the_index_is_a_warning_and_reads_no_server():
    result = own_run(data={"SBIN": share_bars()}, benchmark="NIFTY")
    assert any("data= holds no bars for NIFTY" in text for text in result.warnings)
    assert "beta" not in result.metrics


def test_a_benchmark_missing_a_session_leaves_it_missing_and_warns():
    missing = DAYS[2]
    result = own_run(benchmark=nifty_closes().drop(pd.Timestamp(missing)))
    # 2026-11-04 has no close and 2026-11-05 no previous one: neither has a return.
    assert_figures_against(result, [*BENCH_RETURNS[:2], math.nan, math.nan, BENCH_RETURNS[4]])
    note = next(text for text in result.warnings if text.startswith("benchmark"))
    assert "no close on 1 of the 6 sessions" in note and missing.isoformat() in note


def test_a_benchmark_without_the_close_before_the_run_has_no_first_return():
    result = own_run(benchmark=nifty_closes().iloc[1:])
    assert result.metrics["benchmark_sessions"] == len(DAYS) - 1
    assert any(OPENING_DAY.isoformat() in text for text in result.warnings)


def test_an_intraday_run_reads_the_benchmark_at_its_session_end():
    # One session of 1-minute bars (accounting_runs); the index gains 1 percent on it.
    closes = pd.Series([20_000.0, 20_200.0], index=pd.DatetimeIndex([DAY - ONE_DAY, DAY]))
    figures = round_trip(costs=None, slippage=0, benchmark=closes).metrics
    assert figures["benchmark_total_return"] == pytest.approx(0.01)
    assert figures["benchmark_sessions"] == 1


def test_the_benchmark_never_changes_the_trades_or_the_equity():
    alone, compared = own_run(), own_run(benchmark=nifty_closes())
    pd.testing.assert_frame_equal(alone.fills, compared.fills)
    pd.testing.assert_frame_equal(alone.equity, compared.equity)


# ----- what benchmark= accepts --------------------------------------------------------


@pytest.mark.parametrize(
    ("exchange", "index"),
    [("NSE", "NSE_INDEX"), ("NFO", "NSE_INDEX"), ("BSE", "BSE_INDEX"), ("BFO", "BSE_INDEX")],
)
def test_a_bare_symbol_is_read_on_the_index_exchange_of_the_strategy_exchange(exchange, index):
    assert benchmark_request("NIFTY", exchange) == ("NIFTY", index)


def test_a_symbol_and_exchange_pair_is_read_as_given():
    assert benchmark_request(("SENSEX", "BSE_INDEX"), "NSE") == ("SENSEX", "BSE_INDEX")


def test_a_bare_symbol_on_an_exchange_with_no_index_exchange_is_refused():
    with pytest.raises(ConfigError, match="no index exchange of its own"):
        benchmark_request("NIFTY", "MCX")


@pytest.mark.parametrize(
    ("benchmark", "message"),
    [
        (42, "benchmark takes an index symbol"),
        ("", "benchmark takes an index symbol"),
        (("NIFTY", "NSE"), "the exchange must be one of"),
        (("GOLDIDX", "MCX_INDEX"), "the exchange must be one of"),
        (nifty_closes().pct_change().fillna(0.0), "levels"),
        (
            pd.Series(NIFTY_LEVELS, index=pd.date_range("2026-10-30 15:30", periods=6)),
            "time zone",
        ),
        (pd.DataFrame({"level": NIFTY_LEVELS}), "close column"),
        (nifty_closes().iloc[::-1], "increase"),
    ],
    ids=[
        "a number",
        "an empty symbol",
        "a cash exchange",
        "an exchange the calendar does not know",
        "returns",
        "naive times",
        "no close column",
        "unsorted",
    ],
)
def test_an_unusable_benchmark_is_refused(benchmark, message):
    with pytest.raises(ConfigError, match=message):
        own_run(benchmark=benchmark)


# ----- the reports --------------------------------------------------------------------


def test_the_metrics_table_lists_the_benchmark_figures():
    table = own_run(benchmark=nifty_closes()).metrics_table()
    rows = table.loc[["alpha", "beta", "information_ratio", "benchmark_total_return"]]
    assert set(rows["group"]) == {"Benchmark"}
    assert rows.loc["beta", "value"] == pytest.approx(beta(strategy_returns(), BENCH_RETURNS))


def test_the_yearly_comparison_compounds_the_benchmark():
    comparison = own_run(benchmark=nifty_closes()).performance.benchmark_comparison("year")
    assert comparison.loc[2026, "benchmark"] == pytest.approx(
        NIFTY_LEVELS[-1] / NIFTY_LEVELS[0] - 1
    )


def test_the_tearsheet_compares_with_the_benchmark_and_names_it():
    page = client_run(fake_client(), benchmark="NIFTY").tearsheet()
    assert "Against the benchmark" in page and "Benchmark sessions" in page
    assert "benchmark: NIFTY on NSE_INDEX: 6 daily closes from Historify" in page


# ----- manifest, rerun and sweeps -----------------------------------------------------


def test_the_manifest_records_the_index_its_source_and_its_data():
    record = client_run(fake_client(), benchmark="NIFTY").manifest["arguments"]["benchmark"]
    assert (record["symbol"], record["exchange"], record["source"]) == ("NIFTY", "NSE_INDEX", "db")
    assert record["partitions"]
    assert all("/NSE_INDEX/NIFTY/D" in key for key in record["partitions"])


def test_the_manifest_records_given_closes_by_their_hash_and_round_trips_through_json():
    manifest = own_run(benchmark=nifty_closes()).manifest
    record = manifest["arguments"]["benchmark"]
    assert record["source"] == "given" and record["partitions"]["given"].startswith("sha256:")
    assert json.loads(json.dumps(manifest)) == manifest


def test_a_run_without_a_benchmark_records_none():
    assert own_run().manifest["arguments"]["benchmark"] is None


def test_a_rerun_reproduces_the_benchmark_figures():
    result = client_run(fake_client(), benchmark="NIFTY")
    again = result.rerun()
    assert again.manifest == result.manifest
    assert again.metrics["beta"] == result.metrics["beta"]
    assert again.assumptions["benchmark"] == result.assumptions["benchmark"]


def test_a_rerun_says_when_the_benchmark_closes_changed():
    closes = nifty_closes()
    result = own_run(benchmark=closes)
    closes.iloc[-1] += 1.0
    assert result.rerun().warnings[0].startswith("The benchmark's data differ")


def test_a_rerun_can_drop_the_benchmark():
    rerun = own_run(benchmark=nifty_closes()).rerun(benchmark=None)
    assert "beta" not in rerun.metrics
    assert not any("benchmark" in text for text in rerun.warnings)


def test_a_sweep_measures_every_combination_against_the_same_benchmark():
    # Three combinations and two workers: the first runs here, the other two in workers.
    plan = []
    sweep = bt.optimize(
        Hold,
        data=share_bars(),
        capital=CAPITAL,
        costs=None,
        slippage=0,
        now="2026-11-10",
        benchmark=nifty_closes(),
        objective="beta",
        workers=2,
        progress=plan.append,
    )
    assert "in 2 worker processes" in plan[0]
    expected = [beta(strategy_returns(qty), BENCH_RETURNS) for qty in QTY_RANGE]
    assert sweep.table["objective"].to_numpy() == pytest.approx(np.array(expected))
    selected = sweep.select(qty=QTY_RANGE[1])
    assert selected.metrics["beta"] == pytest.approx(expected[1])
    assert selected.assumptions["benchmark"].startswith("the Series given as benchmark=")
