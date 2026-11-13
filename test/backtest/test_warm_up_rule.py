"""The warm-up rule (decision D25, `_warm_up`) as a specification: the warm-up of an
indicator is a function of the indicator call alone, so no decision depends on a later
bar; a run with too few bars waits through them; history meets the need of every
series, in its own interval; and the wait is counted per symbol and told on screen.

The bars are a seeded walk of NSE sessions (`walk`): each close is the previous one
times e^(0.01 z), z a standard normal draw, from Rs 100.00; the open is the previous
close, the high and low 0.5 percent either side. The strategies trade 1 share, delivery
(CNC), without charges or slippage. The warm-ups used are those of the engine's
reference bars: 64 bars for a 10-bar EMA, 512 for a 50-bar one, 128 for a 20-bar one,
32 for a 20-bar simple average and 16 for a 10-bar one.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.errors import WarmUpWarning

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CALENDAR = bt.calendar.default_calendar()
SETTINGS = {"capital": 1_000_000, "costs": None, "slippage": 0}
PRICES = ("open", "high", "low", "close")


def sessions(first: str, last: str) -> list[dt.date]:
    return list(CALENDAR.sessions("NSE", first, last)["date"])


def walk(first: str, last: str, seed: int = 7) -> pd.DataFrame:
    """Daily bars of the module docstring over the NSE sessions from `first` to `last`."""
    days = sessions(first, last)
    steps = np.random.default_rng(seed).normal(0.0, 0.01, len(days))
    closes = 100.0 * np.exp(np.cumsum(steps))
    opens = np.concatenate(([100.0], closes[:-1]))
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    frame = {
        "open": opens,
        "high": np.maximum(opens, closes) * 1.005,
        "low": np.minimum(opens, closes) * 0.995,
        "close": closes,
        "volume": np.full(len(days), 1_000.0),
    }
    return pd.DataFrame(frame, index=index)


def above_average(length: int, interval: str = "D") -> type[bt.Strategy]:
    """Long while the close is above the EMA of `length` bars of `interval`."""

    class AboveAverage(bt.Strategy):
        symbol, exchange, product = "SBIN", "NSE", "CNC"

        def init(self):
            series = self.close if interval == "D" else self.data["SBIN", interval].close
            self.line = self.I(ta.ema, series, length)

        def on_bar(self, bar):
            if self.position.is_flat and self.close[-1] > self.line[-1]:
                self.buy(qty=1)
            elif self.position.is_long and self.close[-1] < self.line[-1]:
                self.exit()

    AboveAverage.interval = "D"
    return AboveAverage


def decisions(result: bt.Result, before: pd.Timestamp) -> list[tuple]:
    """(decision instant, side, units) of every order decided before `before`."""
    orders = result.orders
    early = orders[orders["decided_at"] < before]
    return list(zip(early["decided_at"], early["side"], early["quantity_units"], strict=True))


def warm_up(result: bt.Result) -> dict:
    return result.manifest["arguments"]["warm_up"]


# ----- no decision depends on a later bar ----------------------------------------------

CALM = walk("2021-01-01", "2023-06-30")


TREND_BARS = 100


@settings(max_examples=10, deadline=None)
@given(cut=st.integers(70, len(CALM) - 2), rate=st.sampled_from([-0.03, -0.01, 0.01, 0.03]))
def test_no_decision_moves_when_the_bars_after_it_change(cut, rate):
    # From bar `cut` on every price is moved by (1 + rate) more each bar, for up to
    # TREND_BARS bars: a crash or a rally of 1 or 3 percent a bar, with no one-day gap a
    # split check would stop. The decisions made at the closes before bar `cut` are the
    # same, and so is the first decision of the 10-bar EMA strategy (its 64th bar).
    moved = CALM.copy()
    steps = np.minimum(np.arange(1, len(CALM) - cut + 1), TREND_BARS)
    columns = [moved.columns.get_loc(name) for name in PRICES]
    moved.iloc[cut:, columns] = (
        moved.iloc[cut:, columns].to_numpy() * ((1 + rate) ** steps)[:, None]
    )
    strategy = above_average(10)
    calm, changed = (
        bt.backtest(strategy, data=CALM, **SETTINGS),
        bt.backtest(strategy, data=moved, **SETTINGS),
    )
    before = CALM.index[cut]
    assert decisions(changed, before) == decisions(calm, before)
    assert warm_up(changed)["first_decision"] == warm_up(calm)["first_decision"]


def test_a_different_end_leaves_the_early_decisions_unchanged():
    strategy = above_average(10)
    short = bt.backtest(strategy, data=CALM, end=CALM.index[300].date(), **SETTINGS)
    whole = bt.backtest(strategy, data=CALM, **SETTINGS)
    before = CALM.index[300]
    assert decisions(short, before) == decisions(whole, before)
    assert len(decisions(whole, before)) > 2


def test_the_need_does_not_depend_on_how_many_bars_the_run_has():
    strategy = above_average(50)
    shorter = bt.backtest(strategy, data=CALM.iloc[:391], **SETTINGS)
    longer = bt.backtest(strategy, data=CALM.iloc[:618], **SETTINGS)
    assert warm_up(shorter)["needs"] == warm_up(longer)["needs"]
    assert warm_up(longer)["needs"][0]["bars"] == 512


# ----- a run with too few bars waits through them --------------------------------------


def test_a_run_with_fewer_bars_than_the_need_makes_no_decision_and_says_so():
    data = CALM.iloc[:391]
    result = bt.backtest(above_average(50), data=data, **SETTINGS)
    first = data.index[0].date()
    assert result.orders.empty
    assert warm_up(result)["first_decision"] == "never"
    assert result.assumptions["warm_up"].endswith("no decision: the indicators were never ready")
    (warning,) = [text for text in result.warnings if "no decision" in text]
    assert warning == (
        "The strategy made no decision: its indicators were never ready within the run's "
        "bars (SBIN waited 391 strategy bars). ema(SBIN.close, 50) needs 512 D bars of SBIN "
        f"and 0 bars closed before {first}: the bars given for SBIN begin on {first}; give "
        f"earlier ones to decide from {first}."
    )


def test_a_run_with_more_bars_than_the_need_decides_at_the_close_of_the_needed_bar():
    result = bt.backtest(above_average(50), data=CALM.iloc[:618], **SETTINGS)
    ready = CALM.index[511].date()
    assert warm_up(result)["first_decision"] == f"{ready} 15:30 IST"
    assert warm_up(result)["waited"] == {"SBIN": 511}


# ----- history meets every series' need, in its own interval ---------------------------

SINCE_2004 = walk("2004-01-01", "2026-06-30", seed=1)


def closed_bars_at_the_first_decision(strategy: type, interval: str, **arguments) -> int:
    seen = []

    class Counting(strategy):
        def on_bar(self, bar):
            if not seen:
                seen.append(len(self.data["SBIN", interval].close))
            super().on_bar(bar)

    bt.backtest(Counting, data=SINCE_2004, **arguments, **SETTINGS)
    return seen[0]


@pytest.mark.parametrize(
    ("interval", "length", "need"), [("W", 50, 512), ("M", 20, 128), ("D", 200, 2048)]
)
def test_a_longer_interval_gets_its_need_from_history(interval, length, need):
    strategy = above_average(length, interval)
    result = bt.backtest(strategy, data=SINCE_2004, start="2024-01-01", **SETTINGS)
    text = result.assumptions["warm_up"]
    series = "SBIN" if interval == "D" else f"SBIN[{interval}]"
    assert f"ema({series}.close, {length}) needs {need} {interval} bars of SBIN" in text
    assert text.endswith("ready at start")
    assert closed_bars_at_the_first_decision(strategy, interval, start="2024-01-01") >= need


def test_the_same_indicator_call_needs_the_same_bars_on_daily_and_on_15_minute_bars():
    class Intraday(above_average(50)):
        interval, product = "15m", "MIS"

    stamps = [
        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=15 * k)
        for day in sessions("2026-01-01", "2026-01-31")
        for k in range(25)
    ]
    closes = 100.0 + np.sin(np.arange(len(stamps)) / 9.0)
    minutes = pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes,
         "volume": 1_000.0},
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )  # fmt: skip
    daily = bt.backtest(above_average(50), data=CALM, **SETTINGS)
    fifteen = bt.backtest(Intraday, data=minutes, **SETTINGS)
    assert warm_up(daily)["needs"][0]["bars"] == warm_up(fifteen)["needs"][0]["bars"] == 512
    assert "ema(SBIN.close, 50) needs 512 15m bars of SBIN" in fifteen.assumptions["warm_up"]


def test_history_ends_at_the_last_bar_before_a_datetime_start():
    # 15-minute bars; the run starts at 11:00 on 2026-01-05: the bars of that morning,
    # 09:15 to 10:45, are history and are seen before the first bar of the run.
    stamps = [
        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=15 * k)
        for day in sessions("2025-12-01", "2026-01-09")
        for k in range(25)
    ]
    closes = 100.0 + np.cos(np.arange(len(stamps)) / 5.0)
    bars = pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes,
         "volume": 1_000.0},
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )  # fmt: skip
    seen = []

    class Probe(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"

        def init(self):
            self.line = self.I(ta.sma, self.close, 3)

        def on_bar(self, bar):
            if not seen:
                seen.extend(self.history(None, 8).index)

    start = dt.datetime(2026, 1, 5, 11, 0, tzinfo=IST)
    bt.backtest(Probe, data=bars, start=start, end="2026-01-09", **SETTINGS)
    expected = [start + dt.timedelta(minutes=15 * k) for k in range(-7, 1)]
    assert list(seen) == expected


def test_a_run_on_one_explicit_contract_keeps_the_contracts_own_bars_before_start():
    days = list(CALENDAR.sessions("NFO", "2026-08-03", "2026-11-12")["date"])
    closes = 25_000.0 + 10.0 * np.arange(len(days))
    frame = pd.DataFrame(
        {"open": closes, "high": closes + 20.0, "low": closes - 20.0, "close": closes,
         "volume": 1_000.0},
        index=pd.DatetimeIndex(
            [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days]
        ),
    )  # fmt: skip

    class Contract(bt.Strategy):
        symbol, exchange, interval, product = "NIFTY23NOV26FUT", "NFO", "D", "NRML"

        def init(self):
            self.line = self.I(ta.sma, self.close, 10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the reference tables' look-ahead notice
        result = bt.backtest(Contract, data=frame, start="2026-10-01", now="2026-11-13", **SETTINGS)
    before = [day for day in days if day < dt.date(2026, 10, 1)]
    assert (
        f"history before start: NIFTY23NOV26FUT 16 strategy bars from {before[-16]} to {before[-1]}"
    ) in result.assumptions["warm_up"]
    assert result.assumptions["warm_up"].endswith("ready at start")


# ----- the wait is counted per symbol ---------------------------------------------------


def pair(order: tuple[str, str]) -> type[bt.Strategy]:
    class Pair(bt.Strategy):
        symbols, exchange, product, interval = order, "NSE", "CNC", "D"

        def init(self):
            self.lines = {s: self.I(ta.sma, self.data[s].close, 20) for s in self.symbols}

    return Pair


@pytest.mark.parametrize("order", [("AAA", "BBB"), ("BBB", "AAA")])
def test_each_symbol_counts_the_strategy_bars_it_waited_whatever_the_order(order):
    # BBB lists on 2026-02-02, within the run; its 20-bar average needs 32 bars, so the
    # strategy decides from the close of BBB's 32nd session. AAA has years of history.
    data = {"AAA": walk("2020-01-01", "2026-06-30", 3), "BBB": walk("2026-02-02", "2026-06-30", 4)}
    result = bt.backtest(pair(order), data=data, start="2026-01-01", **SETTINGS)
    listed = sessions("2026-02-02", "2026-06-30")
    ready = listed[31]
    waited_aaa = len(sessions("2026-01-01", (ready - dt.timedelta(days=1)).isoformat()))
    assert warm_up(result)["waited"] == {"AAA": waited_aaa, "BBB": 31}
    assert warm_up(result)["first_decision"] == f"{ready} 15:30 IST"


def test_own_data_that_lists_late_is_advised_to_begin_earlier():
    data = {"AAA": walk("2020-01-01", "2026-06-30", 3), "BBB": walk("2026-02-02", "2026-06-30", 4)}
    result = bt.backtest(pair(("AAA", "BBB")), data=data, start="2026-01-01", **SETTINGS)
    (warning,) = [text for text in result.warnings if "waited" in text]
    assert warning.endswith(
        "sma(BBB.close, 20) needs 32 D bars of BBB and 0 bars closed before 2026-01-01: the "
        "bars given for BBB begin on 2026-02-02; give earlier ones to decide from 2026-01-01."
    )


def test_a_client_source_without_bars_before_start_says_the_source_has_none(tmp_path):
    from data_fixtures.fakes import FakeClient, historify_shape

    bars = walk("2026-02-02", "2026-06-30", 4)
    epoch = pd.Timestamp("1970-01-01")
    rows = historify_shape(
        [
            {
                "timestamp": (stamp.tz_localize(None).normalize() - epoch).days * 86_400,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": int(row.volume),
            }
            for stamp, row in bars.iterrows()
        ]
    )
    client = FakeClient(db={("BBB", "NSE", "D"): rows})

    class Listed(bt.Strategy):
        symbol, exchange, product, interval = "BBB", "NSE", "CNC", "D"

        def init(self):
            self.line = self.I(ta.sma, self.close, 20)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPENALGO_BT_CACHE", str(tmp_path))
        result = bt.backtest(
            Listed, client, start="2026-01-01", end="2026-06-30", source="db", **SETTINGS
        )
    (warning,) = [text for text in result.warnings if "waited" in text]
    assert warning.endswith(
        "sma(BBB.close, 20) needs 32 D bars of BBB and 0 bars closed before 2026-01-01: the "
        "source has no bars of BBB before 2026-02-02."
    )


# ----- the wait is told on screen ------------------------------------------------------


def test_a_backtest_that_waited_ends_with_one_warm_up_warning():
    data = CALM.iloc[:200]
    with pytest.warns(WarmUpWarning) as caught:
        bt.backtest(above_average(10), data=data, **SETTINGS)
    (warning,) = [w for w in caught if w.category is WarmUpWarning]
    ready = data.index[63].date()
    assert str(warning.message) == (
        "The strategy waited inside the run for its indicators to warm up and made its "
        f"first decision at {ready} 15:30 IST (SBIN waited 63 strategy bars): signals before "
        "then were not taken. See result.warnings for why."
    )
    assert warning.filename == __file__


def test_a_backtest_that_never_decided_says_so_on_screen():
    with pytest.warns(WarmUpWarning) as caught:
        bt.backtest(above_average(50), data=CALM.iloc[:391], **SETTINGS)
    (warning,) = [w for w in caught if w.category is WarmUpWarning]
    assert str(warning.message) == (
        "The strategy made no decision: its indicators did not finish their warm-up within "
        "the run's bars (SBIN waited 391 strategy bars). See result.warnings for why."
    )


def test_a_backtest_ready_at_start_issues_no_warm_up_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarmUpWarning)
        bt.backtest(above_average(10), data=CALM, start=CALM.index[100].date(), **SETTINGS)


def test_a_sweep_issues_no_warm_up_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt.optimize(
            Crossing, data=CALM.iloc[:200], workers=1, progress=lambda line: None,
            fast=(5, 10), **SETTINGS,
        )  # fmt: skip
    assert not [w for w in caught if w.category is WarmUpWarning]


# ----- a sweep keeps what each combination needs ---------------------------------------


class Crossing(bt.Strategy):
    """Long while the fast EMA is above the slow one."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    fast, slow = 10, 20

    def init(self):
        self.fast_line = self.I(ta.ema, self.close, self.fast)
        self.slow_line = self.I(ta.ema, self.close, self.slow)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast_line[-1] > self.slow_line[-1]:
            self.buy(qty=1)
        elif self.position.is_long and self.fast_line[-1] < self.slow_line[-1]:
            self.exit()


def test_every_combination_keeps_the_history_its_single_run_loads():
    # slow=20 needs 128 bars and slow=50 needs 512: the sweep loads the larger once and
    # each combination keeps its own, bar for bar as its single run.
    start = SINCE_2004.index[-300].date()
    swept = bt.optimize(
        Crossing, data=SINCE_2004, start=start, workers=1, progress=lambda line: None,
        slow=(20, 50), **SETTINGS,
    )  # fmt: skip
    for slow in (20, 50):
        params = {"slow": slow}
        alone = bt.backtest(Crossing, data=SINCE_2004, start=start, params=params, **SETTINGS)
        chosen = swept.select(slow=slow)
        assert chosen.manifest["data"] == alone.manifest["data"]
        assert chosen.assumptions["warm_up"] == alone.assumptions["warm_up"]
        assert chosen.trades.equals(alone.trades)


def test_a_sweeps_init_runs_on_the_reference_bars_before_any_data_is_loaded():
    from data_fixtures.fakes import FakeClient

    class Refused(Crossing):
        def init(self):
            self.I(len, self.close)

    client = FakeClient(db={})
    with pytest.raises(bt.errors.ConfigError, match="self.I takes a function of openalgo.ta"):
        bt.optimize(Refused, client, start="2026-01-01", end="2026-02-01", fast=(5, 10), **SETTINGS)
    assert client.history_calls() == []


def test_an_order_in_a_sweeps_init_is_refused_as_in_a_run():
    class OrdersInInit(Crossing):
        def init(self):
            super().init()
            self.buy(qty=1)

    with pytest.raises(bt.errors.BacktestError, match="paused at a decision point"):
        bt.optimize(OrdersInInit, data=CALM.iloc[:200], workers=1, fast=(5, 10), **SETTINGS)


# ----- what the engine cannot measure ---------------------------------------------------


def test_an_indicator_over_an_array_of_your_own_is_not_waited_for_and_the_result_says_so():
    data = CALM.iloc[:120]
    volumes = np.full(len(data), 1_000.0)

    class OwnVolume(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            self.balance = self.I(ta.obv, self.close, volumes)

    result = bt.backtest(OwnVolume, data=data, **SETTINGS)
    (warning,) = [text for text in result.warnings if "values of your own" in text]
    assert warning.startswith("obv(SBIN.close, array(")
    assert warning.endswith(
        "reads values of your own (an array, or a view the reference bars cannot "
        "reproduce), so its warm-up cannot be measured: the strategy did not wait for it."
    )
    assert warm_up(result)["first_decision"] is None


def test_a_from_signals_function_is_computed_over_the_history_of_a_200_bar_ema():
    # The function sees the run's bars in init, then the history and the run's bars
    # together; the strategy does not wait (it has no self.I indicator).
    firsts = []

    def signals(frame):
        firsts.append(frame.index[0].date())
        fast, slow = ta.ema(frame.close, 10), ta.ema(frame.close, 30)
        return ta.crossover(fast, slow), ta.crossunder(fast, slow)

    start = SINCE_2004.index[-300].date()
    strategy = bt.from_signals(signals, "SBIN", "NSE", "D", qty=1, product="CNC")
    result = bt.backtest(strategy, data=SINCE_2004, start=start, **SETTINGS)
    assert firsts[0] == start and min(firsts) < start
    assert "as ema(close, 200) needs 2048 D bars of SBIN" not in result.assumptions["warm_up"]
    assert warm_up(result)["first_decision"] is None
