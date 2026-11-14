"""While a symbol is not ready (decision D25, `_warm_up`), its indicators read NaN, or
false for a condition, wherever they are read: in `on_bar` deciding for another symbol,
through another interval of it or through a futures series. Its bar fields stay
readable.

Why: an indicator is computed over the whole series, so a value shown before its symbol
is ready can depend on how many bars follow. `ta.ema(close, 20)` is not computed on
fewer than 20 bars (every value unknown) and is computed from the first bar on 20 or
more, so on a series of 25 bars its value at the 5th bar exists only when the run ends
after the 20th. Read before the symbol is ready, it would let a later `end` change an
earlier decision. The values themselves are kept: once the symbol is ready its lookbacks
(`line[-2]`) read them.

The bars are the seeded walks of `warm_up_bars`. A 20-bar simple average needs 32 bars of
its series on the engine's reference bars, a 3-bar or 5-bar one 16, and an EMA of 20 and
50 bars 256 and 512 (pinned in test_warm_up_reference.py).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from warm_up_bars import IST, SETTINGS, sessions, walk

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.reference import Instrument

END = "2026-04-30"
# Years of bars: ready at start for an EMA of 20 bars.
LONG = walk("2024-06-03", END, seed=3)
# 25 bars from start: an EMA of 20 bars is computed on them but never ready (256 bars).
SHORT = walk("2026-01-01", END, seed=4).iloc[:25]
# Listed within the run: a 20-bar average is ready at the close of its 32nd session.
LISTED = walk("2026-02-02", END, seed=1)
LISTED_READY = sessions("2026-02-02", END)[31]

pytestmark = pytest.mark.filterwarnings(
    "ignore::openalgo.backtest.errors.WarmUpWarning",
    "ignore::openalgo.backtest.errors.OrdersNotPlacedWarning",
)


def unknown(value: float) -> float | None:
    """A read value for comparison: None for NaN."""
    return None if math.isnan(value) else value


def up_to(strategy: type, result: bt.Result, t: pd.Timestamp) -> dict[str, list]:
    """The decisions recorded in `seen` and the orders decided at or before `t`, and the
    fills before it."""
    orders = result.orders[result.orders["decided_at"] <= t]
    fills = result.fills[result.fills["time"] < t]
    return {
        "decisions": [entry for entry in strategy.seen if entry[0] <= t],
        "orders": list(
            orders[["decided_at", "symbol", "side", "quantity_units"]].itertuples(
                index=False, name=None
            )
        ),
        "fills": list(
            fills[["time", "symbol", "side", "quantity_units"]].itertuples(index=False, name=None)
        ),
    }


# ----- an indicator of a symbol never ready, read for another --------------------------


def strength_pair() -> type[bt.Strategy]:
    """Holds AAA while BBB's 20-bar EMA has a value (a relative-strength filter); each
    decision records its instant, the symbols given and BBB's EMA."""

    class Pair(bt.Strategy):
        symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "D", "CNC"
        seen: list = []

        def init(self):
            self.lines = {s: self.I(ta.ema, self.data[s].close, 20) for s in self.symbols}

        def on_bar(self, bars):
            line = self.lines["BBB"][-1]
            type(self).seen.append((self.now, tuple(sorted(bars)), unknown(line)))
            if not math.isnan(line) and self.positions["AAA"].is_flat:
                self.buy(symbol="AAA", qty=1)

    return Pair


def strength_run(end: str) -> tuple[type, bt.Result]:
    strategy = strength_pair()
    data = {"AAA": LONG, "BBB": SHORT}
    return strategy, bt.backtest(strategy, data=data, start="2026-01-01", end=end, **SETTINGS)


def test_the_ema_of_a_symbol_never_ready_reads_nan_when_deciding_for_another():
    strategy, result = strength_run(END)
    assert result.manifest["arguments"]["warm_up"]["first_decisions"] == {
        "AAA": None,
        "BBB": "never",
    }
    assert {symbols for _, symbols, _ in strategy.seen} == {("AAA",)}
    assert {line for _, _, line in strategy.seen} == {None}
    assert result.orders.empty


def test_a_decision_on_another_symbols_ema_is_the_same_for_a_later_end():
    # Cut on 2026-01-14 BBB has 9 bars, too few to compute its EMA; on 2026-04-30 it has
    # 25 and its EMA has values from its first bar.
    cut = pd.Timestamp("2026-01-14 15:30", tz=IST)
    whole_strategy, whole = strength_run(END)
    cut_strategy, cut_result = strength_run("2026-01-14")
    assert up_to(whole_strategy, whole, cut) == up_to(cut_strategy, cut_result, cut)


# ----- a regime filter on a symbol it does not trade ------------------------------------

FILTER_DATA = {
    "SBIN": walk("2025-01-01", "2026-06-30", seed=2),
    "NIFTYBEES": walk("2025-01-01", "2026-06-30", seed=5),
}


def regime_filter() -> type[bt.Strategy]:
    """Holds SBIN while NIFTYBEES closes above its 50-day EMA. SBIN has no indicator of
    its own, so it is ready at start; NIFTYBEES has fewer than 512 bars, so it is never
    ready."""

    class Filter(bt.Strategy):
        symbols, exchange, interval, product = ("SBIN", "NIFTYBEES"), "NSE", "D", "CNC"
        seen: list = []

        def init(self):
            self.trend = self.I(ta.ema, self.data["NIFTYBEES"].close, 50)

        def on_bar(self, bars):
            line = self.trend[-1]
            type(self).seen.append((self.now, tuple(sorted(bars)), unknown(line)))
            up = self.data["NIFTYBEES"].close[-1] > line
            if up and self.positions["SBIN"].is_flat:
                self.buy(symbol="SBIN", qty=10)
            elif not up and self.positions["SBIN"].is_long:
                self.exit(symbol="SBIN")

    return Filter


def filter_run(end: str) -> tuple[type, bt.Result]:
    strategy = regime_filter()
    return strategy, bt.backtest(
        strategy, data=FILTER_DATA, start="2025-01-01", end=end, **SETTINGS
    )


def test_a_regime_filter_places_no_order_while_its_filter_is_not_ready():
    strategy, result = filter_run("2026-06-30")
    assert result.manifest["arguments"]["warm_up"]["first_decisions"] == {
        "SBIN": None,
        "NIFTYBEES": "never",
    }
    assert {line for _, _, line in strategy.seen} == {None}
    assert result.orders.empty


def test_a_regime_filter_decides_the_same_up_to_a_day_for_any_end():
    # Cut on 2025-03-03 NIFTYBEES has 40 bars, too few to compute a 50-bar EMA.
    cut = pd.Timestamp("2025-03-03 15:30", tz=IST)
    whole_strategy, whole = filter_run("2026-06-30")
    cut_strategy, cut_result = filter_run("2025-03-03")
    assert up_to(whole_strategy, whole, cut) == up_to(cut_strategy, cut_result, cut)


# ----- what a symbol shows while it waits, and once it is ready -------------------------


def reader() -> type[bt.Strategy]:
    """AAA (ready at start) and BBB (listed within the run, a 20-bar average): each
    decision records what it reads of BBB."""

    class Reader(bt.Strategy):
        symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "D", "CNC"
        seen: list = []

        def init(self):
            data = self.data["BBB"]
            self.fast = self.I(ta.sma, data.close, 5)
            self.slow = self.I(ta.sma, data.close, 20)
            # BBB's fast average is below its slow one at every bar from the 20th to its
            # 32nd, and its close crosses above the slow average at its 23rd bar.
            self.below = self.fast < self.slow
            self.gap = self.slow - data.close
            self.crossed = data.close.crossed_above(self.slow)
            self.aaa = self.I(ta.sma, self.data["AAA"].close, 5)

        def on_bar(self, bars):
            data = self.data["BBB"]
            read = {
                "symbols": tuple(sorted(bars)),
                "bars": len(data.close),
                "close": data.close[-1] if len(data.close) else None,
                "high": data.high[-1] if len(data.high) else None,
                "slow": [unknown(value) for value in self.slow[-3:]],
                "slow_bars": len(self.slow),
                "gap": unknown(self.gap[-1]) if len(self.gap) else None,
                "below": self.below[-1] if len(self.below) else None,
                "below_truth": bool(self.below),
                "crossed": bool(self.crossed),
                "aaa": unknown(self.aaa[-1]),
            }
            type(self).seen.append((self.now, read))

    return Reader


@pytest.fixture(scope="module")
def read():
    """(the decisions while BBB waits, the decision at which it joins), each the dict of
    what was read; BBB has bars from 2026-02-02."""
    strategy = reader()
    bt.backtest(
        strategy, data={"AAA": LONG, "BBB": LISTED}, start="2026-01-01", end=END, **SETTINGS
    )
    waiting = [(now, entry) for now, entry in strategy.seen if now.date() < LISTED_READY]
    joined = [entry for now, entry in strategy.seen if now.date() == LISTED_READY]
    return waiting, joined[0]


def test_a_waiting_symbols_bar_fields_stay_readable(read):
    waiting, _ = read
    listed = [(now, entry) for now, entry in waiting if entry["bars"]]
    assert len(listed) == 31
    for now, entry in listed:
        bar = LISTED.loc[LISTED.index.date == now.date()].iloc[0]
        assert (entry["close"], entry["high"]) == (bar["close"], bar["high"])
        assert entry["bars"] == entry["slow_bars"]


def test_a_waiting_symbols_indicators_and_the_views_built_on_them_read_nan(read):
    waiting, _ = read
    assert {tuple(entry["slow"]) for _, entry in waiting if entry["bars"] >= 3} == {
        (None, None, None)
    }
    assert {entry["gap"] for _, entry in waiting} == {None}


def test_a_waiting_symbols_conditions_read_false(read):
    waiting, _ = read
    assert {entry["below"] for _, entry in waiting if entry["bars"]} == {False}
    assert {entry["below_truth"] for _, entry in waiting} == {False}
    assert {entry["crossed"] for _, entry in waiting} == {False}


def test_the_symbol_that_is_ready_reads_its_own_indicators_meanwhile(read):
    waiting, _ = read
    assert all(entry["aaa"] is not None for _, entry in waiting)
    assert {entry["symbols"] for _, entry in waiting} == {("AAA",)}


def test_once_ready_a_symbols_lookbacks_read_the_values_before_it_joined(read):
    _, joined = read
    closes = LISTED["close"].to_numpy()
    # BBB joins at its 32nd bar (index 31): the averages of bars 10 to 29, 11 to 30 and
    # 12 to 31.
    expected = [float(np.mean(closes[k - 19 : k + 1])) for k in (29, 30, 31)]
    assert joined["symbols"] == ("AAA", "BBB")
    assert joined["slow"] == pytest.approx(expected, rel=1e-12)
    assert joined["gap"] == pytest.approx(expected[-1] - closes[31], rel=1e-12)


# ----- another interval of a waiting symbol ---------------------------------------------

# BBB's only indicator is a 5-week average of its weekly bars, which needs 16 weeks: BBB
# waits until its 16th weekly bar has closed, and its average has values from the 5th.
WEEKLY_DATA = {"AAA": LONG, "BBB": walk("2026-01-01", "2026-06-30", seed=4)}


def weekly_reader() -> type[bt.Strategy]:
    class Weekly(bt.Strategy):
        symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "D", "CNC"
        seen: list = []

        def init(self):
            self.weekly = self.data["BBB", "W"]
            self.line = self.I(ta.sma, self.weekly.close, 5)

        def on_bar(self, bars):
            weeks = len(self.weekly.close)
            line = unknown(self.line[-1]) if weeks else None
            type(self).seen.append((self.now, tuple(sorted(bars)), weeks, line))

    return Weekly


def test_a_waiting_symbols_indicator_of_another_interval_reads_nan():
    strategy = weekly_reader()
    bt.backtest(strategy, data=WEEKLY_DATA, start="2026-01-01", end="2026-06-30", **SETTINGS)
    waiting = [entry for entry in strategy.seen if "BBB" not in entry[1]]
    joined = [entry for entry in strategy.seen if "BBB" in entry[1]]
    assert {weeks for _, _, weeks, _ in waiting} == set(range(16))
    assert {line for _, _, _, line in waiting} == {None}
    assert joined[0][2] == 16 and joined[0][3] is not None


# ----- a futures series not ready, read for another underlying ---------------------------

CALENDAR = bt.calendar.default_calendar()
FUTURES_START, FUTURES_END = "2024-08-01", "2024-10-24"


def futures_contract(symbol: str, last_day: dt.date, underlying: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="NFO",
        kind="future",
        lot_size=25,
        tick_size=0.05,
        expiry=last_day,
        underlying=underlying,
        settlement="cash",
    )


def futures_bars(last: str, seed: int, base: float) -> pd.DataFrame:
    """Daily bars of a contract from FUTURES_START to `last`: a seeded walk from `base`."""
    days = list(CALENDAR.sessions("NFO", FUTURES_START, last)["date"])
    closes = base * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(days))))
    opens = np.concatenate(([base], closes[:-1]))
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    frame = {
        "open": opens,
        "high": np.maximum(opens, closes) * 1.003,
        "low": np.minimum(opens, closes) * 0.997,
        "close": closes,
        "volume": 1_000.0,
    }
    return pd.DataFrame(frame, index=index)


CONTRACTS = {
    "NIFTY26SEP24FUT": (dt.date(2024, 9, 26), "NIFTY", 1, 25_000.0),
    "NIFTY31OCT24FUT": (dt.date(2024, 10, 31), "NIFTY", 2, 25_100.0),
    "BANKNIFTY25SEP24FUT": (dt.date(2024, 9, 25), "BANKNIFTY", 3, 51_000.0),
    "BANKNIFTY30OCT24FUT": (dt.date(2024, 10, 30), "BANKNIFTY", 4, 51_200.0),
}


def futures_pair() -> type[bt.Strategy]:
    """NIFTY with a 3-bar average (ready at its 16th session) and BANKNIFTY with a 20-bar
    one (its 32nd): each decision records BANKNIFTY's average."""

    class Pair(bt.Strategy):
        symbols, exchange, interval = ("NIFTY", "BANKNIFTY"), "NFO", "D"
        seen: list = []

        def init(self):
            self.nifty = self.I(ta.sma, self.data["NIFTY"].close, 3)
            self.bank = self.I(ta.sma, self.data["BANKNIFTY"].close, 20)

        def on_bar(self, bars):
            type(self).seen.append((self.now.date(), tuple(sorted(bars)), unknown(self.bank[-1])))

    return Pair


def test_a_futures_series_not_ready_reads_nan_when_deciding_for_another_underlying():
    strategy = futures_pair()
    instruments = {
        symbol: futures_contract(symbol, last_day, underlying)
        for symbol, (last_day, underlying, _, _) in CONTRACTS.items()
    }
    data = {
        symbol: futures_bars(str(last_day), seed, base)
        for symbol, (last_day, _, seed, base) in CONTRACTS.items()
    }
    bt.backtest(
        strategy,
        data=data,
        instrument=instruments,
        start=FUTURES_START,
        end=FUTURES_END,
        roll=1,
        now="2024-12-31",
        **SETTINGS,
    )
    days = list(CALENDAR.sessions("NFO", FUTURES_START, FUTURES_END)["date"])
    waiting = [entry for entry in strategy.seen if entry[0] < days[31]]
    assert waiting[0][0] == days[15]
    assert {symbols for _, symbols, _ in waiting} == {("NIFTY",)}
    assert {line for _, _, line in waiting} == {None}
    joined = [entry for entry in strategy.seen if entry[0] == days[31]]
    assert joined[0][1] == ("BANKNIFTY", "NIFTY") and joined[0][2] is not None


# ----- conditions read what on_bar reads ------------------------------------------------


def crossing_on_bar() -> type[bt.Strategy]:
    class OnBar(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            self.fast = self.I(ta.sma, self.close, 5)
            self.slow = self.I(ta.sma, self.close, 20)

        def on_bar(self, bar):
            if self.position.is_flat and self.fast[-1] > self.slow[-1]:
                self.buy(qty=1)
            elif self.position.is_long and self.fast[-1] < self.slow[-1]:
                self.exit()

    return OnBar


def crossing_conditions() -> type[bt.Strategy]:
    class Conditions(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            fast = self.I(ta.sma, self.close, 5)
            slow = self.I(ta.sma, self.close, 20)
            self.entry_when(fast > slow, qty=1)
            self.exit_when(fast < slow)

    return Conditions


def test_a_condition_strategy_that_waits_decides_as_its_on_bar_twin():
    data = walk("2026-01-01", END, seed=4)
    columns = ["decided_at", "side", "quantity_units"]
    orders = [
        list(
            bt.backtest(make(), data=data, start="2026-01-01", end=END, **SETTINGS)
            .orders[columns]
            .itertuples(index=False, name=None)
        )
        for make in (crossing_on_bar, crossing_conditions)
    ]
    assert orders[0] == orders[1]
    assert orders[0][0][0].date() >= sessions("2026-01-01", END)[31]
