"""Readiness is per symbol and causal (decision D25, `_warm_up`): at each strategy bar a
symbol is ready once its own indicators have closed their warm-up, counting only the
bars up to that bar; a symbol not ready yet gets no bar in `on_bar` and its orders are
skipped (skip reason "not_ready"), it joins at the first bar it is ready, and the
symbols that are ready trade.

The property: over a table of scenarios, cutting `end` earlier or changing every bar
after a time t never changes a decision, an order or a fill at or before t. A decision
is an `on_bar` call (its instant and the symbols it was given), an order is the
(decision instant, symbol, side, units) of every order placed or skipped, and a fill is
(time, symbol, side, units, price) for every fill before t (a fill at the open after t
takes a price after t).

The bars are the seeded walks of `warm_up_bars`. The strategies compare a 5-bar and a
20-bar simple average, whose warm-ups on the engine's reference bars are 16 and 32 bars,
so a symbol is ready at the close of the 32nd bar of its series, history counted.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from warm_up_bars import IST, SETTINGS, intraday, sessions, walk

import openalgo.backtest as bt
from openalgo import ta

PRICES = ["open", "high", "low", "close"]
# The bars after t move by 2 percent more each bar, up to this many bars: a rally or a
# fall with no one-bar gap the data checks would take for a split.
TREND_BARS = 100
RATES = (0.02, -0.02)
FAST, SLOW, READY_BARS = 5, 20, 32


def crossing(symbols: tuple[str, ...], interval: str, product: str) -> type[bt.Strategy]:
    """Long one share of each symbol it decides for while the fast average is above the
    slow one; asks for one share of every symbol it is not given, which is skipped while
    that symbol is not ready. Each decision is recorded in `seen`."""

    class Crossing(bt.Strategy):
        exchange = "NSE"
        seen: list[tuple[pd.Timestamp, tuple[str, ...]]] = []

        def init(self):
            self.fast = {s: self.I(ta.sma, self.data[s].close, FAST) for s in symbols}
            self.slow = {s: self.I(ta.sma, self.data[s].close, SLOW) for s in symbols}

        def on_bar(self, bars):
            deciding = sorted(bars) if isinstance(bars, dict) else list(symbols)
            type(self).seen.append((self.now, tuple(deciding)))
            for s in symbols:
                if s not in deciding:
                    self.target(qty=1, symbol=s)
                elif self.positions[s].is_flat and self.fast[s][-1] > self.slow[s][-1]:
                    self.buy(symbol=s, qty=1)
                elif self.positions[s].is_long and self.fast[s][-1] < self.slow[s][-1]:
                    self.exit(symbol=s)

    Crossing.interval, Crossing.product = interval, product
    if len(symbols) == 1:
        Crossing.symbol = symbols[0]
    else:
        Crossing.symbols = symbols
    return Crossing


def crossing_conditions(interval: str, product: str) -> type[bt.Strategy]:
    """The single-symbol crossing as conditions."""

    class Conditions(bt.Strategy):
        symbol, exchange = "SBIN", "NSE"
        seen: list = []

        def init(self):
            fast = self.I(ta.sma, self.close, FAST)
            slow = self.I(ta.sma, self.close, SLOW)
            self.entry_when(fast > slow, qty=1)
            self.exit_when(fast < slow)

    Conditions.interval, Conditions.product = interval, product
    return Conditions


def sparse(bars: pd.DataFrame, share: float, seed: int) -> pd.DataFrame:
    """About `share` of `bars`, at bars drawn by a seeded generator (a thin share)."""
    return bars[np.random.default_rng(seed).random(len(bars)) < share]


@dataclass(frozen=True)
class Scenario:
    """A run: its strategy (made afresh for each run), its bars by symbol, its `start`
    and `end`, and the instants t at which it is cut (day closes) or changed after."""

    make: object
    data: dict[str, pd.DataFrame]
    start: str
    end: str
    cuts: tuple[pd.Timestamp, ...]
    changes: tuple[pd.Timestamp, ...] = ()

    def run(self, data: dict[str, pd.DataFrame] | None = None, end: str | None = None):
        strategy = self.make()
        frames = self.data if data is None else data
        given = frames if len(frames) > 1 else next(iter(frames.values()))
        result = bt.backtest(
            strategy, data=given, start=self.start, end=end or self.end, **SETTINGS
        )
        return strategy, result


def day_close(day: str) -> pd.Timestamp:
    """15:30 IST on `day`: the close of its last bar."""
    return pd.Timestamp(f"{day} 15:30", tz=IST)


DAILY_CUTS = tuple(day_close(day) for day in ("2026-01-14", "2026-02-18", "2026-04-09"))
# For bars that begin on 2026-02-02: a run cut before then has no bars at all.
LATE_CUTS = tuple(day_close(day) for day in ("2026-02-18", "2026-03-17", "2026-04-09"))
LONG = walk("2025-06-02", "2026-06-30", seed=3)
FROM_START = walk("2026-01-01", "2026-06-30", seed=4)
LISTED_LATE = walk("2026-02-02", "2026-06-30", seed=5)
MINUTES = intraday("2026-01-01", "2026-01-30", minutes=15, seed=6)
EARLIER_MINUTES = intraday("2025-12-15", "2026-01-30", minutes=15, seed=7)
THIN = sparse(intraday("2025-12-01", "2026-01-30", minutes=5, seed=8), 0.3, 1)
INTRADAY_CUTS = tuple(day_close(day) for day in ("2026-01-02", "2026-01-06", "2026-01-15"))
INTRADAY_CHANGES = (pd.Timestamp("2026-01-05 12:00", tz=IST),)


def one(interval: str = "D", product: str = "CNC"):
    return lambda: crossing(("SBIN",), interval, product)


def two(interval: str = "D", product: str = "CNC"):
    return lambda: crossing(("AAA", "BBB"), interval, product)


SCENARIOS = {
    "one symbol, own data from start": Scenario(
        one(), {"SBIN": FROM_START}, "2026-01-01", "2026-06-30", DAILY_CUTS
    ),
    "one symbol, own data before start": Scenario(
        one(), {"SBIN": LONG}, "2026-01-01", "2026-06-30", DAILY_CUTS
    ),
    "one symbol, own data after start": Scenario(
        one(), {"SBIN": LISTED_LATE}, "2026-01-01", "2026-06-30", LATE_CUTS
    ),
    "two symbols, one with own data from start": Scenario(
        two(), {"AAA": LONG, "BBB": FROM_START}, "2026-01-01", "2026-06-30", DAILY_CUTS
    ),
    "two symbols, one listed after start": Scenario(
        two(), {"AAA": LONG, "BBB": LISTED_LATE}, "2026-01-01", "2026-06-30", LATE_CUTS
    ),
    "two symbols, one shorter than its need": Scenario(
        two(), {"AAA": LONG, "BBB": FROM_START.iloc[:25]}, "2026-01-01", "2026-06-30", DAILY_CUTS
    ),
    "two symbols, one too short to compute": Scenario(
        two(), {"AAA": LONG, "BBB": FROM_START.iloc[:12]}, "2026-01-01", "2026-06-30", DAILY_CUTS
    ),
    "intraday, own data from start": Scenario(
        one("15m", "MIS"),
        {"SBIN": MINUTES},
        "2026-01-01",
        "2026-01-30",
        INTRADAY_CUTS,
        INTRADAY_CHANGES,
    ),
    "sparse bars, history before start": Scenario(
        one("5m", "MIS"),
        {"SBIN": THIN},
        "2026-01-02",
        "2026-01-30",
        INTRADAY_CUTS,
        INTRADAY_CHANGES,
    ),
    "two symbols intraday, one sparse from start": Scenario(
        two("15m", "MIS"),
        {"AAA": EARLIER_MINUTES, "BBB": sparse(MINUTES, 0.3, 2)},
        "2026-01-01",
        "2026-01-30",
        INTRADAY_CUTS,
        INTRADAY_CHANGES,
    ),
    "conditions, own data from start": Scenario(
        lambda: crossing_conditions("D", "CNC"),
        {"SBIN": FROM_START},
        "2026-01-01",
        "2026-06-30",
        DAILY_CUTS,
    ),
    "conditions intraday, own data from start": Scenario(
        lambda: crossing_conditions("15m", "MIS"),
        {"SBIN": MINUTES},
        "2026-01-01",
        "2026-01-30",
        INTRADAY_CUTS,
        INTRADAY_CHANGES,
    ),
}


def moved_after(frame: pd.DataFrame, t: pd.Timestamp, rate: float) -> pd.DataFrame:
    """`frame` with every bar that opens at or after `t` moved by (1 + rate) more each bar,
    for up to TREND_BARS bars."""
    moved = frame.copy()
    later = moved.index >= t
    steps = np.minimum(np.arange(1, int(later.sum()) + 1), TREND_BARS)
    moved.loc[later, PRICES] = moved.loc[later, PRICES].to_numpy() * ((1 + rate) ** steps)[:, None]
    return moved


def up_to(strategy: type, result: bt.Result, t: pd.Timestamp) -> dict[str, list]:
    """The decisions, orders and skips at or before `t`, and the fills before it."""
    orders, skipped, fills = result.orders, result.skipped, result.fills
    orders = orders[orders["decided_at"] <= t]
    skipped = skipped[skipped["decided_at"] <= t]
    fills = fills[fills["time"] < t]
    return {
        "decisions": [entry for entry in strategy.seen if entry[0] <= t],
        "orders": list(
            zip(orders["decided_at"], orders["symbol"], orders["side"], orders["quantity_units"])
        ),
        "skipped": list(zip(skipped["decided_at"], skipped["symbol"], skipped["reason"])),
        "fills": list(
            zip(
                fills["time"],
                fills["symbol"],
                fills["side"],
                fills["quantity_units"],
                fills["price_rupees"],
            )
        ),
    }


@pytest.fixture(scope="module", params=list(SCENARIOS))
def whole(request):
    """(the scenario, its strategy class and result over its whole range)."""
    scenario = SCENARIOS[request.param]
    return (scenario, *scenario.run())


def test_every_scenario_decides_for_its_symbols(whole):
    scenario, strategy, result = whole
    assert len(result.orders) + len(result.skipped) > 0


def test_cutting_end_earlier_changes_nothing_at_or_before_the_cut(whole):
    scenario, strategy, result = whole
    for t in scenario.cuts:
        cut_strategy, cut = scenario.run(end=t.date().isoformat())
        assert up_to(cut_strategy, cut, t) == up_to(strategy, result, t), t


def test_changing_the_bars_after_t_changes_nothing_at_or_before_t(whole):
    scenario, strategy, result = whole
    for k, t in enumerate(scenario.cuts + scenario.changes):
        rate = RATES[k % len(RATES)]
        data = {symbol: moved_after(frame, t, rate) for symbol, frame in scenario.data.items()}
        changed_strategy, changed = scenario.run(data=data)
        assert up_to(changed_strategy, changed, t) == up_to(strategy, result, t), t


# ----- each symbol decides from its own readiness ----------------------------------------

LISTED = sessions("2026-02-02", "2026-06-30")
READY = LISTED[READY_BARS - 1]


def listed_late():
    strategy = crossing(("AAA", "BBB"), "D", "CNC")
    result = bt.backtest(
        strategy, data={"AAA": LONG, "BBB": LISTED_LATE}, start="2026-01-01", **SETTINGS
    )
    return strategy, result


def test_a_symbol_ready_at_start_decides_from_start_while_another_waits():
    strategy, result = listed_late()
    first_day, first_symbols = strategy.seen[0]
    assert (first_day.date(), first_symbols) == (dt.date(2026, 1, 1), ("AAA",))
    warm_up = result.manifest["arguments"]["warm_up"]
    assert warm_up["first_decision"] is None
    assert warm_up["waited"] == {"BBB": READY_BARS - 1}
    assert warm_up["first_decisions"] == {"AAA": None, "BBB": f"{READY} 15:30 IST"}


def test_a_symbol_joins_at_the_first_bar_it_is_ready():
    strategy, _ = listed_late()
    joined = [day for day, symbols in strategy.seen if "BBB" in symbols]
    assert joined[0].date() == READY
    assert all("BBB" in symbols for day, symbols in strategy.seen if day.date() >= READY)


def test_orders_on_a_symbol_not_ready_yet_are_skipped_until_it_is():
    _, result = listed_late()
    skipped = result.skipped
    assert set(skipped["reason"]) == {"not_ready"} and set(skipped["symbol"]) == {"BBB"}
    assert skipped["decided_at"].max().date() < READY
    bbb = result.orders[result.orders["symbol"] == "BBB"]
    assert bbb["decided_at"].min().date() >= READY


def test_a_symbol_that_waited_is_told_on_screen_with_its_first_decision():
    with pytest.warns(bt.errors.WarmUpWarning) as caught:
        listed_late()
    (warning,) = [w for w in caught if w.category is bt.errors.WarmUpWarning]
    assert str(warning.message) == (
        "The strategy waited inside the run for the indicators of BBB to warm up (BBB "
        f"waited {READY_BARS - 1} strategy bars and decided from {READY} 15:30 IST): the "
        "signals of a symbol before its first decision were not taken. See result.warnings "
        "for why."
    )


def test_the_assumptions_give_the_first_decision_of_each_symbol():
    _, result = listed_late()
    assert result.assumptions["warm_up"].endswith(
        f"first decisions: AAA at start, BBB {READY} 15:30 IST"
    )
