"""Accounting invariants of whole runs, as property tests (NFR-COR-05).

Hypothesis draws a run: two NSE shares with random 1-minute price paths on the 0.05 tick
grid (2026-11-04 from 09:15 IST), MIS or CNC, with or without charges and slippage, and a
random schedule of buys, sells (sometimes with `reverse=True`, sometimes with a stop and a
target on an entry) and exits; after the schedule every position is closed, at the open of
the next-to-last bar. Orders the engine rejects (a short on CNC, a buy beyond the free
cash) are part of the draw. For every run:

1. At every equity row, cash plus the value of the positions equals equity. Cash is the
   capital plus what sales brought in less what buys paid (from the fills up to that
   slice) less the charges the ledger has posted; a position's value is its signed units
   times the close of its instrument's bar in that slice.
2. Each position equals the sum of its fills: every row of the position history and
   every final position.
3. Each fill's charges are the sum of their six components (brokerage, STT or CTT,
   exchange charge, SEBI fee, stamp duty, GST) plus the DP charge, and each trade's
   charges are the sum of its fills' charges.
4. Over the whole run, the ledger's cash change equals the realized P&L less the charges,
   and the realized P&L is the sum of the trades' gross P&L.

Tolerance: the ledger posts realized P&L rounded to the paisa at each fill (decision D6),
so a sum of fills differs from it by at most half a paisa per fill. Everything else is
compared to 1e-9 rupees (float noise only).

Seeded and bounded: `derandomize=True` fixes the examples, `max_examples` bounds the run
time (about 20 ms a run), and no example database is written.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2026, 11, 4, 9, 15, tzinfo=IST)
ONE_MINUTE = dt.timedelta(minutes=1)
SYMBOLS = ("SBIN", "INFY")
CAPITAL = 1_000_000.0
TICK = 0.05
BARS = 40
# The last bar at which the schedule may act; every position is closed after it.
LAST_ACTION_BAR = BARS - 4
HALF_PAISA = 0.005
FLOAT_NOISE = 1e-9
CHARGE_LINES = ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst")
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@dataclass(frozen=True)
class Scenario:
    """One drawn run: bars by symbol, the schedule {bar: [(symbol, action, units,
    reverse, protect)]} and the run settings."""

    bars: dict[str, pd.DataFrame]
    schedule: dict[int, list[tuple]]
    product: str
    costs: str | None
    slippage: object


@st.composite
def price_path(draw) -> pd.DataFrame:
    """1-minute bars whose prices sit on the tick grid: a random walk of closes, each
    open a few ticks from the previous close, the range a few ticks beyond both."""
    start = draw(st.integers(2_000, 40_000))  # 100.00 to 2,000.00 rupees, in ticks
    steps = draw(st.lists(st.integers(-40, 40), min_size=BARS, max_size=BARS))
    gaps = draw(st.lists(st.integers(-10, 10), min_size=BARS, max_size=BARS))
    reach = draw(st.lists(st.integers(0, 20), min_size=BARS, max_size=BARS))
    closes = start + np.cumsum(steps)
    opens = np.concatenate([[start], closes[:-1]]) + np.array(gaps)
    highs = np.maximum(opens, closes) + np.array(reach)
    lows = np.maximum(np.minimum(opens, closes) - np.array(reach[::-1]), 1)
    frame = pd.DataFrame(
        {
            "open": np.round(np.maximum(opens, 1) * TICK, 2),
            "high": np.round(np.maximum(highs, 1) * TICK, 2),
            "low": np.round(lows * TICK, 2),
            "close": np.round(np.maximum(closes, 1) * TICK, 2),
            "volume": 1_000.0,
        },
        index=pd.DatetimeIndex([SESSION_OPEN + ONE_MINUTE * k for k in range(BARS)]),
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


ACTIONS = st.tuples(
    st.sampled_from(SYMBOLS),
    st.sampled_from(("buy", "sell", "exit")),
    st.integers(1, 300),
    st.booleans(),
    st.booleans(),
)


@st.composite
def scenarios(draw) -> Scenario:
    schedule = draw(
        st.dictionaries(
            st.integers(0, LAST_ACTION_BAR), st.lists(ACTIONS, min_size=1, max_size=3), max_size=12
        )
    )
    return Scenario(
        bars={symbol: draw(price_path()) for symbol in SYMBOLS},
        schedule=schedule,
        product=draw(st.sampled_from(("MIS", "CNC"))),
        costs=draw(st.sampled_from((None, "zerodha"))),
        slippage=draw(st.sampled_from((0, "default"))),
    )


def scheduled_strategy(schedule: dict[int, list[tuple]]) -> type[bt.Strategy]:
    """A 1-minute strategy that acts on the schedule and closes everything after it. A
    stop and a target go only on the first order of a bar for a flat symbol: a later
    order of the same bar would be an add that changes them, which is refused."""

    class Scheduled(bt.Strategy):
        symbols, exchange, interval = SYMBOLS, "NSE", "1m"

        def on_bar(self, bar):
            index = len(self.data[SYMBOLS[0]].close) - 1
            if index == LAST_ACTION_BAR + 1:
                for symbol in SYMBOLS:
                    self.exit(symbol)
            ordered = set()
            for symbol, action, units, reverse, protect in schedule.get(index, ()):
                if action == "exit":
                    self.exit(symbol)
                    continue
                rules = {}
                if protect and symbol not in ordered and self.positions[symbol].is_flat:
                    rules = {"stop": bt.points(2), "target": bt.points(3)}
                ordered.add(symbol)
                order = self.buy if action == "buy" else self.sell
                order(qty=units, symbol=symbol, reverse=reverse, **rules)

    return Scheduled


def run(scenario: Scenario) -> bt.Result:
    return bt.backtest(
        scheduled_strategy(scenario.schedule),
        data=scenario.bars,
        capital=CAPITAL,
        product=scenario.product,
        costs=scenario.costs,
        slippage=scenario.slippage,
        base="1m",
        end=SESSION_OPEN + ONE_MINUTE * (BARS - 1),
    )


def signed_units(fills: pd.DataFrame) -> pd.Series:
    return fills["quantity_units"].where(fills["side"] == "BUY", -fills["quantity_units"])


@given(scenarios())
@SETTINGS
def test_cash_plus_the_value_of_the_positions_equals_equity_at_every_row(scenario):
    result = run(scenario)
    fills = result.fills
    units = signed_units(fills)
    flows = -units * fills["price_rupees"]
    for close_time, row in result.equity.iterrows():
        done = (fills["time"] < close_time).to_numpy()
        cash = CAPITAL + flows[done].sum() - row["costs_rupees"]
        value = sum(
            units[done & (fills["symbol"] == symbol).to_numpy()].sum()
            * scenario.bars[symbol].loc[close_time - ONE_MINUTE, "close"]
            for symbol in SYMBOLS
        )
        allowed = HALF_PAISA * done.sum() + FLOAT_NOISE
        assert abs(cash + value - row["equity_rupees"]) <= allowed, close_time


@given(scenarios())
@SETTINGS
def test_each_position_equals_the_sum_of_its_fills(scenario):
    result = run(scenario)
    fills = result.fills
    units = signed_units(fills)
    for _, row in result.positions.iterrows():
        mine = (fills["symbol"] == row["symbol"]) & (fills["slice"] <= row["slice"])
        assert row["quantity_units"] == units[mine].sum()
    for _, final in result.final_positions.iterrows():
        assert final["quantity_units"] == units[fills["symbol"] == final["symbol"]].sum()


@given(scenarios())
@SETTINGS
def test_charges_are_the_sum_of_their_components_per_fill_and_per_trade(scenario):
    result = run(scenario)
    costs = result.costs
    for _, charged in costs.iterrows():
        lines = sum(charged[name] for name in CHARGE_LINES)
        assert abs(charged["cost"] - lines) <= FLOAT_NOISE
        assert abs(charged["total"] - (lines + charged["dp_charge"])) <= FLOAT_NOISE
    for _, trade in result.trades.iterrows():
        own = costs.loc[costs["trade_id"] == trade["trade_id"], "total"].sum()
        assert abs(trade["costs_rupees"] - own) <= FLOAT_NOISE
        assert abs(trade["net_pnl_rupees"] - (trade["gross_pnl_rupees"] - own)) <= FLOAT_NOISE


@given(scenarios())
@SETTINGS
def test_the_ledgers_cash_change_is_the_realized_pnl_less_the_charges(scenario):
    result = run(scenario)
    fills, equity = result.fills, result.equity
    assert (result.final_positions["quantity_units"] == 0).all()
    cash_change = (-signed_units(fills) * fills["price_rupees"]).sum()
    ledger_costs = equity["costs_rupees"].iloc[-1]
    realized = result.gross_pnl["realized"]
    allowed = HALF_PAISA * len(fills) + FLOAT_NOISE
    assert abs((cash_change - ledger_costs) - (realized - ledger_costs)) <= allowed
    assert abs(equity["equity_rupees"].iloc[-1] - CAPITAL - (realized - ledger_costs)) <= (
        FLOAT_NOISE
    )
    assert abs(result.trades["gross_pnl_rupees"].sum() - realized) <= allowed
