"""A resting order against a strategy's open position in a live session, on the fake
OpenAlgo (2.4.0 item P4-1): a plain order on the closing side (no `reverse=True`), a
LIMIT, SL or SL-M, placed while the position's broker-held stop works.

The stop takes the position's closing room, so the kernel classes such an order as an
entry (`venue_roles`: it could open the other side). At the broker it would work beside
the stop, both on the closing side, for twice the position, and both could fill: the
stop first and the order later leave the account on the other side with no stop. So a
live session refuses it before anything reaches the broker, rejected with `venue`, as it
refuses a resting reversal (2.4.0 item P3-2). A take-profit belongs in `target=` or
`targets=` on the entry (held by the engine, sent at MARKET), and an exit at MARKET is
sent as before. A resting order that fits the position's closing room (no stop working)
and a resting entry placed while flat are sent as before, and a backtest takes every one
of them as before.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades 10 units every 10 s (at :05, :15 and so on) at 500.00
unless a test moves the price. The strategies decide at 14:42:01 (the first decision)
and 14:43:01 (the second); Ctrl+C comes at 14:44:30.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import events, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

#: The position's units, and the stop's distance below (or above) the entry, points: the
#: entry fills at 500.00, so a long's stop is an SL-M sale at 497.00.
UNITS = 10
STOP_POINTS = 3.0
#: The resting sale's limit, above the market: a take-profit the strategy means.
TAKE_PROFIT = 505.0
#: SBIN's prices of the two orders of events, and when each move comes: the first trade
#: at or after `FIRST_MOVE` is at 14:43:35, the first at or after `SECOND_MOVE` at
#: 14:44:05.
THROUGH_THE_STOP, THROUGH_THE_LIMIT = 496.0, 506.0
FIRST_MOVE, SECOND_MOVE = ist(14, 43, 30), ist(14, 44)


def moves(first: float, second: float):
    """SBIN's price: 500.00, then `first` from `FIRST_MOVE`, then `second` from
    `SECOND_MOVE`."""

    def price(now_ns: int) -> float:
        if now_ns >= SECOND_MOVE:
            return second
        return first if now_ns >= FIRST_MOVE else 500.0

    return price


#: The stop's trigger is reached first (14:43:35), then the sale's limit (14:44:05).
STOP_FIRST = moves(THROUGH_THE_STOP, THROUGH_THE_LIMIT)
#: The sale's limit is reached first (14:43:35), then the stop's trigger (14:44:05).
LIMIT_FIRST = moves(THROUGH_THE_LIMIT, THROUGH_THE_STOP)


class LongThenRestingSale(bt.Strategy):
    """Buys `UNITS` with a stop 3 points below at its first decision, then at its second
    rests a plain LIMIT sale of the same units at 505.00, a take-profit (no `reverse`).
    Keeps every order event it was given (the session builds the strategy, so a test
    reads the class)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    seen: list[dict] = []

    def init(self):
        self.calls = 0

    def on_order(self, order):
        type(self).seen.append(dict(order))

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.sell(UNITS, price=TAKE_PROFIT)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The session's world, Ctrl+C at 14:44:30. No check of the session invariants is
    waived: the stop and a resting sale at the broker together would hold 20 units to
    close a long of 10 ("exits within the position")."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    LongThenRestingSale.seen = []
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 44, 30))
    return world


def placed(world) -> list[tuple[str, int, str, str]]:
    """The session's orders at the fake, oldest first: (action, units, type, status)."""
    return [(o.action, o.quantity, o.price_type, o.status) for o in world.orders()]


def refused(outcome, name: str) -> list[tuple]:
    """(side, units, order type, limit, trigger, reject code) of each order of strategy
    `name` the kernel rejected, oldest first."""
    orders = outcome.outputs[name]["orders"]
    columns = (
        "side",
        "quantity_units",
        "order_type",
        "limit_rupees",
        "trigger_rupees",
        "reject_code",
    )
    rows = zip(*(orders[column] for column in columns), strict=True)
    return [row for row in rows if row[-1] is not None]


# ----- the two orders of events, live and in a backtest -------------------------------------


def test_stop_first_live_the_sale_is_never_sent_and_the_stop_leaves_the_account_flat(world):
    # Before 2.4.0 item P4-1 the LIMIT sale rested beside the SL-M sale of 10: the fall to
    # 496.00 filled the stop, the rise to 506.00 the LIMIT, and the account ended short
    # 10 with no stop. Now only the stop works: it fills on the fall, and the rise sells
    # nothing.
    world.script.price = STOP_FIRST
    world.run(LongThenRestingSale, on_stop="keep")

    assert placed(world) == [
        ("BUY", UNITS, "MARKET", "complete"),
        ("SELL", UNITS, "SL-M", "complete"),
    ]
    assert world.fake.position("SBIN", "NSE") == 0


def test_limit_first_live_the_sale_is_never_sent_and_the_stop_leaves_the_account_flat(world):
    # The rise to 506.00 sells nothing (the sale was refused); the fall to 496.00 then
    # fills the stop.
    world.script.price = LIMIT_FIRST
    world.run(LongThenRestingSale, on_stop="keep")

    assert placed(world) == [
        ("BUY", UNITS, "MARKET", "complete"),
        ("SELL", UNITS, "SL-M", "complete"),
    ]
    assert world.fake.position("SBIN", "NSE") == 0


#: The backtest's day and its bars: SBIN on NSE on 2026-11-16 from 09:15, every 1-minute
#: bar 500.00 flat unless a sequence moves one.
BACKTEST_OPEN = dt.datetime(2026, 11, 16, 9, 15, tzinfo=dt.timezone(dt.timedelta(hours=5.5)))
BACKTEST_MINUTES = 12
FLAT = (500.0, 500.0, 500.0, 500.0)
#: A bar that falls through the stop (497.00) and one that rises through the limit
#: (505.00), each from an open of 500.00.
FALLS = (500.0, 500.0, 496.0, 496.0)
RISES = (500.0, 506.0, 500.0, 506.0)


def backtest(moved: dict[int, tuple[float, float, float, float]]):
    """`LongThenRestingSale` backtested over the day's bars, bar k (minutes after 09:15)
    replaced by `moved[k]`. Its first decision is the 09:15 bar's close: the buy fills at
    the 09:16 open, 500.00, with its stop at 497.00; the sale decided at the 09:16 close
    rests from the 09:17 bar."""
    rows = [moved.get(k, FLAT) for k in range(BACKTEST_MINUTES)]
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    bars["volume"] = 1_000.0
    bars.index = pd.DatetimeIndex(
        [BACKTEST_OPEN + dt.timedelta(minutes=k) for k in range(BACKTEST_MINUTES)],
        name="timestamp",
    )
    return bt.backtest(
        LongThenRestingSale,
        data=bars,
        capital=1_000_000,
        costs=None,
        slippage=0,
        base="1m",
        product="MIS",
        end=bars.index[-1],
    )


def fills(result) -> list[tuple[str, float, float]]:
    """(side, units, price) of each fill of a backtest, oldest first."""
    table = result.fills
    rows = zip(table["side"], table["quantity_units"], table["price_rupees"], strict=True)
    return [(side, float(units), float(price)) for side, units, price in rows]


def test_stop_first_in_a_backtest_the_sale_fills_after_the_stop_and_ends_short():
    # The 09:18 bar falls through the stop: it sells the 10 at 497.00. The sale keeps
    # resting, now an entry, and the 09:20 bar rises through its limit: it sells 10 at
    # 505.00 and the backtest ends short 10, as the kernel models one order book.
    result = backtest({3: FALLS, 5: RISES})

    assert fills(result) == [
        ("BUY", 10.0, 500.0),
        ("SELL", 10.0, 497.0),
        ("SELL", 10.0, 505.0),
    ]
    assert result.final_positions["quantity_units"].iloc[0] == -UNITS


def test_limit_first_in_a_backtest_the_sale_closes_the_long_and_the_fall_sells_nothing():
    # The 09:18 bar rises through the limit: the sale closes the long at 505.00 and the
    # leg's stop goes with it, so the 09:20 bar's fall sells nothing.
    result = backtest({3: RISES, 5: FALLS})

    assert fills(result) == [("BUY", 10.0, 500.0), ("SELL", 10.0, 505.0)]
    assert result.final_positions["quantity_units"].sum() == 0


# ----- what the refusal tells the strategy and the trader ----------------------------------


def test_the_strategy_sees_the_resting_sale_rejected_by_the_venue(world):
    outcome = world.run(LongThenRestingSale, on_stop="keep")

    # o1 is the buy, o2 its stop, o3 the sale.
    rejected = [
        (e["order_id"], e["reject_code"])
        for e in LongThenRestingSale.seen
        if e["name"] == "Rejected"
    ]
    assert rejected == [(3, "venue")]
    assert refused(outcome, "LongThenRestingSale") == [
        ("SELL", float(UNITS), "LIMIT", TAKE_PROFIT, None, "venue")
    ]


def test_the_refusal_is_written_to_the_event_stream(world):
    outcome = world.run(LongThenRestingSale, on_stop="keep")

    written = [
        (e["strategy"], e["kernel_order"]) for e in events(outcome, "refused_resting_against")
    ]
    assert written == [("LongThenRestingSale", 3)]


def test_the_log_names_the_refused_sale_as_a_rejection_of_its_order(world, caplog):
    world.run(LongThenRestingSale, on_stop="keep")

    lines = [record.getMessage() for record in caplog.records if record.levelname == "INFO"]
    assert (
        "LongThenRestingSale: order o3 rejected before it was sent: SELL 10 units, venue "
        "(a resting order against the position, LIMIT)"
    ) in lines


def test_the_refusal_says_to_set_a_take_profit_on_the_entry_or_to_exit_at_market(world, caplog):
    # Risk arguments attach to the position the order opens, so a take-profit for the
    # next position goes on its entry: `target=` for one, `targets=` for (distance,
    # fraction) pairs. The open position can still be exited at MARKET.
    outcome = world.run(LongThenRestingSale, on_stop="keep")

    (line,) = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "against the open position" in record.getMessage()
    ]
    assert line.startswith("LongThenRestingSale: ")
    assert "set target= or targets= on the entry" in line
    assert line.endswith("or exit with a MARKET order")
    assert line in outcome.warnings


class TwoRestingSales(bt.Strategy):
    """Buys `UNITS` with a stop at its first decision, then rests a LIMIT sale of them at
    505.00 at its second and again at its third (14:44:01)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls in (2, 3):
            self.sell(UNITS, price=TAKE_PROFIT)


def test_the_warning_is_given_once_per_strategy_however_many_are_refused(world, caplog):
    outcome = world.run(TwoRestingSales, on_stop="keep")

    assert [row[-1] for row in refused(outcome, "TwoRestingSales")] == ["venue", "venue"]
    warned = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "against the open position" in record.getMessage()
    ]
    assert len(warned) == 1


def resting_against(side: str, price: float | None, trigger: float | None) -> type[bt.Strategy]:
    """A strategy that opens `UNITS` on the other side of `side` with a stop 3 points away
    at its first decision, then at its second rests a plain order of `side` for the same
    units at `price` and `trigger` (a LIMIT, an SL-M or an SL)."""

    class RestingAgainst(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.calls = 0

        def on_bar(self, bar):
            self.calls += 1
            opening = self.sell if side == "BUY" else self.buy
            closing = self.buy if side == "BUY" else self.sell
            if self.calls == 1:
                opening(UNITS, stop=bt.points(STOP_POINTS))
            elif self.calls == 2:
                closing(UNITS, price=price, trigger_price=trigger)

    return RestingAgainst


@pytest.mark.parametrize(
    ("side", "price", "trigger", "order_type"),
    [
        ("SELL", TAKE_PROFIT, None, "LIMIT"),
        ("SELL", None, 495.0, "SL-M"),
        ("SELL", 494.0, 495.0, "SL"),
        ("BUY", 495.0, None, "LIMIT"),
        ("BUY", None, 505.0, "SL-M"),
        ("BUY", 506.0, 505.0, "SL"),
    ],
    ids=["long-LIMIT", "long-SL-M", "long-SL", "short-LIMIT", "short-SL-M", "short-SL"],
)
def test_a_resting_order_against_the_position_of_any_type_is_never_sent(
    world, side, price, trigger, order_type
):
    # The price stays at 500.00: the position and its stop (an SL-M on the closing side,
    # `side`) are all the broker holds.
    outcome = world.run(resting_against(side, price, trigger), on_stop="keep")

    opening = "BUY" if side == "SELL" else "SELL"
    assert placed(world) == [
        (opening, UNITS, "MARKET", "complete"),
        (side, UNITS, "SL-M", "trigger pending"),
    ]
    assert [(row[2], row[-1]) for row in refused(outcome, "RestingAgainst")] == [
        (order_type, "venue")
    ]


# ----- what is sent as before ----------------------------------------------------------------


class LongNoStopThenRestingSale(bt.Strategy):
    """Buys `UNITS` with no stop at its first decision, then rests a LIMIT sale of them at
    505.00 at its second: an exit that fits the position's closing room."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(UNITS)
        elif self.calls == 2:
            self.sell(UNITS, price=TAKE_PROFIT)


def test_a_resting_exit_that_fits_the_closing_room_is_sent_and_fills(world, caplog):
    # No stop works, so the sale closes no more than the position: an exit, sent and
    # resting until the rise to 506.00 fills it at the fake.
    world.script.price = LIMIT_FIRST
    outcome = world.run(LongNoStopThenRestingSale, on_stop="keep")

    assert placed(world) == [
        ("BUY", UNITS, "MARKET", "complete"),
        ("SELL", UNITS, "LIMIT", "complete"),
    ]
    assert refused(outcome, "LongNoStopThenRestingSale") == []
    assert world.fake.position("SBIN", "NSE") == 0


class RestingBuyWhileFlat(bt.Strategy):
    """At its first decision, while flat, rests a LIMIT buy of `UNITS` at 495.00 with a
    stop 3 points below its fill."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(UNITS, price=495.0, stop=bt.points(STOP_POINTS))


def test_a_resting_entry_placed_while_flat_is_sent_and_fills(world):
    # SBIN falls to 494.00 from 14:43:35: the LIMIT fills, the leg gets its stop, and
    # Ctrl+C at 14:44:30 cancels the stop and sells the 10.
    world.script.price = lambda now_ns: 494.0 if now_ns >= FIRST_MOVE else 500.0
    outcome = world.run(RestingBuyWhileFlat)

    assert placed(world) == [
        ("BUY", UNITS, "LIMIT", "complete"),
        ("SELL", UNITS, "SL-M", "cancelled"),
        ("SELL", UNITS, "MARKET", "complete"),
    ]
    assert refused(outcome, "RestingBuyWhileFlat") == []


# ----- a resting entry placed while flat never comes to stand against a position ------------


def entry_then_opposite(same_decision: bool, market_first: bool) -> type[bt.Strategy]:
    """A strategy that, while flat, rests a LIMIT sale of `UNITS` at 505.00 (an entry, above
    the market) and buys `UNITS` at MARKET: the buy first when `market_first`, both at its
    first decision when `same_decision`, else the buy at its second."""

    class EntryThenOpposite(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.calls = 0

        def on_bar(self, bar):
            self.calls += 1
            first = [lambda: self.sell(UNITS, price=TAKE_PROFIT), lambda: self.buy(UNITS)]
            if market_first:
                first.reverse()
            if self.calls == 1:
                for submit in first if same_decision else first[:1]:
                    submit()
            elif self.calls == 2 and not same_decision:
                first[1]()

    return EntryThenOpposite


@pytest.mark.parametrize(
    ("same_decision", "market_first", "rejected", "at_the_fake"),
    [
        (False, False, ("BUY", "MARKET"), ("SELL", UNITS, "LIMIT", "cancelled")),
        (True, False, ("BUY", "MARKET"), ("SELL", UNITS, "LIMIT", "cancelled")),
        (True, True, ("SELL", "LIMIT"), ("BUY", UNITS, "MARKET", "complete")),
    ],
    ids=["market-buy-at-the-next-decision", "market-buy-after-it", "market-buy-before-it"],
)
def test_an_opposite_entry_while_a_resting_entry_works_is_refused_in_flight(
    world, same_decision, market_first, rejected, at_the_fake
):
    # The kernel's in-flight gate (`venue_gates`, R16): an entry is refused while any other
    # order of its instrument works but the leg's stop, so the resting sale never stands
    # beside a long. Ctrl+C at 14:44:30 cancels whatever still rests (or sells the long).
    outcome = world.run(entry_then_opposite(same_decision, market_first))

    side, order_type = rejected
    assert [(row[0], row[2], row[-1]) for row in refused(outcome, "EntryThenOpposite")] == [
        (side, order_type, "live_in_flight")
    ]
    assert placed(world)[0] == at_the_fake
    assert world.fake.position("SBIN", "NSE") == 0


def test_stop_first_no_error_alert_names_a_position_left_without_a_stop(world, caplog):
    # Before 2.4.0 item P4-1 the sale filled after the stop and the session alerted
    # "LongThenRestingSale SBIN NSE -10 units: no stop at the broker, close it by hand".
    # Now the refusal leaves the long with its stop alone, the stop closes it, and no
    # error alert comes.
    world.script.price = STOP_FIRST
    world.run(LongThenRestingSale, on_stop="keep")

    assert alerts(caplog, "error") == []
