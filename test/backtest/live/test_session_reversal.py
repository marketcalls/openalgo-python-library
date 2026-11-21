"""A reversal in a live session, on the fake OpenAlgo (2.4.0 item P2-3 a): a short with a
broker-held stop, then `buy(..., reverse=True, stop=...)`. The reversing order is an entry
(`venue_roles`: the part beyond the position opens the other side): its stop is cancelled
first, the buy goes as one order, and the new leg gets its own stop. The session's
invariants (16-session-design 7.1) follow the leg to its new side. A reversal that rests
(a LIMIT, SL or SL-M order with `reverse=True`) is refused and never sent (2.4.0 item
P3-2): beside the position's stop both could fill.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades 10 units every 10 s at 500.00. The strategy decides at
14:42 (the short) and 14:43 (the reversal); Ctrl+C at 14:44:30 flattens the long.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

#: The short's units, and the stop's distance from each entry, points.
UNITS = 10
STOP_POINTS = 3.0


class ShortThenReverse(bt.Strategy):
    """Sells `UNITS` with a stop at its first decision, then reverses to a long of `UNITS`
    (a buy of twice that, `reverse=True`) with a stop at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.buy(2 * UNITS, reverse=True, stop=bt.points(STOP_POINTS))


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 44, 30))
    return world


def placed(world) -> list[tuple[str, int, str, str]]:
    """The session's orders at the fake, oldest first: (action, units, type, status)."""
    return [(o.action, o.quantity, o.price_type, o.status) for o in world.orders()]


def test_a_reversal_cancels_the_short_s_stop_buys_once_and_protects_the_long(world):
    world.run(ShortThenReverse)
    # The short and its stop 3 points above; the stop cancelled before the reversal's buy
    # of 20; the long's stop 3 points below; Ctrl+C cancels it and sells the 10 held.
    assert placed(world) == [
        ("SELL", UNITS, "MARKET", "complete"),
        ("BUY", UNITS, "SL-M", "cancelled"),
        ("BUY", 2 * UNITS, "MARKET", "complete"),
        ("SELL", UNITS, "SL-M", "cancelled"),
        ("SELL", UNITS, "MARKET", "complete"),
    ]


def test_the_reversing_order_is_filed_as_an_entry(world):
    outcome = world.run(ShortThenReverse)
    orders = outcome.orders
    reversal = [order for order in orders if order.quantity == 2 * UNITS]
    assert [(order.action, order.purpose) for order in reversal] == [("BUY", "entry")]


def test_the_reversals_buy_is_placed_only_once_the_shorts_stop_has_ended(world):
    # Never both at the broker: the stop's fill and the buy would leave a long of 20.
    world.run(ShortThenReverse)
    stop, reversal = (world.orders()[at].order_id for at in (1, 2))
    changes = world.trace.changes
    stop_ended = next(c.seq for c in changes if c.order_id == stop and not c.working)
    reversal_placed = next(c.seq for c in changes if c.order_id == reversal)
    assert stop_ended < reversal_placed


def test_the_session_ends_flat_at_the_fake(world):
    world.run(ShortThenReverse)
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_reversal_raises_no_alert_of_a_leg_left_without_a_stop(world, caplog):
    # Between the short's stop ending and the buy's fill the reversal closes the short:
    # the leg is not bare, and no stop is placed again beside the buy.
    world.run(ShortThenReverse)
    assert not [text for text in alerts(caplog, "error") if "had no stop" in text]


def reject_codes(outcome) -> list[tuple[str, float, str]]:
    """(side, units, reject code) of each order the kernel rejected, oldest first."""
    orders = outcome.outputs["ShortThenReverse"]["orders"]
    rows = zip(orders["side"], orders["quantity_units"], orders["reject_code"], strict=True)
    return [(side, units, code) for side, units, code in rows if code is not None]


@pytest.fixture
def locked_world(world):
    """The feed dies at 14:42:58 and OpenAlgo answers the next cancel 8 s late from
    14:43:00: the reversal decided at 14:43:01 waits for its stop's cancel while the
    feed-dead lock is set (10 s without a tick, 14:43:08 or later)."""
    world.script.at(ist(14, 42, 58), world.fake.faults.feed_dead)
    world.script.at(ist(14, 43), lambda: world.fake.faults.slow("cancelorder", 8.0))
    return world


def test_a_reversal_released_once_entries_locked_is_never_sent(locked_world):
    # Its buy would open a long while entries are locked. The short is given its stop
    # again, and Ctrl+C at 14:44:30 cancels it and buys the 10 back.
    locked_world.run(ShortThenReverse)
    assert placed(locked_world) == [
        ("SELL", UNITS, "MARKET", "complete"),
        ("BUY", UNITS, "SL-M", "cancelled"),
        ("BUY", UNITS, "SL-M", "cancelled"),
        ("BUY", UNITS, "MARKET", "complete"),
    ]
    assert locked_world.fake.position("SBIN", "NSE") == 0


def test_a_reversal_released_once_entries_locked_is_rejected_with_the_lock_s_code(locked_world):
    # The code a submission gets while entries are locked (the kernel's entry gates).
    outcome = locked_world.run(ShortThenReverse)
    assert reject_codes(outcome) == [("BUY", 2.0 * UNITS, "live_lock")]


class ShortTwoTargetsThenRestingReverse(bt.Strategy):
    """Sells `UNITS` at its first decision with a stop 3 points above and two targets, half
    at 1 point and the rest at 2 points below; at its third decision (14:44:01), while
    short, it rests a reversal: a LIMIT buy of 20 at 495.00, below the market."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            targets = [(bt.points(1), 0.5), (bt.points(2), 0.5)]
            self.sell(UNITS, stop=bt.points(STOP_POINTS), targets=targets)
        elif self.calls == 3 and self.position.qty < 0:
            self.buy(2 * UNITS, price=495.0, reverse=True)


def test_a_resting_reversal_does_not_cover_a_short_left_without_its_stop(world, caplog):
    # 2.4.0 item M8-X1. SBIN trades at 498.00 at 14:43:55: both targets are reached on one
    # tick, so the kernel cancels the short's stop for the two purchases of 5 (14:43:58),
    # and the broker refuses the first. The second fills, but its words are held until
    # 14:44:01, so 5 units are found short with no stop in the loop step of the decision
    # of 14:44:01. That decision rests the reversal, a buy below the market, which closes
    # nothing on a rise: the leg is bare all the same, so its stop is placed again over
    # the 5 units at once and an ERROR alert says so (not only by the safe stop's
    # protect at 14:44:30). The reversal itself is refused in that same loop step.
    faults = world.fake.faults

    def refuse_the_first():
        world.fake.reject_next("RMS: blocked", price_type="MARKET", at_placement=True)

    def hold_the_words():
        faults.hold_order_updates()
        for _ in range(10):
            faults.gateway_error("orderstatus")

    world.script.at(ist(14, 43, 50), refuse_the_first)
    world.script.at(ist(14, 43, 57) + 500_000_000, hold_the_words)
    world.script.at(ist(14, 44, 1), faults.release_order_updates)
    world.script.price = lambda now_ns: 498.0 if now_ns >= ist(14, 43, 50) else 500.0
    # The resting reversal itself is never sent (2.4.0 item P3-2), so no check of the
    # session invariants is waived any more.
    world.run(ShortTwoTargetsThenRestingReverse, on_stop="keep")

    assert [o for o in world.orders() if o.price_type == "LIMIT"] == []
    stops = [o for o in world.orders() if o.price_type == "SL-M"]
    assert [(o.quantity, o.status) for o in stops] == [
        (UNITS, "cancelled"),
        (UNITS // 2, "trigger pending"),
    ]
    assert ist(14, 44, 1) <= stops[-1].placed_ns < ist(14, 44, 5)
    assert [text for text in alerts(caplog, "error") if "had no stop" in text]


class LateReverse(bt.Strategy):
    """Sells `UNITS` with a stop at its first decision, then orders the reversal at the
    decision of 14:51, after NSE's MIS entry cut-off (14:50 IST)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(UNITS, stop=bt.points(STOP_POINTS))
        elif bar.ts_close == ist(14, 51):
            self.buy(2 * UNITS, reverse=True, stop=bt.points(STOP_POINTS))


def test_a_reversal_after_the_entry_cut_off_is_refused_whole_as_in_a_backtest(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Its opening part obeys the entry window: the kernel refuses an order that opens at
    # submission (`entry_window`), the whole order, as a backtest refuses it. The short
    # keeps its stop at the broker until Ctrl+C at 14:52:30 flattens it.
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 52, 30))
    outcome = world.run(LateReverse)
    orders = outcome.outputs["LateReverse"]["orders"]
    refused = [
        (side, units, code)
        for side, units, code in zip(
            orders["side"], orders["quantity_units"], orders["reject_code"], strict=True
        )
        if code is not None
    ]
    assert refused == [("BUY", 2.0 * UNITS, "entry_window")]
    assert [o.quantity for o in world.orders() if o.price_type == "MARKET"] == [UNITS, UNITS]


# ----- a resting reversal is refused in a live session (2.4.0 item P3-2) ---------------------


class ShortThenRestingReverse(bt.Strategy):
    """Sells `UNITS` with a stop 3 points above at its first decision (14:42:01, filled at
    500.00: its SL-M buy at 503.00), then at its second (14:43:01) rests a reversal below
    the market: a LIMIT buy of twice that at 495.00, `reverse=True`, meant to leave it long
    `UNITS`. Keeps every order event it was given (the session builds the strategy, so a
    test reads the class)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    seen: list[dict] = []

    def init(self):
        self.calls = 0

    def on_order(self, order):
        type(self).seen.append(dict(order))

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.buy(2 * UNITS, price=495.0, reverse=True)


#: SBIN rises through the short's stop (503.00) from 14:43:30, then falls through the
#: reversal's limit (495.00) from 14:44: rested at the broker beside the stop, both fill
#: and the account ends long 20, twice what the strategy meant.
RISE_AT, FALL_AT = ist(14, 43, 30), ist(14, 44)


def rise_then_fall(now_ns: int) -> float:
    if now_ns >= FALL_AT:
        return 494.0
    return 504.0 if now_ns >= RISE_AT else 500.0


@pytest.fixture
def resting_world(world):
    """The rise and the fall, the positions kept at Ctrl+C (14:44:30). No check of the
    session invariants is waived: the stop and the resting reversal at the broker together
    would hold 30 units to close a short of 10 ("exits within the position")."""
    ShortThenRestingReverse.seen = []
    world.script.price = rise_then_fall
    return world


def test_a_resting_reversal_is_never_sent_beside_the_positions_stop(resting_world):
    resting_world.run(ShortThenRestingReverse, on_stop="keep")

    # The short, then its stop, filled on the rise; nothing more reaches the broker, so the
    # fall buys nothing and the account ends flat, not long 20.
    assert placed(resting_world) == [
        ("SELL", UNITS, "MARKET", "complete"),
        ("BUY", UNITS, "SL-M", "complete"),
    ]
    assert resting_world.fake.position("SBIN", "NSE") == 0


def test_the_strategy_sees_the_resting_reversal_rejected(resting_world):
    outcome = resting_world.run(ShortThenRestingReverse, on_stop="keep")

    # o1 is the short, o2 its stop, o3 the reversal: rejected by the venue, the session.
    rejected = [
        (e["order_id"], e["reject_code"])
        for e in ShortThenRestingReverse.seen
        if e["name"] == "Rejected"
    ]
    assert rejected == [(3, "venue")]
    orders = outcome.outputs["ShortThenRestingReverse"]["orders"]
    columns = ("side", "quantity_units", "order_type", "limit_rupees", "reject_code")
    rows = zip(*(orders[name] for name in columns), strict=True)
    refused = [row for row in rows if row[-1] is not None]
    assert refused == [("BUY", 2.0 * UNITS, "LIMIT", 495.0, "venue")]


def test_the_refusal_says_to_send_the_reversal_at_market_or_to_exit_and_enter(
    resting_world, caplog
):
    resting_world.run(ShortThenRestingReverse, on_stop="keep")

    (line,) = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "resting reversal" in record.getMessage()
    ]
    assert "MARKET" in line and "exit and enter separately" in line


def resting_reverse(price: float | None, trigger: float | None, *, short_first: bool = True):
    """A strategy that sells `UNITS` with a stop at its first decision (unless
    `short_first` is False: it stays flat), then at its second buys twice that with
    `reverse=True` at `price` and `trigger` (a LIMIT, an SL-M or an SL)."""

    class RestingReverse(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.calls = 0

        def on_bar(self, bar):
            self.calls += 1
            if self.calls == 1 and short_first:
                self.sell(UNITS, stop=bt.points(STOP_POINTS))
            elif self.calls == 2:
                self.buy(2 * UNITS, price=price, trigger_price=trigger, reverse=True)

    return RestingReverse


@pytest.mark.parametrize(
    ("price", "trigger"),
    [(495.0, None), (None, 506.0), (507.0, 506.0)],
    ids=["LIMIT", "SL-M", "SL"],
)
def test_a_resting_reversal_of_any_order_type_is_never_sent(world, price, trigger):
    world.run(resting_reverse(price, trigger), on_stop="keep")

    assert placed(world) == [
        ("SELL", UNITS, "MARKET", "complete"),
        ("BUY", UNITS, "SL-M", "trigger pending"),
    ]


def test_a_resting_reversal_is_refused_while_flat_too(world):
    # reverse=True says the order may open the other side: refused whatever the position.
    outcome = world.run(resting_reverse(499.0, None, short_first=False), on_stop="keep")

    assert placed(world) == []
    orders = outcome.outputs["RestingReverse"]["orders"]
    assert [code for code in orders["reject_code"] if code is not None] == ["venue"]
