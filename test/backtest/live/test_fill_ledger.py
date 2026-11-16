"""The fill ledger (`_fill_ledger`, 16-session-design 2.6): the one door through which every
fill of an own order reaches the kernel. Cumulative words (filled quantity and average
price of the whole order) become increments, priced from the two cumulative values (put
on the tick where the average's rounding cannot hide one), the average itself used only
when its rounding could move the derived price by more than a tick, a stale or repeated
word ignored, and a terminal word ending the order in the kernel; a refusal by the kernel
is alerted as a mismatch.

The kernel is the real binding's session (`venue_samples`), or a stand-in for a race the
binding cannot be brought to yet; the test plays the broker.
"""

from __future__ import annotations

import pytest
from venue_samples import Alerts, Rig, order, update

from openalgo.backtest._live._fill_ledger import (
    FillLedger,
    FillOutcome,
    FillResult,
    Increment,
    next_increment,
    on_tick,
)
from openalgo.backtest._live._types import AlertKind, OrderStatus
from openalgo.backtest.errors import LiveError, VenueRefusalError

TICK = 0.05


@pytest.mark.parametrize(
    ("known", "word", "expected"),
    [
        ((0, 0.0), (40, 500.10), (FillOutcome.APPLIED, Increment(40, 500.10))),
        # 40 at 500.10 then 60 more: 100 at 500.16 is 50,016, less 20,004, over 60.
        ((40, 20_004.0), (100, 500.16), (FillOutcome.APPLIED, Increment(60, 500.20))),
        # 10 at 500.00, then 3 more with 13 averaged to 500.0115: 6,500.1495 is put on the
        # tick, 6,500.15, so the 3 are 1,500.15, at 500.05.
        ((10, 5_000.0), (13, 500.0115), (FillOutcome.APPLIED, Increment(3, 500.05))),
        # 1 at 500, then 2 averaged to 505: 1,010 less 500.
        ((1, 500.0), (2, 505.0), (FillOutcome.APPLIED, Increment(1, 510.0))),
        # 40 at 100.00, then 60 at 110.00: an average of 106.00, far from the second fill.
        ((40, 4_000.0), (100, 106.0), (FillOutcome.APPLIED, Increment(60, 110.0))),
        # An option: 50 at 1.00, then 50 at 1.05, one tick away: an average of 1.025.
        ((50, 50.0), (100, 1.025), (FillOutcome.APPLIED, Increment(50, 1.05))),
        # One word for 500 at 100.00 and 500 at 100.05: an average, not one trade's price.
        ((0, 0.0), (1000, 100.025), (FillOutcome.APPLIED, Increment(1000, 100.025))),
        # 100 at 500, 1 more averaged to 480: the derived -1520 is not a price.
        (
            (100, 50_000.0),
            (101, 480.0),
            (FillOutcome.APPLIED, Increment(1, 480.0, derived_from_average=True)),
        ),
        # 100 at 500, 1 more averaged to 500.50: an average to the paisa may be off by
        # half a paisa, which moves the one unit's derived 550.50 by 50 paise, ten ticks.
        (
            (100, 50_000.0),
            (101, 500.5),
            (FillOutcome.APPLIED, Increment(1, 500.5, derived_from_average=True)),
        ),
        ((40, 20_004.0), (40, 500.10), (FillOutcome.NOTHING_NEW, None)),
        ((40, 20_004.0), (30, 500.10), (FillOutcome.STALE, None)),
    ],
    ids=[
        "first fill",
        "derived from two averages",
        "a value put on the tick",
        "a small order",
        "a second fill far from the average",
        "a second fill one tick away on an option",
        "a first word of several trades",
        "derived price not above 0",
        "an average too coarse to price the units",
        "same units",
        "fewer units",
    ],
)
def test_a_cumulative_word_gives_the_units_newly_filled_and_their_price(known, word, expected):
    assert next_increment(*known, *word, TICK) == expected


@pytest.mark.parametrize(
    ("price", "tick", "on"),
    [(500.1234, 0.05, 500.10), (500.13, 0.05, 500.15), (82.33, 0.0025, 82.33), (101.4, 1.0, 101)],
)
def test_a_derived_price_is_put_on_the_nearest_tick(price, tick, on):
    assert on_tick(price, tick) == on


def holding_order(rig: Rig, units: int = 100) -> tuple[int, FillLedger]:
    """A working BUY of `units` and its ledger, with the session waiting."""
    kernel_order = rig.submit(order("BUY", float(units)))
    rig.wait()
    return kernel_order, FillLedger(rig.engine, kernel_order, tick=TICK, alerts=rig.alerts)


def test_fills_reach_the_kernel_as_increments_at_their_derived_prices():
    rig = Rig()
    kernel_order, ledger = holding_order(rig)

    ledger.apply(update("B1", filled=40, average=500.10))
    ledger.apply(update("B1", OrderStatus.COMPLETE, filled=100, average=500.16))

    position = rig.engine.position(0, "MIS")
    assert (position["quantity_units"], position["average_price_rupees"]) == (100.0, 500.16)
    assert rig.status(kernel_order) == "Filled"


def test_the_ledger_moves_to_the_brokers_cumulative_figures_its_value_on_the_tick():
    rig = Rig()
    _, ledger = holding_order(rig)

    ledger.apply(update("B1", filled=13, average=500.0115))

    assert (ledger.filled_units, ledger.filled_value) == (13, 6_500.15)


def test_a_fill_seen_by_a_second_source_is_applied_once():
    rig = Rig()
    _, ledger = holding_order(rig)
    ledger.apply(update("B1", filled=40, average=500.10, source="stream"))

    again = ledger.apply(update("B1", filled=40, average=500.10, source="poll"))

    assert again.outcome is FillOutcome.NOTHING_NEW
    assert rig.engine.position(0, "MIS")["quantity_units"] == 40.0


def test_a_stale_word_is_ignored():
    rig = Rig()
    _, ledger = holding_order(rig)
    ledger.apply(update("B1", filled=40, average=500.10))

    stale = ledger.apply(update("B1", filled=20, average=500.10, source="poll"))

    assert (stale.outcome, stale.mismatch) == (FillOutcome.STALE, False)
    assert rig.engine.position(0, "MIS")["quantity_units"] == 40.0


@pytest.mark.parametrize(
    ("status", "kernel_status"),
    [(OrderStatus.CANCELLED, "Canceled"), (OrderStatus.REJECTED, "Rejected")],
)
def test_a_terminal_word_ends_the_order_after_its_fills(status, kernel_status):
    rig = Rig()
    kernel_order, ledger = holding_order(rig)

    result = ledger.apply(update("B1", status, filled=40, average=500.10))

    assert (result.ended, rig.status(kernel_order)) == (status, kernel_status)
    assert rig.engine.order(kernel_order)["filled_units"] == 40.0


def test_a_rejection_carries_the_venue_reject_code():
    rig = Rig()
    kernel_order, ledger = holding_order(rig)

    ledger.apply(update("B1", OrderStatus.REJECTED))

    assert rig.engine.order(kernel_order)["reject_code"] == "venue"


def test_a_complete_word_with_the_orders_units_needs_nothing_more():
    rig = Rig()
    kernel_order, ledger = holding_order(rig)

    result = ledger.apply(update("B1", OrderStatus.COMPLETE, filled=100, average=500.0))

    assert (result.ended, result.mismatch, rig.status(kernel_order)) == (None, False, "Filled")
    assert rig.alerts.sent == []


def test_a_complete_word_short_of_the_orders_units_cancels_the_rest_with_an_alert():
    rig = Rig()
    kernel_order, ledger = holding_order(rig)

    result = ledger.apply(update("B1", OrderStatus.COMPLETE, filled=60, average=500.0))

    assert (result.ended, result.mismatch) == (OrderStatus.COMPLETE, True)
    assert (rig.status(kernel_order), rig.engine.order(kernel_order)["filled_units"]) == (
        "Canceled",
        60.0,
    )
    assert [kind for kind, _ in rig.alerts.sent] == [AlertKind.ERROR]


def test_a_fill_the_kernel_refuses_is_a_mismatch_alerted_and_not_recorded():
    rig = Rig()
    kernel_order, ledger = holding_order(rig, units=10)

    result = ledger.apply(update("B1", quantity=10, filled=11, average=500.0))

    assert (result.outcome, result.refusal, result.mismatch) == (
        FillOutcome.REFUSED,
        "too_many_units",
        True,
    )
    assert (ledger.filled_units, rig.engine.order(kernel_order)["filled_units"]) == (0, 0.0)
    kind, text = rig.alerts.sent[0]
    assert kind is AlertKind.ERROR and "11 filled" in text


class ClosingOnlyEngine:
    """A stand-in kernel with one order that may only close a position of `position`
    units: a fill beyond it records the closing units and refuses the rest, as
    `apply_fill` does for an order that may not reverse the position."""

    def __init__(self, position: int) -> None:
        self.position = position
        self.filled = 0

    def apply_fill(self, order: int, units: float, price: float, at: int) -> None:
        closing = min(units, self.position)
        self.filled += closing
        self.position -= closing
        if closing < units:
            raise VenueRefusalError(f"{units - closing} units beyond the position", "would_reverse")

    def order(self, order: int) -> dict:
        return {"filled_units": float(self.filled)}

    def working_orders(self) -> list[int]:
        return []


def test_a_fill_beyond_the_position_reports_the_units_the_kernel_recorded():
    ledger = FillLedger(ClosingOnlyEngine(position=6), 1, tick=TICK, alerts=Alerts())

    result = ledger.apply(update("B1", OrderStatus.COMPLETE, quantity=10, filled=10, average=500.0))

    assert result == FillResult(
        1, FillOutcome.REFUSED, Increment(6, 500.0), "would_reverse", mismatch=True
    )
    assert ledger.filled_units == 10


def test_a_terminal_word_for_an_order_the_kernel_ended_already_is_the_expected_race():
    rig = Rig()
    kernel_order, ledger = holding_order(rig)
    ledger.apply(update("B1", filled=100, average=500.0))

    result = ledger.apply(update("B1", OrderStatus.CANCELLED, filled=100, average=500.0))

    assert (result.mismatch, rig.status(kernel_order)) == (False, "Filled")
    assert rig.alerts.sent == []


def test_a_ledger_without_a_positive_tick_is_refused():
    rig = Rig()
    with pytest.raises(LiveError, match="tick"):
        FillLedger(rig.engine, 1, tick=0.0, alerts=Alerts())
