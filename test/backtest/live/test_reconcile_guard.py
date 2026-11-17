"""No exit beyond what the broker holds, in a running session on the fake OpenAlgo
(`_live/_reconcile_guard.py`, `_live/reconcile.py`; 16-session-design 5.3, 5.4 "As built"
and 7.2 row F78, unit 12): while OpenAlgo's books show the broker holding fewer units than
the engine, an exit of any owner waits and the leg's stop stays at the broker; before the
safe stop's flatten and the square-off the books are read, so a sale by hand made since
the last read is known before anything is sold.

The market is `stops_samples`': SBIN on NSE from 14:40:05 IST, trading 10 units every
10 s at 500.00. `StopHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00; `Hold`
buys 10 with no stop. Another
client of the account (`manual`) stands for an order placed by hand. Expected values come
from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, Hold, alerts, ist
from stops_samples import StopHold, events, stop_world

from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


@pytest.fixture
def lagging(world):
    """The position book stops showing new fills at 14:41:30, before the entry: it shows
    SBIN flat while the trade book shows the entry, a difference no trade explains."""
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    return world


def by_hand(world, action: str, quantity: int, at_ns: int) -> None:
    """Another client of the account places a MARKET order of SBIN at `at_ns`."""
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action=action,
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=quantity,
        ),
    )


def stops(world) -> list[tuple[int, str]]:
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


def market_sells(world) -> list[tuple[int, int]]:
    return [
        (o.quantity, o.filled)
        for o in world.orders()
        if o.strategy == "StopHold" and o.action == "SELL" and o.price_type == "MARKET"
    ]


# ----- F78: the stop stays while an exit waits for the books ----------------------------


def test_F78_the_stop_stays_at_the_broker_while_the_flatten_waits_for_the_books(lagging):
    lagging.stop_at(ist(14, 46))
    lagging.stop_at(ist(14, 48))
    outcome = lagging.run(StopHold)
    assert stops(lagging) == [(10, "trigger pending")]
    assert market_sells(lagging) == []
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (10, True)
    assert [w for w in outcome.unprotected if w.from_ns >= ist(14, 46)] == []


def test_F78_the_stop_stays_at_the_broker_while_the_square_off_waits_for_the_books(lagging, caplog):
    lagging.stop_at(ist(15, 2))
    lagging.stop_at(ist(15, 2, 10))
    lagging.run(StopHold, on_stop="keep")
    assert stops(lagging) == [(10, "trigger pending")]
    assert market_sells(lagging) == []
    assert any(
        "the cancel of the stop of SBIN NSE (SELL 10) is held back" in text
        for text in alerts(caplog, "error")
    )


def test_F78_the_safe_stops_reminder_says_the_exit_waits_for_the_books(lagging, caplog):
    # `Hold` has no stop: its exit is held, and the reminder a minute on names why.
    lagging.stop_at(ist(14, 46))
    lagging.stop_at(ist(14, 48))
    lagging.run(Hold)
    [reminder] = [text for text in alerts(caplog, "error") if "safe stop:" in text]
    assert "its exit is held back until OpenAlgo's books show the units it closes" in reminder


# ----- the close sends nothing held -----------------------------------------------------


@pytest.fixture
def agreeing_at_the_close(lagging):
    """`Hold` (no stop) is stopped at 14:46 and again at 14:48: its flatten exit is held
    while the book shows 0, and the second Ctrl+C's lines, after its 5 s wait, say it was
    not sent. The book catches up at 14:48:05.5, so the close's final read agrees."""
    lagging.script.at(ist(14, 48, 5) + NS // 2, lagging.fake.catch_up_positionbook)
    lagging.stop_at(ist(14, 46))
    lagging.stop_at(ist(14, 48))
    return lagging


def test_an_exit_held_when_the_session_ends_is_not_sent_when_the_books_agree_at_the_close(
    agreeing_at_the_close,
):
    world = agreeing_at_the_close
    outcome = world.run(Hold)
    closed = [e for e in events(outcome, "reconcile") if e["ts_ns"] > ist(14, 48, 5)]
    assert [e["found"] for e in closed] == ["agree"]
    # What the session handed to the order worker: the entry only.
    placed = [e["purpose"] for e in events(outcome, "request") if e["type"] == "PlaceRequest"]
    assert placed == ["entry"]
    assert world.fake.position("SBIN", "NSE") == 10


def test_the_result_names_an_exit_held_at_the_close_as_the_last_lines_did(agreeing_at_the_close):
    world = agreeing_at_the_close
    outcome = world.run(Hold)
    # The last line of the leg is the second Ctrl+C's (a reminder came at 14:47).
    *_, line = [line for line in world.lines if line.startswith("Hold SBIN NSE 10 units: ")]
    (left,) = outcome.open_positions
    assert left.note == line.removeprefix("Hold SBIN NSE 10 units: ")
    assert left.note == "order o2 not sent; no stop at the broker, close it by hand"


# ----- a sale by hand just before the flatten or the square-off ---------------------------


def test_a_ctrl_c_right_after_a_sale_by_hand_sells_only_what_the_broker_holds(world):
    by_hand(world, "SELL", 6, ist(14, 43))
    world.stop_at(ist(14, 43, 2))
    outcome = world.run(StopHold)
    assert [(e["side"], e["units"]) for e in events(outcome, "adopted")] == [("SELL", 6)]
    assert market_sells(world) == [(4, 4)]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_square_off_soon_after_a_sale_by_hand_sells_only_what_the_broker_holds(world):
    # The reconciler's scheduled reads come at 14:59:35 and 15:00:05: the read asked for
    # 5 s before the 15:00 square-off is the one that sees the sale at 14:59:40.
    by_hand(world, "SELL", 6, ist(14, 59, 40))
    world.stop_at(ist(15, 2))
    world.run(StopHold, on_stop="keep")
    assert market_sells(world) == [(4, 4)]
    assert world.fake.position("SBIN", "NSE") == 0
