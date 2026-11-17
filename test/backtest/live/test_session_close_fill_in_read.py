"""A fill of the leg's own stop that lands inside the close's final book read, on the fake
OpenAlgo, inline and on the session's threads (`_live/reconcile.py`,
`_live/_reconcile_close.py`, `_live/_session_close.py`; 16-session-design 5.4, the round 9
paragraph "A read inside which a leg's own fill lands"; unit 12).

`StopHold` holds 10 SBIN from 14:42:01 IST with an SL-M stop of 10 at 497.00. A read takes
OpenAlgo's position book first, then its order and trade books. Right after the final
read's position-book call answers, SBIN trades at 496.00, through the stop's trigger: the
SL-M for 10 fills at the broker, so the read's trade book carries the fill while its
position book predates it. That read does not judge the leg; the close reads the books once
more within its seconds and judges the leg from that read.

- Nothing sold by hand, one Ctrl+C with on_stop="keep" at 14:43:01: the account ends flat,
  and so does the engine. The close says flat and 0 mismatches.
- 4 sold by hand at 14:43:00, Ctrl+C with on_stop="keep" at the first comparison that
  shows the sale: the account ends short 4 against the engine's 0, and the close lists the
  leg at OpenAlgo's -4 units.

Either way the order-update stream is live or held (the fill then reaches the engine by the
read's trade book alone). When the close's second read brings no books (its position-book
call loses its connection), no read judged the leg after the fill: the close does not say
flat or 0 mismatches unqualified, it lists the leg at the earlier comparison's figure moved
by the fill when there was one, and one WARNING alert (logged at WARNING level, as a
RECONCILE_MISMATCH alert) tells the operator to check OpenAlgo's position book. Nothing is
sent. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import (
    StopHold,
    ctrl_c_at_the_first_sighting,
    handed_after_the_final_read,
    sold_by_hand,
    stop_world,
)
from test_session_close_lines import LinesThen
from test_session_threads import alone, passing, run_threaded  # noqa: F401

import openalgo.account as account
from openalgo.backtest._live._types import LOGGER_NAME

#: The price SBIN trades at to fill the stop (its trigger is 497.00).
BELOW_THE_STOP = 496.0
#: When the other client sells, and when one Ctrl+C comes with nothing sold.
SOLD_AT, PRESSED_AT = ist(14, 43), ist(14, 43, 1)
#: Why the close cannot confirm a leg no read judged after its fill.
WHY = (
    "a fill of its own orders landed during a read of the books and no later read judged "
    "the position"
)


def arm_the_fill_inside_the_read(world: Any, monkeypatch: Any, *, held: bool, then: Any) -> None:
    """Right after the leg's last line, the next position-book call answers, then SBIN
    trades through the stop (the stream's words held first when `held`), then `then` runs
    (a fault for the read after it, or nothing)."""
    answer_book = account.AccountAPI.positionbook
    armed: list[bool] = []

    def position_book(api: Any, **kwargs: Any) -> Any:
        answer = answer_book(api, **kwargs)
        if armed and armed[0]:
            armed[0] = False
            if held:
                world.fake.faults.hold_order_updates()
            world.fake.trade("SBIN", "NSE", BELOW_THE_STOP, volume=10)
            then()
        return answer

    monkeypatch.setattr(account.AccountAPI, "positionbook", position_book)
    world.lines = LinesThen("StopHold SBIN NSE", lambda: armed.append(True))


def scenario(request: Any, monkeypatch: Any, *, then: Any) -> Any:
    """The scenario of `request.param` (seam, units sold by hand, stream held), before the
    run; `then` runs once the stop has filled inside the final read."""
    seam, sold, held = request.param
    request.getfixturevalue("caplog").set_level(logging.INFO, logger=LOGGER_NAME)
    if seam == "inline":
        fake = request.getfixturevalue("make_fake_openalgo")
        world = stop_world(fake, monkeypatch, request.getfixturevalue("tmp_path"))
        world.stop_at(ist(14, 50))
    else:
        world = request.getfixturevalue("alone")
    world.seam, world.sold = seam, sold
    if sold:
        sold_by_hand(world, sold, SOLD_AT)
        ctrl_c_at_the_first_sighting(world, monkeypatch, broker_below=StopHold.qty)
    else:
        world.script.at(PRESSED_AT, world.stop.request)
    arm_the_fill_inside_the_read(world, monkeypatch, held=held, then=then)
    return world


def run(world: Any) -> Any:
    """Run `StopHold` with on_stop="keep" on the world's seam."""
    if world.seam == "inline":
        return world.run(StopHold, on_stop="keep")
    return run_threaded(world, passing(world), "orders", (StopHold,), on_stop="keep")


def summary_lines(caplog: Any) -> dict[str, str]:
    """The DAILY_SUMMARY's lines by label."""
    (text,) = alerts(caplog, "daily_summary")
    pairs = [line.split(" : ", 1) for line in text.splitlines()[1:]]
    return {label.strip(): value for label, value in pairs}


def leg_warnings(caplog: Any) -> list[str]:
    """The alerts that tell the operator to check OpenAlgo's position book, without the mode
    and strategy they begin with."""
    named = [text for text in alerts(caplog, "reconcile_mismatch") if "position book" in text]
    return [text.split(": ", 1)[1] for text in named]


#: Each case: the seam, the units sold by hand, and whether the order stream is held.
CASES = [
    pytest.param((seam, sold, held), id=f"{seam}, sold {sold}, stream {name}")
    for seam in ("inline", "threads")
    for sold in (0, 4)
    for held, name in ((False, "live"), (True, "held"))
]


# ----- the close reads the books again and judges the leg --------------------------------


@pytest.fixture
def filled_inside(request, monkeypatch):
    """The scenario with the close's second read answering as usual."""
    return scenario(request, monkeypatch, then=lambda: None)


@pytest.mark.parametrize("filled_inside", CASES, indirect=True)
def test_the_stop_fills_inside_the_read_and_the_account_ends_at_what_was_not_sold(filled_inside):
    world = filled_inside
    run(world)
    stops = [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == [(StopHold.qty, "complete")]
    assert world.fake.position("SBIN", "NSE") == -world.sold


@pytest.mark.parametrize("filled_inside", CASES, indirect=True)
def test_the_close_lists_the_leg_at_what_openalgos_books_hold_after_the_fill(filled_inside):
    world = filled_inside
    outcome = run(world)
    assert [p.units for p in outcome.open_positions] == ([-world.sold] if world.sold else [])
    assert outcome.summary.flat is (world.sold == 0)


@pytest.mark.parametrize("filled_inside", CASES, indirect=True)
def test_the_daily_summary_names_what_the_read_after_the_fill_showed(filled_inside, caplog):
    world = filled_inside
    run(world)
    lines = summary_lines(caplog)
    if not world.sold:
        assert (lines["Reconcile"], lines["End state"]) == ("0 mismatches", "flat")
        return
    named = "at the close: StopHold SBIN NSE -4 units in OpenAlgo's books, the engine's 0"
    assert lines["Reconcile"] == f"1 mismatch ({named})"
    assert lines["End state"] == f"not flat: a position is still open ({named})"


@pytest.mark.parametrize("filled_inside", CASES, indirect=True)
def test_a_fill_inside_the_read_is_no_difference_of_that_read(filled_inside, caplog):
    world = filled_inside
    outcome = run(world)
    after_the_sale = [(r.kind, r.engine_units, r.broker_units) for r in outcome.reconciliations]
    if world.sold:
        assert after_the_sale == [("explained_fewer", 10, 6), ("more", 0, -4)]
    else:
        assert after_the_sale == []
    assert leg_warnings(caplog) == []


@pytest.mark.parametrize("filled_inside", CASES, indirect=True)
def test_nothing_is_sent_once_the_stop_filled_inside_the_read(filled_inside):
    outcome = run(filled_inside)
    assert handed_after_the_final_read(outcome) == []


# ----- the close's second read brings no books: no read judged the leg after its fill ----

#: The cases with the stream live.
UNJUDGED = [case for case in CASES if not case.values[0][2]]


@pytest.fixture
def unjudged(request, monkeypatch):
    """The scenario with the close's second read losing its position book's connection."""
    world: Any = None

    def drop_the_next_book() -> None:
        world.fake.faults.drop_connection("positionbook")

    world = scenario(request, monkeypatch, then=drop_the_next_book)
    return world


@pytest.mark.parametrize("unjudged", UNJUDGED, indirect=True)
def test_a_leg_no_read_judged_after_its_fill_is_listed_at_the_earlier_figure_moved_by_it(
    unjudged,
):
    world = unjudged
    outcome = run(world)
    if not world.sold:
        assert (outcome.open_positions, outcome.summary.flat) == ((), True)
        return
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (-4, "", False)
    assert (
        "OpenAlgo's books showed 6 units against the engine's 10 at the last read to judge it; "
        "the engine's own fills (10 units sold) came after it, which would leave -4 units in "
        "OpenAlgo's books against the engine's 0, and the session ended before they agreed: "
        f"nothing was sent for the difference; not confirmed, since {WHY}"
    ) in left.note


@pytest.mark.parametrize("unjudged", UNJUDGED, indirect=True)
def test_the_daily_summary_never_calls_a_leg_no_read_judged_after_its_fill_flat_unqualified(
    unjudged, caplog
):
    world = unjudged
    run(world)
    lines = summary_lines(caplog)
    if world.sold:
        named = (
            "at the close: StopHold SBIN NSE -4 units in OpenAlgo's books by the last read to "
            "judge it and the engine's own fills since, the engine's 0, not confirmed, since "
            f"{WHY}"
        )
        assert lines["Reconcile"] == f"1 mismatch ({named})"
        assert lines["End state"] == f"not flat: a position is still open ({named})"
    else:
        named = f"at the close: StopHold SBIN NSE the engine's 0 units, not confirmed, since {WHY}"
        assert lines["Reconcile"] == f"0 mismatches ({named})"
        assert lines["End state"] == f"flat ({named})"


@pytest.mark.parametrize("unjudged", UNJUDGED, indirect=True)
def test_one_warning_alert_asks_to_check_openalgos_position_book(unjudged, caplog):
    world = unjudged
    run(world)
    if world.sold:
        figures = (
            "the engine holds no position against -4 units in OpenAlgo's books by the last "
            "read to judge it and the engine's own fills since"
        )
    else:
        figures = "the engine holds no position"
    assert leg_warnings(caplog) == [
        f"StopHold SBIN NSE: {figures}, not confirmed, since {WHY}: check OpenAlgo's position book"
    ]
    warned = [
        record.levelname
        for record in caplog.records
        if record.getMessage().startswith("alert reconcile_mismatch: ")
        and "position book" in record.getMessage()
    ]
    assert warned == ["WARNING"]


@pytest.mark.parametrize("unjudged", UNJUDGED, indirect=True)
def test_nothing_is_sent_for_a_leg_no_read_judged_after_its_fill(unjudged):
    outcome = run(unjudged)
    assert handed_after_the_final_read(outcome) == []


# ----- part of the stop fills inside the read, and no read judges the leg after it --------


@pytest.fixture
def part_filled_inside(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """Inline, nothing sold by hand, one Ctrl+C with on_stop="keep" at 14:43:01: right after
    the final read's position-book call the broker fills 4 of the stop's 10, and the
    close's second read loses its position book's connection; before the run."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 50))
    world.script.at(PRESSED_AT, world.stop.request)
    answer_book = account.AccountAPI.positionbook
    armed: list[bool] = []

    def position_book(api: Any, **kwargs: Any) -> Any:
        answer = answer_book(api, **kwargs)
        if armed and armed[0]:
            armed[0] = False
            stop = next(o for o in world.orders() if o.price_type == "SL-M")
            world.fake.fill(stop.order_id, 4)
            world.fake.faults.drop_connection("positionbook")
        return answer

    monkeypatch.setattr(account.AccountAPI, "positionbook", position_book)
    world.lines = LinesThen("StopHold SBIN NSE", lambda: armed.append(True))
    return world


def test_an_open_leg_no_read_judged_after_its_fill_is_listed_at_the_engines_units_unconfirmed(
    part_filled_inside, caplog
):
    world = part_filled_inside
    outcome = world.run(StopHold, on_stop="keep")
    assert world.fake.position("SBIN", "NSE") == 6
    (left,) = outcome.open_positions
    assert left.units == 6
    assert f"not confirmed, since {WHY}" in left.note
    named = f"at the close: StopHold SBIN NSE the engine's 6 units, not confirmed, since {WHY}"
    assert summary_lines(caplog)["End state"] == f"not flat: a position is still open ({named})"
    assert leg_warnings(caplog) == [
        f"StopHold SBIN NSE: the engine holds 6 units, not confirmed, since {WHY}: check "
        "OpenAlgo's position book"
    ]
