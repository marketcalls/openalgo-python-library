"""A difference the close's final read is the first to see, on the fake OpenAlgo, inline
and on the session's threads (`_live/_session_close.py`, `_live/_stops_after.py`,
`_live/reconcile.py`, `_live/_alerts_summary.py`; 16-session-design 3.3 step 5 and 5.4
"As built", round 7; unit 12).

`StopHold` holds 10 SBIN from 14:42:01 IST with an SL-M stop of 10 at 497.00. Another
client sells 4 at 14:43:00; the first comparison that shows the sale opens the exit guard
on the leg (OpenAlgo's books show 6 of the engine's 10). Then SBIN trades at 496.00,
through the stop's trigger: the SL-M for 10 fills at the broker, the engine is flat and
the account is short 4. No later read can confirm what the close's final read is the
first to see (OpenAlgo's books at -4 against the engine's 0), so the close says it: one
ERROR alert for the leg naming both figures, the result's open position at OpenAlgo's
units with no confirmed stop, and a DAILY_SUMMARY that counts and names it. Nothing is
sent for it.

- The stop fires at the first sighting and Ctrl+C comes one second later, with
  on_stop="keep" or the default flatten: the final read is the first to see the short.
- Ctrl+C with on_stop="keep" comes at the first sighting and the stop fires right after
  the leg's last line: the alert of the stop that filled after the lines names the short
  the final read saw.
- As that, but the final read brings no books (its position-book call loses its
  connection, or, on the threads, takes 8 s, past the read's seconds): the last sighting
  is the first comparison's, 6 units against the engine's 10, older than the stop's fill.
  That fill was the broker's too, so it moved OpenAlgo's books by the same 10 units: the
  close names the read's figures and the fill, lists the leg at the -4 units they leave,
  and calls no figure the engine's that it no longer holds (round 8).

When all 10 are sold by hand and one Ctrl+C with on_stop="keep" comes a second later,
the final read is the first to see OpenAlgo's books flat while the engine holds 10 and
its SL-M for 10 works at the broker: the leg is no open position, and its alert names the
stop still working with no position to close by hand (round 8). A session that closes flat
with books that agree still says "0 mismatches" and "flat". Expected values come from the
scenario and the fake's books.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import (
    StopHold,
    ctrl_c_at_the_first_sighting,
    handed_after_the_final_read,
    sold_by_hand,
    stop_world,
)
from test_session_close_lines import LinesThen
from test_session_threads import alone, passing, run_threaded  # noqa: F401

from openalgo.backtest._live._reconcile_guard import ExitGuard
from openalgo.backtest._live._types import LOGGER_NAME

#: Units sold by hand at `SOLD_AT`; the price SBIN trades at to fill the stop (its
#: trigger is 497.00); and what the account holds once it filled.
SOLD, SOLD_AT = 4, ist(14, 43)
BELOW_THE_STOP = 496.0
SHORT = StopHold.qty - SOLD - StopHold.qty
#: How the close names the leg's figures and what it did about them.
BOTH_FIGURES = (
    f"OpenAlgo's books showed {SHORT} units against the engine's 0 at the last read, and "
    "the session ended before they agreed: nothing was sent for the difference"
)
NO_STOP_CLOSE_BY_HAND = (
    "no stop at the broker; check the order book before closing it by hand, cancelling "
    "what still works"
)
#: How the DAILY_SUMMARY names the leg.
SUMMARY_WORDS = f"at the close: StopHold SBIN NSE {SHORT} units in OpenAlgo's books, the engine's 0"
#: Seconds the final read's position-book call takes in the slow case: past the close's
#: 5 s wait for that read (`_session_close.FINAL_READ_WAIT_S`).
SLOW_BOOK_S = 8.0


def fall(world: Any) -> None:
    """SBIN trades through the stop's trigger."""
    world.fake.trade("SBIN", "NSE", BELOW_THE_STOP, volume=10)


def fires_at_the_first_sighting(world: Any, monkeypatch: Any) -> None:
    """As the first comparison that shows the sale opens the exit guard, SBIN trades
    through the stop; Ctrl+C comes one second later."""
    opened = ExitGuard.open
    armed: list[bool] = []

    def open_then_fall(guard: ExitGuard, name: str, broker_units: int) -> None:
        if broker_units < StopHold.qty and not armed:
            armed.append(True)
            fall(world)
            world.script.at(world.fake.clock.now() + NS, world.stop.request)
        opened(guard, name, broker_units)

    monkeypatch.setattr(ExitGuard, "open", open_then_fall)


def fires_after_the_line(world: Any, monkeypatch: Any) -> None:
    """Ctrl+C at the first sighting; SBIN trades through the stop right after the leg's
    last line is printed."""
    ctrl_c_at_the_first_sighting(world, monkeypatch, broker_below=StopHold.qty)
    world.lines = LinesThen("StopHold SBIN NSE", lambda: fall(world))


def fires_after_the_line_with_no_books(how: str) -> Any:
    """`fires_after_the_line`, the final read's position book then losing its connection
    ("dropped") or answering in `SLOW_BOOK_S` ("slow")."""

    def arm(world: Any, monkeypatch: Any) -> None:
        ctrl_c_at_the_first_sighting(world, monkeypatch, broker_below=StopHold.qty)

        def fall_then_no_books() -> None:
            fall(world)
            if how == "dropped":
                world.fake.faults.drop_connection("positionbook")
            else:
                world.fake.faults.slow("positionbook", SLOW_BOOK_S)

        world.lines = LinesThen("StopHold SBIN NSE", fall_then_no_books)

    return arm


#: The cases: when the stop fires, on which seam, and what Ctrl+C does with the leg.
CASES = [
    pytest.param((fires_at_the_first_sighting, "inline", "keep"), id="first sighting, keep"),
    pytest.param(
        (fires_at_the_first_sighting, "threads", "keep"), id="first sighting, keep, threads"
    ),
    pytest.param((fires_at_the_first_sighting, "inline", "flatten"), id="first sighting, flatten"),
    pytest.param((fires_after_the_line, "inline", "keep"), id="after the line, keep"),
    pytest.param((fires_after_the_line, "threads", "keep"), id="after the line, keep, threads"),
]


@pytest.fixture
def short(request, monkeypatch):
    """The scenario of the module docstring for the case `request.param`, before the run:
    the world, with the seam and Ctrl+C's on_stop it is run with (`run`)."""
    arm, seam, on_stop = request.param
    request.getfixturevalue("caplog").set_level(logging.INFO, logger=LOGGER_NAME)
    if seam == "inline":
        fake = request.getfixturevalue("make_fake_openalgo")
        world = stop_world(fake, monkeypatch, request.getfixturevalue("tmp_path"))
        world.stop_at(ist(14, 50))
    else:
        world = request.getfixturevalue("alone")
    world.seam, world.on_stop = seam, on_stop
    sold_by_hand(world, SOLD, SOLD_AT)
    arm(world, monkeypatch)
    return world


def run(world: Any) -> Any:
    """Run `StopHold` on the world's seam with its on_stop; the session's outcome."""
    if world.seam == "inline":
        return world.run(StopHold, on_stop=world.on_stop)
    return run_threaded(world, passing(world), "orders", (StopHold,), on_stop=world.on_stop)


def leg_alerts(caplog: Any) -> list[str]:
    """The ERROR alerts that name the leg, without the mode and strategy they begin with
    ("LIVE (StopHold): ")."""
    named = [text for text in alerts(caplog, "error") if "StopHold SBIN NSE" in text]
    return [text.split(": ", 1)[1] for text in named]


def summary_lines(caplog: Any) -> dict[str, str]:
    """The DAILY_SUMMARY's lines by label."""
    (text,) = alerts(caplog, "daily_summary")
    pairs = [line.split(" : ", 1) for line in text.splitlines()[1:]]
    return {label.strip(): value for label, value in pairs}


@pytest.mark.parametrize("short", CASES, indirect=True)
def test_the_stop_for_10_fills_and_leaves_the_account_short_4(short):
    world = short
    run(world)
    stops = [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == [(StopHold.qty, "complete")]
    assert world.fake.position("SBIN", "NSE") == SHORT


@pytest.mark.parametrize("short", CASES[:3], indirect=True)
def test_the_close_alerts_the_short_once_naming_both_figures(short, caplog):
    run(short)
    assert leg_alerts(caplog) == [f"StopHold SBIN NSE: {BOTH_FIGURES}; {NO_STOP_CLOSE_BY_HAND}"]


@pytest.mark.parametrize("short", CASES[3:], indirect=True)
def test_the_alert_of_the_stop_that_filled_after_the_lines_names_the_short(short, caplog):
    world = short
    run(world)
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    *_, last = leg_alerts(caplog)
    assert last == (
        f"StopHold SBIN NSE: its stop {stop.order_id} filled at the broker after the "
        f"session's last lines; {BOTH_FIGURES}; {NO_STOP_CLOSE_BY_HAND}"
    )


@pytest.mark.parametrize("short", CASES, indirect=True)
def test_the_result_lists_the_short_at_openalgos_units_with_no_confirmed_stop(short):
    outcome = run(short)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (SHORT, "", False)
    assert left.note.endswith(f"{BOTH_FIGURES}; {NO_STOP_CLOSE_BY_HAND}")
    assert outcome.summary.flat is False


@pytest.mark.parametrize("short", CASES, indirect=True)
def test_the_daily_summary_counts_and_names_the_short(short, caplog):
    run(short)
    lines = summary_lines(caplog)
    assert lines["Reconcile"] == f"1 mismatch ({SUMMARY_WORDS})"
    assert lines["End state"] == f"not flat: a position is still open ({SUMMARY_WORDS})"


@pytest.mark.parametrize("short", CASES, indirect=True)
def test_nothing_is_sent_for_the_short(short):
    outcome = run(short)
    assert handed_after_the_final_read(outcome) == []


# ----- the final read brings no books: the sighting is older than the stop's fill -------

#: The cases: how the final read brings no books, on which seam.
STALE = [
    pytest.param((fires_after_the_line_with_no_books("dropped"), "inline", "keep"), id="dropped"),
    pytest.param(
        (fires_after_the_line_with_no_books("dropped"), "threads", "keep"),
        id="dropped, threads",
    ),
    pytest.param(
        (fires_after_the_line_with_no_books("slow"), "threads", "keep"), id="slow, threads"
    ),
]
#: How the close names the first comparison's figures, the stop's fill after them, and the
#: figures that fill leaves.
STALE_FIGURES = (
    "OpenAlgo's books showed 6 units against the engine's 10 at the last read; the engine's "
    f"own fills (10 units sold) came after it, which would leave {SHORT} units in OpenAlgo's "
    "books against the engine's 0, and the session ended before they agreed: nothing was "
    "sent for the difference"
)
#: How the DAILY_SUMMARY names the leg then.
STALE_SUMMARY_WORDS = (
    f"at the close: StopHold SBIN NSE {SHORT} units in OpenAlgo's books by the last read and "
    "the engine's own fills since, the engine's 0"
)


def stale_note(world: Any) -> str:
    """The note of the leg once the stop's fill came after the last sighting."""
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    return (
        f"its stop {stop.order_id} filled at the broker after the session's last lines; "
        f"{STALE_FIGURES}; {NO_STOP_CLOSE_BY_HAND}"
    )


@pytest.mark.parametrize("short", STALE, indirect=True)
def test_with_no_books_at_the_close_the_stop_for_10_still_leaves_the_account_short_4(short):
    world = short
    run(world)
    stops = [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == [(StopHold.qty, "complete")]
    assert world.fake.position("SBIN", "NSE") == SHORT


@pytest.mark.parametrize("short", STALE, indirect=True)
def test_a_sighting_older_than_the_stops_fill_lists_the_short_that_fill_leaves(short):
    world = short
    outcome = run(world)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (SHORT, "", False)
    assert left.note == stale_note(world)
    assert outcome.summary.flat is False


@pytest.mark.parametrize("short", STALE, indirect=True)
def test_the_alert_of_a_sighting_older_than_the_stops_fill_names_the_read_and_the_fill(
    short, caplog
):
    world = short
    run(world)
    *_, last = leg_alerts(caplog)
    assert last == f"StopHold SBIN NSE: {stale_note(world)}"


@pytest.mark.parametrize("short", STALE, indirect=True)
def test_the_daily_summary_names_the_short_the_fill_leaves_against_the_engines_units_now(
    short, caplog
):
    run(short)
    lines = summary_lines(caplog)
    assert lines["Reconcile"] == f"1 mismatch ({STALE_SUMMARY_WORDS})"
    assert lines["End state"] == f"not flat: a position is still open ({STALE_SUMMARY_WORDS})"


@pytest.mark.parametrize("short", STALE, indirect=True)
def test_nothing_is_sent_for_a_sighting_older_than_the_stops_fill(short):
    outcome = run(short)
    assert handed_after_the_final_read(outcome) == []


# ----- OpenAlgo's books flat at the first sighting, the stop still working --------------

#: When one Ctrl+C with on_stop="keep" comes once all 10 are sold by hand.
PRESSED_AT = ist(14, 43, 1)
#: How the close names the leg's figures, and what it asks of the operator.
FLAT_FIGURES = (
    "OpenAlgo's books showed no position against the engine's 10 at the last read, and the "
    "session ended before they agreed: nothing was sent for the difference"
)
NOTHING_TO_CLOSE = (
    "no position is left to close by hand: check the order book in OpenAlgo, cancelling "
    "what still works"
)
#: How the DAILY_SUMMARY names the leg.
FLAT_SUMMARY_WORDS = (
    "at the close: StopHold SBIN NSE no position in OpenAlgo's books, the engine's 10"
)


@pytest.fixture(params=["inline", "threads"])
def all_sold(request, monkeypatch):
    """All 10 SBIN sold by hand at 14:43:00 and one Ctrl+C with on_stop="keep" at
    14:43:01, so the close's final read is the first to see OpenAlgo's books flat; before
    the run."""
    request.getfixturevalue("caplog").set_level(logging.INFO, logger=LOGGER_NAME)
    if request.param == "inline":
        fake = request.getfixturevalue("make_fake_openalgo")
        world = stop_world(fake, monkeypatch, request.getfixturevalue("tmp_path"))
        world.stop_at(ist(14, 50))
    else:
        world = request.getfixturevalue("alone")
    world.seam, world.on_stop = request.param, "keep"
    sold_by_hand(world, StopHold.qty, SOLD_AT)
    world.script.at(PRESSED_AT, world.stop.request)
    return world


def test_a_leg_openalgo_shows_flat_keeps_its_stop_at_the_broker_and_nothing_is_sent(all_sold):
    world = all_sold
    outcome = run(world)
    stops = [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == [(StopHold.qty, "trigger pending")]
    assert world.fake.position("SBIN", "NSE") == 0
    assert handed_after_the_final_read(outcome) == []


def test_a_leg_openalgo_shows_flat_is_no_open_position(all_sold):
    outcome = run(all_sold)
    assert (outcome.open_positions, outcome.summary.flat) == ((), True)


def test_the_alert_of_a_leg_openalgo_shows_flat_names_the_stop_and_nothing_to_close(
    all_sold, caplog
):
    world = all_sold
    run(world)
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    assert leg_alerts(caplog) == [
        f"StopHold SBIN NSE: {FLAT_FIGURES}; stop {stop.order_id} confirmed at the broker "
        f"for 10 units; {NOTHING_TO_CLOSE}"
    ]


def test_the_daily_summary_of_a_leg_openalgo_shows_flat_says_flat_and_names_it(all_sold, caplog):
    run(all_sold)
    lines = summary_lines(caplog)
    assert lines["Reconcile"] == f"1 mismatch ({FLAT_SUMMARY_WORDS})"
    assert lines["End state"] == f"flat ({FLAT_SUMMARY_WORDS})"


# ----- a clean close --------------------------------------------------------------------


@pytest.fixture(params=["inline", "threads"])
def clean(request, monkeypatch):
    """`StopHold` to be stopped with Ctrl+C (flatten) at 14:46 with no sale by hand: the
    flatten closes the leg and the books agree; before the run."""
    request.getfixturevalue("caplog").set_level(logging.INFO, logger=LOGGER_NAME)
    if request.param == "inline":
        fake = request.getfixturevalue("make_fake_openalgo")
        world = stop_world(fake, monkeypatch, request.getfixturevalue("tmp_path"))
        world.stop_at(ist(14, 46))
    else:
        world = request.getfixturevalue("alone")
        world.script.at(ist(14, 46), world.stop.request)
    world.seam, world.on_stop = request.param, "flatten"
    return world


def test_a_clean_close_still_says_flat_and_0_mismatches(clean, caplog):
    outcome = run(clean)
    assert (outcome.open_positions, outcome.summary.flat) == ((), True)
    lines = summary_lines(caplog)
    assert (lines["Reconcile"], lines["End state"]) == ("0 mismatches", "flat")
    assert leg_alerts(caplog) == []
