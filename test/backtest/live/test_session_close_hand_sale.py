"""A sale by hand the close does not settle at the broker, on the fake OpenAlgo, inline and
on the session's threads (`_live/_stops_report.py`, `_live/_stops_after.py`,
`_live/_reconcile_rules.py`; 16-session-design 3.3 step 5 "As built", 5.3, 5.4; unit 12).

`StopHold` holds 10 SBIN from 14:42:01 IST with an SL-M stop of 10 at 497.00. Another
client sells 4 at 14:43:00, and Ctrl+C with on_stop="keep" comes as the first comparison
that shows the sale: the exit guard holds the leg (OpenAlgo's books show 6, the engine
counts 10) when the leg's last line is printed, and the close's final read is the second
comparison to show the sale, the one that would adopt it. Either way below, the broker's
SL-M for 10 works on a position of 6, so a trigger would leave the account short 4: no
line, note or result may say that stop protects the position.

- The final read brings no books: right after the leg's line its position-book call loses
  its connection, or, on the threads, takes 8 s, past the read's seconds. The guard still
  holds the leg at the close, the first sighting of the sale was never confirmed, and so
  the result lists the leg at OpenAlgo's units with no confirmed stop and one ERROR alert
  names both figures and the stop (16-session-design 5.4 "As built", round 7).
- The final read adopts the sale, but the stop's cut to 6 never reaches the broker: the
  exit guard keeps it (standing for any reason it is held after the last lines), its
  connection drops, or, inline, the books come after the read's seconds, so the cut is
  handed to the order worker as it closes. The RECONCILE_MISMATCH alert says the cut was
  asked for, not that it was made.

Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from session_trace import ANSWERED
from stops_samples import StopHold, ctrl_c_at_the_first_sighting, sold_by_hand, stop_world
from test_session_close_lines import LinesThen
from test_session_threads import alone, passing, run_threaded  # noqa: F401

from openalgo.backtest._live._reconcile_guard import ExitGuard
from openalgo.backtest._live._types import LOGGER_NAME

#: Units sold by hand at `SOLD_AT`, and what the broker holds then.
SOLD, SOLD_AT = 4, ist(14, 43)
LEFT = StopHold.qty - SOLD
#: Seconds the final read's position-book call takes in the slow cases: past the close's
#: 5 s wait for that read (`_session_close.FINAL_READ_WAIT_S`).
SLOW_BOOK_S = 8.0
#: Why a test's cut of the stop has no answer when the session ends, which the session
#: invariants take instead of an answered modify (16-session-design 3.3 step 5 "As built").
CUT_UNANSWERED = "3.3 step 5 As built: the stop's cut is not sent or not answered by the close"
#: The end of a note whose confirmed stop covers the whole position.
PROTECTS = "it protects the position, so cancel the stop before any sale by hand"
#: How a line names the broker's units the exit guard knows of the leg.
BOOKS_SHOWED_6 = "OpenAlgo's books showed 6 units at the last read"
#: How the close names both figures of a difference no later read confirmed.
BOTH_FIGURES = (
    "OpenAlgo's books showed 6 units against the engine's 10 at the last read, and the "
    "session ended before they agreed: nothing was sent for the difference"
)


def scenario(seam: str, request: Any, monkeypatch: Any) -> Any:
    """The scenario of the module docstring on `seam` ("inline" or "threads"), before the
    run: the sale by hand scheduled and Ctrl+C armed for its first sighting."""
    request.getfixturevalue("caplog").set_level(logging.INFO, logger=LOGGER_NAME)
    if seam == "inline":
        fake = request.getfixturevalue("make_fake_openalgo")
        made = stop_world(fake, monkeypatch, request.getfixturevalue("tmp_path"))
    else:
        made = request.getfixturevalue("alone")
    made.seam = seam
    sold_by_hand(made, SOLD, SOLD_AT)
    ctrl_c_at_the_first_sighting(made, monkeypatch, broker_below=StopHold.qty)
    return made


@pytest.fixture(params=["inline", "threads"])
def world(request, monkeypatch):
    """The scenario on the seam the parameter names, before the run."""
    return scenario(request.param, request, monkeypatch)


def run(world: Any) -> Any:
    """Run `StopHold` with on_stop="keep" on the world's seam."""
    if world.seam == "inline":
        return world.run(StopHold, on_stop="keep")
    return run_threaded(world, passing(world), "orders", (StopHold,), on_stop="keep")


def after_the_line(world: Any, action: Callable[[], None]) -> None:
    """Run `action` right after the leg's last line is printed."""
    world.lines = LinesThen("StopHold SBIN NSE", action)


def stops(world: Any) -> list[tuple[int, str]]:
    """Each SL-M stop at the fake: its units and status."""
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


def the_line(world: Any) -> str:
    (line,) = [line for line in world.lines if line.startswith("StopHold SBIN NSE")]
    return line


# ----- the final read brings no books: the guard holds the leg at the close --------------


def no_books(world: Any, case: str) -> None:
    """The final read's position book loses its connection, or answers in `SLOW_BOOK_S`."""
    faults = world.fake.faults
    if case == "dropped":
        after_the_line(world, lambda: faults.drop_connection("positionbook"))
    else:
        after_the_line(world, lambda: faults.slow("positionbook", SLOW_BOOK_S))


#: The ways the final read brings no books, by seam: inline, a slow position book answers
#: within the same loop step, so the read adopts the sale (the cases further down).
NO_BOOKS = [
    pytest.param(("inline", "dropped"), id="inline, dropped"),
    pytest.param(("threads", "dropped"), id="threads, dropped"),
    pytest.param(("threads", "slow"), id="threads, slow"),
]


@pytest.fixture
def unsettled(request, monkeypatch):
    """The scenario on the seam `request.param[0]`, the final read bringing no books in
    the way `request.param[1]` names, before the run."""
    seam, case = request.param
    made = scenario(seam, request, monkeypatch)
    no_books(made, case)
    return made


@pytest.mark.parametrize("unsettled", NO_BOOKS, indirect=True)
def test_the_broker_keeps_the_stop_for_10_on_a_position_of_6(unsettled):
    world = unsettled
    run(world)
    assert stops(world) == [(StopHold.qty, "trigger pending")]
    assert world.fake.position("SBIN", "NSE") == LEFT


@pytest.mark.parametrize("unsettled", NO_BOOKS, indirect=True)
def test_the_last_line_of_a_leg_the_guard_holds_names_the_books_units_not_a_protection(
    unsettled,
):
    world = unsettled
    run(world)
    line = the_line(world)
    assert PROTECTS not in line
    assert BOOKS_SHOWED_6 in line


@pytest.mark.parametrize("unsettled", NO_BOOKS, indirect=True)
def test_the_result_lists_a_leg_the_guard_holds_at_openalgos_units_and_no_protection(unsettled):
    outcome = run(unsettled)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (LEFT, False)
    assert PROTECTS not in left.note
    assert left.note.startswith(BOTH_FIGURES)


@pytest.mark.parametrize("unsettled", NO_BOOKS, indirect=True)
def test_a_leg_the_guard_holds_at_the_close_is_one_error_alert_naming_both_figures(
    unsettled, caplog
):
    world = unsettled
    outcome = run(world)
    (left,) = outcome.open_positions
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    named = [text for text in alerts(caplog, "error") if BOTH_FIGURES in text]
    assert [text.split(": ", 1)[1] for text in named] == [f"StopHold SBIN NSE: {left.note}"]
    assert left.note == (
        f"{BOTH_FIGURES}; stop {stop.order_id} confirmed at the broker for 10 units; check "
        "the order book before closing it by hand, cancelling what still works"
    )


# ----- the final read adopts the sale, and the stop's cut never reaches the broker -------


def cut_held(world: Any, monkeypatch: Any) -> None:
    """The exit guard keeps the stop's cut after the last lines."""
    monkeypatch.setattr(ExitGuard, "_takes_away", lambda guard, request: False)
    world.trace.waive(ANSWERED, CUT_UNANSWERED)


def cut_dropped(world: Any, monkeypatch: Any) -> None:
    """The stop's cut loses its connection (the session's first modify)."""
    del monkeypatch
    world.fake.faults.drop_connection("modifyorder")
    world.trace.waive(ANSWERED, CUT_UNANSWERED)


@pytest.mark.parametrize("unsent", [cut_held, cut_dropped], ids=["held", "dropped"])
def test_a_cut_not_sent_leaves_the_stop_for_10_at_the_broker(world, monkeypatch, unsent):
    unsent(world, monkeypatch)
    run(world)
    assert stops(world) == [(StopHold.qty, "trigger pending")]
    assert world.fake.position("SBIN", "NSE") == LEFT


@pytest.mark.parametrize("unsent", [cut_held, cut_dropped], ids=["held", "dropped"])
def test_a_stop_whose_cut_was_not_sent_is_never_said_to_protect_what_is_left(
    world, monkeypatch, unsent
):
    unsent(world, monkeypatch)
    outcome = run(world)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (LEFT, False)
    assert PROTECTS not in left.note
    assert "confirmed at the broker for 10 units, more than the 6 held" in left.note


@pytest.fixture
def adopted_late(request, monkeypatch):
    """Inline, the final read's position book answers in `SLOW_BOOK_S`: the read adopts
    the sale once its seconds have run out, and the order worker closes before the cut it
    hands over is sent; before the run."""
    made = scenario("inline", request, monkeypatch)
    no_books(made, "slow")
    made.trace.waive(ANSWERED, CUT_UNANSWERED)
    return made


def test_a_cut_asked_for_as_the_final_read_ends_is_not_sent(adopted_late):
    world = adopted_late
    run(world)
    assert stops(world) == [(StopHold.qty, "trigger pending")]
    assert world.fake.position("SBIN", "NSE") == LEFT


def test_the_mismatch_alert_says_the_cut_is_asked_for_not_made(adopted_late, caplog):
    run(adopted_late)
    (text,) = alerts(caplog, "reconcile_mismatch")
    assert (
        "trades of other orders reduced it at the broker: the engine takes them as its own "
        "and asks for a stop of the leg to be cut down to what is left." in text
    )


def test_a_stop_whose_cut_came_too_late_is_never_said_to_protect_what_is_left(adopted_late):
    outcome = run(adopted_late)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (LEFT, False)
    assert PROTECTS not in left.note
