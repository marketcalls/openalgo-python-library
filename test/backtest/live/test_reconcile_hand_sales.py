"""A sale by hand the reconciler adopts while the exit guard holds the cancel of the leg's
stop, on the fake OpenAlgo, inline and on the session's threads (`_live/_reconcile_act.py`,
`_live/_reconcile_guard.py`, `_live/_stops_report.py`; 16-session-design 3.3 step 5 "As
built", 5.3, 5.4; unit 12).

`StopHold` holds 10 SBIN from 14:42:01 IST with an SL-M stop of 10 at 497.00 (`TargetHold`
also a target at 502.00), and another client sells 10, or 4, at 14:43:00. An exit of the
leg (the flatten's, or the target's) cancels the stop first, and the exit guard holds that
cancel while the books show fewer units than the engine. Once the reconciler adopts the
sale, the stop is fitted to what is left at once: with nothing left it is cancelled; with
6 left it is cancelled and placed again for 6 before the session's last lines (protect by
state), and cut to 6 after them (16-session-design 3.3 step 5 "As built", round 7): the
held cancel is withdrawn, never sent, and the kernel's fit with the adoption cuts the stop
by a modify that only lowers its units, so the 6 left keep a stop. Only once the broker
confirmed the cut does a note say the stop protects them; a cut the broker refuses, or
that has no answer by the close, is named by the ERROR alert. The RECONCILE_MISMATCH alert
says what became of the stop, and no note, result or alert says that a stop larger than
the position protects it.

Three ways the cancel comes to be held when the sale is adopted:

- two Ctrl+C at once at 14:43:01, before any read has seen the sale: the last lines come
  first, and the close's final read adopts it;
- two Ctrl+C at once as the first comparison that shows the sale comes: the flatten's own
  read adopts it, before the last lines;
- with on_stop="keep", SBIN crossing the target right after that first comparison and one
  Ctrl+C two seconds later: the target's exit asked for the cancel, and the close's final
  read adopts the sale.

On the threads seam the adoption may fall on either side of the last lines, so those
tests check the end state either side allows. Expected values come from the scenario and
the fake's books.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fake_openalgo import rest as fake_rest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist, rig
from stops_samples import (
    StopHold,
    TargetHold,
    after_the_final_read,
    ctrl_c_at_the_first_sighting,
    sold_by_hand,
    stop_world,
)
from test_session_threads import passing, run_threaded

from openalgo.backtest._live import _reconcile_act as reconcile_act
from openalgo.backtest._live._reconcile_act import NOT_ADOPTED_WORDS
from openalgo.backtest._live._reconcile_guard import ExitGuard
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import VenueRefusalError

#: When the other client sells, and when two Ctrl+C come at once in the first way.
SOLD_AT = ist(14, 43)
PRESSED_AT = ist(14, 43, 1)
#: A price past `TargetHold`'s target, and how long after the first comparison that shows
#: the sale Ctrl+C comes once SBIN trades there, seconds.
ABOVE_TARGET = 503.0
KEEP_AFTER_S = 2
#: The end of a note whose confirmed stop covers the whole position.
PROTECTS = "it protects the position, so cancel the stop before any sale by hand"
#: What a RECONCILE_MISMATCH alert says the engine asks of the stop, by what is left.
CANCELLED_WITH_THEM = "asks for a stop of the leg to be cancelled with them"
CANCELLED_NOT_CUT = (
    "asks for the stop of the leg to be cancelled (an exit had asked for its cancel, held "
    "back until the books agreed)"
)
CUT_DOWN = "asks for a stop of the leg to be cut down to what is left"
#: What a flat leg's alert asks of the operator.
FLAT_NOW = (
    "the engine holds no position now: check the order book in OpenAlgo, cancelling what "
    "still works"
)

Scenario = Callable[[Any, Any], tuple[type, dict[str, Any]]]


def before_any_read(world: Any, monkeypatch: Any) -> tuple[type, dict[str, Any]]:
    """Two Ctrl+C at once at 14:43:01 (flatten)."""
    del monkeypatch

    def press_twice() -> None:
        world.stop.request()
        world.stop.request()

    world.script.at(PRESSED_AT, press_twice)
    return StopHold, {}


def at_the_first_sighting(world: Any, monkeypatch: Any) -> tuple[type, dict[str, Any]]:
    """Two Ctrl+C at once as the first comparison that shows the sale comes (flatten)."""
    ctrl_c_at_the_first_sighting(world, monkeypatch, broker_below=StopHold.qty, presses=2)
    return StopHold, {}


def target_between_the_sightings(world: Any, monkeypatch: Any) -> tuple[type, dict[str, Any]]:
    """As the first comparison that shows the sale comes, SBIN trades past the target from
    then on; one Ctrl+C `KEEP_AFTER_S` later, with on_stop="keep"."""
    opened = ExitGuard.open
    crossed: list[int] = []

    def open_then_cross(guard: ExitGuard, name: str, broker_units: int) -> None:
        if broker_units < StopHold.qty and not crossed:
            crossed.append(broker_units)
            now = world.fake.clock.now()
            world.script.price = lambda now_ns: ABOVE_TARGET
            world.script.at(
                now + 1, lambda: world.fake.trade("SBIN", "NSE", ABOVE_TARGET, volume=10.0)
            )
            world.script.at(now + KEEP_AFTER_S * NS, world.stop.request)
        opened(guard, name, broker_units)

    monkeypatch.setattr(ExitGuard, "open", open_then_cross)
    return TargetHold, {"on_stop": "keep"}


#: The scenarios whose adoption comes after the last lines on the inline seam.
AFTER_THE_LINES = [before_any_read, target_between_the_sightings]
AFTER_IDS = ["two Ctrl+C before any read", "target crossed, then keep"]


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    made.stop_at(ist(14, 50))
    return made


def play(world: Any, monkeypatch: Any, scenario: Scenario, sold: int) -> Any:
    """Run `scenario` on the inline seam with `sold` SBIN sold by hand at 14:43:00."""
    sold_by_hand(world, sold, SOLD_AT)
    strategy, arguments = scenario(world, monkeypatch)
    return world.run(strategy, **arguments)


def stops(world: Any) -> list[tuple[int, str]]:
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


def first_stop(world: Any) -> Any:
    return next(o for o in world.orders() if o.price_type == "SL-M")


def after_the_lines(caplog: Any) -> list[str]:
    """The ERROR alerts that say what happened after the session's last lines, without the
    mode and strategy they begin with ("LIVE (StopHold): ")."""
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    return [text.split(": ", 1)[1] for text in named]


def errors_after_the_adoption(caplog: Any) -> list[str]:
    """The ERROR alerts sent after the RECONCILE_MISMATCH alert of the adoption: a line
    printed before it, while the exit guard held the leg, names the stop beside the units
    OpenAlgo's books showed."""
    texts = [record.getMessage() for record in caplog.records if record.name == LOGGER_NAME]
    adopted = max(
        i for i, text in enumerate(texts) if text.startswith("alert reconcile_mismatch: ")
    )
    tag = "alert error: "
    return [text[len(tag) :] for text in texts[adopted:] if text.startswith(tag)]


# ----- inline: the stop at the broker ---------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "sold", "stops_left"),
    [
        (before_any_read, 10, [(10, "cancelled")]),
        (before_any_read, 4, [(6, "trigger pending")]),
        (at_the_first_sighting, 10, [(10, "cancelled")]),
        (at_the_first_sighting, 4, [(10, "cancelled"), (6, "trigger pending")]),
        (target_between_the_sightings, 10, [(10, "cancelled")]),
        (target_between_the_sightings, 4, [(6, "trigger pending")]),
    ],
    ids=[
        "two Ctrl+C before any read, all sold",
        "two Ctrl+C before any read, 4 sold: cut to 6",
        "two Ctrl+C at the first sighting, all sold",
        "two Ctrl+C at the first sighting, 4 sold: placed again for 6",
        "target crossed, then keep, all sold",
        "target crossed, then keep, 4 sold: cut to 6",
    ],
)
def test_a_sale_by_hand_adopted_while_the_stops_cancel_is_held_fits_the_stop_at_the_broker(
    world, monkeypatch, scenario, sold, stops_left
):
    play(world, monkeypatch, scenario, sold)
    assert stops(world) == stops_left
    assert world.fake.position("SBIN", "NSE") == StopHold.qty - sold


# ----- inline: what the operator is told --------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "sold", "words"),
    [
        (before_any_read, 10, CANCELLED_WITH_THEM),
        (before_any_read, 4, CUT_DOWN),
        (at_the_first_sighting, 10, CANCELLED_WITH_THEM),
        (at_the_first_sighting, 4, CANCELLED_NOT_CUT),
        (target_between_the_sightings, 10, CANCELLED_WITH_THEM),
        (target_between_the_sightings, 4, CUT_DOWN),
    ],
    ids=[
        "two Ctrl+C before any read, all sold",
        "two Ctrl+C before any read, 4 sold",
        "two Ctrl+C at the first sighting, all sold",
        "two Ctrl+C at the first sighting, 4 sold",
        "target crossed, then keep, all sold",
        "target crossed, then keep, 4 sold",
    ],
)
def test_the_mismatch_alert_asks_for_the_stops_cancel_or_its_cut(
    world, monkeypatch, caplog, scenario, sold, words
):
    play(world, monkeypatch, scenario, sold)
    (text,) = alerts(caplog, "reconcile_mismatch")
    assert words in text


@pytest.mark.parametrize("scenario", AFTER_THE_LINES, ids=AFTER_IDS)
def test_a_sale_of_all_adopted_after_the_last_lines_is_one_alert_naming_the_cancelled_stop(
    world, monkeypatch, caplog, scenario
):
    outcome = play(world, monkeypatch, scenario, 10)
    assert outcome.open_positions == ()
    strategy = outcome.strategies[0]
    assert after_the_lines(caplog) == [
        f"{strategy} SBIN NSE: its position went from 10 to 0 units after the session's last "
        f"lines, and its stop {first_stop(world).order_id} cancelled at the broker; {FLAT_NOW}"
    ]


@pytest.mark.parametrize(
    ("scenario", "unsent"),
    [(before_any_read, ""), (target_between_the_sightings, "order o3 not sent; ")],
    ids=AFTER_IDS,
)
def test_a_sale_of_part_adopted_after_the_last_lines_cuts_the_stop_and_says_it_protects_the_rest(
    world, monkeypatch, caplog, scenario, unsent
):
    # The exit that asked for the stop's cancel waits in the engine, never sent: the
    # flatten's is no order yet, the target's (o3) is named as not sent. The stop cut to 6
    # at the broker protects the 6 units left.
    outcome = play(world, monkeypatch, scenario, 4)
    (left,) = outcome.open_positions
    stop = first_stop(world).order_id
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (6, stop, True)
    assert left.note == (
        f"its position went from 10 to 6 units after the session's last lines; {unsent}"
        f"stop {stop} confirmed at the broker: {PROTECTS}"
    )
    assert after_the_lines(caplog) == [f"{left.strategy} SBIN NSE 6 units: {left.note}"]


def test_a_sale_of_part_adopted_before_the_last_lines_leaves_the_rest_with_a_stop_for_it(
    world, monkeypatch
):
    outcome = play(world, monkeypatch, at_the_first_sighting, 4)
    placed_again = [o for o in world.orders() if o.price_type == "SL-M"][-1]
    (left,) = outcome.open_positions
    assert (placed_again.quantity, placed_again.status) == (6, "trigger pending")
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (6, placed_again.order_id, True)
    note = f"stop {placed_again.order_id} confirmed at the broker: {PROTECTS}"
    assert left.note == note
    assert [line for line in world.lines if line.startswith("StopHold SBIN")] == [
        f"StopHold SBIN NSE 6 units: {note}"
    ]


def test_a_sale_of_all_adopted_before_the_last_lines_leaves_nothing_to_tell(
    world, monkeypatch, caplog
):
    # The stop is cancelled before the last lines: no order of the flat leg works on.
    outcome = play(world, monkeypatch, at_the_first_sighting, 10)
    assert outcome.open_positions == ()
    assert [text for text in alerts(caplog, "error") if "StopHold SBIN NSE" in text] == []
    assert [line for line in world.lines if line.startswith("StopHold SBIN")] == []


@pytest.mark.parametrize(
    ("scenario", "sold"),
    [
        (before_any_read, 10),
        (at_the_first_sighting, 10),
        (at_the_first_sighting, 4),
        (target_between_the_sightings, 10),
    ],
    ids=[
        "two Ctrl+C before any read, all sold",
        "two Ctrl+C at the first sighting, all sold",
        "two Ctrl+C at the first sighting, 4 sold",
        "target crossed, then keep, all sold",
    ],
)
def test_no_alert_or_note_names_the_first_stop_as_confirmed_once_the_sale_is_adopted(
    world, monkeypatch, caplog, scenario, sold
):
    # The stop of 10 is cancelled with the adoption: no ERROR alert and no note of the
    # result may say it is confirmed at the broker (and so protects what is left).
    outcome = play(world, monkeypatch, scenario, sold)
    claim = f"stop {first_stop(world).order_id} confirmed at the broker"
    texts = errors_after_the_adoption(caplog) + [left.note for left in outcome.open_positions]
    assert [text for text in texts if claim in text] == []


# ----- on the session's threads -------------------------------------------------------------


@pytest.fixture
def threaded(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """A rig for the session's threads (`test_session_threads.run_threaded`), started at
    14:40:05 IST."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 5))


def play_threaded(world: Any, monkeypatch: Any, scenario: Scenario, sold: int) -> Any:
    """Run `scenario` on the threads seam with `sold` SBIN sold by hand at 14:43:00."""
    sold_by_hand(world, sold, SOLD_AT)
    world.stop_at(ist(14, 50))
    strategy, arguments = scenario(world, monkeypatch)
    return run_threaded(world, passing(world), "orders", (strategy,), **arguments)


@pytest.mark.parametrize("sold", [10, 4])
@pytest.mark.parametrize(
    "scenario",
    [before_any_read, at_the_first_sighting, target_between_the_sightings],
    ids=["two Ctrl+C before any read", "two Ctrl+C at the first sighting", *AFTER_IDS[1:]],
)
def test_on_threads_the_stop_ends_fitted_to_what_is_left_and_the_result_says_so(
    threaded, monkeypatch, caplog, scenario, sold
):
    # Either side of the last lines, the stop of 10 ends cancelled, or cut to what is left
    # after them; what is left has a stop of exactly its units that the result says
    # protects it, or no stop at all and a note that says so; and the mismatch alert asks
    # for the stop's cancel, or its cut.
    outcome = play_threaded(threaded, monkeypatch, scenario, sold)
    left_units = StopHold.qty - sold
    stops_now = [o for o in threaded.orders() if o.price_type == "SL-M"]
    working = [o for o in stops_now if o.status == "trigger pending"]
    assert stops_now[0].status in ("cancelled", "trigger pending")
    assert threaded.fake.position("SBIN", "NSE") == left_units
    assert [o.quantity for o in working] in ([], [left_units] if left_units else [])
    (mismatch,) = alerts(caplog, "reconcile_mismatch")
    assert "of the leg to be cancelled" in mismatch or CUT_DOWN in mismatch
    if not left_units:
        assert outcome.open_positions == ()
        return
    (left,) = outcome.open_positions
    if working:
        assert (left.stop_order_id, left.stop_confirmed) == (working[0].order_id, True)
        assert left.note.endswith(PROTECTS)
    else:
        assert (left.stop_order_id, left.stop_confirmed) == ("", False)
        assert left.note.endswith("no stop at the broker, close it by hand")


# ----- the probe's case: 4 sold, two Ctrl+C before any read, the cut at the broker ---------

#: What the broker answers a modify of the stop when a test has it refuse the cut.
REFUSED_WORDS = "Order cannot be modified"


def refusing_modifies(monkeypatch: Any) -> None:
    """The fake broker refuses every modify (HTTP 400 with its words)."""

    def refuse(venue: Any, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        del venue, payload
        return {"status": "error", "message": REFUSED_WORDS}, 400

    monkeypatch.setitem(fake_rest.ROUTES, "modifyorder", refuse)


@pytest.fixture(params=["inline", "threads"])
def cut_world(request, monkeypatch, caplog):
    """The probe's case on the seam the parameter names, before the run: 4 SBIN sold by
    hand at 14:43:00, two Ctrl+C at once at 14:43:01, the close's final read adopting the
    sale after the last lines."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    fake, tmp_path = (
        request.getfixturevalue("make_fake_openalgo"),
        request.getfixturevalue("tmp_path"),
    )
    if request.param == "inline":
        made = stop_world(fake, monkeypatch, tmp_path)
    else:
        made = rig(fake, monkeypatch, tmp_path, start=(14, 40, 5))
    made.seam = request.param
    sold_by_hand(made, 4, SOLD_AT)
    made.stop_at(ist(14, 50))
    before_any_read(made, monkeypatch)
    return made


def play_cut(world: Any) -> Any:
    """Run `StopHold` (flatten) on the cut world's seam."""
    if world.seam == "inline":
        return world.run(StopHold)
    return run_threaded(world, passing(world), "orders", (StopHold,))


def adopted_after_the_lines(outcome: Any) -> bool:
    """Whether the close's final read is the one that adopted the sale."""
    return "adopted" in [event["kind"] for event in after_the_final_read(outcome)]


def test_the_stop_cut_after_the_last_lines_ends_for_the_6_units_left(cut_world):
    outcome = play_cut(cut_world)
    assert adopted_after_the_lines(outcome)
    assert stops(cut_world) == [(6, "trigger pending")]
    assert cut_world.fake.position("SBIN", "NSE") == 6


def test_the_result_says_the_stop_cut_to_6_protects_the_rest_once_the_broker_confirmed_it(
    cut_world,
):
    outcome = play_cut(cut_world)
    (left,) = outcome.open_positions
    stop = first_stop(cut_world).order_id
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (6, stop, True)
    assert left.note.endswith(f"stop {stop} confirmed at the broker: {PROTECTS}")


def test_a_cut_the_broker_refuses_leaves_the_stop_for_10_and_nothing_claims_protection(
    cut_world, monkeypatch, caplog
):
    refusing_modifies(monkeypatch)
    outcome = play_cut(cut_world)
    (left,) = outcome.open_positions
    stop = first_stop(cut_world).order_id
    assert stops(cut_world) == [(10, "trigger pending")]
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (6, stop, False)
    texts = [left.note, *alerts(caplog, "error")]
    assert [text for text in texts if PROTECTS in text] == []
    assert f"stop {stop} confirmed at the broker for 10 units, more than the 6 held" in left.note


def test_the_error_alert_names_a_cut_the_broker_refused(cut_world, monkeypatch, caplog):
    refusing_modifies(monkeypatch)
    outcome = play_cut(cut_world)
    (left,) = outcome.open_positions
    stop = first_stop(cut_world).order_id
    refused = f"the cut of its stop {stop} to 6 units was refused"
    assert [text for text in after_the_lines(caplog) if refused in text] == [
        f"StopHold SBIN NSE 6 units: {left.note}"
    ]
    assert refused in left.note


def no_answer_by_the_close(world: Any) -> None:
    """The cut gets no answer before the close ends: inline its connection drops, so the
    order worker has it to settle by a read; on the threads its call takes 8 s, past the
    close's 5 s wait for the final read (the broker cuts the stop meanwhile)."""
    if world.seam == "inline":
        world.fake.faults.drop_connection("modifyorder")
    else:
        world.fake.faults.slow("modifyorder", 8.0)


def test_the_error_alert_names_a_cut_with_no_answer_by_the_close_and_claims_no_protection(
    cut_world, caplog
):
    no_answer_by_the_close(cut_world)
    # The cut has no answer by the close (3.3 step 5 "As built"): the ERROR alert names it
    # and no confirmed stop is claimed, which the session invariants take instead of an
    # answered modify and a confirmed stop.
    cut_world.trace.leave_unprotected("StopHold", "a cut with no answer by the close")
    outcome = play_cut(cut_world)
    (left,) = outcome.open_positions
    stop = first_stop(cut_world).order_id
    unanswered = f"the cut of its stop {stop} to 6 units got no answer before the session closed"
    assert unanswered in left.note
    assert [text for text in after_the_lines(caplog) if unanswered in text] == [
        f"StopHold SBIN NSE 6 units: {left.note}"
    ]
    assert (left.stop_confirmed, PROTECTS in left.note) == (False, False)


# ----- the probe's case, the kernel refusing the adoption after the last lines -------------


def refusing_adoptions(monkeypatch: Any) -> None:
    """The kernel refuses the reconciler's adoption of the sale: `apply_external_fill`
    raises VenueRefusalError (here "session_over", the kernel session having finished)."""
    real = reconcile_act.call

    def refuse(func: Any, *args: Any) -> Any:
        if getattr(func, "__name__", "") == "apply_external_fill":
            raise VenueRefusalError("the session has finished", "session_over")
        return real(func, *args)

    monkeypatch.setattr(reconcile_act, "call", refuse)


def test_an_adoption_the_kernel_refuses_leaves_the_stop_for_10_and_sends_nothing(
    cut_world, monkeypatch
):
    refusing_adoptions(monkeypatch)
    outcome = play_cut(cut_world)
    assert stops(cut_world) == [(10, "trigger pending")]
    assert cut_world.fake.position("SBIN", "NSE") == 6
    assert "adopted" not in [event["kind"] for event in after_the_final_read(outcome)]


def test_the_close_names_the_held_cancel_an_adoption_the_kernel_refused_made_way_for(
    cut_world, monkeypatch, caplog
):
    # The cancel withdrawn for a cut that never came is held again: the close's alert names
    # it among what was not sent, as it would had no adoption been tried.
    refusing_adoptions(monkeypatch)
    play_cut(cut_world)
    not_sent = f"not sent: the cancel of stop order {first_stop(cut_world).order_id}"
    named = [text for text in alerts(caplog, "error") if "StopHold SBIN NSE 10 units" in text]
    assert [text for text in named if not_sent in text] != []


def test_the_mismatch_alert_of_an_adoption_the_kernel_refused_claims_no_adoption(
    cut_world, monkeypatch, caplog
):
    refusing_adoptions(monkeypatch)
    play_cut(cut_world)
    (text,) = alerts(caplog, "reconcile_mismatch")
    assert NOT_ADOPTED_WORDS in text
    assert "the engine takes them as its own" not in text
