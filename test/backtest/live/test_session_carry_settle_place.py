"""A stop GTT's place under way when the process died, settled at the restart, on the fake
OpenAlgo (20-positional-live 3.1 "Unanswered calls"; the lead's ruling N5a-5): the journal
shows it recorded and bound to no trigger ID, so the restart settles it by the rule of an
unanswered place before anything is sent for it. One of its shape resting at the broker is
bound, never placed a second time; none, with a settled read against a reference from
before the place showing no fall, is placed again once; a fall is unresolved (entries
locked, an ERROR alert, never placed again while it may have fired).

The days and the strategies are `carry_samples`'. On Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with its stop GTT at 497.00 (child limit 492.00). Tuesday's session
continues it; at 14:44 its stop GTT is cancelled by hand, so the engine places it again,
and the process dies inside that place (a hook raises in the GTT call, before or after the
call reaches OpenAlgo). The restart comes at 14:50, past the window in which OpenAlgo could
still be working on a place sent before the crash (`_order_resolve.BROKER_WINDOW_NS`), its
reference the leg's 10 units Monday's close read. A trailing modify that times out, which
OpenAlgo may still apply, is cancelled and placed again (REPLACE_PLACE), and the process
dies in that place.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY2, CncKept, CncKeptTrail, carried_world, held, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtts
from restart_samples import Crash, crashing, restart, run_to_crash
from session_samples import NS, alerts
from stops_samples import at_price

from openalgo.backtest._live._gtt_calls import GttCalls
from openalgo.backtest._live._types import LOGGER_NAME

DAY2_START = on(DAY2, 14, 40, 5)
#: When the stop GTT is cancelled by hand, and when the session restarts.
BY_HAND, RESTART = on(DAY2, 14, 44), on(DAY2, 14, 50)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def crash_in_place(monkeypatch, *, reached: bool):
    """The process dies inside the next GTT place: after the call reached OpenAlgo
    (`reached`) or before it was sent; later places go as usual."""
    original = GttCalls.place
    armed = [True]

    def place(calls_, request):
        if not armed[0]:
            return original(calls_, request)
        armed[0] = False
        if reached:
            original(calls_, request)
        raise Crash("the process dies inside a GTT place")

    return place


def tuesday_until_the_crash(world, monkeypatch, strategy, *, reached):
    """Tuesday's session of `strategy` until it dies inside the place of its stop GTT after
    the one resting is cancelled by hand at 14:44."""
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    crashing(monkeypatch)
    to_day(world, DAY2)
    (resting,) = [g.trigger_id for g in world.fake.gtts()]
    world.script.at(BY_HAND, lambda: world.fake.set_gtt_status(resting, "cancelled"))
    world.script.at(
        BY_HAND + NS,
        lambda: monkeypatch.setattr(
            GttCalls, "place", crash_in_place(monkeypatch, reached=reached)
        ),
    )
    run_to_crash(world, strategy)
    world.script.sleep((RESTART - world.fake.clock.now()) / NS)


def tuesday_until_a_replace_crashes(world, monkeypatch):
    """Tuesday's session of `CncKeptTrail` until it dies inside a replace's place (module
    docstring), the clock moved to the restart."""
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    crashing(monkeypatch)
    to_day(world, DAY2)
    at_price(world, on(DAY2, 14, 41), 503.0)
    world.fake.faults.timeout("modifygttorder")
    monkeypatch.setattr(GttCalls, "place", crash_in_place(monkeypatch, reached=False))
    run_to_crash(world, CncKeptTrail)
    assert [c.what for c in calls(world, "modifygttorder", "cancelgttorder")][-2:] == [
        "modifygttorder",
        "cancelgttorder",
    ]
    world.script.sleep((RESTART - world.fake.clock.now()) / NS)


def sold_by_hand(world):
    """Another client of the account sells the leg's 10 SBIN CNC units at market."""
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )


def places_after(world, since_ns):
    return [c for c in calls(world, "placegttorder") if c.ts_ns >= since_ns]


@MODES
@BOOKS
def test_a_place_that_reached_the_broker_before_the_crash_is_bound_never_placed_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    tuesday_until_the_crash(world, monkeypatch, CncKept, reached=True)
    assert len(places_after(world, DAY2_START)) == 1

    outcome = restart(world, CncKept, stop_ns=on(DAY2, 14, 55))

    assert outcome.stopped_by == "ctrl_c"
    assert len(places_after(world, DAY2_START)) == 1
    assert sorted((status, units) for status, units, *_ in gtts(world)) == [
        ("active", 10),
        ("cancelled", 10),
    ]
    (position,) = outcome.open_positions
    assert position.stop_confirmed is True
    assert not [a for a in alerts(caplog, "error") if "entries are locked" in a]


@MODES
@BOOKS
def test_a_place_that_never_reached_the_broker_is_placed_again_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    tuesday_until_the_crash(world, monkeypatch, CncKept, reached=False)
    assert places_after(world, DAY2_START) == []

    outcome = restart(world, CncKept, stop_ns=on(DAY2, 14, 55))

    assert len(places_after(world, RESTART)) == 1
    assert sorted(status for status, *_ in gtts(world)) == ["active", "cancelled"]
    (position,) = outcome.open_positions
    assert position.stop_confirmed is True
    assert held(world) == 10
    assert not [a for a in alerts(caplog, "error") if "entries are locked" in a]


@MODES
@BOOKS
def test_a_place_never_sent_while_the_position_fell_by_its_units_is_unresolved(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The leg's units fell by the GTT's (a child of its shape may have sold them): it may
    # have landed and fired, so it is never placed again.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    world.trace.left_unprotected_by_decision("CncKept", "a place under way that may have fired")
    tuesday_until_the_crash(world, monkeypatch, CncKept, reached=False)
    sold_by_hand(world)
    world.script.sleep(60)

    restart(world, CncKept, stop_ns=on(DAY2, 14, 57))

    assert places_after(world, RESTART) == []
    (locked,) = [a for a in alerts(caplog, "error") if "entries are locked" in a]
    assert "it may have landed and fired, so it is not placed again" in locked


@MODES
@BOOKS
def test_a_crash_in_a_replaces_place_is_settled_and_the_replacement_placed_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Tuesday's trail moves the stop to 500.00; the modify times out before it reaches
    # OpenAlgo, which may still apply it, so the GTT is cancelled and placed again with the
    # modify's terms, and the process dies inside that place.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, history=history, analyzer=analyzer
    )
    tuesday_until_a_replace_crashes(world, monkeypatch)

    outcome = restart(world, CncKeptTrail, stop_ns=on(DAY2, 14, 55))

    (placed,) = places_after(world, RESTART)
    assert placed.detail["request"]["triggerprice_sl"] == 500.0
    (position,) = outcome.open_positions
    assert position.stop_confirmed is True
    assert not [a for a in alerts(caplog, "error") if "entries are locked" in a]


@MODES
@BOOKS
def test_a_crash_in_a_replaces_place_while_the_position_fell_is_unresolved_never_placed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The leg's units fell by the GTT's while the process was down (as when the old GTT
    # fired before its cancel): the replacement may have landed and fired, so it is never
    # placed again.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, history=history, analyzer=analyzer
    )
    world.trace.left_unprotected_by_decision(
        "CncKeptTrail", "a replace's place that may have fired"
    )
    tuesday_until_a_replace_crashes(world, monkeypatch)
    sold_by_hand(world)
    world.script.sleep(60)

    restart(world, CncKeptTrail, stop_ns=on(DAY2, 14, 57))

    assert places_after(world, RESTART) == []
    (locked,) = [a for a in alerts(caplog, "error") if "entries are locked" in a]
    assert "it may have landed and fired, so it is not placed again" in locked
