"""What the start of a session says and checks of what it continues from an earlier day, on
the fake OpenAlgo (20-positional-live 4.2 steps 1 to 3, 3.1 "expired"; the lead's rulings
N5a-1 and N5a-3): the banner names each continued leg with its stop GTT, and each position
the other mode's ledger carries, never restored into this one; the preflight's funds check
sets the cash against the capital less what the carried positions hold; and a carried stop
GTT the broker expires (one older than 360 days) is placed again by the engine.

The days and the strategies are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with its stop GTT at 497.00, and `FutKept` (when it runs) one lot of
65 units of the NIFTY future at 120.00 NRML with its stop GTT 10 points below; their legs
are carried at NSE's close. Tuesday's session starts at 14:40:05 IST and is stopped at
14:46. Each strategy's capital is 1,00,000.00 rupees; OpenAlgo reports 2,50,000.00 rupees
of available cash in the broker's book.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY1, DAY2, CncKept, FutKept, carried_world, held, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, calls, gtts
from session_samples import alerts

from openalgo.backtest._live._types import LOGGER_NAME

DAY2_START = on(DAY2, 14, 40, 5)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def tuesday(world, strategy=CncKept):
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))
    return world.run(strategy)


def preflight(caplog, name):
    """The message of the preflight check `name` of the latest start."""
    tag = f"preflight {name}: "
    found = [r.getMessage() for r in caplog.records if r.getMessage().startswith(tag)]
    return found[-1][len(tag) :]


def test_the_banner_names_each_continued_leg_with_its_stop_gtt(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    tuesday(world)

    assert (
        f"CncKept: continued from its day-close record of {DAY1}: 10 SBIN NSE, stop GTT "
        f"{trigger_id} confirmed at the broker, trigger 497.00 for 10 units"
    ) in world.lines


@pytest.mark.parametrize(
    ("monday_in_sandbox", "line"),
    [
        (True, "CncKept: 10 SBIN NSE (CNC) held in sandbox, not continued in live"),
        (False, "CncKept: 10 SBIN NSE (CNC) held in live, not continued in sandbox"),
    ],
    ids=["sandbox then live", "live then sandbox"],
)
def test_a_position_the_other_ledger_carries_is_named_and_never_restored(
    make_fake_openalgo, monkeypatch, tmp_path, monday_in_sandbox, line
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=monday_in_sandbox)
    world.trace.waive("books agree", "the other mode's book holds Monday's position")
    world.fake.set_analyzer(not monday_in_sandbox)

    outcome = tuesday(world)

    assert line in world.lines
    assert not [entry for entry in world.lines if "continued from its day-close record" in entry]
    # Started afresh in this mode: its own entry and stop GTT, beside the other book's.
    buys = [c for c in calls(world, "placeorder") if c.ts_ns >= DAY2_START]
    assert [c.detail["request"]["action"] for c in buys] == ["BUY"]
    assert outcome.open_positions.__len__() == 1


def test_the_funds_check_sets_the_cash_against_the_capital_less_what_the_carry_holds(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # By hand: the CNC leg holds its cost, 10 x 500.00 = 5,000.00 rupees; the future its
    # margin, 20 percent of 65 x 120.00 = 1,560.00 rupees; the capital of the two
    # strategies, 2,00,000.00 rupees, less 6,560.00 leaves 1,93,440.00 to find.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, [CncKept, FutKept], future=True
    )

    tuesday(world, [CncKept, FutKept])

    assert preflight(caplog, "funds") == (
        "2,50,000.00 rupees of available cash cover the capital of 2,00,000.00 rupees less "
        "the 6,560.00 rupees the carried positions hold (CncKept SBIN CNC 5,000.00, FutKept "
        "NIFTY29DEC26FUT NRML 1,560.00), 1,93,440.00 rupees"
    )


@BOOKS
def test_a_carried_stop_gtt_the_broker_expires_is_placed_again_by_the_engine(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, history
):
    # A GTT older than 360 days ends at the broker (20-positional-live 3.1, "expired"):
    # the engine places the leg's stop again, as for any stop GTT that ends unasked.
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    (old,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.set_gtt_status(old, "expired")

    outcome = tuesday(world)

    places = [c for c in calls(world, "placegttorder") if c.ts_ns >= DAY2_START]
    assert len(places) == 1
    assert ("active", 10, 497.0, "SELL") in gtts(world)
    (position,) = outcome.open_positions
    assert position.stop_confirmed is True and position.stop_order_id != old
    assert held(world) == 10
    assert not [a for a in alerts(caplog, "error") if "entries are locked" in a]
