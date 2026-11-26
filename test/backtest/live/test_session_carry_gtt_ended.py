"""A carried stop GTT that the journal ended on an earlier day, after the close that carried
it, at the next day's start on the fake OpenAlgo (20-positional-live 3.1, 3.4 and 4.2; review
round 2 of unit N5a): it ended outside any session, so the kernel places the leg's stop again
as a new GTT, sent once, and the session runs on; while the start's books show the leg short,
the exit guard holds the place back and nothing is sent.

The days and the strategy are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with a stop GTT at 497.00 (sell 10) and carries it at NSE's close.
After the close the GTT is cancelled at the broker and the journal ends it. Tuesday's
session starts at 14:40:05 IST and is stopped at 14:46.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY2, CncKept, carried_world, held, journal_of, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtts
from session_samples import alerts

from openalgo.backtest._live._journal_gtt_rows import GttState
from openalgo.backtest._live._types import LOGGER_NAME

#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def sell_by_hand(world):
    """Another client of the account sells the 10 SBIN CNC units at market."""
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )


def gtts_active_trigger(world):
    """The trigger ID of the one GTT active at the fake."""
    (active,) = [gtt.trigger_id for gtt in world.fake.gtts() if gtt.status == "active"]
    return active


def ended_after_the_close(make_fake_openalgo, monkeypatch, tmp_path, *, sold=False, **options):
    """Monday's carry, its stop GTT cancelled at the broker after the close and ended in the
    journal then, and the 10 units sold by hand that evening when `sold`; the world moved
    to Tuesday's start, stopped at 14:46."""
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, **options)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.set_gtt_status(trigger_id, "cancelled")
    with journal_of(world) as journal:
        seq = journal.gtt_by_trigger(trigger_id).seq
        journal.set_gtt_state(seq, GttState.ENDED, note="cancelled by hand after the close")
    if sold:
        sell_by_hand(world)
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))
    return world


@MODES
@BOOKS
def test_a_carried_stop_gtt_ended_after_the_close_is_placed_again_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = ended_after_the_close(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )
    sent = len(calls(world, *SENDS))
    caplog.clear()

    tuesday = world.run(CncKept)

    assert tuesday.stopped_by == "ctrl_c"
    assert [c.what for c in calls(world, *SENDS)[sent:]] == ["placegttorder"]
    assert gtts(world) == [("cancelled", 10, 497.0, "SELL"), ("active", 10, 497.0, "SELL")]
    with journal_of(world) as journal:
        (bound,) = journal.open_gtts()
    assert bound.trigger_id == gtts_active_trigger(world)
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_carried_stop_gtt_ended_after_the_close_of_a_leg_sold_by_hand_is_not_placed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = ended_after_the_close(
        make_fake_openalgo, monkeypatch, tmp_path, sold=True, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "the leg was sold by hand on Monday evening")
    world.trace.waive("protected", "the leg was sold by hand: the guard holds its stop back")
    assert held(world) == 0
    sent = len(calls(world, *SENDS))
    caplog.clear()

    tuesday = world.run(CncKept)

    assert tuesday.stopped_by == "ctrl_c"
    assert calls(world, *SENDS)[sent:] == []
    assert not [text for text in alerts(caplog, "error") if "stops on an error" in text]
