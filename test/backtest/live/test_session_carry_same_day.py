"""A same-day restart on a day that continued a carry, on the fake OpenAlgo (20-positional-live
4.2, 16-session-design 3.6; review round 3 of unit N5a): a Ctrl+C that keeps a leg carried
from an earlier day writes the day's record, and a restart that day resumes the leg from the
day's save with its stop GTT of the earlier day taken back, never judged as a GTT no record
carries; a start after that day's close is refused for the hour alone, and the next day
continues the day's record. A crash on that day restarted before the close is continued the
next day from the record the restart's close writes: the way to keep a leg the crash would
otherwise leave without a record. Nothing is sent.

The days and the strategies are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with its stop GTT at 497.00, or `FutKept` one lot (65 units) of
NIFTY29DEC26FUT NRML at 120.00 with its stop GTT at 110.00, and the leg is carried at the
close. Tuesday's session starts at 14:40:05 IST and continues it.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import (
    DAY2,
    DAY3,
    FUTURE,
    FUTURE_LOT,
    CncKept,
    FutKept,
    carried_world,
    journal_of,
    on,
    to_day,
)
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtts
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import StopFlag, alerts

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import PreflightError

#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)
#: Each leg of the tests: (strategy, the fake's future to list or None, symbol, units, the
#: stop GTT's trigger in rupees).
LEGS = pytest.mark.parametrize(
    ("strategy", "future", "symbol", "units", "trigger"),
    [
        (CncKept, None, "SBIN", 10, 497.0),
        (FutKept, FUTURE, FUTURE, FUTURE_LOT, 110.0),
    ],
    ids=["CNC", "NRML"],
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def tuesday(world, strategy, *, stop_ns=None):
    """Tuesday's session of `strategy` continuing Monday's carry, stopped by a Ctrl+C at
    `stop_ns` (None: run to its end)."""
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY2)
    if stop_ns is not None:
        world.stop_at(stop_ns)
    return world.run(strategy)


def resumed(strategy, symbol, units, trigger_id, trigger):
    """The restart banner's line of a leg resumed with its stop GTT confirmed."""
    return (
        f"{strategy.__name__}: resumed {units:g} {symbol} {strategy.exchange}, stop GTT "
        f"{trigger_id} confirmed at the broker, trigger {trigger:.2f} for {units} units"
    )


@MODES
@BOOKS
@LEGS
def test_a_restart_after_a_kept_ctrl_c_on_a_continued_day_resumes_the_leg_and_its_stop_gtt(
    make_fake_openalgo,
    monkeypatch,
    tmp_path,
    caplog,
    analyzer,
    history,
    strategy,
    future,
    symbol,
    units,
    trigger,
):
    world = carried_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        strategy,
        future=future,
        analyzer=analyzer,
        history=history,
    )
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    assert tuesday(world, strategy, stop_ns=on(DAY2, 14, 46)).stopped_by == "ctrl_c"
    with journal_of(world) as journal:
        assert journal.day_close(strategy.__name__).day == DAY2
    sent = len(calls(world, *SENDS))
    caplog.clear()

    outcome = restart(world, strategy, stop_ns=on(DAY2, 14, 50))

    assert outcome.stopped_by == "ctrl_c"
    assert resumed(strategy, symbol, units, trigger_id, trigger) in world.lines
    assert calls(world, *SENDS)[sent:] == []
    assert gtts(world) == [("active", units, trigger, "SELL")]
    assert not alerts(caplog, "error")


@MODES
@BOOKS
@LEGS
def test_a_start_after_the_close_of_a_continued_day_is_refused_only_for_the_hour_and_sends_nothing(
    make_fake_openalgo,
    monkeypatch,
    tmp_path,
    caplog,
    analyzer,
    history,
    strategy,
    future,
    symbol,
    units,
    trigger,
):
    world = carried_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        strategy,
        future=future,
        analyzer=analyzer,
        history=history,
    )
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    assert tuesday(world, strategy).stopped_by == "session_end"
    sent = len(calls(world, *SENDS))
    caplog.clear()

    with pytest.raises(PreflightError) as refused:
        restart(world, strategy)

    # The day's record and save are not judged as a GTT no record carries: the start is
    # refused for the hour alone, as any start after the close.
    (failure,) = refused.value.failures
    assert failure.startswith(f"the {strategy.exchange} session of {DAY2} closed at ")
    assert failure.endswith(" IST: start the session on a trading day, before the close")
    assert calls(world, *SENDS)[sent:] == []
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 46))

    assert world.run(strategy).stopped_by == "ctrl_c"
    assert calls(world, *SENDS)[sent:] == []
    assert gtts(world) == [("active", units, trigger, "SELL")]
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_on_a_continued_day_restarted_before_the_close_is_continued_the_next_day(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, analyzer=analyzer, history=history
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    crashing(monkeypatch)
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY2)
    crash_at(world, on(DAY2, 14, 46))
    run_to_crash(world, CncKept)
    assert restart(world, CncKept).stopped_by == "session_end"
    with journal_of(world) as journal:
        assert journal.day_close("CncKept").day == DAY2
    sent = len(calls(world, *SENDS))
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 46))
    caplog.clear()

    assert world.run(CncKept).stopped_by == "ctrl_c"
    assert calls(world, *SENDS)[sent:] == []
    assert gtts(world) == [("active", 10, 497.0, "SELL")]
    assert not alerts(caplog, "error")
