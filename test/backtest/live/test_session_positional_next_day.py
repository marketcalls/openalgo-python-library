"""A strategy that holds a position or a GTT from an earlier day is refused at the start, in
this version, before anything is sent (20-positional-live 4.2; the brief of unit N3b: the
restart across days and the reading of the day-close record are unit N5's; review
finding 1 of round 3 of unit N3b).

Started afresh on the next day, such a strategy would enter again beside the position it
carries, and the GTT follower, which follows the journal's GTTs of every day, would apply
a finding about the earlier day's stop GTT to the new session's GTT of the same kernel id
(kernel GTT ids start again at 1 each day). So the start is refused, naming what the
strategy holds and what to do. A day-close record of an earlier day that carries nothing
refuses nothing.

The market and the strategies are `positional_samples`': on Monday 2026-11-16 `CncHold`
buys 10 SBIN CNC at 14:42:01 with its stop GTT 261116000001 (sell 10 at 497.00) and the
session ends at NSE's close with the leg carried. The day ends at the fake (the CNC units
move into the holdings, the GTT stays), and the same strategy is started again on Tuesday
2026-11-17 at 14:40:05 IST.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_openalgo import BROKER
from fake_openalgo.clock import ist_ns
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, STOP_AT, CncHold, calls, gtt_world, gtts
from session_samples import DAY, NS, TRADE_EVERY_S

from openalgo.backtest._live._journal_gtt_rows import GttTerms
from openalgo.backtest._live._types import Mode, Purpose
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import PreflightError

#: The next trading day.
NEXT_DAY = dt.date(2026, 11, 17)
#: The second session's start, UTC nanoseconds.
NEXT_START = ist_ns(2026, 11, 17, 14, 40, 5)
#: The calls that place, change or cancel anything at the broker.
SENDS = ("placeorder", "modifyorder", "cancelorder", "placegttorder", "modifygttorder")
#: What the refusal asks the operator to do.
WHAT_TO_DO = (
    "This version does not continue a position or a GTT from an earlier day: close the "
    "position and cancel its GTT in OpenAlgo by hand, then start under another account "
    "name (account=), or wait for the version that continues them"
)

MINUTE_NS = 60 * NS


def flat_minutes(start_ns, end_ns):
    """Broker candles of SBIN at 500.00 for each minute from `start_ns` to `end_ns`."""
    return [(t, 500.0, 500.0, 500.0, 500.0, 100.0) for t in range(start_ns, end_ns, MINUTE_NS)]


def to_the_next_day(world):
    """End the fake's day and move it to the second session's start, with SBIN's candles
    of both days and a trade every 10 s from then."""
    fake = world.fake
    fake.end_day()
    calendar = default_calendar()
    _, first_close = calendar.session("NSE", DAY)
    next_open, _ = calendar.session("NSE", NEXT_DAY)
    candles = flat_minutes(ist_ns(2026, 11, 16, 14, 40), first_close)
    candles += flat_minutes(next_open, ist_ns(2026, 11, 17, 14, 40))
    fake.seed_candles("SBIN", "NSE", candles)
    world.script.sleep((NEXT_START - fake.clock.now()) / NS)
    world.script.trade()
    world.script.trades(NEXT_START + TRADE_EVERY_S * NS, ist_ns(2026, 11, 17, 15, 32))


def journal_of(world, mode=Mode.LIVE):
    """The account's journal, created when missing, in the ledger of `mode`."""
    folder = world.folder / "live" / BROKER
    folder.mkdir(parents=True, exist_ok=True)
    return Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=mode,
        now=world.fake.clock.now,
    )


@MODES
@BOOKS
def test_a_strategy_carrying_a_position_from_an_earlier_day_is_refused_with_nothing_sent(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    world = gtt_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        history=history,
        analyzer=analyzer,
        trades_until=(15, 32),
    )
    first = world.run(CncHold)
    assert first.stopped_by == "session_end"
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    book = gtts(world)
    to_the_next_day(world)

    with pytest.raises(PreflightError) as refused:
        world.run(CncHold)

    assert refused.value.failures == (
        f"CncHold: its day-close record of {DAY} carries 10 SBIN NSE (CNC); its stop GTT "
        f"{trigger_id} of {DAY} on NSE:SBIN (CNC) has not ended. {WHAT_TO_DO}",
    )
    assert [c for c in calls(world, *SENDS, "cancelgttorder") if c.ts_ns >= NEXT_START] == []
    assert gtts(world) == book
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_gtt_of_an_earlier_day_that_has_not_ended_refuses_the_strategy(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # No day-close record: a session that stopped without one left the GTT recorded.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    earlier = DAY - dt.timedelta(days=3)
    with journal_of(world) as journal:
        journal.start_session(earlier, Mode.LIVE, BROKER)
        terms = GttTerms("BUY", 490.0, 490.0, 5)
        seq = journal.record_gtt(
            "CncHold", 1, 1, "SBIN", "NSE", "CNC", terms, purpose=Purpose.ENTRY
        )
        journal.bind_gtt(seq, "261113000001")

    with pytest.raises(PreflightError) as refused:
        world.run(CncHold)

    assert refused.value.failures == (
        f"CncHold: its GTT order 261113000001 of {earlier} on NSE:SBIN (CNC) has not "
        f"ended. {WHAT_TO_DO}",
    )
    assert calls(world, *SENDS) == []


def test_an_earlier_days_record_that_carries_nothing_refuses_nothing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)
    earlier = DAY - dt.timedelta(days=3)
    flat = {"instruments": [], "positions": [], "orders": []}
    with journal_of(world) as journal:
        journal.start_session(earlier, Mode.LIVE, BROKER)
        journal.put_day_close("CncHold", flat, {}, "v1", day=earlier)

    outcome = world.run(CncHold)

    assert outcome.stopped_by == "ctrl_c"
    assert len(calls(world, "placegttorder")) == 1
