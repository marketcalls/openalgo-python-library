"""Daily and weekly strategies in `bt.live`, on the fake OpenAlgo (design
20-positional-live section 8, decision 11.1, review M8; the lead's ruling N5b-1; the 13.3 row
"a daily bar decided once across two restarts"): such a strategy decides once, before the
open, on the broker's candle of its interval of the session that closed; its MARKET entry
goes at the open; the journal keeps the last bar it decided, so a later start decides
nothing again; bars of days no session ran feed its series only and the banner names them;
a start after the open sends the decision's orders at once, flagged `late_decision`; a
weekly strategy decides only the week whose last session has closed. A daily strategy in
MIS is refused.

The market and the strategies are `daily_samples`'. Expected values come from the candles
and the calendar.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from daily_samples import (
    OPEN,
    SESSIONS,
    UNITS,
    Recorded,
    WeeklyRecorded,
    close_of,
    daily_world,
    open_of,
)
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import candles, specify
from positional_samples import MODES
from restart_samples import restart
from session_samples import WAIT_MAX_S, alerts, ist

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal

WEDNESDAY, THURSDAY, FRIDAY = dt.date(2026, 11, 11), dt.date(2026, 11, 12), dt.date(2026, 11, 13)
MONDAY_BEFORE = dt.date(2026, 11, 9)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class DailyTarget(Recorded):
    """Recorded whose entry has a target 3 points above, held by the session on ticks."""

    decided: list[tuple[int, int, int]] = []
    rules = {"target": bt.points(3)}


def journal(world):
    """The rig's journal in the ledger of the fake's mode, opened as the session opens it (its
    account is the broker's)."""
    folder = world.folder / "live" / BROKER
    folder.mkdir(parents=True, exist_ok=True)
    return Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=world.fake.mode,
        now=world.fake.clock.now,
    )


def buys(world) -> list[tuple[str, int, str, str]]:
    return [
        (o.price_type, o.quantity, o.product, o.status) for o in world.orders() if o.action == "BUY"
    ]


@MODES
def test_a_daily_strategy_started_before_the_open_decides_fridays_bar_and_buys_at_the_open(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.stop_at(ist(9, 30))

    world.run(Recorded)

    # Friday's bar, the latest whole session's, with the four daily bars of the week
    # before it in its series.
    assert Recorded.decided == [(open_of(FRIDAY), close_of(FRIDAY), 4)]
    assert buys(world) == [("MARKET", UNITS, "CNC", "complete")]
    (bought,) = world.orders()
    # The order waited for the open: placed in the loop's turn of its first tick, 09:15:00,
    # which waits at most WAIT_MAX_S (1 s) in these tests.
    assert OPEN <= bought.placed_ns <= OPEN + WAIT_MAX_S * 10**9
    with journal(world) as kept:
        assert kept.decided_bar("Recorded") == close_of(FRIDAY)


def test_a_daily_bar_is_decided_once_across_two_starts_the_same_morning(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # 13.3 "a daily bar decided once across two restarts": the first start decides
    # Friday's bar before the open and is stopped (Ctrl+C) at 09:08, before its order
    # goes; the stop says it cancelled the order, and the second start finds the bar
    # decided, says so and decides nothing.
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 8))
    first = world.run(Recorded)

    restart(world, Recorded, stop_ns=ist(9, 30))

    assert [bar_close for _, bar_close, _ in Recorded.decided] == [close_of(FRIDAY)]
    assert buys(world) == []
    cancelled = (
        "Recorded: the stop cancels the orders of its decision on its D bar closing on "
        "2026-11-13 (BUY 10 MARKET) before they went to the broker, and that bar is not "
        "decided again today"
    )
    assert f"Warning: {cancelled}" in world.lines
    result = session_result(first)
    assert cancelled in result.warnings
    assert f"Warning     : {cancelled}" in result.summary_text().splitlines()
    assert (
        "Warning: Recorded: its D bar closing on 2026-11-13 was decided by an earlier start "
        "today and is not decided again today"
    ) in world.lines


def test_a_start_after_the_open_sends_the_decision_at_once_flagged_late_decision(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path, start=(9, 20, 5))
    # SBIN rises to 504.00 at 09:25, past the 503.00 target of the entry at 500.00.
    world.script.price = lambda now_ns: 504.0 if now_ns >= ist(9, 25) else 500.0
    world.stop_at(ist(9, 30))

    outcome = world.run(DailyTarget)

    assert [bar_close for _, bar_close, _ in DailyTarget.decided] == [close_of(FRIDAY)]
    (entry, *_) = [text for text in alerts(caplog, "entry")]
    assert entry.endswith("(entry, order 261116000000001, flagged late_decision)")
    (trade,) = session_result(outcome).results["DailyTarget"].trades.itertuples()
    assert (trade.exit_reason, trade.flags) == ("leg_target", ["late_decision"])


def test_bars_of_days_no_session_ran_feed_the_series_and_only_the_latest_is_decided(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The last bar decided is Wednesday's: Thursday's closed while no session ran.
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    with journal(world) as kept:
        kept.start_session(WEDNESDAY, Mode.LIVE, BROKER)
        kept.put_decided_bar("Recorded", close_of(WEDNESDAY))
    world.stop_at(ist(9, 30))

    world.run(Recorded)

    assert (
        "Warning: Recorded: its D bars closing on 2026-11-12 closed while no session ran: "
        "they feed its indicators only, and it decides the bar closing on 2026-11-13"
    ) in world.lines
    assert Recorded.decided == [(open_of(FRIDAY), close_of(FRIDAY), 4)]


def test_a_weekly_strategy_decides_the_week_whose_last_session_has_closed(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 30))

    world.run(WeeklyRecorded)

    # The week of Monday 2026-11-09: from that Monday's open to Friday's close.
    assert WeeklyRecorded.decided == [(open_of(MONDAY_BEFORE), close_of(FRIDAY), 1)]
    assert buys(world) == [("MARKET", UNITS, "CNC", "complete")]


@MODES
def test_a_daily_strategy_whose_last_daily_candle_is_missing_decides_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # The broker has not served Friday's candle: the bar due at Monday's open is Friday's,
    # so Thursday's is not decided in its place, at an open a backtest never fills it at.
    world = daily_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        analyzer=analyzer,
        sessions=(MONDAY_BEFORE, WEDNESDAY, THURSDAY),
    )
    world.stop_at(ist(9, 30))

    world.run(Recorded)

    assert Recorded.decided == []
    assert buys(world) == []
    assert (
        "Warning: Recorded: the broker has not served the daily candle of 2026-11-13 (the "
        "last session) yet, so its D bar closing on 2026-11-13 is not decided now, nor an older "
        "bar in its place; start again once the broker serves it to decide that bar (after "
        "the open its orders are flagged late_decision)"
    ) in world.lines
    with journal(world) as kept:
        assert kept.decided_bar("Recorded") is None


@MODES
def test_a_start_once_the_broker_serves_the_missing_candle_decides_that_bar_late(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # The 09:05 start finds no candle of Friday and decides nothing, as the test above; the
    # broker serves it by 09:20, and a start then decides Friday's bar, its entry flagged
    # late_decision, as the warning promised: each start asks the broker for the recent
    # daily candles again, so the first start's empty answer kept in the local cache does
    # not hide the candle served since.
    world = daily_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        analyzer=analyzer,
        sessions=(MONDAY_BEFORE, WEDNESDAY, THURSDAY),
    )
    world.stop_at(ist(9, 20))
    world.run(Recorded)
    world.fake.seed_candles("SBIN", "NSE", candles(SESSIONS, today_until=None))

    restart(world, Recorded, stop_ns=ist(9, 40))

    assert Recorded.decided == [(open_of(FRIDAY), close_of(FRIDAY), 4)]
    assert buys(world) == [("MARKET", UNITS, "CNC", "complete")]
    (entry,) = alerts(caplog, "entry")
    (bought,) = world.orders()
    assert entry.endswith(f"(entry, order {bought.order_id}, flagged late_decision)")
    missing = [line for line in world.lines if "has not served the daily candle" in line]
    assert len(missing) == 1
    with journal(world) as kept:
        assert kept.decided_bar("Recorded") == close_of(FRIDAY)


@MODES
def test_a_weekly_strategy_first_started_in_the_middle_of_a_week_decides_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # Friday 2026-11-13: the week of 2026-11-02 closed at Sunday 2026-11-08's session and
    # was due at Monday 2026-11-09's open; this week has not closed.
    weeks = tuple(dt.date(2026, 11, day) for day in (2, 3, 4, 5, 6, 8)) + SESSIONS[:-1]
    world = daily_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, day=FRIDAY, sessions=weeks
    )
    world.stop_at(ist(9, 30, day=FRIDAY))

    world.run(WeeklyRecorded)

    assert WeeklyRecorded.decided == []
    assert buys(world) == []
    assert (
        "Warning: WeeklyRecorded: its W bar closing on 2026-11-08 was due at the open of "
        "2026-11-09, not this one: it feeds its indicators only and is not decided; the next "
        "is decided at the first start after its period's last session"
    ) in world.lines
    with journal(world) as kept:
        assert kept.decided_bar("WeeklyRecorded") is None


def test_a_strategy_raising_at_its_decision_before_the_open_stops_the_session_safely(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # As an exception of the strategy's code in the loop does (16-session-design 3.5): an
    # ERROR alert, then the safe stop by on_stop, and the exception raised again.
    class RaisesBeforeOpen(Recorded):
        def on_bar(self, bar):
            raise RuntimeError("the strategy's own error")

    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(RaisesBeforeOpen)

    assert alerts(caplog, "error") == [
        "LIVE (RaisesBeforeOpen): RaisesBeforeOpen raised RuntimeError: the strategy's own "
        "error. The session stops safely"
    ]
    assert buys(world) == []


def test_a_daily_strategy_in_mis_is_refused():
    class DailyIntraday(Recorded):
        product = "MIS"

    with pytest.raises(NotImplementedError) as refused:
        specify(DailyIntraday)

    assert str(refused.value) == (
        "DailyIntraday: a D strategy in MIS in a live session (it decides before the open, "
        "and an MIS position is squared off the same day) is not supported in this version: "
        "trade it in CNC or NRML to hold its positions from one decision to the next, or give "
        "it an intraday interval, such as '5m'"
    )
