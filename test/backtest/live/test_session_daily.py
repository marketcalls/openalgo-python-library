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
    UNITS,
    Recorded,
    WeeklyRecorded,
    close_of,
    daily_world,
    open_of,
)
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import specify
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
    # goes; the second start finds the bar decided and decides nothing.
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 8))
    world.run(Recorded)

    restart(world, Recorded, stop_ns=ist(9, 30))

    assert [bar_close for _, bar_close, _ in Recorded.decided] == [close_of(FRIDAY)]
    assert buys(world) == []


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
