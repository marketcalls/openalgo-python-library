"""`bt.live(positional_stops="rearm_daily")` on the fake OpenAlgo (design 20-positional-live
section 6, decisions P2 and P5; the lead's ruling N5b-2; the 13.3 rows "broker without
GTT, with and without rearm_daily" and "gap through the stop at the open"): a CNC or NRML
position's stop is a day stop order, as an MIS position's, placed as each session reaches
its open; the start, the close, the daily summary and the result say what it leaves
unprotected, in the design's words.

Two markets. `positional_samples`': SBIN from 14:40:05 IST on Monday 2026-11-16, where
`CncHold` buys 10 at 500.00 at 14:42:01 with a 3-point stop (497.00). `rearm_samples`': a
start at 09:05, before the open, holding 10 SBIN bought on Friday 2026-11-13 at 500.00 with
the same stop, carried through the binding's `restore_carried_state`. Each runs in the
broker's book and the sandbox's. Expected values come from the scenario.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES, CncHold, calls, gtt_world
from rearm_samples import OPEN, RearmHold, rearm_world
from session_samples import alerts, ist

from openalgo.backtest._live._journal_carry import DayClose
from openalgo.backtest._live._preflight_gtt import REARM_DAILY_SENTENCE
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal

REARM = {"positional_stops": "rearm_daily"}
#: The carried leg's line at the start, before the open.
WAITS = "RearmHold 10 SBIN NSE (CNC), stop 497.00: its day stop waits for the open"
#: The design's sentence for rearm_daily (20-positional-live 6), written out.
SENTENCE = (
    "Positions held overnight have no stop at the broker from the close until your next "
    "start; start bt.live before the open on every trading day you hold a position."
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def live_orders(world) -> list[tuple[str, str, int, str, float]]:
    """(action, price type, units, product, trigger) of each SBIN order at the fake."""
    return [
        (o.action, o.price_type, o.quantity, o.product, o.trigger_price)
        for o in world.orders()
        if o.symbol == "SBIN"
    ]


def test_the_sentence_is_the_designs():
    assert REARM_DAILY_SENTENCE == SENTENCE


@MODES
def test_a_cnc_strategy_on_a_broker_without_gtt_runs_with_rearm_daily(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # 13.3 "broker without GTT, with rearm_daily": the start passes and the stop is a day
    # SL-M of the position's 10 units at 497.00, never a GTT. (Without rearm_daily the
    # start is refused: `test_session_positional_start`.)
    build = lambda **o: make_fake_openalgo(gtt=False, **o)  # noqa: E731
    world = gtt_world(build, monkeypatch, tmp_path, analyzer=analyzer)
    world.stop_at(ist(14, 46))

    world.run(CncHold, **REARM)

    assert live_orders(world) == [
        ("BUY", "MARKET", 10, "CNC", 0.0),
        ("SELL", "SL-M", 10, "CNC", 497.0),
    ]
    assert calls(world, "placegttorder", "modifygttorder", "cancelgttorder") == []


def test_a_live_start_with_rearm_daily_on_a_broker_with_gtt_says_gtts_would_protect(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 46))

    world.run(CncHold, **REARM)

    assert (
        f"Warning: {BROKER} holds GTT orders, which would protect your positions overnight; "
        'positional_stops="rearm_daily" places a day stop at each start instead'
    ) in world.lines


@MODES
def test_a_carried_legs_day_stop_waits_for_the_open_and_is_placed_there(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = rearm_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.stop_at(ist(9, 30))

    world.run(RearmHold, **REARM)

    assert f"Warning: {SENTENCE}" in world.lines
    assert f"Warning: {WAITS}" in world.lines
    # The day stop goes as the session reaches the open: an SL-M of the 10 units at 497.00.
    (stop,) = world.orders()
    terms = (stop.action, stop.price_type, stop.quantity, stop.product)
    assert terms == ("SELL", "SL-M", 10, "CNC")
    assert (stop.trigger_price, stop.status) == (497.0, "trigger pending")
    (placed,) = calls(world, "placeorder")
    assert placed.ts_ns >= OPEN
    # Waiting for the open is no alert.
    assert alerts(caplog, "error") == []
    assert alerts(caplog, "reject") == []


@MODES
def test_a_gap_through_the_stop_at_the_open_exits_flagged_stop_crossed(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # 13.3 "gap through the stop at the open" under rearm_daily: SBIN opens at 495.00,
    # beyond the 497.00 stop, so no stop is placed: the 10 units are sold at market at the
    # open, 495.00, the trade's loss 10 x (495.00 - 500.00) = -50.00.
    world = rearm_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, open_price=495.0
    )
    world.stop_at(ist(9, 30))

    outcome = world.run(RearmHold, **REARM)

    assert live_orders(world) == [("SELL", "MARKET", 10, "CNC", 0.0)]
    (trade,) = session_result(outcome).results["RearmHold"].trades.itertuples()
    assert (trade.exit_reason, trade.flags) == ("leg_sl", ["stop_crossed"])
    assert (trade.exit_price_rupees, trade.gross_pnl_rupees) == (495.0, -50.0)
    assert trade.exit_time == dt.datetime(2026, 11, 16, 9, 15, tzinfo=trade.exit_time.tzinfo)


@MODES
def test_the_close_warns_of_each_carried_leg_without_a_stop_until_the_next_start(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)

    outcome = world.run(CncHold, **REARM)

    line = "CncHold 10 SBIN NSE (CNC), stop 497.00: no stop at the broker until your next start"
    mode = "SANDBOX" if analyzer else "LIVE"
    assert (
        f'{mode} (CncHold): WARNING: positional_stops="rearm_daily": {line}. {SENTENCE}'
    ) in alerts(caplog, "session")
    assert outcome.stopped_by == "session_end"
    result = session_result(outcome)
    assert {SENTENCE, line} <= set(result.warnings)
    assert f"Warning     : {line}" in result.summary_text().splitlines()
    assert (
        "CncHold SBIN NSE 10 units: no stop at the broker until your next start, which places "
        'its day stop at the open (positional_stops="rearm_daily")'
    ) in world.lines
    # The day stop's end at the close is the design's, no ERROR.
    assert alerts(caplog, "error") == []
    assert [o.status for o in world.orders()] == ["complete", "cancelled"]


def test_the_start_names_the_trading_days_no_session_ran_since_the_last_close(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The last close was Thursday 2026-11-12, flat; Friday 2026-11-13 ran no session.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    folder = tmp_path / "live" / BROKER
    folder.mkdir(parents=True)
    thursday = dt.date(2026, 11, 12)
    with Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=Mode.LIVE,
        now=world.fake.clock.now,
    ) as journal:
        journal.start_session(thursday, Mode.LIVE, BROKER)
        flat = {"instruments": [], "positions": []}
        journal.put_day_close("CncHold", flat, {}, "v1", day=thursday)
        assert isinstance(journal.day_close("CncHold"), DayClose)
    world.stop_at(ist(14, 46))

    world.run(CncHold, **REARM)

    assert (
        "Warning: CncHold: no session ran on 2026-11-13, the trading days since its last "
        "close on 2026-11-12"
    ) in world.lines
