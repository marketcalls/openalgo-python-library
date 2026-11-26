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
from expiry_samples import FUTURE, FUTURE_DAY, LOT, OPTION, OPTION_DAY, expiry_world
from expiry_samples import ist as ist_on
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES, CncHold, calls, gtt_world
from rearm_samples import OPEN, RearmHold, rearm_world
from restart_samples import restart
from session_samples import alerts, ist
from test_session_expiry import FutureHold, OptionHold
from test_session_positional_carry import HoldAndRest

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
    assert (
        'CNC and NRML with positional_stops="rearm_daily" (CncHold): each position\'s stop is '
        "a day stop order at the broker, placed at each session's open and ended by its close; "
        "its targets are held by the session while it runs"
    ) in world.lines
    assert (
        "Ctrl+C: exits the MIS positions and keeps the CNC and NRML positions open with their "
        "stops at the broker, a rearm_daily day stop until the close only (on_stop default; "
        "on_stop='flatten' exits them all)"
    ) in world.lines


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


def test_a_mode_switch_with_a_carried_leg_says_to_check_its_stop_and_sends_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # 20-positional-live 10.4: no MIS square-off closes a CNC leg, so the alert does not
    # say one does; nothing is sent after the switch, the day stop at the open neither.
    world = rearm_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("books agree", "the switch shows the sandbox's book, which holds none")
    world.script.at(ist(9, 10), lambda: world.fake.set_analyzer(True))
    world.stop_at(ist(9, 30))

    world.run(RearmHold, **REARM)

    assert (
        "LIVE (RearmHold): OpenAlgo's mode changed: the session sends nothing more. Positions "
        "as last known: RearmHold SBIN NSE 10 units, no stop at the broker. CNC and NRML "
        "positions stay open overnight, so check each one's stop at the broker; stop the "
        "session with Ctrl+C"
    ) in alerts(caplog, "error")
    assert world.fake.orders(Mode.LIVE) == []


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


@MODES
def test_a_resting_gtt_order_of_the_strategy_does_not_hold_up_the_close_of_a_rearm_daily_leg(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # HoldAndRest rests a GTT buy of 5 at 490.00 overnight by design: only the leg's day
    # stop ends at the close, so the session ends there, the leg named as left by design.
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 40)
    )

    outcome = world.run(HoldAndRest, **REARM)

    assert outcome.stopped_by == "session_end"
    assert outcome.ended_ns < ist(15, 31)
    assert alerts(caplog, "error") == []
    assert (
        "HoldAndRest SBIN NSE 10 units: no stop at the broker until your next start, which "
        'places its day stop at the open (positional_stops="rearm_daily")'
    ) in world.lines
    assert [(g.action, g.quantity, g.status) for g in world.fake.gtts()] == [("BUY", 5, "active")]


def test_a_ctrl_c_before_the_close_says_the_day_stop_works_until_the_close(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Ctrl+C at 14:46 keeps the CNC leg (on_stop's default) with its day SL-M working: the
    # close's WARNING says it protects the leg until the close, and nothing after.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 46))

    outcome = world.run(CncHold, **REARM)

    line = (
        "CncHold 10 SBIN NSE (CNC), stop 497.00: its day stop works at the broker until the "
        "close, then no stop until your next start"
    )
    assert f'LIVE (CncHold): WARNING: positional_stops="rearm_daily": {line}. {SENTENCE}' in (
        alerts(caplog, "session")
    )
    assert line in session_result(outcome).warnings


class CncHoldOnce(CncHold):
    """CncHold that buys only once in the day, across restarts (its `self.state`)."""

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(self.qty, **type(self).rules)


def test_a_restart_of_the_day_says_the_carried_legs_day_stop_is_placed(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 46))
    world.run(CncHoldOnce, **REARM)

    restart(world, CncHoldOnce, stop_ns=ist(14, 50), **REARM)

    assert (
        "Warning: CncHoldOnce 10 SBIN NSE (CNC), stop 497.00: its day stop is placed at the broker"
    ) in world.lines
    # The restart places no second stop: the day SL-M of the first start still works.
    assert live_orders(world) == [
        ("BUY", "MARKET", 10, "CNC", 0.0),
        ("SELL", "SL-M", 10, "CNC", 497.0),
    ]


def test_a_carried_leg_is_uncovered_while_the_market_is_closed_in_the_unprotected_table(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = rearm_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 30))

    outcome = world.run(RearmHold, **REARM)

    closed, sent = outcome.unprotected
    reason = "no day stop while the market is closed"
    assert (closed.strategy, closed.units, closed.reason) == ("RearmHold", 10, reason)
    # From the start at 09:05:00 to the open at 09:15:00; then the day stop is sent at the
    # open and uncovered until the broker confirms it, the loop's next turn.
    assert (closed.seconds(outcome.ended_ns), closed.to_ns) == (600.0, OPEN)
    assert (sent.from_ns, sent.reason) == (OPEN, "stop unconfirmed")


@pytest.mark.parametrize(
    ("strategy", "symbol", "day", "stop"),
    [
        (OptionHold, OPTION, OPTION_DAY, ("SL", 110.0, 104.5)),
        (FutureHold, FUTURE, FUTURE_DAY, ("SL-M", 70.0, 0.0)),
    ],
    ids=["option", "future"],
)
def test_a_rearm_daily_nrml_leg_on_nfo_gets_the_day_stop_of_its_segment(
    make_fake_openalgo, monkeypatch, tmp_path, strategy, symbol, day, stop
):
    # 16-session-design 4.1: an option takes no SL-M, so its day stop is a stop-limit 5
    # percent beyond the trigger (110.00 x 0.95 = 104.50); a future's is an SL-M. Bought
    # at 120.00, a stop 10 points (the option) or 50 (the future) below.
    world = expiry_world(
        make_fake_openalgo, monkeypatch, tmp_path, symbol=symbol, day=day, start=(14, 0, 5)
    )
    world.stop_at(ist_on(day, 14, 5))

    world.run(strategy, positional_stops="rearm_daily", capital=100_000, max_daily_loss=5_000)

    (placed,) = [o for o in world.orders() if o.action == "SELL"]
    terms = (placed.price_type, placed.quantity, placed.product)
    assert (*terms, placed.trigger_price, placed.price) == (stop[0], LOT, "NRML", *stop[1:])
    assert world.fake.gtts() == []


def test_a_kept_gtt_refusal_does_not_refuse_a_strategy_that_chose_rearm_daily(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # 20-positional-live 3.5: a stop GTT refusal kept from an earlier day refuses its
    # strategy at every start until the user passes rearm_daily, which places no GTT.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    earlier = dt.date(2026, 11, 13)
    with journal_of(world) as kept:
        kept.start_session(earlier, Mode.LIVE, BROKER)
        kept.keep_refusal("CncHold", "SBIN", "NSE", "CNC", "no GTT on this segment", day=earlier)
    world.stop_at(ist(14, 46))

    world.run(CncHold, **REARM)

    assert live_orders(world) == [
        ("BUY", "MARKET", 10, "CNC", 0.0),
        ("SELL", "SL-M", 10, "CNC", 497.0),
    ]


def journal_of(world):
    """The rig's live journal, opened as the session opens it."""
    folder = world.folder / "live" / BROKER
    folder.mkdir(parents=True, exist_ok=True)
    return Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=Mode.LIVE,
        now=world.fake.clock.now,
    )


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
