"""The account's daily loss limit and the kill switch of a live session, on the fake
OpenAlgo (`_live/watch.py`, `_live/_kill_switch.py`; 16-session-design 5.4, 6.3 and 7.2
row F46, unit 11): the limit fires the kill switch of every strategy once, which locks
their entries for the day and exits each position by the quantity the engine and the
broker agree on, or alerts and sends nothing when they do not, or after a mode change.

The market and the script are `session_samples`': SBIN and INFY trade 10 units every
10 s from 14:40:05 IST at 500.00, falling to 492.00 from 14:43:40 unless a test says
otherwise. `HoldSbin` and `HoldInfy` buy 10 units at their first decision (14:42:01) and
again at any decision at which they are flat; each strategy's own daily loss limit is
Rs 5,000, so only the account's limit acts.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist, rig
from watch_samples import HoldInfy, HoldSbin, events, run

import openalgo.backtest as bt
from openalgo.backtest._live._kill_switch import KILL_REASONS, KillSwitch, exit_agreed
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import ConfigError

#: The account's limit: below either strategy's loss of 80.00 at 492.00, at or below both.
ACCOUNT_LIMIT = 150.0
FALL_AT = ist(14, 43, 40)
#: The first trade at 492.00: its tick reaches the kernel (tick mode), so the loss is
#: marked there, not at the bar's close.
FIRST_FALL_TICK = ist(14, 43, 45)
#: More refusals of a cancel than its retries (1, 2 and 5 s, then every 5 s) make
#: between the kill switch's fire and 14:44:30.
CANCELS_REFUSED = 20


@pytest.mark.parametrize(
    ("engine", "broker", "agreed"),
    [
        (10, 10, True),
        (10, 15, True),
        (-10, -10, True),
        (10, 4, False),
        (-10, -4, False),
        (10, 0, False),
        (10, -10, False),
        (10, None, False),
        (0, 0, False),
    ],
)
def test_an_exit_is_agreed_only_when_the_broker_holds_at_least_the_engines_units_same_side(
    engine, broker, agreed
):
    assert exit_agreed(engine, broker) is agreed


def test_the_kill_switch_refuses_a_reason_that_is_not_its_own():
    assert "daily_loss" not in KILL_REASONS
    with pytest.raises(ConfigError, match="kill switch"):
        KillSwitch(session=None).fire(trader=None, reason="daily_loss")


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.script.price = lambda now_ns: 492.0 if now_ns >= FALL_AT else 500.0
    world.stop_at(ist(14, 47))
    return world


def placed_after(world, action: str, start_ns: int) -> list[str]:
    return [o.symbol for o in world.orders() if o.action == action and o.placed_ns >= start_ns]


def test_F46_a_combined_loss_at_the_account_limit_fires_the_kill_switch_of_every_strategy(
    world,
):
    outcome = run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    fired = [(e["strategy"], e["reason"]) for e in events(outcome, "kill_switch")]
    assert fired == [("HoldSbin", "account_loss"), ("HoldInfy", "account_loss")]


def test_F46_the_kill_switch_exits_both_positions_as_risk_exits(world):
    outcome = run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    exits = [
        e["strategy"]
        for e in events(outcome, "request")
        if e["type"] == "PlaceRequest" and e["purpose"] == "risk_exit"
    ]
    assert exits == ["HoldSbin", "HoldInfy"]
    # The first tick at 492.00 marks the fall, before the 14:43 bar closes; the exits
    # fill long before the Ctrl+C.
    sells = [o.placed_ns for o in world.orders() if o.action == "SELL"]
    assert len(sells) == 2 and all(FIRST_FALL_TICK < placed < ist(14, 44) for placed in sells)
    assert (world.fake.position("SBIN", "NSE"), world.fake.position("INFY", "NSE")) == (0, 0)


def test_F46_entries_stay_locked_for_the_day_after_the_kill_switch(world):
    outcome = run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    assert placed_after(world, "BUY", FALL_AT) == []
    for name in ("HoldSbin", "HoldInfy"):
        assert "live_lock" in outcome.outputs[name]["orders"]["reject_code"]
    assert [e for e in events(outcome, "unlock") if e["reason"] == "account_loss"] == []


def test_F46_the_account_limit_is_alerted_once(world, caplog):
    run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    errors = [text for text in alerts(caplog, "error") if "account's daily loss" in text]
    assert len(errors) == 1


def test_F46_the_kill_switch_reports_each_strategy_flat_at_the_broker(world, caplog):
    run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    reports = [text for text in alerts(caplog, "error") if "flat at the broker too" in text]
    assert len(reports) == 2


def test_F46_a_flat_report_waits_for_a_read_begun_after_the_strategy_went_flat(world, caplog):
    # OpenAlgo's limiter refuses SBIN's exit twice, so it fills on a retry at about
    # 14:44:08, after INFY's exit (14:44:03) and after the read asked once INFY was flat
    # began (14:44:05), while SBIN still held its 10 units.
    world.script.at(ist(14, 44), lambda: world.fake.faults.rate_limit("placeorder"))
    world.script.at(ist(14, 44, 4), lambda: world.fake.faults.rate_limit("placeorder"))
    run(world, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT)
    assert (world.fake.position("SBIN", "NSE"), world.fake.position("INFY", "NSE")) == (0, 0)
    reports = [text for text in alerts(caplog, "error") if " is flat in " in text]
    assert len(reports) == 2
    assert all("flat at the broker too" in text for text in reports)


def test_a_partly_filled_kill_switch_exit_is_never_sent_twice(world):
    # The exit's first fill takes 4 units; its other 6 fill at the next trade.
    world.script.at(ist(14, 43, 50), lambda: world.fake.split_next_fill(4))
    run(world, HoldSbin, account_max_daily_loss=50.0)
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert [(o.quantity, o.filled) for o in sells] == [(10, 10)]
    assert world.fake.position("SBIN", "NSE") == 0


class HoldAndAdd(bt.Strategy):
    """Buys 10 SBIN at its first decision, and at its second rests a LIMIT buy of 10 more
    at 491.00."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.decisions = 0

    def on_bar(self, bar):
        self.decisions += 1
        if self.decisions == 1:
            self.buy(10)
        elif self.decisions == 2:
            self.buy(10, price=491.0)


def test_an_add_that_fills_before_its_cancel_is_exited_with_the_position(world):
    # The first tick at 492.00 (FIRST_FALL_TICK) marks the loss past the limit: the kill
    # switch fires in that loop step and cancels the resting add; a trade at 490.00 half
    # a second later fills it before the cancel reaches the broker.
    world.script.at(
        FIRST_FALL_TICK + NS // 2, lambda: world.fake.trade("SBIN", "NSE", 490.0, volume=10.0)
    )
    outcome = run(world, HoldAndAdd, account_max_daily_loss=50.0)
    asked = [(e["units"], e["outcome"]) for e in events(outcome, "kill_exit")]
    assert asked == [(20.0, "sent")]
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert [(o.quantity, o.filled) for o in sells] == [(20, 20)]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_position_that_opens_again_after_the_kill_switch_exit_is_exited_too(world):
    # OpenAlgo's limiter refuses the kill switch's cancel of the resting add and every
    # time it is sent again (16-session-design 4.5), so the add still works when the kill
    # switch exits the first 10 units after its wait; a trade at 490.00 at 14:44:30 fills
    # the add, opening 10 units again.
    for _ in range(CANCELS_REFUSED):
        world.fake.faults.rate_limit("cancelorder")
    world.script.at(ist(14, 44, 30), lambda: world.fake.trade("SBIN", "NSE", 490.0, volume=10.0))
    outcome = run(world, HoldAndAdd, account_max_daily_loss=50.0)
    asked = [(e["units"], e["outcome"]) for e in events(outcome, "kill_exit")]
    assert asked == [(10.0, "sent"), (10.0, "sent")]
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert [(o.quantity, o.filled) for o in sells] == [(10, 10), (10, 10)]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_loss_below_the_account_limit_fires_nothing(world):
    outcome = run(world, [HoldSbin, HoldInfy], account_max_daily_loss=200.0)
    assert events(outcome, "kill_switch") == []


def test_without_an_account_limit_nothing_fires(world):
    outcome = run(world, [HoldSbin, HoldInfy])
    assert events(outcome, "kill_switch") == []


@pytest.fixture
def sold_by_hand(world):
    """Another client of the account sells 6 of the 10 SBIN the strategy bought at
    14:43:46, right after the engine's loss passed the limit: the trade book explains the
    broker's 4 units, so the reconciler takes the sale as the engine's (16-session-design
    5.3 row 2)."""
    other = world.fake.client()
    world.script.at(
        FIRST_FALL_TICK + NS,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=6,
        ),
    )
    return world


@pytest.fixture
def lagging(world):
    """OpenAlgo's position book stops showing new fills at 14:41:30, before the strategy's
    entry: it shows SBIN flat while the engine holds 10 units and the trade book shows the
    entry, an unexplained difference (5.3 row 3). A second Ctrl+C follows the first at
    14:47:10, so the session ends once its exits are known."""
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    world.stop_at(ist(14, 47, 10))
    return world


def strategy_sells(world) -> list[tuple[int, int]]:
    return [
        (o.quantity, o.filled)
        for o in world.orders()
        if o.strategy == "HoldSbin" and o.action == "SELL"
    ]


def test_after_a_sale_by_hand_the_kill_switch_exits_only_what_is_left(sold_by_hand):
    outcome = run(sold_by_hand, HoldSbin, account_max_daily_loss=50.0, on_stop="keep")
    assert [(e["side"], e["units"]) for e in events(outcome, "adopted")] == [("SELL", 6)]
    assert strategy_sells(sold_by_hand) == [(4, 4)]
    assert sold_by_hand.fake.position("SBIN", "NSE") == 0


def test_the_kill_switch_alerts_once_when_the_broker_and_the_engine_disagree(lagging, caplog):
    run(lagging, HoldSbin, account_max_daily_loss=50.0, on_stop="keep")
    errors = [text for text in alerts(caplog, "error") if "no exit is sent until" in text]
    assert len(errors) == 1
    assert "holds 10 units of SBIN NSE" in errors[0] and "shows 0 units" in errors[0]


def test_with_on_stop_keep_the_disagreement_alert_names_only_the_square_off(lagging, caplog):
    run(lagging, HoldSbin, account_max_daily_loss=50.0, on_stop="keep")
    [error] = [text for text in alerts(caplog, "error") if "no exit is sent until" in text]
    assert "no exit of the session (the square-off at 15:00 IST)" in error
    assert "Ctrl+C" not in error


def test_while_the_book_shows_fewer_units_no_exit_of_any_owner_goes_beyond_them(lagging, caplog):
    # The default on_stop="flatten": the Ctrl+C at 14:47 asks the engine to exit its 10
    # units, which the position book does not show, so the exit is held back.
    outcome = run(lagging, HoldSbin, account_max_daily_loss=50.0)
    [error] = [text for text in alerts(caplog, "error") if "no exit is sent until" in text]
    assert 'a Ctrl+C (on_stop="flatten") and the square-off at 15:00 IST' in error
    assert events(outcome, "kill_exit") == []
    assert [e["strategy"] for e in events(outcome, "exit_held")] == ["HoldSbin"]
    assert strategy_sells(lagging) == []


def test_after_a_mode_change_the_kill_switch_alerts_and_sends_nothing(world, caplog):
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    run(world, HoldSbin, account_max_daily_loss=50.0)
    assert placed_after(world, "SELL", ist(14, 43, 30)) == []
    errors = [text for text in alerts(caplog, "error") if "cannot act" in text]
    assert len(errors) == 1


def test_after_a_mode_change_the_account_limit_alert_promises_no_exit(world, caplog):
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    run(world, HoldSbin, account_max_daily_loss=50.0)
    [error] = [text for text in alerts(caplog, "error") if "account's daily loss" in text]
    assert "exits every strategy's position" not in error
    assert "OpenAlgo's mode changed" in error
