"""Every alert of 08-live-trading section 8 ("Operations and safety": entry, exit, stop moves,
rejects, mode breach, reconciliation mismatch, feed dead, daily summary), sent in a fault
scenario through the session's alert path to the fake OpenAlgo's Telegram endpoint, and
asserted by its words as Telegram took them (16-session-design 7.2 and plan step 8, unit
14). Each scenario is the row's that names it: F31 (entry), F09 (exit), F25 (stop moves),
F11 (rejects), F35 (mode breach), F48 (reconciliation mismatch), F40 (feed dead); the
daily summary ends every session, here F29's.

The market is `stops_samples`': SBIN on NSE on Monday 2026-11-16 from 14:40:05 IST, 10 units
every 10 s at 500.00 unless a test moves the price. `StopHold` buys 10 SBIN at 14:42:01 with
an SL-M stop at 497.00. Alerts post under the test seam as they would on the sender's
thread, each run of routine alerts in one post and each urgent alert alone, with the
session's prefix ("LIVE (StopHold): ") and Telegram's Markdown escapes; `sent` reads them
back as plain text with the priority they were posted at.
"""

from __future__ import annotations

import logging
import re

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import ist
from stops_samples import STOP_TRIGGER, StopHold, at_price, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.alerts import ALERT_PRIORITY, AlertKind

#: The session's prefix of every alert of a strategy.
PREFIX = "LIVE ({name}): "
#: Telegram's legacy Markdown escapes (`_alerts_telegram.telegram_text`).
ESCAPED = re.compile(r"\\([_*`\[])")


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def sent(world, kind: AlertKind) -> list[str]:
    """The texts the fake's Telegram endpoint took at the priority of `kind`, unescaped."""
    priority = ALERT_PRIORITY[kind]
    return [
        ESCAPED.sub(r"\1", message.message)
        for message in world.fake.telegram_sent
        if message.priority == priority
    ]


def holding(texts: list[str], words: str) -> list[str]:
    return [text for text in texts if words in text]


def times(texts: list[str], words: str) -> int:
    """How many times `words` stand in `texts`: routine alerts that wait together leave in
    one post, so a post can hold an alert twice."""
    return sum(text.count(words) for text in texts)


class StopRoundTrip(StopHold):
    """StopHold that exits at its fourth decision (14:45:01)."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 4 and self.position.qty:
            self.exit()


def test_F31_the_entry_alert_names_the_fill_once_though_its_word_came_twice(world):
    world.fake.faults.duplicate_next_order_update()
    world.fake.faults.duplicate_next_order_update()
    world.run(StopRoundTrip)
    entry = world.orders()[0].order_id
    words = f"BUY 10 SBIN NSE at 500.00 (entry, order {entry})"
    assert times(sent(world, AlertKind.ENTRY), words) == 1


def test_F09_the_exit_alert_names_the_exit_that_filled_after_a_rate_limited_try(world):
    world.script.at(ist(14, 45, 2), lambda: world.fake.faults.rate_limit("placeorder"))
    world.run(StopRoundTrip)
    exit_order = world.orders()[-1].order_id
    # The strategy's exit() was refused at the rate limit; the session asked for it again
    # as a risk exit, which the kernel records as a manual exit.
    words = f"SELL 10 SBIN NSE at 500.00 (risk_exit: manual exit, order {exit_order})"
    assert times(sent(world, AlertKind.EXIT), words) == 1


class Trails(bt.Strategy):
    """Buys 10 SBIN at its first decision with a stop 3 points below that trails 3 points
    behind the best price."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, stop=bt.points(3), trail=bt.trail_points(3))


def test_F25_the_stop_moved_alert_names_the_brokers_new_trigger(world):
    at_price(world, ist(14, 43), 510.0)
    world.stop_at(ist(14, 46))
    world.run(Trails, on_stop="keep")
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    words = f"Trails: stop of SBIN NSE moved to {stop.trigger_price:.2f} (order {stop.order_id})"
    assert stop.trigger_price > STOP_TRIGGER
    assert holding(sent(world, AlertKind.STOP_MOVED), words) != []


def test_F11_the_reject_alert_names_the_stop_and_the_brokers_reason(world):
    world.fake.reject_next("trigger invalid", price_type="SL-M")
    world.stop_at(ist(14, 45))
    world.run(StopHold, on_stop="keep")
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    words = f"StopHold: SELL 10 SBIN NSE (stop) order {stop.order_id} rejected: trigger invalid"
    assert holding(sent(world, AlertKind.REJECT), words) != []


def test_F35_the_mode_breach_alert_says_the_switch_changed_and_nothing_more_is_sent(world):
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    world.stop_at(ist(14, 47))
    outcome = world.run(StopRoundTrip)
    world.trace.leave_unprotected("StopRoundTrip", "F35")
    breach = sent(world, AlertKind.MODE_BREACH)
    words = "OpenAlgo's analyzer switch changed during the session: it started in LIVE mode"
    assert times(breach, words) == 1
    assert holding(breach, "The session sends nothing more.") != []
    assert outcome.open_positions[0].stop_confirmed is False
    assert world.fake.orders(Mode.LIVE)[1].status == "trigger pending"


def test_F48_the_reconciliation_mismatch_alert_names_both_figures(world):
    other = world.fake.client()
    world.script.at(
        ist(14, 43),
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        ),
    )
    world.stop_at(ist(14, 46))
    world.run(StopHold, on_stop="keep")
    words = "StopHold SBIN NSE: the engine holds 10 units, OpenAlgo's books show 20"
    mismatch = holding(sent(world, AlertKind.RECONCILE_MISMATCH), words)
    assert times(mismatch, words) == 1
    assert "Entries of the strategy are locked for the rest of the session" in mismatch[0]


def test_F40_the_feed_dead_alert_says_entries_are_locked_and_quotes_watch_the_position(world):
    world.script.at(ist(14, 43), world.fake.faults.feed_dead)
    world.script.at(ist(14, 43, 15), world.fake.faults.feed_alive)
    world.stop_at(ist(14, 46))
    world.run(StopHold, on_stop="keep")
    words = (
        "no tick on any instrument for 10 s: entries are locked until the feed has ticked for "
        "a whole minute again; REST quotes of SBIN NSE every 2 s keep the targets and the "
        "daily loss watched"
    )
    assert times(sent(world, AlertKind.FEED_DEAD), words) == 1


def test_F29_the_daily_summary_comes_last_with_the_days_figures(world):
    world.fake.faults.silence_order_updates()
    world.run(StopRoundTrip)
    summaries = holding(sent(world, AlertKind.DAILY_SUMMARY), "Session of 2026-11-16 (IST)")
    assert times(summaries, "Session of 2026-11-16 (IST)") == 1
    text = summaries[0]
    assert f"{PREFIX.format(name='StopRoundTrip')}Session of 2026-11-16 (IST): LIVE mode" in text
    assert "Trades      : 1 closed" in text
    assert "End state   : flat" in text
