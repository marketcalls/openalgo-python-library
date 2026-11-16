"""The clock skew of a live session from its feed, on the fake OpenAlgo (`_live/watch.py`,
`_live/_clock_skew.py`; 16-session-design 6.4 and 7.2 row F61, unit 11): the session's
clock runs 3 s ahead of the fake's (the exchange's), on a server on this computer, which
the preflight's clock check cannot measure.

- Variant a: zerodha, whose feed gives the server's own clock as the last trade time in
  quote mode: no estimate; the banner and the result's warnings name the broker.
- Variant b: a broker whose feed gives the trade's exchange time: the estimate is about
  3 s, one ERROR alert in its first minute, and a result warning.

The market and the script are `session_samples`', with SBIN also trading every 2 s from
14:40:06 IST, so it is liquid in every window; watching begins at about 14:40:15.
"""

from __future__ import annotations

import functools
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, BuyAndHold, alerts, ist, rig
from watch_samples import events, on_this_computer, run

from openalgo.backtest._live._types import LOGGER_NAME

AHEAD_S = 3.0
#: Seconds between the extra trades that make SBIN liquid.
EVERY_S = 2


def liquid_world(make_fake_openalgo, monkeypatch, tmp_path, **fake_options):
    world = rig(functools.partial(make_fake_openalgo, **fake_options), monkeypatch, tmp_path)
    for instant in range(ist(14, 40, 6), ist(14, 45), EVERY_S * NS):
        world.script.at(instant, world.script.trade)
    return world


@pytest.fixture
def zerodha(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """Variant a, stopped at 14:43."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = liquid_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 43))
    return world


@pytest.fixture
def exchange_times(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """Variant b, the fake's broker "angel", stopped at 14:41:45, after the first window
    and before the second."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = liquid_world(make_fake_openalgo, monkeypatch, tmp_path, broker="angel")
    world.stop_at(ist(14, 41, 45))
    return world


def run_ahead(world):
    return run(world, BuyAndHold, ahead_s=AHEAD_S, clients=on_this_computer(world.clients()))


UNMEASURED = (
    "the clock against the exchange cannot be checked through zerodha's feed (its last "
    "trade time is the OpenAlgo server's own clock): keep this computer's clock "
    "synchronised"
)


def test_F61a_the_banner_warns_that_zerodhas_feed_cannot_check_the_clock(zerodha):
    run_ahead(zerodha)
    assert f"Warning: {UNMEASURED}" in zerodha.lines


def test_F61a_the_results_warnings_repeat_it(zerodha):
    outcome = run_ahead(zerodha)
    assert UNMEASURED in outcome.warnings


def test_F61a_no_estimate_is_made_and_nothing_is_alerted(zerodha, caplog):
    outcome = run_ahead(zerodha)
    assert events(outcome, "clock_skew") == []
    assert [text for text in alerts(caplog, "error") if "local clock" in text] == []


def test_F61b_the_estimate_is_about_3_s(exchange_times):
    outcome = run_ahead(exchange_times)
    [estimate] = events(outcome, "clock_skew")
    assert estimate["seconds"] == pytest.approx(AHEAD_S, abs=0.01)


def test_F61b_a_clock_3_s_ahead_is_alerted_once(exchange_times, caplog):
    run_ahead(exchange_times)
    errors = [text for text in alerts(caplog, "error") if "local clock" in text]
    assert len(errors) == 1
    assert "about 3.0 s ahead of the exchange's" in errors[0]


def test_F61b_the_results_warnings_keep_the_skew(exchange_times):
    outcome = run_ahead(exchange_times)
    assert [note for note in outcome.warnings if "3.0 s ahead" in note]


def test_F61b_a_broker_with_exchange_times_gets_no_unmeasured_warning(exchange_times):
    outcome = run_ahead(exchange_times)
    assert not [note for note in outcome.warnings if "cannot be checked" in note]


def test_a_clock_in_step_with_the_exchange_is_not_alerted(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = liquid_world(make_fake_openalgo, monkeypatch, tmp_path, broker="angel")
    world.stop_at(ist(14, 41, 45))
    outcome = run(world, BuyAndHold)
    [estimate] = events(outcome, "clock_skew")
    assert estimate["seconds"] == pytest.approx(0.0, abs=0.01)
    assert [text for text in alerts(caplog, "error") if "local clock" in text] == []
