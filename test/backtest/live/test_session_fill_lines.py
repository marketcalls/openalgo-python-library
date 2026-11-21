"""The ENTRY and EXIT lines of a live session's fills, on the console and in the alerts
(`_live/_session_news.py`; 2.4.0 item P5-1): in a session of more than one strategy each
line names the strategy whose order filled, right after its kind, so a trader reading the
screen or a phone can tell which strategy bought or sold; a session of one strategy keeps
its lines as they were.

The market and the scripts are `session_samples`' and `shared_samples`': the session
starts at 14:40:05 IST on Monday 2026-11-16 and SBIN (and INFY, where a test adds it)
trades 10 units every 10 s at 500.00 unless a test moves it; the fake fills a MARKET order
at the last trade's price. On one symbol, A (`SbinTrend`, 100 units, stop 497.00, target
502.00) and B (`SbinBreakout`, 50 units, stop 497.00) buy at 14:42:01, A first; SBIN
trades at 502.00 from 14:44, which reaches A's target, and B is squared off at 15:00.
"""

from __future__ import annotations

import logging
import re

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import BuyAndHold, alerts, ist, rig
from shared_samples import SbinBreakout, SbinTrend
from stops_samples import at_price, stop_world
from watch_samples import HoldInfy, HoldSbin

from openalgo.backtest._live._types import LOGGER_NAME

#: A fill's line after its kind: the strategy when the session names one, the order, and
#: the order's broker ID.
FILL = re.compile(
    r"(?:(?P<strategy>\w+) )?(?P<action>BUY|SELL) (?P<units>\d+) (?P<symbol>\w+) NSE at "
    r"(?P<price>[\d.]+) \((?P<why>[^,]+), order (?P<order_id>\w+)\)"
)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def fill_lines(lines: list[str]) -> list[tuple[str, re.Match]]:
    """The console's ENTRY and EXIT lines, each as (its kind, its text matched by `FILL`)."""
    found = []
    for line in lines:
        kind, _, text = line.partition(": ")
        if kind in ("ENTRY", "EXIT"):
            matched = FILL.fullmatch(text)
            assert matched is not None, line
            found.append((kind, matched))
    return found


def shown(found: list[tuple[str, re.Match]]) -> list[str]:
    """Each fill line up to its order's broker ID, which the fake numbers."""
    return [f"{kind}: {matched.string.split(', order ')[0]}" for kind, matched in found]


def test_in_a_session_of_two_strategies_on_one_symbol_each_fill_line_names_its_strategy(world):
    at_price(world, ist(14, 44), 502.0)
    world.run([SbinTrend, SbinBreakout])
    assert shown(fill_lines(world.lines)) == [
        "ENTRY: SbinTrend BUY 100 SBIN NSE at 500.00 (entry",
        "ENTRY: SbinBreakout BUY 50 SBIN NSE at 500.00 (entry",
        "EXIT: SbinTrend SELL 100 SBIN NSE at 502.00 (exit: target level",
        "EXIT: SbinBreakout SELL 50 SBIN NSE at 502.00 (square_off: square-off time",
    ]


def test_the_strategy_a_fill_line_names_placed_the_order_it_names(world):
    # The fake's book says which strategy placed each order (the order's strategy field).
    at_price(world, ist(14, 44), 502.0)
    world.run([SbinTrend, SbinBreakout])
    placed_by = {order.order_id: order.strategy for order in world.orders()}
    found = fill_lines(world.lines)
    assert len(found) == 4
    for _, matched in found:
        assert placed_by[matched["order_id"]] == matched["strategy"]


def test_in_a_session_of_two_strategies_each_alert_names_its_strategy_after_the_prefix(
    world, caplog
):
    at_price(world, ist(14, 44), 502.0)
    world.run([SbinTrend, SbinBreakout])
    prefix = "LIVE (SbinTrend, SbinBreakout): "
    entries = [text.split(", order ")[0] for text in alerts(caplog, "entry")]
    exits = [text.split(", order ")[0] for text in alerts(caplog, "exit")]
    assert entries == [
        f"{prefix}SbinTrend BUY 100 SBIN NSE at 500.00 (entry",
        f"{prefix}SbinBreakout BUY 50 SBIN NSE at 500.00 (entry",
    ]
    assert exits == [
        f"{prefix}SbinTrend SELL 100 SBIN NSE at 502.00 (exit: target level",
        f"{prefix}SbinBreakout SELL 50 SBIN NSE at 502.00 (square_off: square-off time",
    ]


def test_in_a_session_of_two_strategies_on_two_symbols_each_fill_line_names_its_strategy(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # HoldSbin and HoldInfy each buy 10 at their first decision at 500.00; Ctrl+C at 14:44
    # sells both.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    made.stop_at(ist(14, 44))
    made.run([HoldSbin, HoldInfy])
    assert shown(fill_lines(made.lines)) == [
        "ENTRY: HoldSbin BUY 10 SBIN NSE at 500.00 (entry",
        "ENTRY: HoldInfy BUY 10 INFY NSE at 500.00 (entry",
        "EXIT: HoldSbin SELL 10 SBIN NSE at 500.00 (risk_exit: Ctrl+C or a retried exit",
        "EXIT: HoldInfy SELL 10 INFY NSE at 500.00 (risk_exit: Ctrl+C or a retried exit",
    ]


def test_a_session_of_one_strategy_keeps_its_fill_lines_without_a_name(world, caplog):
    # BuyAndHold buys 10 at its first decision and sells them three bars later itself.
    world.stop_at(ist(14, 47, 30))
    world.run(BuyAndHold)
    assert shown(fill_lines(world.lines)) == [
        "ENTRY: BUY 10 SBIN NSE at 500.00 (entry",
        "EXIT: SELL 10 SBIN NSE at 500.00 (exit: strategy exit",
    ]
    [entry] = alerts(caplog, "entry")
    assert entry.startswith("LIVE (BuyAndHold): BUY 10 SBIN NSE at 500.00 (entry, order ")
