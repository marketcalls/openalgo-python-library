"""The two entry gates the session sets on the members of a shared instrument, on the fake
OpenAlgo (`_live/_groups.py`; 18-multi-strategy 4.13 and 4.14a item 1, rows F95, F113 and
F119, unit M6): an entry, add or reversal that would put a member on the other side of
another member's position or pending entry is refused (`group_other_side`), a member is
refused every entry while another reverses, and a strategy's own resting orders are
refused on a shared instrument (`shared_resting`). A group of one sets neither.

The market and the strategies are `shared_samples`': SBIN at 500.00, A and B decide first
at 14:42:01 IST, A first. Every test runs the session invariants, I6 (no two members on
opposite sides at any change) among them.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import ist
from shared_samples import A_UNITS, B_UNITS, SbinBreakout, SbinShort, SbinTrend, sbin_orders
from stops_samples import STOP_POINTS, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._groups import GROUP_REVERSING_LOCK
from openalgo.backtest._live._types import LOGGER_NAME


class ShortLater(bt.Strategy):
    """A B that sells 50 SBIN at its second decision (14:43:01), when A holds its long."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    #: Every order event the strategy was given (the session builds the strategy, so a
    #: test reads the class).
    seen: list[dict] = []

    def init(self):
        self.calls = 0

    def on_order(self, order):
        type(self).seen.append(dict(order))

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.sell(B_UNITS, stop=bt.points(STOP_POINTS))


class LongThenReverse(bt.Strategy):
    """An A that buys 100 SBIN at its first decision and orders a reversal to a short of
    100 (a sale of 200, `reverse=True`) at its second, then exits at its third."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(A_UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.sell(2 * A_UNITS, reverse=True)
        elif self.calls == 3:
            self.exit()


class LongAlong(SbinBreakout):
    """A B that buys 100 SBIN at its first decision with a stop 3 points below."""

    qty = A_UNITS


class BuyAtSecond(bt.Strategy):
    """A flat B that buys 50 SBIN at its second decision (14:43:01)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.buy(B_UNITS, stop=bt.points(STOP_POINTS))


class LimitBuy(bt.Strategy):
    """Buys 100 at a LIMIT of 499.00 at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(A_UNITS, price=499.0)


class InfyLimitBuy(LimitBuy):
    """`LimitBuy` alone on INFY."""

    symbol = "INFY"


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def reject_codes(outcome, strategy: str) -> list[str]:
    """The reject codes of `strategy`'s kernel orders, oldest first."""
    codes = outcome.outputs[strategy]["orders"]["reject_code"]
    return [code for code in codes if code is not None]


def test_F95_an_entry_on_the_other_side_of_a_members_position_is_refused(world):
    outcome = world.run([SbinTrend, ShortLater])

    assert reject_codes(outcome, "ShortLater") == ["group_other_side"]
    assert sbin_orders(world, "ShortLater") == []
    # A keeps its long and its stop; the square-off closes it.
    assert [o.quantity for o in sbin_orders(world, "SbinTrend") if o.action == "BUY"] == [100]
    assert world.fake.position("SBIN", "NSE") == 0


def test_F95_variant_an_entry_decided_after_a_members_entry_in_one_slice_is_refused(world):
    outcome = world.run([SbinTrend, SbinShort])

    # A's long entry was decided first in the same step and had not filled: it counts.
    assert reject_codes(outcome, "SbinShort") == ["group_other_side"]
    assert sbin_orders(world, "SbinShort") == []
    assert [o.quantity for o in sbin_orders(world, "SbinTrend") if o.action == "BUY"] == [100]


def test_F95_the_refused_entry_reaches_the_strategy_in_on_order(world):
    ShortLater.seen = []
    world.run([SbinTrend, ShortLater])

    rejected = [(e["name"], e["reject_code"]) for e in ShortLater.seen]
    assert rejected == [("Rejected", "group_other_side")]


def test_F113_a_reversal_across_another_members_long_is_refused_whole(world):
    outcome = world.run([LongThenReverse, LongAlong])

    assert reject_codes(outcome, "LongThenReverse") == ["group_other_side"]
    sells = [o for o in sbin_orders(world, "LongThenReverse") if o.price_type == "MARKET"]
    # No sale of 200: A keeps its long until its own exit, which is accepted.
    assert [(o.action, o.quantity) for o in sells] == [("BUY", 100), ("SELL", 100)]
    assert outcome.outputs["LongThenReverse"]["trades"]["exit_reason"] == ["signal"]


def test_a_member_is_refused_every_entry_while_another_member_reverses(world):
    # A's reversal (a sale of 200 against its long of 100) is an entry: the invariants
    # hold its closing part as an exit and follow A's leg to the short (2.4.0 item P2-3).
    outcome = world.run([LongThenReverse, BuyAtSecond])

    # A's reversal works when B decides in the same step: B's buy would end long against
    # A's short, so it is refused on the reversal's side, and B is locked meanwhile.
    assert reject_codes(outcome, "BuyAtSecond") == ["group_other_side"]
    assert sbin_orders(world, "BuyAtSecond") == []
    reversal = [o for o in sbin_orders(world, "LongThenReverse") if o.quantity == 200]
    assert [(o.action, o.status) for o in reversal] == [("SELL", "complete")]
    locks = [e for e in outcome_events(outcome, "lock") if e["strategy"] == "BuyAtSecond"]
    assert GROUP_REVERSING_LOCK in [e["reason"] for e in locks]


def test_F119_a_resting_entry_on_a_shared_instrument_is_refused(world):
    outcome = world.run([LimitBuy, SbinBreakout])

    assert reject_codes(outcome, "LimitBuy") == ["shared_resting"]
    assert sbin_orders(world, "LimitBuy") == []
    # B's market entry and its resting stop go: the stop is the leg's, not a strategy order.
    assert [(o.price_type, o.status) for o in sbin_orders(world, "SbinBreakout")][:2] == [
        ("MARKET", "complete"),
        ("SL-M", "cancelled"),
    ]


def test_F119_variant_a_strategy_alone_on_its_instrument_keeps_its_resting_orders(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    outcome = world.run([InfyLimitBuy, SbinBreakout])

    assert reject_codes(outcome, "InfyLimitBuy") == []
    limits = [o for o in world.orders() if o.symbol == "INFY"]
    assert [(o.price_type, o.price) for o in limits] == [("LIMIT", 499.0)]


def test_two_members_on_one_side_both_enter(world):
    outcome = world.run([SbinTrend, SbinBreakout])

    assert reject_codes(outcome, "SbinTrend") == reject_codes(outcome, "SbinBreakout") == []
    buys = [(o.strategy, o.quantity) for o in world.orders() if o.action == "BUY"]
    assert buys == [("SbinTrend", A_UNITS), ("SbinBreakout", B_UNITS)]
    assert ist(14, 42) < world.orders()[0].placed_ns < ist(14, 42, 5)


def outcome_events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]
