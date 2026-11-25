"""A strategy's GTT order (validity "gtt") placed while its entries are locked is refused,
never sent, as an entry order is (`_session_orders`, `_session_unsent`; review finding 10
of unit N3b): a GTT entry the strategy decided in the loop step that locked its entries
(the kill switch, an unresolved GTT, Ctrl+C) is answered to the kernel as refused, so it
ends rejected, and no GTT request reaches the order worker.

The kernel session is played by hand (no server), as `test_kernel_gtts` plays it: SBIN on
NSE, 1-minute slices of Tuesday 2025-07-01, flat at 500.00. The strategy's part of the
session around the kernel is a minimal double of the loop's `Trader`.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

from test_kernel_gtts import new_session, release_minute

from openalgo.backtest._live._session_orders import TraderOrders
from openalgo.backtest._live._venue_gtts import GttMap

STRATEGY = "GttEntry"


class Orders(TraderOrders):
    """The orders part of a `Trader` of `STRATEGY` over the kernel session `engine`."""

    def __init__(self, engine: Any) -> None:
        self.name, self.product, self.engine = STRATEGY, "CNC", engine
        self.part = SimpleNamespace(context=SimpleNamespace(resting_reversals=set()))
        self.venue = SimpleNamespace(withdraw_locked=list, requests=None)
        self.gtts = GttMap(
            engine,
            strategy=STRATEGY,
            markets=[("SBIN", "NSE")],
            product="CNC",
            request_ids=itertools.count(1),
            now=lambda: 0,
        )
        self.finished = False
        self.reasons: dict[int, str] = {}
        self.units_at_place: dict[int, int] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._event = lambda kind, **fields: self.events.append((kind, fields))
        self._warn = lambda text: None


def gtt_entry() -> dict[str, Any]:
    """A GTT buy of 10 SBIN CNC at 499.00, resting until SBIN trades at it."""
    return {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": 10.0},
        "order_type": "LIMIT",
        "price": 499.0,
        "validity": "GTT",
        "product": "CNC",
        "reduce_only": False,
        "reverse": False,
        "rules": {},
    }


def decided() -> Any:
    """A session whose strategy decided a GTT entry at the close of minute 0."""
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, _, _ = engine.submit(gtt_entry())
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    return engine


def test_a_gtt_entry_is_sent_while_entries_are_allowed():
    orders = Orders(decided())

    (request,) = orders.commands()

    assert (request.action.value, request.purpose.value) == ("place", "entry")


def test_a_gtt_entry_decided_before_its_entries_locked_is_refused_and_never_sent():
    engine = decided()
    engine.lock_entries("kill_switch")
    orders = Orders(engine)

    assert orders.commands() == []

    (gtt,) = engine.gtts(0, "CNC")
    assert gtt["state"] == "Rejected"
    (refused,) = [fields for kind, fields in orders.events if kind == "refused_locked"]
    assert (refused["gtt"], refused["locks"]) == (gtt["gtt_id"], ["kill_switch"])
