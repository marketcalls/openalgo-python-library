"""Two strategies of one session on one instrument, on the fake OpenAlgo, shared by the
tests of shared instruments (`test_session_shared_*.py`; 18-multi-strategy 4 and 5.2,
unit M6).

The market is `session_samples`': SBIN on NSE on Monday 2026-11-16, the session started at
14:40:05 IST, SBIN trading 10 units every 10 seconds at the script's price (500.00 unless
a test moves it). "A" is `SbinTrend` and "B" `SbinBreakout`, both on SBIN in MIS, A first
in call order: at their first decision (14:42:01, both filled at once at 500.00) A buys
100 units with a stop 3 points below (an SL-M at 497.00) and a target 2 points above
(502.00), B buys 50 units with a stop 3 points below (497.00); then neither trades again.
`LateBreakout` is a B that buys its 50 at its second decision (14:43:01), so that its
position opened after A's, and `LateTrend` an A without a target that buys its 100 so;
`SbinShort` is a B that sells 50 instead, for the side gate.
`ShortTrend` is an A that sells 100 with a stop 3 points above (an SL-M buy at 503.00),
and `LateShort` a B that sells 50 so at its second decision.
`OnceA` and `OnceB` buy A's and B's units once a day, flat or after a restart (they
remember it in `self.state`), and `NeverB` never trades, for the restarts.

Expected values come from the fake's books and the scenario, never from the session's
code.
"""

from __future__ import annotations

from typing import Any

from session_samples import ist
from stops_samples import STOP_POINTS, StopHold

import openalgo.backtest as bt

#: A's and B's units, A's target in points above the entry.
A_UNITS = 100
B_UNITS = 50
TARGET_POINTS = 2.0
#: The first decision of both strategies, when their entries are decided.
DECIDED = ist(14, 42, 1)


class SbinTrend(StopHold):
    """A: buys 100 SBIN at its first decision with a stop 3 points below and a target 2
    points above."""

    qty = A_UNITS
    rules: dict[str, Any] = {"stop": bt.points(STOP_POINTS), "target": bt.points(TARGET_POINTS)}


class SbinBreakout(StopHold):
    """B: buys 50 SBIN at its first decision with a stop 3 points below."""

    qty = B_UNITS


class LateBreakout(StopHold):
    """B as `SbinBreakout`, buying its 50 units at its second decision (14:43:01) instead:
    the position of the two that opened most recently."""

    qty = B_UNITS

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.buy(self.qty, **type(self).rules)


class LateTrend(StopHold):
    """A as `SbinTrend` without its target, buying its 100 units at its second decision
    (14:43:01) instead: the position of the two that opened most recently when B buys at
    its first."""

    qty = A_UNITS

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.buy(self.qty, **type(self).rules)


class SbinShort(bt.Strategy):
    """A B that sells 50 SBIN at its first decision with a stop 3 points above, then
    never trades again."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(B_UNITS, stop=bt.points(STOP_POINTS))


class ShortTrend(bt.Strategy):
    """A short: sells 100 SBIN at its first decision with a stop 3 points above (an SL-M
    buy at 503.00), then never trades again."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(A_UNITS, stop=bt.points(STOP_POINTS))


class LateShort(ShortTrend):
    """B short: sells 50 SBIN at its second decision (14:43:01) with a stop 3 points above,
    so its position opened after `ShortTrend`'s."""

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.sell(B_UNITS, stop=bt.points(STOP_POINTS))


class OnceA(bt.Strategy):
    """A that buys 100 SBIN with a stop 3 points below at its first decision of the day,
    flat, and never again (it remembers across a restart, in `self.state`)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty = A_UNITS

    def on_bar(self, bar):
        if not self.state.get("bought") and not self.position.qty:
            self.state["bought"] = True
            self.buy(type(self).qty, stop=bt.points(STOP_POINTS))


class OnceB(OnceA):
    """B as `OnceA`, of 50 units."""

    qty = B_UNITS


class NeverB(bt.Strategy):
    """A B on SBIN that never trades."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        return None


def sbin_orders(world: Any, strategy: str) -> list[Any]:
    """The fake's orders of `strategy` on SBIN, oldest first."""
    return [o for o in world.orders() if o.strategy == strategy and o.symbol == "SBIN"]


def terms(orders: list[Any]) -> list[tuple[str, int, str, str]]:
    """(action, units, price type, status) of each order."""
    return [(o.action, o.quantity, o.price_type, o.status) for o in orders]


__all__ = [
    "A_UNITS",
    "B_UNITS",
    "DECIDED",
    "TARGET_POINTS",
    "LateBreakout",
    "LateShort",
    "LateTrend",
    "NeverB",
    "OnceA",
    "OnceB",
    "SbinBreakout",
    "SbinShort",
    "SbinTrend",
    "ShortTrend",
    "sbin_orders",
    "terms",
]
