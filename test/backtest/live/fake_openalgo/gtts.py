"""The fake OpenAlgo's GTT orders: SINGLE triggers resting at the broker or in the sandbox,
their IDs, the rows of the GTT book, and the rule that fires one on a trade.

Written by hand from the server's code: the sandbox's GTT engine (sandbox/gtt_manager.py:
`_generate_gtt_id`, `leg_is_triggered_by`, `_serialize`, `_fire_claimed_leg`), the GTT book
service (services/gtt_orderbook_service.py, active rows first) and the reference broker's
mapping (broker/zerodha/mapping/gtt_data.py, `map_gtt_book`).

- A GTT fires when a trade reaches its trigger from the side it was placed on: a trigger
  sent in `triggerprice_sl` sits below the price and fires on a trade at or below it, one
  in `triggerprice_tg` above it and fires at or above it (`fires`). Firing ends it as
  "triggered" and places its child, a LIMIT order at the GTT's limit for its symbol,
  exchange, product, side and units, which the book then works like any order.
- The book (`gtt_rows`): "active" lists the resting GTTs only; "all" adds the ended ones
  where the book keeps history (the sandbox, and a broker whose fake was built with
  `gtt_history=True`); a broker without history lists the active ones whatever is asked,
  so a GTT that fired or was cancelled is simply gone from it. Active rows come first,
  each group newest first. Only the sandbox's rows name the child (`triggered_order_id`)
  and the strategy.

The fake models no GTT margin and no expiry date: the sandbox's `margin_blocked` is left
out of its rows and `expires_at` is empty, as for a broker that does not give one. Prices
are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openalgo.backtest._live._types import Mode

from .clock import book_time, ist_moment

#: The statuses a GTT takes at the fake (services/gtt_orderbook_service.py lists the
#: brokers' own words as they come).
ACTIVE, TRIGGERED, CANCELLED = "active", "triggered", "cancelled"
#: Where a trigger sits against the price it was placed from: the sandbox's leg
#: directions (sandbox/gtt_manager.py, `_build_legs`).
BELOW, ABOVE = "below", "above"
#: The trigger type of a SINGLE GTT as the book writes it.
SINGLE_ROW = "single"
#: The child order's price type: the fake's GTTs carry a LIMIT child.
CHILD_PRICE_TYPE = "LIMIT"


@dataclass
class FakeGtt:
    """A SINGLE GTT: its child's side, units and limit (rupees), its trigger (rupees) and
    `direction` (`BELOW` or `ABOVE`), its status, the last price when it was placed or
    last modified, when it was made and last changed (UTC nanoseconds, None before any
    change), and the child order it placed when it fired (None before)."""

    trigger_id: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    product: str
    quantity: int
    price: float
    trigger: float
    direction: str
    created_ns: int
    last_price: float = 0.0
    status: str = ACTIVE
    updated_ns: int | None = None
    child_order_id: str | None = None

    @property
    def active(self) -> bool:
        """True while the GTT can still fire, be modified or be cancelled."""
        return self.status == ACTIVE


class GttIds:
    """Trigger IDs in the format of a book: the sandbox's "GTT-YYMMDD-<8 hex>" (from a
    counter, where the sandbox draws random hex, so a test can name them), a broker's
    digits after the IST date."""

    def __init__(self, mode: Mode) -> None:
        self._mode = mode
        self._count = 0

    def trigger_id(self, now_ns: int) -> str:
        """A new trigger ID at `now_ns` (UTC nanoseconds)."""
        self._count += 1
        day = ist_moment(now_ns).strftime("%y%m%d")
        if self._mode is Mode.SANDBOX:
            return f"GTT-{day}-{self._count:08x}"
        return f"{day}{self._count:06d}"


def fires(gtt: FakeGtt, price: float) -> bool:
    """Whether a trade at `price` fires an active `gtt`: at or below a trigger that sits
    below, at or above one that sits above (sandbox/gtt_manager.py, `leg_is_triggered_by`:
    inclusive)."""
    if not gtt.active:
        return False
    return price <= gtt.trigger if gtt.direction == BELOW else price >= gtt.trigger


def gtt_rows(
    gtts: list[FakeGtt], mode: Mode, *, status: str, history: bool
) -> list[dict[str, Any]]:
    """The rows of a GTT book answer for `status` ("active" or "all"): active rows first,
    then the ended ones where the book keeps `history` (always in the sandbox), each group
    newest first (module docstring)."""
    keeps = mode is Mode.SANDBOX or history
    newest = sorted(gtts, key=lambda gtt: gtt.created_ns, reverse=True)
    rows = [gtt for gtt in newest if gtt.active]
    if status == "all" and keeps:
        rows += [gtt for gtt in newest if not gtt.active]
    return [gtt_row(gtt, mode) for gtt in rows]


def gtt_row(gtt: FakeGtt, mode: Mode) -> dict[str, Any]:
    """One row of the GTT book: the reference broker's mapping (times as its book writes
    them), or the sandbox's `_serialize` (ISO times, the child and the strategy)."""
    leg: dict[str, Any] = {
        "action": gtt.action,
        "quantity": gtt.quantity,
        "price": float(gtt.price),
        "pricetype": CHILD_PRICE_TYPE,
        "product": gtt.product,
    }
    sandbox = mode is Mode.SANDBOX
    if sandbox:
        leg["triggered_order_id"] = gtt.child_order_id
    row = {
        "trigger_id": gtt.trigger_id,
        "trigger_type": SINGLE_ROW,
        "status": gtt.status,
        "symbol": gtt.symbol,
        "exchange": gtt.exchange,
        "trigger_prices": [float(gtt.trigger)],
        "last_price": float(gtt.last_price),
        "legs": [leg],
        "created_at": _time(gtt.created_ns, sandbox),
        "updated_at": "" if gtt.updated_ns is None else _time(gtt.updated_ns, sandbox),
        "expires_at": "",
    }
    if sandbox:
        row["strategy"] = gtt.strategy
    return row


def _time(ts_ns: int, sandbox: bool) -> str:
    return ist_moment(ts_ns).replace(tzinfo=None).isoformat() if sandbox else book_time(ts_ns)


__all__ = [
    "ABOVE",
    "ACTIVE",
    "BELOW",
    "CANCELLED",
    "CHILD_PRICE_TYPE",
    "SINGLE_ROW",
    "TRIGGERED",
    "FakeGtt",
    "GttIds",
    "fires",
    "gtt_row",
    "gtt_rows",
]
