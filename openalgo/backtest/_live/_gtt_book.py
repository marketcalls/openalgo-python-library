"""OpenAlgo's GTT book read for the live layer (20-positional-live 3.1, 3.6; the lead's
rulings N3a-1 and N3a-2): each row of a `gttorderbook` answer as a `GttRow`, what its
status means to the engine (`Seen`), and the comparisons of a row with a GTT's terms, by
value whatever field the trigger was sent in.

The answer is read as the SDK returns it (openalgo/gtt.py, `gttorderbook`: `data`, a list
of rows with `trigger_id`, `status`, `symbol`, `exchange`, `trigger_prices` and `legs`,
the sandbox's legs naming `triggered_order_id`; services/gtt_orderbook_service.py). A
SINGLE GTT has one trigger and one leg. A book that is not a success, or a row that
cannot be read, makes the whole read unreadable: a failed read is never taken as a GTT
gone (7.1).

| Status | `Seen` | Meaning to the engine |
|---|---|---|
| active, transit | RESTING | it rests at the broker |
| triggered | TRIGGERED | it fired: find its child |
| (absent from a read that succeeded) | GONE | fired, cancelled, or ended outside |
| cancelled, deleted, disabled | ENDED | ended without firing |
| expired | EXPIRED | ended by the broker's expiry |
| rejected | REJECTED | refused by the broker after placement |
| anything else | UNKNOWN | taken as resting, with an alert once per GTT |

Prices are rupees, quantities whole units.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._journal_gtt_rows import GttTerms
from ._venue_requests import SAME_PRICE_RUPEES


class Seen(enum.Enum):
    """What a GTT's row says of it (module docstring's table)."""

    RESTING = "resting"
    TRIGGERED = "triggered"
    GONE = "gone"
    ENDED = "ended"
    EXPIRED = "expired"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @property
    def rests(self) -> bool:
        """True when the GTT is taken as resting (an unknown status included)."""
        return self in (Seen.RESTING, Seen.UNKNOWN)


#: Each status word a broker or the sandbox writes, as the engine reads it (lower case).
STATUS_WORDS = {
    "active": Seen.RESTING,
    "transit": Seen.RESTING,
    "triggered": Seen.TRIGGERED,
    "cancelled": Seen.ENDED,
    "deleted": Seen.ENDED,
    "disabled": Seen.ENDED,
    "expired": Seen.EXPIRED,
    "rejected": Seen.REJECTED,
}


@dataclass(frozen=True)
class GttRow:
    """One row of the GTT book: its broker `trigger_id`, `status` word, instrument, its
    trigger prices (rupees, as the book lists them), its first leg's side, units, limit
    (rupees) and product, and in the sandbox the child order it placed (None before it
    fired, and on a broker's book, which never names it)."""

    trigger_id: str
    status: str
    symbol: str
    exchange: str
    triggers: tuple[float, ...]
    side: str
    quantity: int
    limit: float
    product: str
    child_order_id: str | None = None

    @property
    def seen(self) -> Seen:
        """What the row's status means (module docstring)."""
        return STATUS_WORDS.get(self.status.strip().lower(), Seen.UNKNOWN)

    def shows(self, terms: GttTerms) -> bool:
        """True when the row holds `terms`' trigger (among its trigger prices, by value)
        and units; the limit is the broker's to set (20-positional-live 3.6)."""
        return self.quantity == terms.quantity and any(
            _same(price, terms.trigger_price) for price in self.triggers
        )

    def has_shape(self, symbol: str, exchange: str, product: str, terms: GttTerms) -> bool:
        """True when the row is a GTT of `terms` on `exchange:symbol` in `product`, its
        side included: the shape an unanswered place is looked for by (3.1)."""
        place = (self.symbol, self.exchange, self.product, self.side)
        return place == (symbol, exchange, product, terms.side) and self.shows(terms)


def seen_of(rows: Mapping[str, GttRow], trigger_id: str) -> Seen:
    """What a read that succeeded says of the GTT `trigger_id`: its row's meaning, or GONE
    when the book has no row for it."""
    row = rows.get(trigger_id)
    return Seen.GONE if row is None else row.seen


def gtt_rows(answer: Mapping[str, Any]) -> dict[str, GttRow] | None:
    """The rows of a `gttorderbook` answer by trigger ID, or None when it is not a
    success or a row cannot be read (module docstring)."""
    data = answer.get("data") if answer.get("status") == "success" else None
    if not isinstance(data, list):
        return None
    rows: dict[str, GttRow] = {}
    for raw in data:
        row = _row(raw)
        if row is None:
            return None
        rows[row.trigger_id] = row
    return rows


def _row(raw: Any) -> GttRow | None:
    if not isinstance(raw, Mapping):
        return None
    legs, triggers = raw.get("legs"), raw.get("trigger_prices")
    if not isinstance(legs, list) or not legs or not isinstance(legs[0], Mapping):
        return None
    if not isinstance(triggers, list) or not triggers:
        return None
    leg = legs[0]
    try:
        prices = tuple(float(price) for price in triggers)
        quantity, limit = float(leg.get("quantity")), float(leg.get("price") or 0.0)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (*prices, quantity, limit)) or not quantity.is_integer():
        return None
    trigger_id = str(raw.get("trigger_id") or "")
    if not trigger_id:
        return None
    child = leg.get("triggered_order_id")
    return GttRow(
        trigger_id=trigger_id,
        status=str(raw.get("status") or ""),
        symbol=str(raw.get("symbol") or ""),
        exchange=str(raw.get("exchange") or ""),
        triggers=prices,
        side=str(leg.get("action") or "").upper(),
        quantity=int(quantity),
        limit=limit,
        product=str(leg.get("product") or "").upper(),
        child_order_id=str(child) if child else None,
    )


def _same(seen: float, sent: float) -> bool:
    return math.isclose(seen, sent, rel_tol=0.0, abs_tol=SAME_PRICE_RUPEES)


__all__ = ["STATUS_WORDS", "GttRow", "Seen", "gtt_rows", "seen_of"]
