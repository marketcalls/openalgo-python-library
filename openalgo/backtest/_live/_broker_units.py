"""The broker's units of a leg, by its product (20-positional-live 7.1; the lead's ruling
N2-1): the one place the live layer turns OpenAlgo's books into a leg's units, used by the
kill switch, the exit guard, held cancels, the reconciler and the instrument groups.

- **MIS and NRML**: the position book's net quantity of the symbol, exchange and product
  (a broker lists one line per product; every line of the leg's product adds up).
- **CNC**: the holdings' quantity of the symbol and exchange plus the CNC line of the
  position book. On the day of a buy the units are in the position book only; from a later
  day the broker shows them in the holdings, and a sale of held units shows as a negative
  CNC line until it settles. Whether a broker's holdings quantity includes the units still
  settling is checked per broker before release (20-positional-live 12, V1 and V2), not
  here.

A read that failed, or was not made, is never taken as zero units: the reader answers
None (unknown), and every caller reads again on its back-off (16-session-design 4.11). A
read that succeeded with no row of the symbol is zero. The answers are read as the SDK
returns them (openalgo/account.py: `positionbook()`, and `holdings()`, whose rows sit
under `data.holdings`). Quantities are signed whole units (long above 0).

**CNC is checked one-sided** (20-positional-live 7.2, decision 11.3): the broker's units
must be at least the strategy's for a long; units beyond them are the user's own
(`own_units`), never a mismatch, so a comparison takes the broker's units as the
strategy's (`strategy_units`). Every exit and GTT is for the strategy's units only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .._products import CNC, PRODUCTS
from ..errors import LiveError


def broker_units(
    positionbook: Mapping[str, Any],
    holdings: Mapping[str, Any] | None,
    symbol: str,
    exchange: str,
    product: str,
) -> int | None:
    """The broker's signed units of the leg `exchange:symbol` in `product` (module
    docstring), from a position-book answer and a holdings answer (None when holdings were
    not read, as a session with no CNC strategy reads none); None when a book the product
    needs failed or cannot be read.

    Raises LiveError for a product that is not MIS, CNC or NRML."""
    if product not in PRODUCTS:
        raise LiveError(
            f"the broker's units of {exchange}:{symbol} cannot be read for product "
            f"{product!r}: a live leg trades MIS, CNC or NRML"
        )
    position = _position_units(positionbook, symbol, exchange, product)
    if product != CNC or position is None:
        return position
    held = None if holdings is None else _holding_units(holdings, symbol, exchange)
    return None if held is None else held + position


def _position_units(
    answer: Mapping[str, Any], symbol: str, exchange: str, product: str
) -> int | None:
    """The net units of the position book's lines of the leg; None when it cannot be read."""
    rows = answer.get("data") if answer.get("status") == "success" else None
    if not isinstance(rows, list):
        return None
    return _units(rows, (symbol, exchange, product), ("symbol", "exchange", "product"))


def _holding_units(answer: Mapping[str, Any], symbol: str, exchange: str) -> int | None:
    """The units of the holdings' rows of `exchange:symbol`; None when they cannot be
    read. Every holding is delivery (CNC), whatever product a broker writes on it."""
    data = answer.get("data") if answer.get("status") == "success" else None
    rows = data.get("holdings") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        return None
    return _units(rows, (symbol, exchange), ("symbol", "exchange"))


def _units(rows: Sequence[Any], key: tuple[str, ...], fields: tuple[str, ...]) -> int | None:
    """The sum of the `quantity` of the rows whose `fields` are `key`, each rounded to
    whole units (brokers write it as a number or as text); None when a row is not a row or
    a quantity of the leg is not a finite number (a guess is never made)."""
    total = 0
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if tuple(row.get(field) for field in fields) != key:
            continue
        try:
            quantity = float(row.get("quantity", 0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(quantity):
            return None
        total += round(quantity)
    return total


def own_units(product: str, strategy_units: int, broker: int | None) -> int:
    """The units the broker holds beyond a CNC strategy's long or flat position (the
    user's own: module docstring); 0 for any other product, a short, or none beyond."""
    if product != CNC or broker is None or strategy_units < 0:
        return 0
    return max(0, broker - strategy_units)


def strategy_units(product: str, strategy: int, broker: int | None) -> int | None:
    """The broker's units of a leg as a comparison with the strategy's `strategy` units
    takes them: for CNC, without the user's own units beyond the strategy's (module
    docstring); for any other product, as they are (None stays None)."""
    if broker is None:
        return None
    return broker - own_units(product, strategy, broker)


__all__ = ["broker_units", "own_units", "strategy_units"]
