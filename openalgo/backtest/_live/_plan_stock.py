"""A stock future or option named by a strategy of a live session, which `bt.live` does not
trade yet (the lead's ruling N5c-4 (f); review round 1 of unit N5c, finding 5): the plan
refuses it in these words when the reference tables cannot resolve it
(`_plan_strategies._instruments_today`), with what a backtest can do instead.

Only an exchange that lists stock futures and options has them
(`_futures_contracts.STOCK_FUTURE_EXCHANGES`, NFO and BFO): an MCX or CDS contract is a
commodity or a currency one, and keeps the words of its own refusal. Used by the main
thread at the start.
"""

from __future__ import annotations

import warnings
from typing import Any

from .._futures_contracts import STOCK_FUTURE_EXCHANGES
from ..errors import ReferenceDataError
from ..reference import ReferenceDataWarning, default_tables
from ..reference.symbols import SymbolError, parse_symbol

#: The refusal of a strategy on a stock future or option, with what a run can do instead.
STOCK_DERIVATIVES_NOT_LIVE = (
    "{symbol}: stock futures and stock options cannot be traded in bt.live yet. A backtest "
    "can trade one: pass it with instrument=Instrument(symbol=..., exchange=..., "
    'kind=..., underlying=..., expiry=..., lot_size=..., tick_size=..., settlement="physical") '
    "(Instrument from openalgo.backtest.reference); a stock future or option settles by "
    "delivery, so the engine exits it at the square-off of its last trading day"
)
#: What the reference tables call an underlying that makes a derivative a stock one; None
#: is an underlying they cannot tell, which may be a stock.
_STOCK_KINDS = ("stock", None)


def stock_derivative(plan: Any) -> str | None:
    """The first symbol of `plan` (a `_settings.RunPlan`: its `symbols`, `exchange` and
    `start` day) that is a future or an option on a stock, or None (module docstring): on
    NFO or BFO, an underlying the reference tables call a stock or cannot tell (an index
    or an ETF they list is not one)."""
    if plan.exchange not in STOCK_FUTURE_EXCHANGES:
        return None
    for symbol in plan.symbols:
        try:
            parsed = parse_symbol(symbol, plan.exchange)
        except SymbolError:
            continue
        if parsed.underlying is None:
            continue
        if _underlying_kind(parsed.underlying, plan) in _STOCK_KINDS:
            return symbol
    return None


def _underlying_kind(underlying: str, plan: Any) -> str | None:
    """What the reference tables call `underlying` on `plan`'s exchange and day, or None
    when they cannot tell. Only the refusal's words depend on it, after the run's own
    lookup was refused, so a row's warning (unverified, past its check date) is not
    issued a second time here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        try:
            return default_tables().underlying_type(underlying, plan.exchange, plan.start)
        except ReferenceDataError:
            return None


__all__ = ["STOCK_DERIVATIVES_NOT_LIVE", "stock_derivative"]
