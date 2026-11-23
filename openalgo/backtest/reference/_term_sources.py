"""Where each term of an instrument comes from: the sources `_resolve` asks in order.

A source answers a term (lot size, multiplier, tick size, freeze quantity, settlement)
with a `Found` value and the reference row it came from, or None when it does not carry
the term: the row for the exact symbol (`SymbolRowSource`), the rules for a share or an
ETF (`CashSource`), the tables for the underlying of an NFO or BFO contract
(`ListedDerivativeSource`), and the contract-terms row of the underlying of an MCX or NSE
currency future (`ContractTableSource`).
"""

import datetime as _dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..errors import ReferenceDataError
from . import tick_rules
from ._rows import RefRow
from ._terms import Sources, contract_row, lot_size_row, underlying_kind
from .instruments import SINGLE_UNIT_LOT, UNIT_MULTIPLIER
from .symbols import ParsedSymbol

__all__ = [
    "CashSource",
    "ContractTableSource",
    "Found",
    "ListedDerivativeSource",
    "Request",
    "Source",
    "SymbolRowSource",
]


@dataclass(frozen=True)
class Found:
    """A term value, with the row it came from or the reason it holds without one."""

    value: object
    row: RefRow | None = None
    basis: str = ""


class Source(Protocol):
    """Anything that may answer a term of an instrument."""

    def find(self, term: str) -> Found | None:
        """The term from this source, or None when the source does not carry it."""
        ...


@dataclass
class Request:
    """One instrument being resolved: where its terms come from, its parsed symbol, the
    trade date, the reference price that picks a tick by price slab, the terms allowed to
    be missing, and the provenance of each term found so far."""

    sources: Sources
    parsed: ParsedSymbol
    day: _dt.date
    reference_price: float | None
    allow_missing: frozenset[str]
    provenance: list[tuple[str, str, str]] = field(default_factory=list)


class SymbolRowSource:
    """Terms stated for the exact symbol (a user row or a trusted snapshot row)."""

    def __init__(self, row: RefRow, terms: Sequence[str]) -> None:
        self._row = row
        self._terms = terms

    def find(self, term: str) -> Found | None:
        value = self._row.values.get(term) if term in self._terms else None
        return None if value is None else Found(value, self._row)


def _table_tick(request: Request, segment: str) -> Found:
    """Tick size from the slab rule of a segment; value None when the slab is undecided."""
    store, exchange, day = request.sources.store, request.parsed.exchange, request.day
    rows = tick_rules.slab_rows(store, exchange, segment, day)
    if not tick_rules.is_flat(rows) and request.reference_price is None:
        return Found(None, basis="by price slab; no reference price was given")
    row = tick_rules.slab_row(rows, exchange, segment, day, request.reference_price)
    return Found(row.values["tick_size"], row)


class CashSource:
    """Terms of a share or an ETF where no exact-symbol row states them."""

    def __init__(self, request: Request) -> None:
        self._request = request

    def find(self, term: str) -> Found | None:
        if term == "lot_size":
            return Found(SINGLE_UNIT_LOT, basis="cash equity trades in single shares")
        if term == "freeze_qty":
            return Found(None)  # no per-order quantity freeze is modelled for cash equity
        request = self._request
        parsed = request.parsed
        try:
            kind = underlying_kind(request.sources, parsed.symbol, parsed.exchange, request.day)
        except ReferenceDataError:
            return Found(None, basis="share or ETF not known; no 'underlyings' row")
        if kind != "etf":
            return _table_tick(request, "equity")
        try:
            return _table_tick(request, tick_rules.ETF_SEGMENT)
        except ReferenceDataError:
            return Found(None, basis="ETF; the share tick rule does not apply")


class ListedDerivativeSource:
    """Terms of an NFO or BFO contract from the tables for its underlying."""

    def __init__(self, request: Request, kind: str) -> None:
        self._request = request
        self._segment = f"{kind}_{request.parsed.kind}"
        self._finders: dict[str, Callable[[], Found]] = {
            "lot_size": self._lot_size,
            "multiplier": lambda: Found(UNIT_MULTIPLIER, basis=_PER_UNIT),
            "tick_size": lambda: _table_tick(request, self._segment),
            "freeze_qty": self._freeze_qty,
            "settlement": self._settlement,
        }

    def find(self, term: str) -> Found | None:
        return self._finders[term]()

    def _lot_size(self) -> Found:
        request, parsed = self._request, self._request.parsed
        row = lot_size_row(
            request.sources, parsed.symbol, parsed.exchange, request.day, None, parsed
        )
        return Found(row.values["lot_size"], row)

    def _freeze_qty(self) -> Found:
        request, parsed = self._request, self._request.parsed
        key = (parsed.underlying, parsed.exchange)
        row = request.sources.store.require("freeze_qty", key, request.day)
        return Found(row.values[f"freeze_qty_{parsed.kind}s"], row)

    def _settlement(self) -> Found:
        request = self._request
        key = (request.parsed.exchange, self._segment)
        row = request.sources.store.require("settlement", key, request.day)
        return Found(row.values["settlement"], row)


_PER_UNIT = "price is quoted per unit of the underlying"


class ContractTableSource:
    """Terms of an MCX or NSE currency futures contract from the contract-terms row of its
    underlying (`mcx_contracts`, `currency_contracts`)."""

    def __init__(self, request: Request, table: str) -> None:
        self._request = request
        self._table = table
        self._key = (request.parsed.underlying, request.parsed.exchange)

    def find(self, term: str) -> Found | None:
        store, day = self._request.sources.store, self._request.day
        expiry = self._request.parsed.expiry
        row = contract_row(store, self._table, self._key, day, expiry, term)
        value = row.values.get(term)
        if value is None:
            raise store.missing(self._table, self._key, day, detail=f" with a {term} value")
        return Found(value, row)
