"""Assembling an `Instrument` from an OpenAlgo symbol and the reference rows.

Each term of an instrument (lot size, multiplier, tick size, freeze quantity,
settlement) is asked of a list of sources in order, and the first source that has it
wins: the row for the exact symbol (a user `symbols` row, or a symbol-master snapshot
from a broker whose adapter was read), then the tables for its underlying. A snapshot
that disagrees with a table is reconciled in the open (see `_terms.reconcile`).

On MCX, CDS and BCD an exact-symbol row replaces the tables altogether and must state
`lot_size`, `multiplier` and `settlement` itself: the server source shows that the lot
unit on those exchanges differs between brokers, so a lot from one place cannot be
combined with a multiplier from another.

The contract itself is checked before any term: the expiry in an NFO or BFO symbol must
be the listed expiry of a contract on the trade date (see `expiry.contract_for`), and
the instrument carries that contract's last trading day and identity.
"""

import datetime as _dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .._protocols import SessionCalendar
from ..errors import ConfigError, ReferenceDataError
from . import tick_rules
from ._rows import RefRow
from ._store import how_to_supply, warn
from ._terms import (
    MCX,
    MCX_TABLE,
    Sources,
    contract_row,
    lot_size_row,
    symbol_row,
    underlying_kind,
)
from .expiry import Expiry, alias_for, contract_for, has_rules
from .instruments import SINGLE_UNIT_LOT, UNIT_MULTIPLIER, Instrument
from .symbols import (
    LISTED_DERIVATIVE_EXCHANGES,
    ParsedSymbol,
    SymbolError,
    format_symbol,
    parse_symbol,
)

# Terms that may be absent from the reference data when the caller allows it.
OPTIONAL_TERMS = ("freeze_qty", "tick_size")
_EQUITY_TERMS = ("lot_size", "tick_size", "freeze_qty")
_DERIVATIVE_TERMS = ("lot_size", "multiplier", "tick_size", "freeze_qty", "settlement")
# What a snapshot row may state besides the lot: the tick only when the store trusts it
# (otherwise the row holds None for it).
_SNAPSHOT_TERMS = ("tick_size",)


@dataclass(frozen=True)
class _Found:
    """A term value, with the row it came from or the reason it holds without one."""

    value: object
    row: RefRow | None = None
    basis: str = ""


class _Source(Protocol):
    def find(self, term: str) -> _Found | None:
        """The term from this source, or None when the source does not carry it."""
        ...


@dataclass
class _Request:
    sources: Sources
    parsed: ParsedSymbol
    day: _dt.date
    reference_price: float | None
    allow_missing: frozenset[str]
    provenance: list[tuple[str, str, str]] = field(default_factory=list)


class _SymbolRowSource:
    """Terms stated for the exact symbol (a user row or a trusted snapshot row)."""

    def __init__(self, row: RefRow, terms: Sequence[str]) -> None:
        self._row = row
        self._terms = terms

    def find(self, term: str) -> _Found | None:
        value = self._row.values.get(term) if term in self._terms else None
        return None if value is None else _Found(value, self._row)


def _table_tick(request: _Request, segment: str) -> _Found:
    """Tick size from the slab rule of a segment; value None when the slab is undecided."""
    store, exchange, day = request.sources.store, request.parsed.exchange, request.day
    rows = tick_rules.slab_rows(store, exchange, segment, day)
    if not tick_rules.is_flat(rows) and request.reference_price is None:
        return _Found(None, basis="by price slab; no reference price was given")
    row = tick_rules.slab_row(rows, exchange, segment, day, request.reference_price)
    return _Found(row.values["tick_size"], row)


class _CashSource:
    """Terms of a share or an ETF where no exact-symbol row states them."""

    def __init__(self, request: _Request) -> None:
        self._request = request

    def find(self, term: str) -> _Found | None:
        if term == "lot_size":
            return _Found(SINGLE_UNIT_LOT, basis="cash equity trades in single shares")
        if term == "freeze_qty":
            return _Found(None)  # no per-order quantity freeze is modelled for cash equity
        request = self._request
        parsed = request.parsed
        try:
            kind = underlying_kind(request.sources, parsed.symbol, parsed.exchange, request.day)
        except ReferenceDataError:
            return _Found(None, basis="share or ETF not known; no 'underlyings' row")
        if kind != "etf":
            return _table_tick(request, "equity")
        try:
            return _table_tick(request, tick_rules.ETF_SEGMENT)
        except ReferenceDataError:
            return _Found(None, basis="ETF; the share tick rule does not apply")


class _ListedDerivativeSource:
    """Terms of an NFO or BFO contract from the tables for its underlying."""

    def __init__(self, request: _Request, kind: str) -> None:
        self._request = request
        self._segment = f"{kind}_{request.parsed.kind}"
        self._finders: dict[str, Callable[[], _Found]] = {
            "lot_size": self._lot_size,
            "multiplier": lambda: _Found(UNIT_MULTIPLIER, basis=_PER_UNIT),
            "tick_size": lambda: _table_tick(request, self._segment),
            "freeze_qty": self._freeze_qty,
            "settlement": self._settlement,
        }

    def find(self, term: str) -> _Found | None:
        return self._finders[term]()

    def _lot_size(self) -> _Found:
        request, parsed = self._request, self._request.parsed
        row = lot_size_row(
            request.sources, parsed.symbol, parsed.exchange, request.day, None, parsed
        )
        return _Found(row.values["lot_size"], row)

    def _freeze_qty(self) -> _Found:
        request, parsed = self._request, self._request.parsed
        key = (parsed.underlying, parsed.exchange)
        row = request.sources.store.require("freeze_qty", key, request.day)
        return _Found(row.values[f"freeze_qty_{parsed.kind}s"], row)

    def _settlement(self) -> _Found:
        request = self._request
        key = (request.parsed.exchange, self._segment)
        row = request.sources.store.require("settlement", key, request.day)
        return _Found(row.values["settlement"], row)


_PER_UNIT = "price is quoted per unit of the underlying"


class _McxFutureSource:
    """Terms of an MCX futures contract from the `mcx_contracts` row of its underlying."""

    def __init__(self, request: _Request) -> None:
        self._request = request
        self._key = (request.parsed.underlying, MCX)

    def find(self, term: str) -> _Found | None:
        store, day = self._request.sources.store, self._request.day
        expiry = self._request.parsed.expiry
        row = contract_row(store, MCX_TABLE, self._key, day, expiry, term)
        value = row.values.get(term)
        if value is None:
            raise store.missing(MCX_TABLE, self._key, day, detail=f" with a {term} value")
        return _Found(value, row)


def _no_source_error(request: _Request, term: str) -> ReferenceDataError:
    parsed = request.parsed
    return ReferenceDataError(
        f"No {term} is known for {parsed.symbol} on {parsed.exchange} on "
        f"{request.day.isoformat()}. The built-in tables cover NFO and BFO contracts and MCX "
        "futures only. Elsewhere the lot unit differs between brokers, so a 'symbols' row "
        "must state lot_size, multiplier (rupees per one-rupee move per unit of quantity) "
        f"and settlement for the exact contract. {how_to_supply('symbols')}"
    )


def _term(request: _Request, sources: Sequence[_Source], term: str) -> object:
    """One term of the instrument, recorded in the provenance of the request."""
    try:
        found = next((hit for hit in (s.find(term) for s in sources) if hit is not None), None)
        if found is None:
            raise _no_source_error(request, term)
    except ReferenceDataError as exc:
        if term not in request.allow_missing:
            raise
        parsed = request.parsed
        warn(
            f"reference data: {term} of {parsed.symbol} on {parsed.exchange} on "
            f"{request.day.isoformat()} is not covered and was allowed to be missing ({exc})",
            stacklevel=5,
        )
        request.provenance.append((term, "not covered", "missing"))
        return None
    if found.row is not None:
        request.sources.store.note_use(found.row, request.day)
        request.provenance.append((term, found.row.source, found.row.status))
    elif found.basis:
        status = "unresolved" if found.value is None else "assumed"
        request.provenance.append((term, found.basis, status))
    return found.value


def _listed_contract(request: _Request, calendar: SessionCalendar) -> Expiry | None:
    """The contract an NFO or BFO symbol names on the trade date; None without rules.

    A symbol that names a re-dated contract by its listed expiry on other trade dates (the
    way stored data often holds it) is an alias: the contract is used with the dates and
    the symbol in force on the trade date, and the alias is recorded in the provenance.
    Raises `SymbolError` when no contract listed on the trade date has that expiry, and
    `ReferenceDataError` when an alias could name two contracts.
    """
    parsed, store, day = request.parsed, request.sources.store, request.day
    if parsed.exchange not in LISTED_DERIVATIVE_EXCHANGES:
        return None
    if not has_rules(store, parsed.underlying, parsed.exchange):
        return None
    lookup = (store, parsed.underlying, parsed.exchange, parsed.expiry, day, calendar)
    contract = contract_for(*lookup)
    if contract is None:
        contract = alias_for(*lookup)
        if contract is None:
            raise _not_listed(parsed, day)
        symbol = format_symbol(
            parsed.underlying, contract.expiry, parsed.strike, parsed.option_type
        )
        request.provenance.append(("symbol", f"{parsed.symbol} is an alias of {symbol}", "alias"))
        request.parsed = parse_symbol(symbol, parsed.exchange)
    request.provenance.append(("last_trading_day", contract.source, contract.status))
    return contract


def _unruled_dates(request: _Request, exact: RefRow | None) -> dict:
    """Contract date and last trading day of a derivative without expiry rules.

    Raises `ReferenceDataError` for an NFO or BFO contract no exact-symbol row lists.
    """
    parsed, day = request.parsed, request.day
    if exact is None and parsed.exchange in LISTED_DERIVATIVE_EXCHANGES:
        raise ReferenceDataError(
            f"Whether {parsed.symbol} was listed on {day.isoformat()} cannot be checked: "
            f"there are no expiry rules for {parsed.underlying} on {parsed.exchange} and no "
            f"row for the exact contract. {how_to_supply('symbols')}"
        )
    stated = None if exact is None else exact.values.get("last_trading_day")
    basis = "the expiry in the symbol" if stated is None else exact.source
    request.provenance.append(("last_trading_day", basis, "assumed" if stated is None else "user"))
    return {"contract_date": parsed.expiry, "last": stated or parsed.expiry}


def _not_listed(parsed: ParsedSymbol, day: _dt.date) -> SymbolError:
    return SymbolError(
        f"{parsed.symbol}: {parsed.expiry.isoformat()} is not an expiry date of "
        f"{parsed.underlying} on {parsed.exchange} for contracts listed on {day.isoformat()} "
        "according to the expiry rules, the contract dates and the calendar"
    )


def _is_snapshot(row: RefRow | None) -> bool:
    return row is not None and row.origin_class == "snapshot"


def _derivative_sources(request: _Request, exact: RefRow | None) -> tuple[list[_Source], bool]:
    """Sources for a future or an option, and whether the MCX table supplies the lot."""
    parsed = request.parsed
    if parsed.exchange in LISTED_DERIVATIVE_EXCHANGES:
        stated = None if exact is None else exact.values.get("underlying_type")
        kind = stated or underlying_kind(
            request.sources, parsed.underlying, parsed.exchange, request.day
        )
        # The lot always goes through lot_size_row, which reconciles user, snapshot and table.
        terms = _SNAPSHOT_TERMS if _is_snapshot(exact) else _DERIVATIVE_TERMS[1:]
        sources: list[_Source] = [] if exact is None else [_SymbolRowSource(exact, terms)]
        return [*sources, _ListedDerivativeSource(request, kind)], False
    sources = [] if exact is None else [_SymbolRowSource(exact, _DERIVATIVE_TERMS)]
    from_mcx_table = exact is None and parsed.exchange == MCX and parsed.kind == "future"
    return sources + ([_McxFutureSource(request)] if from_mcx_table else []), from_mcx_table


def _check_mcx_lot_unit(request: _Request, lot_size: int) -> None:
    """Stop when a snapshot shows the broker counts this MCX contract in another unit."""
    parsed, snapshots = request.parsed, request.sources.snapshots
    record = None if snapshots is None else snapshots.record(parsed.symbol, MCX, request.day)
    if record is None or record.lotsize is None or record.lotsize == lot_size:
        return
    raise ReferenceDataError(
        f"The symbol-master snapshot lists lot size {record.lotsize} for {parsed.symbol} on "
        f"MCX, but the 'mcx_contracts' table has {lot_size} for {parsed.underlying} on "
        f"{request.day.isoformat()}. Either the broker counts quantity in a different unit or "
        "the contract has changed, and the price multiplier cannot be told from the symbol "
        "master. Supply a user 'symbols' row for this contract with lot_size, multiplier "
        "(rupees per one-rupee move per unit of quantity) and settlement."
    )


def _cash_instrument(request: _Request) -> Instrument:
    parsed = request.parsed
    exact = symbol_row(request.sources, parsed.symbol, parsed.exchange, request.day)
    terms = ("lot_size", *_SNAPSHOT_TERMS) if _is_snapshot(exact) else _EQUITY_TERMS
    sources: list[_Source] = [] if exact is None else [_SymbolRowSource(exact, terms)]
    sources.append(_CashSource(request))
    terms = {name: _term(request, sources, name) for name in _EQUITY_TERMS}
    return Instrument(
        symbol=parsed.symbol,
        exchange=parsed.exchange,
        kind="equity",
        settlement="delivery",
        provenance=tuple(request.provenance),
        **terms,
    )


def _derivative(request: _Request, calendar: SessionCalendar) -> Instrument:
    contract = _listed_contract(request, calendar)
    parsed, day = request.parsed, request.day
    exact = symbol_row(request.sources, parsed.symbol, parsed.exchange, day)
    sources, from_mcx_table = _derivative_sources(request, exact)
    if contract is None:
        dates = _unruled_dates(request, exact)
    else:
        dates = {"contract_date": contract.contract_date, "last": contract.last_trading_day}
    if dates["last"] < day:
        raise ConfigError(
            f"{parsed.symbol} expired on {dates['last'].isoformat()}; "
            f"it has no terms on {day.isoformat()}"
        )
    terms = {name: _term(request, sources, name) for name in _DERIVATIVE_TERMS}
    if from_mcx_table:
        _check_mcx_lot_unit(request, terms["lot_size"])
    return Instrument(
        symbol=parsed.symbol,
        exchange=parsed.exchange,
        kind=parsed.kind,
        expiry=parsed.expiry,
        strike=parsed.strike,
        option_type=parsed.option_type,
        underlying=parsed.underlying,
        last_trading_day=dates["last"],
        contract_date=dates["contract_date"],
        provenance=tuple(request.provenance),
        **terms,
    )


def resolve_instrument(
    sources: Sources,
    symbol: str,
    exchange: str,
    day: _dt.date,
    reference_price: float | None = None,
    calendar: SessionCalendar | None = None,
    allow_missing: Sequence[str] = (),
    strict: bool = True,
) -> Instrument:
    """The `Instrument` for `symbol` on `exchange` as it stood on trade date `day`.

    See `ReferenceTables.instrument` for the arguments. Raises `SymbolError` for a
    malformed symbol or an expiry that was not listed on `day`, `ConfigError` for an
    expired contract or a bad argument, and `ReferenceDataError` for a term the
    reference data does not cover.
    """
    unknown = [name for name in allow_missing if name not in OPTIONAL_TERMS]
    if unknown:
        raise ConfigError(f"allow_missing accepts {OPTIONAL_TERMS}, got {unknown}")
    if reference_price is not None:
        reference_price = tick_rules.check_reference_price(reference_price)
    parsed = parse_symbol(symbol, exchange, strict=strict)
    if parsed.kind == "index":
        return Instrument(symbol=symbol, exchange=parsed.exchange, kind="index", settlement="none")
    request = _Request(sources, parsed, day, reference_price, frozenset(allow_missing))
    if parsed.kind == "equity":
        return _cash_instrument(request)
    if calendar is None:
        from ..calendar import default_calendar

        calendar = default_calendar()
    return _derivative(request, calendar)


__all__ = ["OPTIONAL_TERMS", "resolve_instrument"]
