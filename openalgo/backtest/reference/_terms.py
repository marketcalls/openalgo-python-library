"""Terms of an underlying or a contract: kind, lot size, multiplier and freeze quantity.

Lot sizes depend on the trade date and on the contract expiry: when an exchange revises
a market lot, contracts already listed usually keep the old lot until they expire, and
MCX publishes a specification per batch of contracts in the same way. A `lot_sizes` or
`mcx_contracts` row therefore applies to a range of trade dates (`valid_from`,
`valid_to`) and a range of contract expiries (`expiry_from`, `expiry_to`, both
inclusive, empty meaning unbounded).

Where the tables and a symbol-master snapshot both answer, they are reconciled openly:
a snapshot never overrides a verified built-in row or a user row (a disagreement is
announced and the table is used); it does override an unverified built-in row, again
with a warning naming both values; and a snapshot is a source at all only when its
broker's adapter was read (see `broker_units`). Each function takes either an
underlying (NIFTY) or a full contract symbol (NIFTY28NOV2424000CE); `exchange` is the
derivatives exchange (NFO, BFO, MCX) and `day` the trade date.
"""

import datetime as _dt
from dataclasses import dataclass

from ..errors import ConfigError, ReferenceDataError
from ._rows import RefRow
from ._store import RowStore, best_origin, how_to_supply, warn
from .instruments import UNIT_MULTIPLIER
from .snapshot_store import TRUSTED_EXCHANGES, SnapshotStore
from .symbols import LISTED_DERIVATIVE_EXCHANGES, ParsedSymbol, SymbolError, parse_symbol

MCX = "MCX"
MCX_TABLE = "mcx_contracts"
LOT_TABLE = "lot_sizes"
FREEZE_TABLE = "freeze_qty"
UNDERLYINGS_TABLE = "underlyings"
# The cash exchange whose share list tells a stock underlying of a derivatives exchange.
_CASH_EXCHANGE_OF = {"NFO": "NSE", "BFO": "BSE"}


@dataclass(frozen=True)
class Sources:
    """Where terms come from: the row store, the snapshots and the brokers they may trust."""

    store: RowStore
    snapshots: SnapshotStore | None
    trusted_brokers: frozenset[str]

    def snapshot_row(self, symbol: str, exchange: str, day: _dt.date) -> RefRow | None:
        """The snapshot row of one exact symbol on `day`, from any broker, or None."""
        if self.snapshots is None or exchange not in TRUSTED_EXCHANGES:
            return None
        return self.snapshots.lookup(symbol, exchange, day)

    def trusted(self, row: RefRow | None) -> bool:
        """Whether a snapshot row comes from a broker whose adapter was read."""
        return row is not None and row.values.get("broker") in self.trusted_brokers


# ---- contract rows by expiry -------------------------------------------------------------


def _applies_to(row: RefRow, expiry: _dt.date) -> bool:
    after_low = row.span_low is None or row.span_low <= expiry
    before_high = row.span_high is None or expiry <= row.span_high
    return after_low and before_high


def _describe_choices(rows: list[RefRow], column: str) -> str:
    parts = []
    for row in sorted(rows, key=lambda item: item.span_low or _dt.date.min):
        low = row.span_low.isoformat() if row.span_low else "any"
        high = row.span_high.isoformat() if row.span_high else "any"
        parts.append(f"{row.values[column]} for expiries {low} to {high}")
    return "; ".join(parts)


def contract_row(
    store: RowStore,
    table: str,
    key: tuple[str, ...],
    day: _dt.date,
    expiry: _dt.date | None,
    column: str,
) -> RefRow:
    """The row of `table` for (underlying, exchange) `key` on trade date `day`.

    `expiry` picks the contract. Without it the row is returned only when every row
    covering `day` gives the same value for `column`. Raises `ReferenceDataError` when no
    row covers the date (or that expiry), and `ConfigError` when `column` depends on the
    contract and no expiry was given.
    """
    rows = store.covering(table, key, day)
    if not rows:
        raise store.missing(table, key, day)
    if expiry is not None:
        matching = [row for row in rows if _applies_to(row, expiry)]
        chosen = best_origin(matching)
        if not chosen:
            raise store.missing(
                table, key, day, f" for a contract expiring on {expiry.isoformat()}"
            )
        store.announce_overrides(chosen[0], matching)
        return chosen[0]
    best = best_origin(rows)
    if any(row.span_low is None and row.span_high is None for row in best):
        rows = best  # a higher-precedence row for every expiry overrides those below it
    if len({row.values[column] for row in rows}) > 1:
        underlying, exchange = key
        raise ConfigError(
            f"The {column} of {underlying} on {exchange} on {day.isoformat()} depends on the "
            f"contract: {_describe_choices(rows, column)}. Pass expiry=, or the full contract "
            "symbol."
        )
    store.announce_overrides(rows[0], store.covering(table, key, day))
    return rows[0]


# ---- what an underlying is ----------------------------------------------------------------


def underlying_kind(sources: Sources, underlying: str, exchange: str, day: _dt.date) -> str:
    """`index`, `stock`, `etf`, `commodity` or `currency` for an underlying on `day`.

    MCX underlyings are commodities and CDS and BCD underlyings currencies. Otherwise the
    `underlyings` table decides; failing that, a trusted symbol-master snapshot that lists
    the underlying as a share on the cash exchange makes it a stock. Nothing else is
    assumed: an underlying neither source knows raises `ReferenceDataError`.
    """
    if exchange == MCX:
        return "commodity"
    if exchange in ("CDS", "BCD"):
        return "currency"
    row = sources.store.first(UNDERLYINGS_TABLE, (underlying.upper(), exchange), day)
    if row is not None:
        sources.store.note_use(row, day)
        return row.values["kind"]
    cash = _CASH_EXCHANGE_OF.get(exchange)
    share = sources.snapshot_row(underlying.upper(), cash, day) if cash else None
    if sources.trusted(share):
        return "stock"
    raise ReferenceDataError(
        f"Whether {underlying} on {exchange} is an index, a stock or an ETF is not known on "
        f"{day.isoformat()}: no 'underlyings' row covers it and no trusted symbol-master "
        f"snapshot lists it as a share. {how_to_supply(UNDERLYINGS_TABLE)}"
    )


# ---- exact symbols and reconciliation -------------------------------------------------------


def symbol_row(sources: Sources, symbol: str, exchange: str, day: _dt.date) -> RefRow | None:
    """The row for one exact symbol on `day`: a user row first, then a trusted snapshot row."""
    row = sources.store.first("symbols", (symbol, exchange), day)
    if row is not None:
        return row
    snapshot = sources.snapshot_row(symbol, exchange, day)
    return snapshot if sources.trusted(snapshot) else None


def split_contract(name: str, exchange: str) -> tuple[str, ParsedSymbol | None]:
    """(underlying, parsed symbol) when `name` is a contract symbol, else (name, None)."""
    try:
        parsed = parse_symbol(name, exchange)
    except SymbolError:
        return name.upper(), None
    return (parsed.underlying, parsed) if parsed.underlying else (name.upper(), None)


def reconcile(
    sources: Sources,
    term: str,
    table_row: RefRow | None,
    table_value: object,
    snapshot: RefRow | None,
    day: _dt.date,
) -> tuple[object, RefRow | None]:
    """Choose between a table value and a snapshot value of one term, announcing conflicts.

    Returns (value, row). The table wins unless it is an unverified built-in row and the
    snapshot comes from a trusted broker. A snapshot is used alone only when trusted.
    """
    snap_value = None if snapshot is None else snapshot.values.get(term)
    if table_row is None:
        return (snap_value, snapshot) if sources.trusted(snapshot) else (None, None)
    if snap_value is None or snap_value == table_value:
        return table_value, table_row
    snapshot_wins = (
        sources.trusted(snapshot)
        and table_row.origin_class == "builtin"
        and table_row.status == "unverified"
    )
    winner = "the snapshot" if snapshot_wins else "the table row"
    warn(
        f"reference data: {term} on {day.isoformat()}: {snapshot.source} gives {snap_value}, "
        f"{table_row.describe()} ({table_row.source}) gives {table_value}; {winner} is used.",
        stacklevel=5,
    )
    return (snap_value, snapshot) if snapshot_wins else (table_value, table_row)


def _table_lot(sources: Sources, underlying: str, exchange: str, day, expiry) -> RefRow | None:
    table = MCX_TABLE if exchange == MCX else LOT_TABLE
    if not sources.store.covering(table, (underlying, exchange), day):
        return None
    return contract_row(sources.store, table, (underlying, exchange), day, expiry, "lot_size")


def _table_lot_if_any(sources: Sources, underlying: str, exchange: str, day, expiry):
    """The table lot row for an override warning, or None when the tables cannot say."""
    try:
        return _table_lot(sources, underlying, exchange, day, expiry)
    except (ReferenceDataError, ConfigError):
        return None


def lot_size_row(
    sources: Sources, name: str, exchange: str, day: _dt.date, expiry, parsed=None
) -> RefRow:
    """The row whose `lot_size` (units) answers a lookup; the caller records its use.

    `expiry` picks the contract when `name` is an underlying; `parsed` is the parsed symbol
    when the caller already has it (a lenient parse keeps its spelling). A user row for the exact
    contract wins (announced when it differs from a built-in row); otherwise the tables
    and a snapshot are reconciled (`reconcile`). Raises `ReferenceDataError` when nothing
    covers the lookup and `ConfigError` when the lot depends on the contract and neither
    a contract symbol nor an expiry is given.
    """
    underlying, parsed = (
        split_contract(name, exchange) if parsed is None else (parsed.underlying, parsed)
    )
    exact = snapshot = None
    if parsed is not None:
        expiry = parsed.expiry if expiry is None else expiry
        exact = sources.store.first("symbols", (parsed.symbol, exchange), day)
        snapshot = sources.snapshot_row(parsed.symbol, exchange, day)
    if exact is not None:
        table_row = _table_lot_if_any(sources, underlying, exchange, day, expiry)
        announce_exact_override(exact, table_row, "lot_size")
        return exact
    table_row = _table_lot(sources, underlying, exchange, day, expiry)
    table_value = None if table_row is None else table_row.values["lot_size"]
    _, row = reconcile(sources, "lot_size", table_row, table_value, snapshot, day)
    if row is None:
        table = MCX_TABLE if exchange == MCX else LOT_TABLE
        raise sources.store.missing(table, (underlying, exchange), day)
    return row


def announce_exact_override(exact: RefRow, table_row: RefRow | None, term: str) -> None:
    """Warn when a user row for one exact symbol overrides a different built-in value."""
    if table_row is None or table_row.origin_class != "builtin":
        return
    if table_row.values.get(term) == exact.values.get(term):
        return
    warn(
        f"reference data: user row {exact.describe()} from {exact.source} overrides built-in "
        f"row {table_row.describe()} ({table_row.source}): {term} {exact.values.get(term)} "
        f"instead of {table_row.values.get(term)}.",
        stacklevel=5,
    )


def multiplier_on(sources: Sources, name: str, exchange: str, day: _dt.date, expiry) -> float:
    """Rupee value of a one-rupee price move for one unit of quantity.

    Raises `ReferenceDataError` for an uncovered MCX underlying and for an exchange other
    than NFO, BFO and MCX, and `ConfigError` when MCX contracts listed on `day` differ and
    no contract symbol or expiry is given.
    """
    underlying, parsed = split_contract(name, exchange)
    if exchange in LISTED_DERIVATIVE_EXCHANGES:
        return UNIT_MULTIPLIER
    if exchange != MCX:
        raise ReferenceDataError(
            f"No multiplier is known for {underlying} on {exchange}. {how_to_supply('symbols')}"
        )
    expiry = parsed.expiry if parsed is not None and expiry is None else expiry
    row = contract_row(sources.store, MCX_TABLE, (underlying, exchange), day, expiry, "multiplier")
    return sources.store.note_use(row, day).values["multiplier"]


def freeze_qty_on(sources: Sources, name: str, exchange: str, day: _dt.date, kind=None) -> int:
    """Largest quantity in units the exchange accepts in one order.

    `kind` is "future" or "option"; a contract symbol supplies it. It may be left out when
    the limit is the same for both. Raises `ReferenceDataError` when no row covers the
    lookup and `ConfigError` when the limit depends on the kind and none is given.
    """
    underlying, parsed = split_contract(name, exchange)
    if parsed is not None:
        kind = parsed.kind
        exact = sources.store.first("symbols", (parsed.symbol, exchange), day)
        if exact is not None and exact.values["freeze_qty"] is not None:
            return sources.store.note_use(exact, day).values["freeze_qty"]
    key = (underlying, exchange)
    if exchange == MCX:
        expiry = None if parsed is None else parsed.expiry
        row = contract_row(sources.store, MCX_TABLE, key, day, expiry, "freeze_qty")
        if row.values["freeze_qty"] is None:
            raise sources.store.missing(MCX_TABLE, key, day, " with a freeze_qty value")
        return sources.store.note_use(row, day).values["freeze_qty"]
    row = sources.store.note_use(sources.store.require(FREEZE_TABLE, key, day), day)
    futures, options = row.values["freeze_qty_futures"], row.values["freeze_qty_options"]
    if kind is None and futures != options:
        raise ConfigError(
            f"The freeze quantity of {underlying} on {exchange} differs between futures "
            f"({futures}) and options ({options}). Pass kind='future' or kind='option', or a "
            "contract symbol."
        )
    return futures if kind == "future" else options


__all__ = [
    "LOT_TABLE",
    "MCX",
    "MCX_TABLE",
    "Sources",
    "announce_exact_override",
    "contract_row",
    "freeze_qty_on",
    "lot_size_row",
    "multiplier_on",
    "reconcile",
    "split_contract",
    "symbol_row",
    "underlying_kind",
]
