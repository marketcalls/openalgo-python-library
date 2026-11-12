"""`ReferenceTables`: dated lookups over the built-in reference data, user files and
symbol-master snapshots (FR-MKT-09, FR-DATA-11, FR-INST-02).

Rules this class keeps:

* Nothing falls back silently. A lookup no row covers raises `ReferenceDataError`
  naming the table, the key and the date, and saying how to supply the missing data.
* A user file wins over a built-in row, and says so with a `ReferenceDataWarning`
  naming both rows and both values. A snapshot never wins over a verified built-in row
  or a user row, and is used at all only for brokers whose adapters were read.
* Using an unverified row warns on every use; using a built-in row for a date after its
  table's as-of date (see `data/tables.csv`) warns the first time.

Units: quantity, lot size and freeze quantity are in units, never lots; tick sizes,
strikes and prices are in rupees; every date is an IST session date.
"""

import datetime as _dt
from collections.abc import Sequence
from functools import lru_cache

from .._protocols import SessionCalendar
from ..errors import ConfigError
from . import _terms, expiry, tick_rules
from ._loader import UserFiles, builtin_brokers, load_builtin, load_user_files
from ._resolve import resolve_instrument
from ._rows import RefRow, status_counts
from ._schema import TABLES, DateLike, as_date, table_spec
from ._store import RowStore
from .instruments import Instrument
from .snapshot_store import SnapshotStore
from .snapshots import PathLike
from .symbols import derivative_exchange, normalise_exchange

# Hex digits of a SHA-256 digest shown in the version string.
_DIGEST_LENGTH = 12
BROKER_TABLE = "broker_units"


class ReferenceTables:
    """Built-in reference rows plus any user files and symbol-master snapshots.

    user_files: CSV files that extend or override the built-in rows: a path, a list of
    paths (the table is inferred from the column names) or of `(table, path)` pairs, or
    a mapping of table name to path or paths. snapshots: a folder of symbol-master
    snapshots, a list of snapshot files, or a `SnapshotStore`. builtin: False leaves the
    built-in rows out, so only user rows and snapshots count. calendar: the session
    calendar `instrument` checks contracts with when it is given none; when this is None
    too, `openalgo.backtest.calendar.default_calendar()` is used.

    Raises `ReferenceDataError` when a file is unreadable or malformed, or when two rows
    of the same origin overlap for one key.
    """

    def __init__(
        self,
        user_files: UserFiles = (),
        snapshots: SnapshotStore | PathLike | Sequence[PathLike] | None = None,
        builtin: bool = True,
        calendar: SessionCalendar | None = None,
    ) -> None:
        self._store = RowStore()
        self._calendar = calendar
        self._builtin_digest = load_builtin(self._store) if builtin else None
        self._user_file_count, self._user_digest = load_user_files(self._store, user_files)
        if snapshots is not None and not isinstance(snapshots, SnapshotStore):
            snapshots = SnapshotStore(snapshots)
        user_brokers = {row.key[0] for row in self._store.rows(BROKER_TABLE)}
        trusted = builtin_brokers() | frozenset(user_brokers)
        self._sources = _terms.Sources(self._store, snapshots, trusted)

    # ---- identity ----------------------------------------------------------------

    @property
    def as_of(self) -> _dt.date | None:
        """The latest date a built-in table was checked on (None without built-in rows)."""
        dates = self._store.as_of_dates().values()
        return max(dates) if dates else None

    def table_as_of(self, table: str) -> _dt.date | None:
        """The date up to which a built-in table was checked against exchange sources."""
        return self._store.as_of(table)

    @property
    def version(self) -> str:
        """Version string of the reference data, for run manifests.

        `builtin:<as-of date>:<digest>`, then `;user:<digest>` when user files are
        loaded and `;snapshots:<count>:<digest>` when snapshots are attached. Each digest
        is the first 12 hex digits of a SHA-256 of the file contents.
        """
        if self._builtin_digest is None:
            parts = ["builtin:none"]
        else:
            parts = [f"builtin:{self.as_of.isoformat()}:{self._builtin_digest[:_DIGEST_LENGTH]}"]
        if self._user_file_count:
            parts.append(f"user:{self._user_digest[:_DIGEST_LENGTH]}")
        snapshots = self._sources.snapshots
        if snapshots is not None and snapshots.dates:
            digest = snapshots.digest()[:_DIGEST_LENGTH]
            parts.append(f"snapshots:{len(snapshots.dates)}:{digest}")
        return ";".join(parts)

    def rows(self, table: str, key: Sequence[object] | None = None) -> list[RefRow]:
        """Rows of a table (all keys or one), by key, precedence and start date.

        Raises `ConfigError` for an unknown table or a key of the wrong length.
        """
        return self._store.rows(table, key)

    def used_rows(self, status: str | None = None) -> list[RefRow]:
        """Rows that lookups have used so far, optionally only those of one status."""
        return self._store.used_rows(status)

    def find(self, table: str, key: Sequence[object], on_date: DateLike) -> RefRow:
        """The row of a table without a second range that covers `on_date` for `key`.

        Raises `ConfigError` for a table with a second range (use its own lookup) and
        `ReferenceDataError` when no row covers the lookup.
        """
        spec = table_spec(table)
        if spec.span is not None:
            raise ConfigError(f"'{table}' rows also depend on {spec.span[0]}; use its own lookup")
        day = as_date(on_date)
        return self._store.note_use(self._store.require(table, spec.normalise_key(key), day), day)

    # ---- terms -----------------------------------------------------------------------

    def lot_size(
        self, underlying: str, exchange: str, on_date: DateLike, expiry: DateLike | None = None
    ) -> int:
        """Lot size in units of an underlying, or of a full contract symbol, on a trade date.

        `expiry` (the listed expiry) picks the contract; it may be left out when every
        contract listed on `on_date` has the same lot. Raises `ReferenceDataError` when no
        row covers the lookup and `ConfigError` when the lot depends on the contract.
        """
        day, exchange = as_date(on_date), derivative_exchange(exchange)
        expiry_day = None if expiry is None else as_date(expiry, "expiry")
        row = _terms.lot_size_row(self._sources, underlying, exchange, day, expiry_day)
        return self._store.note_use(row, day).values["lot_size"]

    def multiplier(
        self, underlying: str, exchange: str, on_date: DateLike, expiry: DateLike | None = None
    ) -> float:
        """Rupee value of a one-rupee price move for one unit of quantity.

        1.0 on NFO and BFO, from `mcx_contracts` on MCX. Raises `ReferenceDataError` for
        an uncovered MCX underlying and for any other exchange.
        """
        expiry_day = None if expiry is None else as_date(expiry, "expiry")
        exchange, day = derivative_exchange(exchange), as_date(on_date)
        return _terms.multiplier_on(self._sources, underlying, exchange, day, expiry_day)

    def freeze_qty(
        self, underlying: str, exchange: str, on_date: DateLike, kind: str | None = None
    ) -> int:
        """Largest quantity in units the exchange accepts in one order.

        `kind` is "future" or "option"; a full contract symbol supplies it. Raises
        `ReferenceDataError` when no row covers the lookup and `ConfigError` when the
        limit depends on the kind and none is given.
        """
        exchange, day = derivative_exchange(exchange), as_date(on_date)
        return _terms.freeze_qty_on(self._sources, underlying, exchange, day, kind)

    def strike_step(self, underlying: str, exchange: str, on_date: DateLike) -> float:
        """Strike interval in rupees of the near contracts of an underlying.

        Raises `ReferenceDataError` when no row covers the lookup.
        """
        exchange = derivative_exchange(exchange)
        name, _ = _terms.split_contract(underlying, exchange)
        return self.find("strike_steps", (name, exchange), on_date).values["strike_step"]

    def underlying_type(self, underlying: str, exchange: str, on_date: DateLike) -> str:
        """`index`, `stock`, `etf`, `commodity` or `currency` on a date.

        Raises `ReferenceDataError` when the `underlyings` table and the trusted
        snapshots cannot tell (nothing is assumed).
        """
        exchange, day = normalise_exchange(exchange), as_date(on_date)
        return _terms.underlying_kind(self._sources, underlying.upper(), exchange, day)

    # ---- tick size -----------------------------------------------------------------

    def tick_slabs(
        self, exchange: str, segment: str, on_date: DateLike
    ) -> tuple[tick_rules.TickSlab, ...]:
        """The price slabs of the tick-size rule of a segment on a date, lowest first.

        Raises `ConfigError` for an unknown segment and `ReferenceDataError` when no rule
        covers the date.
        """
        exchange, day = normalise_exchange(exchange), as_date(on_date)
        return tick_rules.tick_slabs_on(self._store, exchange, segment, day)

    def tick_size(
        self, exchange: str, segment: str, on_date: DateLike, reference_price: float | None = None
    ) -> float:
        """Tick size in rupees of a segment on a date.

        Where the exchange defines the tick by price slab, `reference_price` (rupees, the
        price named in `TickSlab.basis`) selects the slab. Raises `ConfigError` when a
        slab rule applies and no valid reference price is given, and `ReferenceDataError`
        when no rule or no slab covers the lookup.
        """
        exchange, day = normalise_exchange(exchange), as_date(on_date)
        return tick_rules.tick_size_on(self._store, exchange, segment, day, reference_price)

    def round_price(
        self,
        price: float,
        instrument: Instrument,
        on_date: DateLike,
        reference_price: float | None = None,
        mode: str = "nearest",
    ) -> float:
        """Round a price in rupees to the valid tick of an instrument on a date.

        Uses `instrument.tick_size` when it is set. Otherwise the slab is chosen by
        `reference_price`, the price the rule names (a share's previous month-end close;
        for a stock future or option, the underlying share's close), never by `price`.
        `mode` is as in `rounding.round_to_tick`. Raises the errors of `tick_size`, and
        `ReferenceDataError` when what the instrument is cannot be told.
        """
        day = as_date(on_date)
        return tick_rules.round_price_on(
            self._sources, price, instrument, day, reference_price, mode
        )

    # ---- expiries ------------------------------------------------------------------

    def expiry_rule(
        self, underlying: str, exchange: str, on_date: DateLike, series: str = "monthly"
    ) -> str:
        """Expiry weekday (MON to FRI, or NONE) of a series for expiry dates around a date.

        Raises `ConfigError` for an unknown series and `ReferenceDataError` when no rule
        covers the date.
        """
        exchange, day = derivative_exchange(exchange), as_date(on_date)
        return expiry.expiry_rule(self._store, underlying.upper(), exchange, day, series)

    def expiry_schedule(
        self,
        underlying: str,
        exchange: str,
        start: DateLike,
        end: DateLike,
        calendar: SessionCalendar,
        series: str = expiry.ALL_SERIES,
        as_of: DateLike | None = None,
        record_use: bool = True,
    ) -> list[expiry.Expiry]:
        """Contracts whose last trading day falls from `start` to `end` inclusive.

        `as_of` is a trade date: the listed expiries and last trading days are those in
        force on it (None: the final ones). `series` is "all", "weekly" or "monthly".
        `record_use=False` looks ahead without recording the rows as used or warning
        about them. See `expiry.expiry_schedule` for the rules and the errors raised.
        """
        first, last = as_date(start, "start"), as_date(end, "end")
        trade_date = None if as_of is None else as_date(as_of, "as_of")
        exchange = derivative_exchange(exchange)
        return expiry.expiry_schedule(
            self._store,
            underlying.upper(),
            exchange,
            first,
            last,
            calendar,
            series,
            trade_date,
            record_use,
        )

    def expiries(
        self,
        underlying: str,
        exchange: str,
        start: DateLike,
        end: DateLike,
        calendar: SessionCalendar,
        series: str = expiry.ALL_SERIES,
        as_of: DateLike | None = None,
    ) -> list[_dt.date]:
        """Last trading days of an underlying's contracts from `start` to `end`, in order.

        The dates of `expiry_schedule`, which documents the rules and the errors.
        """
        found = self.expiry_schedule(underlying, exchange, start, end, calendar, series, as_of)
        return sorted({item.last_trading_day for item in found})

    # ---- instruments ---------------------------------------------------------------

    def instrument(
        self,
        symbol: str,
        exchange: str,
        on_date: DateLike,
        reference_price: float | None = None,
        calendar: SessionCalendar | None = None,
        allow_missing: Sequence[str] = (),
        strict: bool = True,
    ) -> Instrument:
        """The `Instrument` for an OpenAlgo symbol as it stood on trade date `on_date`.

        The expiry in an NFO or BFO symbol is checked against the contracts listed on
        `on_date` (expiry rules, contract dates and the calendar), and the instrument
        carries that contract's last trading day (`calendar` None: the calendar the tables
        were built with, else the built-in one). Terms come from a user `symbols` row, a
        trusted symbol-master snapshot and the built-in tables. reference_price: the
        rupee price that selects the tick slab where the exchange defines slabs (without
        it `tick_size` is None for such an instrument). allow_missing: names from
        ("freeze_qty", "tick_size") that may be absent; the field is then None and a
        `ReferenceDataWarning` says so. strict: False also accepts a strike that is not
        written in canonical form.

        Raises `SymbolError` for a malformed symbol or an expiry not listed on
        `on_date`, `ConfigError` for an expired contract or a bad argument, and
        `ReferenceDataError` for anything the reference data does not cover.
        """
        day = as_date(on_date)
        calendar = self._calendar if calendar is None else calendar
        return resolve_instrument(
            self._sources, symbol, exchange, day, reference_price, calendar, allow_missing, strict
        )

    # ---- reports -------------------------------------------------------------------

    def broker_units(self, broker: str | None = None) -> list[dict[str, object]]:
        """What the OpenAlgo server source shows about quantity and tick units per broker.

        The rows describe source code, not live broker behaviour: each is unverified
        until it has been checked against the live broker.
        """
        rows = self._store.rows(BROKER_TABLE)
        chosen = [row for row in rows if broker is None or row.key[0] == broker.lower()]
        return [
            {"broker": row.key[0], "segment": row.key[1], **row.values, "status": row.status}
            | {"source": row.source, "note": row.note}
            for row in chosen
        ]

    def coverage(self) -> list[dict[str, object]]:
        """One entry per table and key: the dates covered and the verification counts.

        Keys of an entry: `table`, `key`, `valid_from`, `valid_to` (None: open-ended),
        `rows`, `verified`, `unverified`, `user` (row counts by status), `gaps` (uncovered
        date ranges inside the covered span), `as_of` and `sources`. Snapshot coverage
        follows, one entry per exchange, under the table name `symbols`.
        """
        report = [
            self._store.coverage_entry(table, key)
            for table in TABLES
            for key in self._store.keys(table)
        ]
        snapshots = self._sources.snapshots
        return report + ([] if snapshots is None else snapshots.coverage())

    def verification_summary(self) -> dict[str, dict[str, int]]:
        """Row counts per table and status, for the tables that have rows."""
        summary = {table: status_counts(self._store.rows(table)) for table in TABLES}
        return {table: counts for table, counts in summary.items() if counts}


@lru_cache(maxsize=1)
def default_tables() -> ReferenceTables:
    """The built-in tables with no user files or snapshots.

    This is the package's one piece of module state: the tables are read from the CSV
    files on the first call and the same object is returned afterwards.
    """
    return ReferenceTables()


def load_tables(
    user_files: UserFiles = (),
    snapshots: SnapshotStore | PathLike | Sequence[PathLike] | None = None,
    builtin: bool = True,
    calendar: SessionCalendar | None = None,
) -> ReferenceTables:
    """Reference tables from the built-in rows plus user files and snapshots.

    Arguments and errors are those of `ReferenceTables`. Without any extras this is
    `default_tables()`.
    """
    if not user_files and snapshots is None and builtin and calendar is None:
        return default_tables()
    return ReferenceTables(user_files, snapshots, builtin, calendar)


__all__ = ["ReferenceTables", "default_tables", "load_tables"]
