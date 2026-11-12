"""Dated lookups over a folder of symbol-master snapshots.

What a snapshot is used for
---------------------------
A snapshot row stands for one exact contract on one date; a contract absent from the
snapshot never takes another contract's value. Lot sizes on NSE, BSE, NFO and BFO are
read from snapshots (the server source shows `lotsize` is the exchange lot in units
there for every broker read; see `broker_units`), but only when the snapshot's broker
is one of those brokers; for any other broker a snapshot is a cross-check that warns
on disagreement. A snapshot never overrides a verified built-in row (see `_terms`).
Not used for terms at all:

* `tick_size`: stored, but ignored unless the store is built with `trust_tick_size=True`.
* MCX, CDS and BCD rows: stored and readable through `record`; the lot unit and the
  price multiplier of those contracts cannot be told from the symbol master.

How long a snapshot answers
---------------------------
A snapshot taken on day D answers lookups from D until the day before the next
snapshot that includes the exchange, for at most `max_gap_days` days.
"""

import datetime as _dt
import hashlib
import os
from collections import OrderedDict
from collections.abc import Sequence

from ..errors import ConfigError, ReferenceDataError
from ._rows import RefRow
from ._schema import DateLike, as_date
from .snapshots import (
    PathLike,
    SnapshotRecord,
    SymbolMasterSnapshot,
    header_exchanges,
    is_snapshot_name,
    read_snapshot_header,
    read_symbol_master_snapshot,
    snapshot_date,
)
from .symbols import EQUITY_EXCHANGES, LISTED_DERIVATIVE_EXCHANGES

# Exchanges on which a snapshot lot size may be used for instrument terms.
TRUSTED_EXCHANGES = (*EQUITY_EXCHANGES, *LISTED_DERIVATIVE_EXCHANGES)
# A daily job that misses a few sessions is still bridged; a longer silence is not.
DEFAULT_MAX_GAP_DAYS = 7
# Snapshots kept in memory at once; a full master is roughly a hundred thousand rows.
DEFAULT_CACHE_SIZE = 4
SNAPSHOT_STATUS = "snapshot"
_ONE_DAY = _dt.timedelta(days=1)
_READ_CHUNK = 1 << 20


def snapshot_files(source: PathLike | Sequence[PathLike]) -> list[tuple[_dt.date, str]]:
    """(date, path) of every snapshot in a folder, a file or a list of files, in date order.

    Raises `ReferenceDataError` for a missing location, a file name that is not a
    snapshot name, or two snapshots for one date.
    """
    if isinstance(source, (str, os.PathLike)):
        root = os.fspath(source)
        if os.path.isdir(root):
            paths = [
                os.path.join(root, name) for name in os.listdir(root) if is_snapshot_name(name)
            ]
        elif os.path.exists(root):
            paths = [root]
        else:
            raise ReferenceDataError(f"snapshot location {root!r} does not exist")
    else:
        paths = [os.fspath(path) for path in source]
    files = sorted((snapshot_date(path), path) for path in paths)
    for (first_day, first), (second_day, second) in zip(files, files[1:], strict=False):
        if first_day == second_day:
            raise ReferenceDataError(
                f"two snapshots for {first_day.isoformat()}: {first} and {second}"
            )
    return files


class SnapshotStore:
    """A set of dated snapshot files, read on demand.

    source: a folder of snapshot files, one file, or a list of files. max_gap_days: how
    many days after its date a snapshot keeps answering when no later snapshot exists.
    trust_tick_size: use the stored `tick_size` (rupees assumed; off by default, because
    its unit is not confirmed for every broker). cache_size: snapshots kept in memory.

    Raises `ReferenceDataError` for a missing location, a file that is not a snapshot or
    two snapshots for one date, and `ConfigError` for a negative `max_gap_days`.
    """

    def __init__(
        self,
        source: PathLike | Sequence[PathLike],
        max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
        trust_tick_size: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if max_gap_days < 0:
            raise ConfigError(f"max_gap_days must be 0 or more, got {max_gap_days!r}")
        files = snapshot_files(source)
        self._dates = [day for day, _ in files]
        self._paths = dict(files)
        self._headers: dict[_dt.date, dict[str, str]] = {}
        self._cache: OrderedDict[_dt.date, SymbolMasterSnapshot] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._reach = _dt.timedelta(days=max_gap_days)
        self.max_gap_days = max_gap_days
        self.trust_tick_size = trust_tick_size

    @property
    def dates(self) -> list[_dt.date]:
        """The dates snapshots were taken on, in order."""
        return list(self._dates)

    def digest(self) -> str:
        """SHA-256 hex digest of the contents of every snapshot file, in date order."""
        digest = hashlib.sha256()
        for day in self._dates:
            digest.update(day.isoformat().encode("ascii"))
            with open(self._paths[day], "rb") as handle:
                for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _header(self, day: _dt.date) -> dict[str, str]:
        if day not in self._headers:
            self._headers[day] = read_snapshot_header(self._paths[day])
        return self._headers[day]

    def exchanges(self, day: _dt.date) -> tuple[str, ...]:
        """Exchange codes included in the snapshot taken on `day`."""
        return header_exchanges(self._header(day)) or self.snapshot(day).exchanges

    def snapshot(self, day: _dt.date) -> SymbolMasterSnapshot:
        """The snapshot taken on exactly `day`. Raises `ReferenceDataError` if there is none."""
        if day not in self._paths:
            raise ReferenceDataError(f"no symbol-master snapshot was taken on {day.isoformat()}")
        if day in self._cache:
            self._cache.move_to_end(day)
            return self._cache[day]
        loaded = read_symbol_master_snapshot(self._paths[day])
        self._cache[day] = loaded
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return loaded

    def covering(self, exchange: str, on_date: DateLike) -> _dt.date | None:
        """Date of the snapshot that answers a lookup for `exchange` on `on_date`, or None."""
        day = as_date(on_date)
        for taken in reversed(self._dates):
            if taken > day:
                continue
            if day - taken > self._reach:
                return None
            if exchange in self.exchanges(taken):
                return taken
        return None

    def valid_to(self, taken: _dt.date, exchange: str) -> _dt.date:
        """Last date the snapshot taken on `taken` answers for `exchange`."""
        limit = taken + self._reach
        later = (day for day in self._dates if day > taken and exchange in self.exchanges(day))
        following = next(later, None)
        return limit if following is None else min(limit, following - _ONE_DAY)

    def record(self, symbol: str, exchange: str, on_date: DateLike) -> SnapshotRecord | None:
        """The stored record of a symbol on a date, on any exchange, or None."""
        taken = self.covering(exchange, on_date)
        return None if taken is None else self.snapshot(taken).get(symbol, exchange)

    def lookup(self, symbol: str, exchange: str, on_date: DateLike) -> RefRow | None:
        """The `symbols` row a snapshot gives for one exact symbol on a date, or None.

        None when no snapshot covers the date, the symbol is not in it, or the exchange
        is not NSE, BSE, NFO or BFO (see the module docstring). The row's
        values carry the snapshot's broker label under "broker", so that the caller can
        decide whether to trust it.
        """
        taken = None if exchange not in TRUSTED_EXCHANGES else self.covering(exchange, on_date)
        if taken is None:
            return None
        loaded = self.snapshot(taken)
        record = loaded.get(symbol, exchange)
        if record is None or record.lotsize is None:
            return None
        label = f"symbol-master snapshot {taken.isoformat()}"
        return RefRow(
            table="symbols",
            key=(record.symbol, record.exchange),
            valid_from=taken,
            valid_to=self.valid_to(taken, exchange),
            values={
                "lot_size": record.lotsize,
                "tick_size": record.tick_size if self.trust_tick_size else None,
                "freeze_qty": record.freeze_qty,
                "multiplier": None,
                "underlying_type": None,
                "settlement": None,
                "last_trading_day": None,
                "broker": loaded.broker,
            },
            source=label + (f" ({loaded.broker})" if loaded.broker else " (broker not named)"),
            status=SNAPSHOT_STATUS,
            origin=f"snapshot:{taken.isoformat()}",
        )

    def coverage(self) -> list[dict[str, object]]:
        """One coverage entry per exchange, in the shape of `ReferenceTables.coverage()`."""
        by_exchange: dict[str, list[_dt.date]] = {}
        for day in self._dates:
            for exchange in self.exchanges(day):
                by_exchange.setdefault(exchange, []).append(day)
        report = []
        for exchange in sorted(by_exchange):
            days = by_exchange[exchange]
            gaps = [
                ((first + self._reach + _ONE_DAY).isoformat(), (second - _ONE_DAY).isoformat())
                for first, second in zip(days, days[1:], strict=False)
                if second - first > self._reach + _ONE_DAY
            ]
            used = "" if exchange in TRUSTED_EXCHANGES else " (stored, not used for terms)"
            report.append(
                {
                    "table": "symbols",
                    "key": f"snapshot {exchange}",
                    "valid_from": days[0].isoformat(),
                    "valid_to": (days[-1] + self._reach).isoformat(),
                    "rows": len(days),
                    "verified": 0,
                    "unverified": 0,
                    "user": 0,
                    "gaps": gaps,
                    "as_of": days[-1].isoformat(),
                    "sources": [f"symbol-master snapshots{used}"],
                }
            )
        return report


__all__ = ["DEFAULT_MAX_GAP_DAYS", "TRUSTED_EXCHANGES", "SnapshotStore"]
