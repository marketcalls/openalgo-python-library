"""Symbol-master snapshot files: writing one from a client, reading one back.

The OpenAlgo server holds today's contracts only, so a later backtest cannot ask it
what the lot size of a contract was. A snapshot is a dated copy of
`client.instruments(...)`. Run `write_symbol_master_snapshot(client, directory)` once
per trading day; the client call is the only network access and it is the caller's
client that makes it. `snapshot_store.SnapshotStore` answers dated lookups from a
folder of these files.

File format: `symbol_master_YYYY-MM-DD.csv.gz` (gzip, UTF-8), or `.csv` uncompressed.

    line 1    # openalgo symbol-master snapshot v1; date=...; broker=...; exchanges=A|B; rows=N
    line 2    CSV header: the columns of SNAPSHOT_COLUMNS
    then      one row per symbol, sorted by exchange and symbol

Column meanings (values are stored as the server returned them, not reinterpreted):

    symbol, exchange    the OpenAlgo symbol and exchange code
    name                the server's name column (the underlying for most derivatives)
    instrumenttype      the server's instrument type (EQ, FUT, CE, PE, ...)
    expiry              DD-MMM-YY as the server gives it; empty for non-derivatives
    strike              strike in rupees; empty when there is none
    lotsize             the server's lot size; its unit differs by broker on MCX and CDS
    tick_size           the server's tick size; its unit is not confirmed for every broker
    freeze_qty          present only if the client supplied it
"""

import csv
import datetime as _dt
import gzip
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigError, DataError, ReferenceDataError
from ._schema import DateLike, as_date
from ._store import warn
from .symbols import format_strike

SNAPSHOT_PREFIX = "symbol_master_"
SNAPSHOT_FORMAT = "openalgo symbol-master snapshot v1"
SNAPSHOT_COLUMNS = (
    "symbol",
    "exchange",
    "name",
    "instrumenttype",
    "expiry",
    "strike",
    "lotsize",
    "tick_size",
    "freeze_qty",
)
DEFAULT_EXCHANGES = ("NSE", "BSE", "NFO", "BFO", "MCX")
GZIP_SUFFIX = ".gz"

_NAME_RE = re.compile(r"^" + SNAPSHOT_PREFIX + r"(\d{4}-\d{2}-\d{2})\.csv(\.gz)?$")
_IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
# Spellings of "no value" seen in symbol masters.
_MISSING_TEXT = ("", "nan", "none", "null", "nat")
# Characters that would break the one-line header if they appeared in the broker label.
_HEADER_RESERVED = ("\n", "\r", ";", "=")

PathLike = str | os.PathLike


@dataclass(frozen=True)
class SnapshotRecord:
    """One symbol of a snapshot, values as the server returned them (see the module docstring)."""

    symbol: str
    exchange: str
    name: str
    instrumenttype: str
    expiry: str
    strike: float | None
    lotsize: int | None
    tick_size: float | None
    freeze_qty: int | None


@dataclass(frozen=True)
class SymbolMasterSnapshot:
    """A snapshot file read into memory: its date, broker label, exchanges and records."""

    path: str
    date: _dt.date
    broker: str
    exchanges: tuple[str, ...]
    records: Mapping[tuple[str, str], SnapshotRecord]

    def get(self, symbol: str, exchange: str) -> SnapshotRecord | None:
        """The record of a symbol on an exchange, or None when the snapshot lacks it."""
        return self.records.get((symbol, exchange))


# ---- cells -------------------------------------------------------------------------------


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING_TEXT


def _text(value: object) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _finite(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_text(value: object) -> str:
    number = _finite(value)
    if number is None:
        return ""
    return str(int(number)) if number == int(number) else repr(number)


def _strike_text(value: object) -> str:
    number = _finite(value)
    # Masters mark "no strike" with 0 or a negative number as well as with a blank.
    return "" if number is None or number <= 0 else format_strike(number)


def _positive(text: str) -> float | None:
    number = _finite(text)
    return number if number is not None and number > 0 else None


def _whole(text: str) -> int | None:
    number = _positive(text)
    return int(number) if number is not None and number == int(number) else None


# ---- writing -----------------------------------------------------------------------------


def _instrument_records(result: object, exchange: str) -> list[Mapping[str, object]]:
    """Rows of one `client.instruments(exchange=...)` result, as mappings."""
    call = f"client.instruments(exchange={exchange!r})"
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        records = list(result.to_dict("records"))
    elif isinstance(result, Mapping):
        if not isinstance(result.get("data"), list) or result.get("status") == "error":
            message = result.get("message") or result.get("error") or dict(result)
            raise DataError(f"{call} did not return instruments: {message}")
        records = list(result["data"])
    elif isinstance(result, (list, tuple)):
        records = list(result)
    else:
        raise DataError(f"{call} returned {type(result).__name__}; expected a table of rows")
    if not any(_text(record.get("symbol")) for record in records):
        raise DataError(f"{call} returned no rows with a symbol")
    return records


def _snapshot_row(record: Mapping[str, object], exchange: str) -> tuple[str, ...]:
    return (
        _text(record.get("symbol")),
        _text(record.get("exchange")) or exchange,
        _text(record.get("name")),
        _text(record.get("instrumenttype")),
        _text(record.get("expiry")).upper(),
        _strike_text(record.get("strike")),
        _number_text(record.get("lotsize")),
        _number_text(record.get("tick_size")),
        _number_text(record.get("freeze_qty")),
    )


def _collect(client: object, exchanges: Sequence[str], skip_failed: bool) -> tuple[list, list[str]]:
    """Fetch every exchange. Returns the rows and the exchanges that supplied them."""
    rows: set[tuple[str, ...]] = set()
    included: list[str] = []
    for exchange in exchanges:
        try:
            records = _instrument_records(client.instruments(exchange=exchange), exchange)
        except DataError as exc:
            if not skip_failed:
                raise
            warn(f"symbol-master snapshot: {exchange} left out ({exc})", stacklevel=3)
            continue
        rows.update(
            _snapshot_row(record, exchange) for record in records if _text(record.get("symbol"))
        )
        included.append(exchange)
    if not included:
        raise DataError("no exchange returned instruments; no snapshot was written")
    return sorted(rows, key=lambda row: (row[1], row[0])), included


def _encode(header_line: str, rows: Sequence[tuple[str, ...]], compress: bool) -> bytes:
    buffer = io.StringIO(newline="")
    buffer.write(header_line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SNAPSHOT_COLUMNS)
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    # mtime=0 keeps the bytes of the file a pure function of its contents.
    return gzip.compress(payload, mtime=0) if compress else payload


def _snapshot_paths(folder: Path, day: _dt.date, compress: bool) -> tuple[Path, Path]:
    """The file to write, and the other spelling (compressed or not) of the same date."""
    plain_name = f"{SNAPSHOT_PREFIX}{day.isoformat()}.csv"
    packed_name = plain_name + GZIP_SUFFIX
    names = (packed_name, plain_name) if compress else (plain_name, packed_name)
    return folder / names[0], folder / names[1]


def _replace_file(target: Path, twin: Path, payload: bytes) -> None:
    """Write `payload` to `target`, removing the other spelling of the same date."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed, so a reader never sees half a file.
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(payload)
    if twin.exists():
        twin.unlink()
    os.replace(temporary, target)


def write_symbol_master_snapshot(
    client: object,
    directory: PathLike,
    exchanges: Sequence[str] = DEFAULT_EXCHANGES,
    as_of: DateLike | None = None,
    broker: str = "",
    compress: bool = True,
    overwrite: bool = False,
    skip_failed: bool = False,
) -> Path:
    """Store the symbol master from `client.instruments(exchange=...)` in a dated file.

    client: any object with `instruments(exchange=...)` returning a pandas DataFrame or a
    list of rows with at least `symbol` and `lotsize` (the OpenAlgo SDK `api` object does);
    this function makes no network call of its own. directory: folder for the files.
    as_of: the IST date the snapshot is for (default: today in India). broker: the broker
    the server is connected to; it is stored because the meaning of `lotsize` and
    `tick_size` differs by broker, and snapshot lot sizes are trusted only for brokers
    whose adapters were read. compress: `.csv.gz` (default) or `.csv`. overwrite: replace
    a snapshot of the same date. skip_failed: leave out an exchange that fails, with a
    `ReferenceDataWarning`, instead of stopping.

    Returns the path written. Raises `ConfigError` for a bad argument or an existing
    snapshot for the date, and `DataError` when an exchange fails or returns no rows
    (nothing is written then).
    """
    if isinstance(exchanges, str):
        exchanges = (exchanges,)
    if not callable(getattr(client, "instruments", None)):
        raise ConfigError("client must have an instruments(exchange=...) method")
    if not exchanges:
        raise ConfigError("exchanges must name at least one exchange")
    if any(mark in broker for mark in _HEADER_RESERVED):
        raise ConfigError(f"broker label {broker!r} may not contain ';', '=' or a line break")
    day = as_date(as_of, "as_of") if as_of is not None else _dt.datetime.now(_IST).date()
    target, twin = _snapshot_paths(Path(directory), day, compress)
    existing = next((path for path in (target, twin) if path.exists()), None)
    if existing is not None and not overwrite:
        raise ConfigError(
            f"a snapshot for {day.isoformat()} already exists ({existing}); "
            "pass overwrite=True to replace it"
        )
    rows, included = _collect(client, exchanges, skip_failed)
    header_line = (
        f"# {SNAPSHOT_FORMAT}; date={day.isoformat()}; broker={broker}; "
        f"exchanges={'|'.join(included)}; rows={len(rows)}"
    )
    _replace_file(target, twin, _encode(header_line, rows, compress))
    return target


# ---- reading -----------------------------------------------------------------------------


def is_snapshot_name(name: str) -> bool:
    """Whether a file name is that of a snapshot file."""
    return _NAME_RE.match(name) is not None


def snapshot_date(path: PathLike) -> _dt.date:
    """The IST date in a snapshot file name. Raises `ReferenceDataError` for any other name."""
    match = _NAME_RE.match(os.path.basename(os.fspath(path)))
    if not match:
        raise ReferenceDataError(
            f"{os.fspath(path)!r} is not a snapshot file name "
            f"(expected {SNAPSHOT_PREFIX}YYYY-MM-DD.csv.gz)"
        )
    try:
        return _dt.date.fromisoformat(match.group(1))
    except ValueError:
        raise ReferenceDataError(f"{os.fspath(path)!r} does not carry a real date") from None


def _open_text(path: PathLike) -> io.TextIOBase:
    name = os.fspath(path)
    if name.endswith(GZIP_SUFFIX):
        return gzip.open(name, "rt", encoding="utf-8", newline="")
    return open(name, encoding="utf-8", newline="")


def read_snapshot_header(path: PathLike) -> dict[str, str]:
    """The `key=value` pairs of a snapshot's first line (date, broker, exchanges, rows).

    Raises `ReferenceDataError` when the file cannot be read or is not a snapshot.
    """
    try:
        with _open_text(path) as handle:
            first = handle.readline()
    except (OSError, EOFError) as exc:
        raise ReferenceDataError(f"cannot read the snapshot {os.fspath(path)!r}: {exc}") from None
    if not first.startswith(f"# {SNAPSHOT_FORMAT}"):
        raise ReferenceDataError(
            f"{os.fspath(path)!r} does not start with the snapshot header line "
            f"('# {SNAPSHOT_FORMAT}; ...')"
        )
    pairs = (part.split("=", 1) for part in first.strip().split(";") if "=" in part)
    return {key.strip(): value.strip() for key, value in pairs}


def header_exchanges(header: Mapping[str, str]) -> tuple[str, ...]:
    """The exchange codes named in a snapshot header."""
    return tuple(part for part in header.get("exchanges", "").split("|") if part)


def _read_records(path: PathLike) -> dict[tuple[str, str], SnapshotRecord]:
    records: dict[tuple[str, str], SnapshotRecord] = {}
    with _open_text(path) as handle:
        handle.readline()  # the header line, read by read_snapshot_header
        reader = csv.DictReader(handle)
        if not {"symbol", "exchange", "lotsize"} <= set(reader.fieldnames or ()):
            raise ReferenceDataError(
                f"{os.fspath(path)!r}: the CSV header lacks symbol, exchange or lotsize"
            )
        for row in reader:
            records[(row["symbol"], row["exchange"])] = SnapshotRecord(
                symbol=row["symbol"],
                exchange=row["exchange"],
                name=row.get("name") or "",
                instrumenttype=row.get("instrumenttype") or "",
                expiry=row.get("expiry") or "",
                strike=_positive(row.get("strike") or ""),
                lotsize=_whole(row.get("lotsize") or ""),
                tick_size=_positive(row.get("tick_size") or ""),
                freeze_qty=_whole(row.get("freeze_qty") or ""),
            )
    return records


def read_symbol_master_snapshot(path: PathLike) -> SymbolMasterSnapshot:
    """Read one snapshot file written by `write_symbol_master_snapshot`.

    Raises `ReferenceDataError` when the name is not a snapshot name, the file cannot be
    read, or the date in the header disagrees with the date in the name.
    """
    day = snapshot_date(path)
    header = read_snapshot_header(path)
    if header.get("date") and header["date"] != day.isoformat():
        raise ReferenceDataError(
            f"{os.fspath(path)!r}: the file name says {day.isoformat()} "
            f"but the header says {header['date']}"
        )
    try:
        records = _read_records(path)
    except (OSError, EOFError, csv.Error) as exc:
        raise ReferenceDataError(f"cannot read the snapshot {os.fspath(path)!r}: {exc}") from None
    exchanges = header_exchanges(header) or tuple(sorted({exchange for _, exchange in records}))
    return SymbolMasterSnapshot(os.fspath(path), day, header.get("broker", ""), exchanges, records)


__all__ = [
    "DEFAULT_EXCHANGES",
    "SNAPSHOT_COLUMNS",
    "PathLike",
    "SnapshotRecord",
    "SymbolMasterSnapshot",
    "header_exchanges",
    "is_snapshot_name",
    "read_snapshot_header",
    "read_symbol_master_snapshot",
    "snapshot_date",
    "write_symbol_master_snapshot",
]
