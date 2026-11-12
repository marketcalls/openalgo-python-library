"""Reference rows: the `RefRow` record, CSV text to rows, overlaps, gaps and counts."""

import csv
import datetime as _dt
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import ReferenceDataError
from ._schema import (
    BUILTIN_STATUSES,
    SPAN_DATE,
    SPAN_PRICE,
    USER_STATUS,
    TableSpec,
    parse_date,
    parse_price_bound,
)

COMMENT_PREFIX = "#"
_ONE_DAY = _dt.timedelta(days=1)


@dataclass(frozen=True, eq=False)
class RefRow:
    """One row of a reference table.

    `valid_from` and `valid_to` are inclusive dates (`valid_to` None: open-ended).
    `values` holds the parsed value columns, None for an empty optional cell.
    `span_low` and `span_high` are the second range of the row where the table has
    one (expiry dates, or prices in rupees). `origin` is "builtin", "user:<file>" or
    "snapshot:<date>". Rows compare by identity.
    """

    table: str
    key: tuple[str, ...]
    valid_from: _dt.date
    valid_to: _dt.date | None
    values: Mapping[str, object]
    source: str
    status: str
    origin: str
    note: str = ""
    span_low: object = None
    span_high: object = None

    @property
    def content_key(self) -> tuple:
        """What the row says, independent of where it is held: equal rows share it."""
        values = tuple(sorted((name, repr(value)) for name, value in self.values.items()))
        span = (self.span_low, self.span_high)
        return (self.table, self.key, self.valid_from, self.valid_to, span, values, self.origin)

    @property
    def origin_class(self) -> str:
        """`builtin`, `user` or `snapshot`."""
        return self.origin.split(":", 1)[0]

    def covers(self, day: _dt.date) -> bool:
        """Whether the validity range includes `day`."""
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)

    def describe(self) -> str:
        """A one-line label for messages."""
        until = self.valid_to.isoformat() if self.valid_to else "open"
        return (
            f"{self.table}[{' '.join(self.key)}] {self.valid_from.isoformat()} to {until} "
            f"({self.origin}, {self.status})"
        )


def read_records(text: str, label: str) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Split CSV text into its header and (line number, record) pairs.

    Blank lines and lines starting with `#` are skipped. Raises `ReferenceDataError`
    for a missing header, a repeated column name or a row of the wrong width.
    """
    kept: list[tuple[int, str]] = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith(COMMENT_PREFIX)
    ]
    if not kept:
        raise ReferenceDataError(f"{label}: the file has no header row")
    rows = list(csv.reader(io.StringIO("\n".join(line for _, line in kept))))
    header = [cell.strip() for cell in rows[0]]
    if len(set(header)) != len(header):
        raise ReferenceDataError(f"{label}: the header repeats a column name: {header}")
    records = []
    for (number, _), cells in zip(kept[1:], rows[1:], strict=False):
        if len(cells) != len(header):
            raise ReferenceDataError(
                f"{label} line {number}: expected {len(header)} columns, found {len(cells)}"
            )
        cleaned = {name: cell.strip() for name, cell in zip(header, cells, strict=True)}
        records.append((number, cleaned))
    return header, records


def _check_header(spec: TableSpec, header: Sequence[str], label: str, strict: bool) -> None:
    if strict:
        needed = list(spec.header)
    else:
        required = [column.name for column in spec.columns if column.required]
        needed = [*spec.key, "valid_from", *required]
    missing = [name for name in needed if name not in header]
    if missing:
        raise ReferenceDataError(
            f"{label}: missing column(s) {missing} for the '{spec.name}' table "
            f"(expected: {', '.join(spec.header)})"
        )


def _validity(record: Mapping[str, str], where: str) -> tuple[_dt.date, _dt.date | None]:
    try:
        valid_from = parse_date(record["valid_from"])
        valid_to = parse_date(record["valid_to"]) if record.get("valid_to") else None
    except ValueError:
        raise ReferenceDataError(
            f"{where}: valid_from and valid_to must be ISO dates (YYYY-MM-DD); "
            f"got {record.get('valid_from')!r} and {record.get('valid_to')!r}"
        ) from None
    if valid_to is not None and valid_to < valid_from:
        raise ReferenceDataError(f"{where}: valid_to {valid_to} is before valid_from {valid_from}")
    return valid_from, valid_to


def _span(spec: TableSpec, record: Mapping[str, str], where: str) -> tuple[object, object]:
    if spec.span is None:
        return None, None
    low_name, high_name, kind = spec.span
    parse = parse_date if kind == SPAN_DATE else parse_price_bound
    try:
        low = parse(record[low_name]) if record.get(low_name) else None
        high = parse(record[high_name]) if record.get(high_name) else None
    except ValueError as exc:
        raise ReferenceDataError(f"{where}: bad {low_name} or {high_name}: {exc}") from None
    if kind == SPAN_PRICE and low is None:
        low = 0.0
    if low is not None and high is not None:
        empty = high <= low if kind == SPAN_PRICE else high < low
        if empty:
            raise ReferenceDataError(f"{where}: {high_name} is not above {low_name}")
    return low, high


def _values(spec: TableSpec, record: Mapping[str, str], where: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in spec.columns:
        cell = record.get(column.name, "")
        if not cell:
            if column.required:
                raise ReferenceDataError(f"{where}: the column {column.name!r} is empty")
            values[column.name] = None
            continue
        try:
            values[column.name] = column.parse(cell)
        except ValueError as exc:
            raise ReferenceDataError(f"{where}: column {column.name!r}: {exc}") from None
    return values


def _status(record: Mapping[str, str], where: str, strict: bool) -> str:
    """The status of a row: verified or unverified for a built-in row, user for a user row."""
    status = (record.get("status") or "").lower()
    if strict:
        if status not in BUILTIN_STATUSES:
            raise ReferenceDataError(
                f"{where}: status must be one of {BUILTIN_STATUSES}, got {status!r}"
            )
        return status
    if status not in ("", USER_STATUS):
        raise ReferenceDataError(
            f"{where}: a user row cannot carry status {status!r}; leave the status empty or "
            f"write {USER_STATUS!r} (only rows shipped with the engine are verified)"
        )
    return USER_STATUS


def _key(spec: TableSpec, record: Mapping[str, str], where: str) -> tuple[str, ...]:
    empty = [name for name in spec.key if not record.get(name)]
    if empty:
        raise ReferenceDataError(f"{where}: the key column {empty[0]!r} is empty")
    try:
        return spec.key_of(record)
    except ValueError as exc:
        raise ReferenceDataError(f"{where}: {exc}") from None


def parse_table(
    spec: TableSpec,
    text: str,
    label: str,
    origin: str,
    strict: bool,
    default_source: str = "",
) -> list[RefRow]:
    """Turn CSV text into rows of `spec`.

    `strict` is for the files shipped in the package: every column of the table must
    be present, `status` must be `verified` or `unverified`, and `source` may not be
    empty. A user file (`strict=False`) may leave out `valid_to`, `source`, `status`,
    `note`, the span columns and optional value columns; its rows always have status
    "user", and a user row that claims another status is refused.

    Raises `ReferenceDataError` naming `label` and the line for any malformed cell.
    """
    header, records = read_records(text, label)
    _check_header(spec, header, label, strict)
    rows: list[RefRow] = []
    for number, record in records:
        where = f"{label} line {number}"
        key = _key(spec, record, where)
        valid_from, valid_to = _validity(record, where)
        span_low, span_high = _span(spec, record, where)
        status = _status(record, where, strict)
        source = record.get("source") or default_source
        if strict and not source:
            raise ReferenceDataError(f"{where}: a built-in row needs a source")
        rows.append(
            RefRow(
                table=spec.name,
                key=key,
                valid_from=valid_from,
                valid_to=valid_to,
                values=_values(spec, record, where),
                source=source,
                status=status,
                origin=origin,
                note=record.get("note", ""),
                span_low=span_low,
                span_high=span_high,
            )
        )
    return rows


def _ends_before(high, low, half_open: bool) -> bool:
    """Whether a range ending at `high` lies wholly before one starting at `low`."""
    if high is None or low is None:
        return False
    return high <= low if half_open else high < low


def _ranges_overlap(low_a, high_a, low_b, high_b, half_open: bool) -> bool:
    """Whether two ranges intersect. None is an open bound."""
    return not (_ends_before(high_a, low_b, half_open) or _ends_before(high_b, low_a, half_open))


def rows_overlap(spec: TableSpec, first: RefRow, second: RefRow) -> bool:
    """True when two rows of one key apply to the same date (and the same span value)."""
    if first.key != second.key:
        return False
    if not _ranges_overlap(
        first.valid_from, first.valid_to, second.valid_from, second.valid_to, half_open=False
    ):
        return False
    if spec.span is None:
        return True
    return _ranges_overlap(
        first.span_low,
        first.span_high,
        second.span_low,
        second.span_high,
        half_open=spec.span[2] == SPAN_PRICE,
    )


def find_overlaps(spec: TableSpec, rows: Sequence[RefRow]) -> list[tuple[RefRow, RefRow]]:
    """Every pair of rows with the same key whose ranges overlap."""
    by_key: dict[tuple[str, ...], list[RefRow]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
    clashes = []
    for group in by_key.values():
        for index, first in enumerate(group):
            clashes.extend(
                (first, second)
                for second in group[index + 1 :]
                if rows_overlap(spec, first, second)
            )
    return clashes


def status_counts(rows: Sequence[RefRow]) -> dict[str, int]:
    """Number of rows per status."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def uncovered_ranges(rows: Sequence[RefRow]) -> list[tuple[str, str]]:
    """ISO date ranges between the first and the last row of a key that no row covers."""
    gaps: list[tuple[str, str]] = []
    reach: _dt.date | None = None
    for row in sorted(rows, key=lambda item: item.valid_from):
        if reach is not None and row.valid_from > reach + _ONE_DAY:
            gaps.append(((reach + _ONE_DAY).isoformat(), (row.valid_from - _ONE_DAY).isoformat()))
        if row.valid_to is None:
            break  # open-ended: nothing after this row can be uncovered
        reach = row.valid_to if reach is None else max(reach, row.valid_to)
    return gaps


__all__ = [
    "RefRow",
    "find_overlaps",
    "parse_table",
    "read_records",
    "rows_overlap",
    "status_counts",
    "uncovered_ranges",
]
