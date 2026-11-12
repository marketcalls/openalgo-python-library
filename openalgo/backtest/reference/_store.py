"""The row store: every reference row in memory, with precedence, lookup and warnings.

Precedence when rows of different origins cover the same lookup: a user file wins over
a built-in row, and the override is announced with a warning naming both rows and both
values. (Symbol-master snapshots are reconciled with the tables in `_terms`; they are
never held here.) Nothing is chosen silently beyond that: a lookup no row covers raises
`ReferenceDataError`; an unverified row warns on every use; a built-in row used for a
date after its table's as-of date warns the first time.
"""

import datetime as _dt
import warnings
from collections.abc import Iterable, Sequence

from ..errors import ReferenceDataError
from ._rows import RefRow, status_counts, uncovered_ranges
from ._schema import table_spec

# Earlier in this tuple wins.
ORIGIN_PRECEDENCE = ("user", "builtin")
_RANK = {origin: rank for rank, origin in enumerate(ORIGIN_PRECEDENCE)}

# How many keys an error message lists when it shows what a table does cover.
_MAX_KEYS_SHOWN = 40
# Frames between warnings.warn and the caller of the public lookup method.
WARNING_STACK_LEVEL = 4

_HOW_TO_SUPPLY = (
    "Supply it with a user file (a CSV with the same columns as the built-in '{table}' "
    "table) passed as load_tables(user_files=[path]), or record a symbol-master snapshot "
    "with write_symbol_master_snapshot(client, directory) and pass "
    "load_tables(snapshots=directory)."
)


class ReferenceDataWarning(UserWarning):
    """A lookup used a row that is unverified, overridden, or beyond its checked date."""


def how_to_supply(table: str) -> str:
    """The sentence an error message ends with, saying how to add missing reference data."""
    return _HOW_TO_SUPPLY.format(table=table)


def warn(message: str, stacklevel: int = WARNING_STACK_LEVEL) -> None:
    """Issue a `ReferenceDataWarning`."""
    warnings.warn(message, ReferenceDataWarning, stacklevel=stacklevel)


def best_origin(rows: Sequence[RefRow]) -> list[RefRow]:
    """The rows of the highest-precedence origin among `rows`."""
    if not rows:
        return []
    best = min(_RANK[row.origin_class] for row in rows)
    return [row for row in rows if _RANK[row.origin_class] == best]


class RowStore:
    """Rows of every table, keyed by table and normalised key."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[tuple[str, ...], list[RefRow]]] = {}
        self._as_of: dict[str, _dt.date] = {}
        self._used: dict[tuple, RefRow] = {}
        self._announced: set[tuple] = set()

    # ---- contents ----------------------------------------------------------------

    def add(self, rows: Iterable[RefRow]) -> None:
        """Add rows (no overlap check here; the loader checks each file)."""
        for row in rows:
            self._rows.setdefault(row.table, {}).setdefault(row.key, []).append(row)

    def set_as_of(self, table: str, day: _dt.date) -> None:
        """Record the date up to which a built-in table was checked against its sources."""
        self._as_of[table] = day

    def as_of(self, table: str) -> _dt.date | None:
        """The as-of date of a built-in table, or None for a table without built-in rows."""
        return self._as_of.get(table)

    def as_of_dates(self) -> dict[str, _dt.date]:
        """As-of date per built-in table."""
        return dict(self._as_of)

    def keys(self, table: str) -> list[tuple[str, ...]]:
        """Sorted keys that have at least one row in `table`."""
        return sorted(self._rows.get(table, {}))

    def rows(self, table: str, key: Sequence[object] | None = None) -> list[RefRow]:
        """Rows of a table (all keys, or one key), by key, precedence, then start date.

        Raises `ConfigError` for an unknown table or a key of the wrong length.
        """
        spec = table_spec(table)
        groups = self._rows.get(table, {})
        if key is None:
            found = [row for group in groups.values() for row in group]
        else:
            found = list(groups.get(spec.normalise_key(key), ()))
        return sorted(found, key=lambda row: (row.key, _RANK[row.origin_class], row.valid_from))

    # ---- lookup ------------------------------------------------------------------

    def covering(self, table: str, key: tuple[str, ...], day: _dt.date) -> list[RefRow]:
        """Rows of every origin whose validity range includes `day` (key already normalised)."""
        return [row for row in self._rows.get(table, {}).get(key, ()) if row.covers(day)]

    def first(self, table: str, key: tuple[str, ...], day: _dt.date) -> RefRow | None:
        """The row that answers a lookup in a table without a span, or None.

        A user row wins over a built-in row; when their values differ the override is
        announced with a warning.
        """
        rows = self.covering(table, key, day)
        chosen = best_origin(rows)
        if not chosen:
            return None
        self.announce_overrides(chosen[0], rows)
        return chosen[0]

    def require(self, table: str, key: tuple[str, ...], day: _dt.date, detail: str = "") -> RefRow:
        """Like `first`, but raises `ReferenceDataError` when no row covers the lookup."""
        row = self.first(table, key, day)
        if row is None:
            raise self.missing(table, key, day, detail)
        return row

    def missing(
        self, table: str, key: tuple[str, ...], day: _dt.date, detail: str = ""
    ) -> ReferenceDataError:
        """The error for a lookup no row covers: what is missing, what exists, how to supply it."""
        spec = table_spec(table)
        named = ", ".join(f"{name}={value}" for name, value in zip(spec.key, key, strict=True))
        return ReferenceDataError(
            f"No '{table}' reference row covers {named} on {day.isoformat()}{detail}. "
            f"{self._covered_text(table, key)} {how_to_supply(table)}"
        )

    def _covered_text(self, table: str, key: tuple[str, ...]) -> str:
        rows = self._rows.get(table, {}).get(key)
        if rows:
            first = min(row.valid_from for row in rows)
            open_ended = any(row.valid_to is None for row in rows)
            last = "the present" if open_ended else max(row.valid_to for row in rows).isoformat()
            return f"Rows for this key cover {first.isoformat()} to {last}."
        keys = [" ".join(key) for key in self.keys(table)]
        if not keys:
            return f"The '{table}' table has no rows."
        shown = ", ".join(keys[:_MAX_KEYS_SHOWN]) + (" ..." if len(keys) > _MAX_KEYS_SHOWN else "")
        return f"The '{table}' table has rows for: {shown}."

    def coverage_entry(self, table: str, key: tuple[str, ...]) -> dict[str, object]:
        """What the rows of one key cover; the keys are those of `ReferenceTables.coverage`."""
        rows = self.rows(table, key)
        counts, as_of = status_counts(rows), self._as_of.get(table)
        open_ended = any(row.valid_to is None for row in rows)
        return {
            "table": table,
            "key": " ".join(key),
            "valid_from": min(row.valid_from for row in rows).isoformat(),
            "valid_to": None if open_ended else max(row.valid_to for row in rows).isoformat(),
            "rows": len(rows),
            "verified": counts.get("verified", 0),
            "unverified": counts.get("unverified", 0),
            "user": counts.get("user", 0),
            "gaps": uncovered_ranges(rows),
            "as_of": as_of.isoformat() if as_of else None,
            "sources": sorted({row.source for row in rows}),
        }

    # ---- warnings ----------------------------------------------------------------

    def announce_overrides(self, chosen: RefRow, candidates: Iterable[RefRow]) -> None:
        """Warn once per pair when a user row is used in place of a different built-in row.

        `candidates` are the rows that answer the same lookup as `chosen`.
        """
        if chosen.origin_class != "user":
            return
        for other in candidates:
            if other.origin_class != "builtin" or other.values == chosen.values:
                continue
            pair = ("override", chosen.content_key, other.content_key)
            if pair in self._announced:
                continue
            self._announced.add(pair)
            differences = ", ".join(
                f"{name} {chosen.values[name]} instead of {other.values[name]}"
                for name in chosen.values
                if chosen.values[name] is not None and chosen.values[name] != other.values.get(name)
            )
            warn(
                f"reference data: user row {chosen.describe()} from {chosen.source} overrides "
                f"built-in row {other.describe()} ({other.source}): {differences}."
            )

    def note_use(self, row: RefRow, day: _dt.date) -> RefRow:
        """Record that a lookup for `day` used `row`, and warn where the row calls for it.

        An unverified row warns on every use. A built-in row used for a date after its
        table's as-of date warns the first time (once per distinct row).
        """
        self._used[row.content_key] = row
        if row.status == "unverified":
            warn(
                f"reference data: {row.describe()} is unverified "
                f"(source: {row.source or 'none'}). Confirm it against the exchange circular, "
                "or override it with a user file or a symbol-master snapshot."
            )
        as_of = self._as_of.get(row.table)
        past = as_of is not None and day > as_of and row.origin_class == "builtin"
        mark = ("past_as_of", row.content_key)
        if past and mark not in self._announced:
            self._announced.add(mark)
            warn(
                f"reference data: {row.describe()} was last checked on {as_of.isoformat()} "
                f"and is being used for {day.isoformat()}. The exchange may have changed it "
                "since: update openalgo, or supply a user file or a symbol-master snapshot."
            )
        return row

    def used_rows(self, status: str | None = None) -> list[RefRow]:
        """Rows that lookups have used so far, optionally only those of one status."""
        rows = list(self._used.values())
        return rows if status is None else [row for row in rows if row.status == status]


__all__ = [
    "ORIGIN_PRECEDENCE",
    "WARNING_STACK_LEVEL",
    "ReferenceDataWarning",
    "RowStore",
    "best_origin",
    "how_to_supply",
    "warn",
]
