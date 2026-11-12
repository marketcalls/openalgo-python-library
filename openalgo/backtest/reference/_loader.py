"""Reading the reference CSV files into a `RowStore`: the shipped ones and user files.

The shipped files live under `data/` and are located with `importlib.resources`, so
they are found in an installed wheel as well as in a source checkout. `data/tables.csv`
is the registry: one line per table with its file name and the date the table was last
checked against exchange sources (its as-of date).
"""

import hashlib
import os
from collections.abc import Iterable, Mapping
from importlib import resources

from ..errors import ReferenceDataError
from ._rows import find_overlaps, parse_table, read_records
from ._schema import TABLES, parse_date, table_spec
from ._store import RowStore

DATA_DIRECTORY = "data"
REGISTRY_FILE = "tables.csv"
BROKER_TABLE = "broker_units"
_PACKAGE = __package__ or "openalgo.backtest.reference"

# A column that occurs in exactly one table once the tables listed before it are ruled
# out. Order matters: `symbols` and `mcx_contracts` also carry a lot_size column.
_TABLE_MARKERS = (
    ("symbol", "symbols"),
    ("broker", "broker_units"),
    ("price_from", "tick_sizes"),
    ("weekday", "expiry_rules"),
    ("last_trading_day", "contract_dates"),
    ("strike_step", "strike_steps"),
    ("multiplier", "mcx_contracts"),
    ("lot_size", "lot_sizes"),
    ("kind", "underlyings"),
    ("freeze_qty_futures", "freeze_qty"),
    ("settlement", "settlement"),
)

# What `load_user_files` accepts: a path, a list of paths or (table, path) pairs, or a
# mapping of table name to a path or a list of paths.
UserFiles = str | os.PathLike | Iterable | Mapping


def builtin_text(filename: str) -> str:
    """Text of a data file shipped inside the package."""
    return (
        resources.files(_PACKAGE)
        .joinpath(DATA_DIRECTORY)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


def builtin_registry() -> dict[str, dict[str, str]]:
    """The registry in `data/tables.csv`: table name to its `file`, `as_of` and `description`."""
    _, records = read_records(builtin_text(REGISTRY_FILE), REGISTRY_FILE)
    return {record["table"]: record for _, record in records}


def builtin_brokers() -> frozenset[str]:
    """Brokers whose adapters the built-in `broker_units` table reads (lower case)."""
    name = builtin_registry()[BROKER_TABLE]["file"]
    rows = parse_table(table_spec(BROKER_TABLE), builtin_text(name), name, "builtin", True)
    return frozenset(row.key[0] for row in rows)


def _normalised(text: str) -> bytes:
    """File text with one line-ending convention, so a checkout's line endings do not matter."""
    return text.replace("\r\n", "\n").encode("utf-8")


def load_builtin(store: RowStore) -> str:
    """Load every shipped table into `store`. Returns the SHA-256 hex digest of the files.

    Raises `ReferenceDataError` when a table has no registry entry, a file is malformed,
    or two rows of one key overlap.
    """
    registry = builtin_registry()
    digest = hashlib.sha256()
    for name, spec in TABLES.items():
        if not spec.builtin:
            continue
        entry = registry.get(name)
        if entry is None:
            raise ReferenceDataError(f"{REGISTRY_FILE} has no entry for the table {name!r}")
        text = builtin_text(entry["file"])
        rows = parse_table(spec, text, entry["file"], origin="builtin", strict=True)
        clashes = find_overlaps(spec, rows)
        if clashes:
            first, second = clashes[0]
            raise ReferenceDataError(
                f"{entry['file']}: rows overlap: {first.describe()} and {second.describe()}"
            )
        store.add(rows)
        store.set_as_of(name, parse_date(entry["as_of"]))
        digest.update(name.encode("ascii"))
        digest.update(_normalised(text))
    return digest.hexdigest()


def infer_table(header: Iterable[str]) -> str:
    """Which table a user file belongs to, judged by its column names.

    Raises `ReferenceDataError` when no table matches.
    """
    columns = set(header)
    for marker, table in _TABLE_MARKERS:
        if marker in columns:
            return table
    raise ReferenceDataError(
        f"cannot tell which reference table the columns {sorted(columns)} belong to; "
        f"pass the table name explicitly, one of {sorted(TABLES)}"
    )


def _is_path(item: object) -> bool:
    return isinstance(item, (str, os.PathLike))


def _user_file_items(user_files: UserFiles) -> list[tuple[str | None, str]]:
    """Flatten the accepted shapes of `user_files` into (table or None, path) pairs."""
    if not user_files:
        return []
    if _is_path(user_files):
        return [(None, os.fspath(user_files))]
    items: list[tuple[str | None, str]] = []
    if isinstance(user_files, Mapping):
        for table, paths in user_files.items():
            for path in [paths] if _is_path(paths) else paths:
                items.append((table, os.fspath(path)))
        return items
    for item in user_files:
        table, path = (None, item) if _is_path(item) else item
        items.append((table, os.fspath(path)))
    return items


def _load_user_file(store: RowStore, path: str, table: str | None) -> tuple[str, str]:
    """Load one user file. Returns (table name, file text)."""
    label = os.path.basename(path)
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        raise ReferenceDataError(f"cannot read the reference user file {path!r}: {exc}") from None
    if table is None:
        header, _ = read_records(text, label)
        table = infer_table(header)
    if table not in TABLES:
        raise ReferenceDataError(
            f"{label}: unknown reference table {table!r}; expected one of {sorted(TABLES)}"
        )
    spec = table_spec(table)
    rows = parse_table(
        spec,
        text,
        label,
        origin=f"user:{label}",
        strict=False,
        default_source=f"user file {label}",
    )
    existing = [row for row in store.rows(table) if row.origin_class == "user"]
    clashes = find_overlaps(spec, existing + rows)
    if clashes:
        first, second = clashes[0]
        raise ReferenceDataError(
            f"{label}: user rows overlap for the same key and date: "
            f"{first.describe()} and {second.describe()}"
        )
    store.add(rows)
    return table, text


def load_user_files(store: RowStore, user_files: UserFiles) -> tuple[int, str]:
    """Load user files into `store`. Returns (number of files, SHA-256 hex digest).

    User rows may overlap built-in rows (they override them) but not each other.
    Raises `ReferenceDataError` for an unreadable or malformed file, an unknown table,
    or overlapping user rows.
    """
    digest = hashlib.sha256()
    items = _user_file_items(user_files)
    for table, path in items:
        loaded_table, text = _load_user_file(store, path, table)
        digest.update(loaded_table.encode("ascii"))
        digest.update(_normalised(text))
    return len(items), digest.hexdigest()


__all__ = [
    "DATA_DIRECTORY",
    "REGISTRY_FILE",
    "UserFiles",
    "builtin_brokers",
    "builtin_registry",
    "builtin_text",
    "infer_table",
    "load_builtin",
    "load_user_files",
]
