"""The file of the live journal (`journal.Journal`): where it lives, its layout and version,
and the checks made each time it is opened (08-live-trading section 7).

A journal is bound to the account and the broker it was created for. Its layout version
is kept in `PRAGMA user_version`, which SQLite reads as 0 in a file never given one.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from ..errors import JournalError

#: The layout of the journal file this version reads and writes.
SCHEMA_VERSION = 1
#: The environment variable naming the folder that holds the journals instead of
#: `~/.openalgo/live`.
LIVE_DIR_VARIABLE = "OPENALGO_LIVE_DIR"
JOURNAL_FILE_NAME = "journal.sqlite"
# Path separators, and the drive mark that would move the path to another drive on Windows.
_NOT_IN_FOLDER_NAME = ("/", "\\", ":")

# Each table has a `ledger` column, the `Mode` value of the handle that wrote the row.
# An intent's `state` is "recorded" until its outcome is known, then "bound" to its
# order, "unbound" (outcome unknown) or "not_placed".
_SCHEMA = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE intents (
        seq INTEGER PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE, ledger TEXT NOT NULL,
        strategy TEXT NOT NULL, symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        action TEXT NOT NULL, quantity INTEGER NOT NULL, price_type TEXT NOT NULL,
        product TEXT NOT NULL, purpose TEXT NOT NULL, created_ns INTEGER NOT NULL,
        price REAL NOT NULL, trigger_price REAL NOT NULL, state TEXT NOT NULL,
        order_id TEXT, reason TEXT NOT NULL, changed_ns INTEGER NOT NULL)""",
    "CREATE UNIQUE INDEX intents_by_order ON intents (ledger, order_id)",
    """CREATE TABLE updates (
        seq INTEGER PRIMARY KEY, ledger TEXT NOT NULL, order_id TEXT NOT NULL,
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, action TEXT NOT NULL,
        quantity INTEGER NOT NULL, filled_quantity INTEGER NOT NULL,
        pending_quantity INTEGER NOT NULL, average_price REAL NOT NULL,
        status TEXT NOT NULL, terminal INTEGER NOT NULL, rejection_reason TEXT NOT NULL,
        mode TEXT NOT NULL, ts_recv_ns INTEGER NOT NULL, source TEXT NOT NULL)""",
    "CREATE INDEX updates_by_order ON updates (ledger, order_id, terminal, seq)",
    """CREATE TABLE stops (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT NOT NULL, order_id TEXT NOT NULL, set_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, strategy, symbol, exchange))""",
    """CREATE TABLE states (
        ledger TEXT NOT NULL, kind TEXT NOT NULL, strategy TEXT NOT NULL,
        state TEXT NOT NULL, saved_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, kind, strategy))""",
    """CREATE TABLE sessions (
        ledger TEXT NOT NULL, day TEXT NOT NULL, broker TEXT NOT NULL,
        first_started_ns INTEGER NOT NULL, last_started_ns INTEGER NOT NULL,
        starts INTEGER NOT NULL, PRIMARY KEY (ledger, day))""",
)


def default_journal_path(account: str) -> Path:
    """Where an account's journal lives: `~/.openalgo/live/<account>/journal.sqlite`, or
    `<folder>/<account>/journal.sqlite` when the environment variable `OPENALGO_LIVE_DIR`
    names a folder (a blank value counts as unset).

    Raises `JournalError` when `account` is not a plain folder name.
    """
    check_name("account", account)
    if any(mark in account for mark in _NOT_IN_FOLDER_NAME) or account in (".", ".."):
        raise JournalError(
            f"account {account!r} must be a plain folder name, with no / \\ or : in it"
        )
    folder = os.environ.get(LIVE_DIR_VARIABLE, "").strip()
    root = Path(folder).expanduser() if folder else Path.home() / ".openalgo" / "live"
    return root / account / JOURNAL_FILE_NAME


def connect(path: Path, account: str, broker: str) -> sqlite3.Connection:
    """Open the journal file at `path` for `account` bound to `broker`, creating its
    folders and, in a new file, its layout. The connection commits every statement as it
    runs, syncs each commit to disk and gives rows as `sqlite3.Row`.

    Raises `JournalError` when the file cannot be opened, is not a journal, has a newer
    or unknown schema version, belongs to another account or is bound to another broker.
    """
    check_name("account", account)
    check_name("broker", broker)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, autocommit=True)
    except (OSError, sqlite3.Error) as error:
        raise JournalError(
            f"cannot open the live journal at {path} ({error}): choose a folder you can "
            f"write, for example with the {LIVE_DIR_VARIABLE} environment variable"
        ) from error
    try:
        connection.row_factory = sqlite3.Row
        _prepare(connection, path, account, broker)
    except BaseException:
        connection.close()
        raise
    return connection


def check_name(what: str, value: str) -> None:
    """Raise `JournalError` unless `value` is a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise JournalError(f"{what} must be a non-empty string, not {value!r}")


def _prepare(connection: sqlite3.Connection, path: Path, account: str, broker: str) -> None:
    try:
        version = _check_version(connection, path)
        if connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
            raise JournalError(
                f"SQLite cannot keep a write-ahead log at {path}: put the journal on a "
                f"local disk with the {LIVE_DIR_VARIABLE} environment variable"
            )
        # FULL syncs every commit to disk, so an intent survives a power cut.
        connection.execute("PRAGMA synchronous = FULL")
        if version == 0:
            _create(connection, account, broker)
        _check_binding(connection, path, account, broker)
    except sqlite3.Error as error:
        raise JournalError(
            f"cannot open the live journal at {path} (SQLite: {error}): if the file is not a "
            "journal, move it away; otherwise check the disk and that no other program holds it"
        ) from error


def _check_version(connection: sqlite3.Connection, path: Path) -> int:
    """The file's schema version: 0 for a new, empty file, else `SCHEMA_VERSION`."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 0 and connection.execute("SELECT 1 FROM sqlite_master").fetchone():
        raise JournalError(
            f"{path} is a SQLite file but not an OpenAlgo live journal: move it away or "
            "open another path"
        )
    if version > SCHEMA_VERSION:
        raise JournalError(
            f"the live journal at {path} was written by a newer openalgo (schema version "
            f"{version}; this one reads {SCHEMA_VERSION}): upgrade with pip install -U openalgo"
        )
    if version not in (0, SCHEMA_VERSION):
        raise JournalError(
            f"the live journal at {path} has the unknown schema version {version}: move it "
            "away or open another path"
        )
    return version


def _create(connection: sqlite3.Connection, account: str, broker: str) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO meta VALUES (?, ?)", (("account", account), ("broker", broker))
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _check_binding(connection: sqlite3.Connection, path: Path, account: str, broker: str) -> None:
    meta = {row["key"]: row["value"] for row in connection.execute("SELECT * FROM meta")}
    if meta.get("account") != account:
        raise JournalError(
            f"the live journal at {path} belongs to account {meta.get('account')!r}, not "
            f"{account!r}: open the journal of {account!r} (default_journal_path)"
        )
    bound = meta.get("broker")
    if bound != broker:
        raise JournalError(
            f"the live journal at {path} is bound to broker {bound!r}, and OpenAlgo is now "
            f"logged in to {broker!r}. An exit sent to {broker!r} would open a new position "
            f"instead of closing one held at {bound!r}: close the positions at {bound!r}, "
            f"then move this journal away to start a new one for {broker!r}"
        )


__all__ = [
    "JOURNAL_FILE_NAME",
    "LIVE_DIR_VARIABLE",
    "SCHEMA_VERSION",
    "check_name",
    "connect",
    "default_journal_path",
]
