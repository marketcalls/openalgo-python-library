"""The file of the live journal (`journal.Journal`): where it lives, its layout and version,
the checks made each time it is opened (08-live-trading section 7), how long a write
waits for a lock another connection holds, and what a SQLite error met while using it
means.

A journal is bound to the account and the broker it was created for. Its layout version
is kept in `PRAGMA user_version`, which SQLite reads as 0 in a file never given one. A file
of an earlier layout this version knows is upgraded in place when it is opened, in the
transaction that opens it, through each later version in turn, its rows kept
(`_journal_layout.UPGRADES`): version 1 (openalgo 2.3.0) gains the strategies' versions of
each start and the version id of each intent (18-multi-strategy 12.1), version 2 (openalgo
2.4) the tables of positions held overnight (20-positional-live). A newer file is refused.
"""

from __future__ import annotations

import math
import numbers
import os
import sqlite3
from pathlib import Path
from types import MappingProxyType

from ..errors import JournalError
from ._journal_layout import SCHEMA, SCHEMA_VERSION, UPGRADES

#: Seconds a statement waits for a lock another connection holds before SQLite gives up
#: with SQLITE_BUSY: 5, Python's own default for `sqlite3.connect(timeout=...)`. A
#: session writes one file from two threads, each through its own handle, and every
#: write commits in milliseconds, so one waits on the other well within it.
DEFAULT_BUSY_TIMEOUT_S = 5.0
# Python's sqlite3 hands the wait to SQLite (`sqlite3_busy_timeout`) as whole
# milliseconds in a C int, and SQLite takes 0 or less as no wait at all: a wait under
# 1 ms, or past 2**31 - 1 ms (about 24.8 days), would silently become none.
#: The shortest busy timeout SQLite keeps as a wait, in seconds: one millisecond.
MIN_BUSY_TIMEOUT_S = 0.001
#: The longest, in seconds: the largest C int of milliseconds.
MAX_BUSY_TIMEOUT_S = (2**31 - 1) / 1_000
#: The environment variable naming the folder (absolute) that holds the journals instead
#: of `~/.openalgo/live`.
LIVE_DIR_VARIABLE = "OPENALGO_LIVE_DIR"
#: The name of the journal file in an account's folder.
JOURNAL_FILE_NAME = "journal.sqlite"

# Windows' rules for a file or folder name (its file naming conventions): these
# characters, control characters, a trailing dot or space, and the device names, with
# or without an extension, their digit also written as a superscript.
_NOT_IN_FOLDER_NAME = '<>:"/\\|?*'
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_SUPERSCRIPT_DIGITS = MappingProxyType(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789"))
_FIRST_PRINTABLE = 32

# What to do after each kind of SQLite error, by its primary result code; SQLITE_BUSY's
# advice names the busy timeout (`_busy_advice`).
_BUSY = "SQLITE_BUSY"
_REFUSED = "a value was refused: pass the field SQLite names, as the type the journal keeps"
_DISK = "check that the disk works and the file can be written, then start the session again"
_ADVICE = MappingProxyType(
    {
        "SQLITE_LOCKED": "the file is locked by this program: start the session again",
        "SQLITE_FULL": "the disk is full: free some space, then start the session again",
        "SQLITE_READONLY": (
            f"the file is read-only: make it writable, or point {LIVE_DIR_VARIABLE} at a "
            "folder you can write, then start the session again"
        ),
        "SQLITE_CORRUPT": (
            "the file is damaged: move it away and start the session again (the broker's "
            "order book and positions still show what is open)"
        ),
        "SQLITE_NOTADB": "the file is not a journal: move it away or open another path",
    }
)


def default_journal_path(account: str) -> Path:
    """Where an account's journal lives: `~/.openalgo/live/<account>/journal.sqlite`, or
    `<folder>/<account>/journal.sqlite` when the environment variable `OPENALGO_LIVE_DIR`
    names a folder (a blank value counts as unset).

    Raises `JournalError` when `account` cannot be a folder name on Windows, Linux and
    macOS alike, or when `OPENALGO_LIVE_DIR` is a relative path.
    """
    _check_folder_name(account)
    folder = os.environ.get(LIVE_DIR_VARIABLE, "").strip()
    if not folder:
        return Path.home() / ".openalgo" / "live" / account / JOURNAL_FILE_NAME
    root = Path(folder).expanduser()
    if not root.is_absolute():
        raise JournalError(
            f"{LIVE_DIR_VARIABLE} must name an absolute folder, not {folder!r}: a relative "
            "one would move with the folder the session is started from"
        )
    return root / account / JOURNAL_FILE_NAME


def connect(
    path: Path, account: str, broker: str, *, busy_timeout_s: float = DEFAULT_BUSY_TIMEOUT_S
) -> sqlite3.Connection:
    """Open the journal file at `path` for `account` bound to `broker`, creating its
    folders and, in a new file, its layout. The connection commits every statement as it
    runs, syncs each commit to disk, gives rows as `sqlite3.Row`, and waits up to
    `busy_timeout_s` seconds for a lock another connection holds.

    Raises `JournalError` when `busy_timeout_s` is not a number of seconds from
    `MIN_BUSY_TIMEOUT_S` to `MAX_BUSY_TIMEOUT_S`, or the file cannot be opened, is not a
    journal, has a newer or unknown schema version, belongs to another account or is
    bound to another broker.
    """
    check_name("account", account)
    check_name("broker", broker)
    _check_busy_timeout(busy_timeout_s)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, autocommit=True, timeout=busy_timeout_s)
    except (OSError, sqlite3.Error) as error:
        raise JournalError(
            f"cannot open the live journal at {path} ({error}): choose a folder you can "
            f"write, for example with the {LIVE_DIR_VARIABLE} environment variable"
        ) from error
    try:
        connection.row_factory = sqlite3.Row
        _prepare(connection, path, account, broker, busy_timeout_s)
    except BaseException:
        connection.close()
        raise
    return connection


def _check_busy_timeout(busy_timeout_s: float) -> None:
    """Raise `JournalError` unless `busy_timeout_s` is a number of seconds SQLite keeps as
    a wait: from `MIN_BUSY_TIMEOUT_S` to `MAX_BUSY_TIMEOUT_S`."""
    number = isinstance(busy_timeout_s, numbers.Real) and not isinstance(busy_timeout_s, bool)
    if not (number and math.isfinite(busy_timeout_s) and busy_timeout_s > 0):
        raise JournalError(
            "busy_timeout_s is how many seconds a write waits for a lock another connection "
            f"holds, a number above 0 such as {DEFAULT_BUSY_TIMEOUT_S:g}, not {busy_timeout_s!r}"
        )
    if not MIN_BUSY_TIMEOUT_S <= busy_timeout_s <= MAX_BUSY_TIMEOUT_S:
        raise JournalError(
            f"busy_timeout_s is {busy_timeout_s!r} seconds, which SQLite would keep as no wait "
            f"at all (it counts whole milliseconds): give from {MIN_BUSY_TIMEOUT_S:g} (one "
            f"millisecond) to {MAX_BUSY_TIMEOUT_S} seconds, such as {DEFAULT_BUSY_TIMEOUT_S:g}"
        )


def check_name(what: str, value: str) -> None:
    """Raise `JournalError` unless `value` is a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise JournalError(f"{what} must be a non-empty string, not {value!r}")


def failure(
    error: sqlite3.Error,
    action: str,
    path: Path,
    busy_timeout_s: float = DEFAULT_BUSY_TIMEOUT_S,
) -> JournalError:
    """The `JournalError` for a SQLite error met while trying to `action`: what happened
    (a refused value; a busy, locked, full, read-only, damaged or foreign file; another
    disk problem) and what to do. `busy_timeout_s` is how long the connection waited for
    a lock, which a busy file's advice names."""
    name = getattr(error, "sqlite_errorname", None) or ""
    # An extended code (SQLITE_READONLY_DBMOVED) names its primary one first.
    primary = "_".join(name.split("_")[:2])
    refused = (sqlite3.IntegrityError, sqlite3.DataError, sqlite3.InterfaceError)
    if isinstance(error, refused) or primary == "SQLITE_CONSTRAINT":
        advice = _REFUSED
    elif isinstance(error, sqlite3.ProgrammingError) and not name:
        advice = _REFUSED  # raised by Python's sqlite3 for a value it cannot bind
    elif primary == _BUSY:
        advice = _busy_advice(busy_timeout_s)
    else:
        advice = _ADVICE.get(primary, _DISK)
    return JournalError(
        f"cannot {action} in the live journal at {path} (SQLite: {error}): {advice}"
    )


def _busy_advice(busy_timeout_s: float) -> str:
    return (
        "another program holds the file (another session of this account, or a tool "
        f"reading it) and kept it locked past the {busy_timeout_s:g} seconds a write waits "
        "(busy_timeout_s): stop it, then start the session again"
    )


def _check_folder_name(account: str) -> None:
    check_name("account", account)
    unfit = [
        mark for mark in account if mark in _NOT_IN_FOLDER_NAME or ord(mark) < _FIRST_PRINTABLE
    ]
    if unfit or account.endswith((".", " ")):
        raise JournalError(
            f"account {account!r} must be a plain folder name: no {_NOT_IN_FOLDER_NAME} or "
            "control characters, and no dot or space at the end"
        )
    stem = account.split(".")[0].rstrip(" ").translate(_SUPERSCRIPT_DIGITS).upper()
    if stem in _DEVICE_NAMES:
        raise JournalError(
            f"account {account!r} is a device name on Windows (CON, PRN, AUX, NUL, COM1 to "
            "COM9, LPT1 to LPT9, their digit also as a superscript): choose another name"
        )


def _prepare(
    connection: sqlite3.Connection, path: Path, account: str, broker: str, busy_timeout_s: float
) -> None:
    try:
        # FULL syncs every commit to disk, so an intent survives a power cut.
        connection.execute("PRAGMA synchronous = FULL")
        # The version is read in the transaction that creates the layout, so two
        # processes opening a new file cannot both create it.
        connection.execute("BEGIN IMMEDIATE")
        try:
            version = _check_version(connection, path)
            if version == 0:
                _create(connection, account, broker)
            _check_binding(connection, path, account, broker)
            if version in UPGRADES:
                _upgrade(connection, version)
            # A write that changes nothing: a file that cannot be written (read-only)
            # is refused now, before the session, not at its first order.
            connection.execute("UPDATE meta SET value = value WHERE key = 'broker'")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        # After the checks, so a file that is not a journal is left as it was.
        if connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
            raise JournalError(
                f"SQLite cannot keep a write-ahead log at {path}: put the journal on a "
                f"local disk with the {LIVE_DIR_VARIABLE} environment variable"
            )
    except sqlite3.Error as error:
        raise failure(error, "open the journal", path, busy_timeout_s) from error


def _check_version(connection: sqlite3.Connection, path: Path) -> int:
    """The file's schema version: 0 for a new, empty file, else `SCHEMA_VERSION` or an
    earlier one `UPGRADES` takes to it."""
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
    if version not in (0, SCHEMA_VERSION, *UPGRADES):
        raise JournalError(
            f"the live journal at {path} has the unknown schema version {version}: move it "
            "away or open another path"
        )
    return version


def _create(connection: sqlite3.Connection, account: str, broker: str) -> None:
    for statement in SCHEMA:
        connection.execute(statement)
    connection.executemany(
        "INSERT INTO meta VALUES (?, ?)", (("account", account), ("broker", broker))
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _upgrade(connection: sqlite3.Connection, version: int) -> None:
    """Take a file of the earlier schema `version` to `SCHEMA_VERSION` in the transaction
    that opens it, through each later version in turn, its rows kept; it is checked as the
    account's and the broker's first, so a file of another account is left as it was."""
    for step in range(version, SCHEMA_VERSION):
        for statement in UPGRADES[step]:
            connection.execute(statement)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


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
            f"instead of closing one held at {bound!r}: close the positions at {bound!r} "
            f"first, then give {broker!r} a journal of its own by setting the "
            f"{LIVE_DIR_VARIABLE} environment variable to another folder"
        )


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_S",
    "JOURNAL_FILE_NAME",
    "LIVE_DIR_VARIABLE",
    "MAX_BUSY_TIMEOUT_S",
    "MIN_BUSY_TIMEOUT_S",
    "SCHEMA_VERSION",
    "check_name",
    "connect",
    "default_journal_path",
    "failure",
]
