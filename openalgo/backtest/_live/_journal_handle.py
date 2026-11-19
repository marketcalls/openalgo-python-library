"""The open handle under the live journal (`journal.Journal`): it owns the SQLite
connection and guards each use of it. A closed handle, a call from another thread, a
missing trading day or a SQLite error each raises `JournalError` saying what to do.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Self

from ..errors import JournalError
from ._journal_file import DEFAULT_BUSY_TIMEOUT_S, failure
from ._journal_rows import day_key, whole_number
from ._types import Mode, Now


class JournalHandle:
    """A journal file opened in one mode at one broker, used from the thread that opened
    it (the session's loop). It keeps the trading day of the session started on it."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path,
        mode: Mode,
        broker: str,
        now: Now,
        busy_timeout_s: float = DEFAULT_BUSY_TIMEOUT_S,
    ) -> None:
        """Wrap `connection`, already checked by `_journal_file.connect` and waiting
        `busy_timeout_s` seconds for a lock; `now` gives UTC nanoseconds."""
        self._connection: sqlite3.Connection | None = connection
        self._path = path
        self._busy_timeout_s = busy_timeout_s
        self._mode = mode
        self._ledger = mode.value
        self._broker = broker
        self._now = now
        self._thread = threading.get_ident()
        self._session_day: date | None = None

    @property
    def path(self) -> Path:
        """The journal file."""
        return self._path

    def close(self) -> None:
        """Close the file. A closed journal raises `JournalError`; closing twice is harmless."""
        if self._connection is None:
            return
        self._check_thread("close the journal")
        connection, self._connection = self._connection, None
        with self._guard("close the journal"):
            connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def _sql(self, action: str) -> Iterator[sqlite3.Connection]:
        """The connection, to `action` with; a SQLite error raised in the block becomes the
        `JournalError` that says what happened."""
        connection = self._usable(action)
        with self._guard(action):
            yield connection

    def _usable(self, action: str) -> sqlite3.Connection:
        """The connection, or `JournalError` when the handle is closed or is called from
        another thread than the one that opened it."""
        if self._connection is None:
            raise JournalError(
                f"cannot {action}: the live journal at {self._path} is closed; open it again "
                "with Journal.open"
            )
        self._check_thread(action)
        return self._connection

    @contextmanager
    def _guard(self, action: str) -> Iterator[None]:
        try:
            yield
        except sqlite3.Error as error:
            raise failure(error, action, self._path, self._busy_timeout_s) from error

    def _check_thread(self, action: str) -> None:
        if threading.get_ident() != self._thread:
            raise JournalError(
                f"cannot {action} from this thread: the live journal at {self._path} belongs "
                "to the session's thread, which opened it; a feed callback queues what it "
                "receives for the session loop instead"
            )

    def _day_key(self, day: date | None, action: str) -> str:
        """The key of `day`, or of the session's day when `day` is None."""
        self._usable(action)
        if day is not None:
            return day_key(day)
        if self._session_day is None:
            raise JournalError(
                f"cannot {action} with no trading day: call start_session first, or pass day="
            )
        return day_key(self._session_day)

    def _now_ns(self) -> int:
        return whole_number("now()", self._now())


__all__ = ["JournalHandle"]
