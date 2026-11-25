"""The strategies' versions in the live journal (`journal.Journal`; 18-multi-strategy 12.1):
at each start of a day, each strategy's version id and its run manifest, so the journal
shows which code and parameters placed each order (each intent carries the version id of
the start that made it).

A row is kept per strategy per start, oldest first, in the ledger of the handle's mode and
under the session's trading day. The manifest is kept as JSON text and read back as the
dict it was given; one that JSON cannot keep is refused. Times are UTC nanoseconds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..errors import JournalError
from ._journal_file import check_name
from ._journal_handle import JournalHandle
from ._journal_rows import day_key, state_json, unreadable

_RECORD = "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?)"
_OF_DAY = "SELECT * FROM versions WHERE ledger = ? AND day = ? ORDER BY rowid"
_LAST_RAN = "SELECT MAX(day) AS day FROM versions WHERE ledger = ? AND strategy = ? AND day < ?"


@dataclass(frozen=True)
class StartVersion:
    """One strategy's version at one start: its `version_id`, its run `manifest` (a dict
    of JSON types) and when the start recorded it (`started_ns`)."""

    strategy: str
    version_id: str
    manifest: dict[str, Any]
    started_ns: int


class JournalVersions(JournalHandle):
    """The journal's part for the strategies' versions (module docstring), mixed into
    `journal.Journal`."""

    def record_version(
        self,
        strategy: str,
        version_id: str,
        manifest: Mapping[str, Any],
        *,
        day: date | None = None,
    ) -> None:
        """Record `strategy`'s version at this start of the day (the session's, or `day`):
        `version_id` and its run `manifest`. Raises `JournalError` for a blank strategy or
        version id, a manifest whose values JSON cannot keep, or with no `day` before
        `start_session`."""
        check_name("strategy", strategy)
        check_name("version_id", version_id)
        key = self._day_key(day, "record a strategy's version")
        text, now = state_json(dict(manifest), "manifest", strategy), self._now_ns()
        with self._sql("record the strategy's version") as sql:
            sql.execute(_RECORD, (self._ledger, key, strategy, version_id, text, now))

    def versions(self, *, day: date | None = None) -> list[StartVersion]:
        """Every strategy's version at each start of the day (the session's, or `day`),
        oldest first. Raises `JournalError` with no `day` before `start_session`, and for
        a row it cannot read."""
        key = self._day_key(day, "read the strategies' versions")
        with self._sql("read the strategies' versions") as sql:
            rows = sql.execute(_OF_DAY, (self._ledger, key)).fetchall()
        try:
            return [
                StartVersion(
                    row["strategy"],
                    row["version_id"],
                    json.loads(row["manifest"]),
                    row["started_ns"],
                )
                for row in rows
            ]
        except ValueError as error:
            raise JournalError(unreadable("a strategy's version", error)) from error

    def last_ran(self, strategy: str, before: date) -> date | None:
        """The latest trading day before `before` on which a session of `strategy` started
        in the handle's ledger (a start records its version), or None for none: a session
        that ran after a day-close record may have changed what the record carries
        (20-positional-live 4.2). Raises `JournalError` for a row it cannot read."""
        with self._sql("read the days a strategy ran") as sql:
            row = sql.execute(_LAST_RAN, (self._ledger, strategy, day_key(before))).fetchone()
        try:
            return None if row["day"] is None else date.fromisoformat(row["day"])
        except ValueError as error:
            raise JournalError(unreadable("a strategy's version", error)) from error


__all__ = ["JournalVersions", "StartVersion"]
