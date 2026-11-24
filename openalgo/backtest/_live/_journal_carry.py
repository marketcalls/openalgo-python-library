"""What the live journal (`journal.Journal`) carries from one trading day's session to the
next (20-positional-live 3.5, 4 and 8; the lead's ruling N2-2).

- **The day-close record** (`put_day_close`, `day_close`): per strategy and trading day,
  the state the next day's session restores (the kernel's carried state, the strategy's
  `self.state` and its version id), written after the day's last GTT book read, or by a
  Ctrl+C that keeps the positions. A strategy has one record a day: a later write of the
  same day (a close after such a Ctrl+C) replaces it. The latest is read back, in the
  handle's own ledger only, so a live session never restores a sandbox position nor the
  reverse; the other ledger's latest records are read only to name their carried
  positions in the start banner (`other_ledger_day_closes`).
- **Kept refusals** (`keep_refusal`, `kept_refusals`, `clear_refusals`): a broker's
  refusal of a stop GTT that no price explains (a segment or product the broker takes no
  GTT on), kept so the strategy is refused at every later start with that message until
  the user changes its product or `positional_stops` (3.5); the session clears them then.
- **The last decided bar** (`put_decided_bar`, `decided_bar`): per strategy, the close of
  the last bar it decided, so a daily strategy never decides one bar twice across starts
  (8).

States are kept as JSON text and read back as the dicts they were given; one JSON cannot
keep is refused. Times are UTC nanoseconds; every failure raises JournalError saying what
to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from ..errors import JournalError
from ._journal_file import check_name
from ._journal_handle import JournalHandle
from ._journal_rows import state_from, state_json, unreadable, whole_number
from ._types import Mode

_LATEST = (
    "SELECT * FROM day_closes AS c WHERE c.ledger = ? AND c.day = (SELECT MAX(d.day) FROM "
    "day_closes AS d WHERE d.ledger = c.ledger AND d.strategy = c.strategy) ORDER BY c.strategy"
)


@dataclass(frozen=True)
class DayClose:
    """A strategy's day-close record (module docstring): the trading `day` it closed, the
    `mode` of its ledger, its `carried` state, its `user_state` (`self.state`), its
    `version_id`, and when it was saved."""

    strategy: str
    day: date
    mode: Mode
    carried: dict[str, Any]
    user_state: dict[str, Any]
    version_id: str
    saved_ns: int


@dataclass(frozen=True)
class KeptRefusal:
    """A broker's refusal of a strategy's stop GTT, kept across starts (module docstring):
    the leg it refused, the broker's `message`, the trading `day` it was kept on and when."""

    strategy: str
    symbol: str
    exchange: str
    product: str
    message: str
    day: date
    kept_ns: int


class JournalCarry(JournalHandle):
    """The journal's part for what carries across days (module docstring), mixed into
    `journal.Journal`."""

    def put_day_close(
        self,
        strategy: str,
        carried: dict[str, Any],
        user_state: dict[str, Any],
        version_id: str,
        *,
        day: date | None = None,
    ) -> None:
        """Write `strategy`'s day-close record of the day (the session's, or `day`),
        replacing one written earlier that day. Raises JournalError for a blank strategy or
        version id, a state JSON cannot keep, or with no `day` before `start_session`."""
        check_name("strategy", strategy)
        check_name("version_id", version_id)
        key = self._day_key(day, "write the day-close record")
        states = (
            state_json(carried, "the carried state", strategy),
            state_json(user_state, "self.state", strategy),
        )
        with self._sql("write the day-close record") as sql:
            row = (self._ledger, strategy, key, *states, version_id, self._now_ns())
            sql.execute("INSERT OR REPLACE INTO day_closes VALUES (?, ?, ?, ?, ?, ?, ?)", row)

    def day_close(self, strategy: str) -> DayClose | None:
        """`strategy`'s latest day-close record in the handle's ledger, or None."""
        query = (
            "SELECT * FROM day_closes WHERE ledger = ? AND strategy = ? ORDER BY day DESC LIMIT 1"
        )
        with self._sql("read the day-close record") as sql:
            row = sql.execute(query, (self._ledger, strategy)).fetchone()
        return None if row is None else _day_close(row)

    def day_closes(self) -> dict[str, DayClose]:
        """Each strategy's latest day-close record in the handle's ledger, by strategy."""
        return self._latest(self._ledger, "read the day-close records")

    def other_ledger_day_closes(self) -> dict[str, DayClose]:
        """Each strategy's latest day-close record in the other mode's ledger, by strategy:
        only for naming the positions held there (20-positional-live 4.2 step 1), never
        to restore them into this one."""
        other = Mode.LIVE if self._mode is Mode.SANDBOX else Mode.SANDBOX
        return self._latest(other.value, "read the other ledger's day-close records")

    def keep_refusal(
        self,
        strategy: str,
        symbol: str,
        exchange: str,
        product: str,
        message: str,
        *,
        day: date | None = None,
    ) -> None:
        """Keep the broker's refusal (`message`) of the stop GTT of `strategy`'s leg, on the
        day (the session's, or `day`), replacing one kept for the leg before. Raises
        JournalError for a blank name or message, or with no `day` before `start_session`."""
        leg = (strategy, symbol, exchange, product)
        for name, value in zip(("strategy", "symbol", "exchange", "product"), leg, strict=True):
            check_name(name, value)
        check_name("message", message)
        key = self._day_key(day, "keep the refusal")
        with self._sql("keep the refusal") as sql:
            row = (self._ledger, *leg, message, key, self._now_ns())
            sql.execute("INSERT OR REPLACE INTO kept_refusals VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)

    def kept_refusals(self) -> list[KeptRefusal]:
        """Every refusal kept in the handle's ledger, oldest first."""
        query = "SELECT * FROM kept_refusals WHERE ledger = ? ORDER BY kept_ns, strategy"
        with self._sql("read the kept refusals") as sql:
            rows = sql.execute(query, (self._ledger,)).fetchall()
        try:
            return [
                KeptRefusal(
                    row["strategy"],
                    row["symbol"],
                    row["exchange"],
                    row["product"],
                    row["message"],
                    date.fromisoformat(row["day"]),
                    row["kept_ns"],
                )
                for row in rows
            ]
        except ValueError as error:
            raise JournalError(unreadable("a kept refusal", error)) from error

    def clear_refusals(self, strategy: str) -> None:
        """Forget every refusal kept for `strategy` in the handle's ledger (nothing happens
        when it has none): the user changed its product or `positional_stops`."""
        with self._sql("clear the kept refusals") as sql:
            sql.execute(
                "DELETE FROM kept_refusals WHERE ledger = ? AND strategy = ?",
                (self._ledger, strategy),
            )

    def put_decided_bar(self, strategy: str, bar_ns: int) -> None:
        """Note that `strategy` decided its bar closing at `bar_ns` (UTC nanoseconds); the
        same bar again changes nothing. Raises JournalError for a blank strategy, a time
        that is not a whole number, and a bar before the one it last decided: a strategy
        decides its bars in order, each once."""
        check_name("strategy", strategy)
        closed = whole_number("bar_ns", bar_ns)
        with self._sql("note the decided bar") as sql:
            last = _decided(sql, self._ledger, strategy)
            if last == closed:
                return
            if last is not None and closed < last:
                raise JournalError(
                    f"{strategy} decided its bar closing at {last} already, after the bar "
                    f"closing at {closed}: a strategy decides its bars in order, each once"
                )
            row = (self._ledger, strategy, closed, self._now_ns())
            sql.execute("INSERT OR REPLACE INTO decided_bars VALUES (?, ?, ?, ?)", row)

    def decided_bar(self, strategy: str) -> int | None:
        """The close (UTC nanoseconds) of the last bar `strategy` decided, or None."""
        with self._sql("read the decided bar") as sql:
            return _decided(sql, self._ledger, strategy)

    def _latest(self, ledger: str, action: str) -> dict[str, DayClose]:
        with self._sql(action) as sql:
            rows = sql.execute(_LATEST, (ledger,)).fetchall()
        return {row["strategy"]: _day_close(row) for row in rows}


def _decided(sql: Any, ledger: str, strategy: str) -> int | None:
    query = "SELECT bar_ns FROM decided_bars WHERE ledger = ? AND strategy = ?"
    row = sql.execute(query, (ledger, strategy)).fetchone()
    return None if row is None else row["bar_ns"]


def _day_close(row: Any) -> DayClose:
    """The record a `day_closes` row holds. Raises JournalError for one it cannot read."""
    strategy = row["strategy"]
    try:
        day, mode = date.fromisoformat(row["day"]), Mode(row["ledger"])
    except ValueError as error:
        raise JournalError(unreadable("a day-close record", error)) from error
    return DayClose(
        strategy,
        day,
        mode,
        state_from(row["carried"], "the carried state", strategy),
        state_from(row["user_state"], "self.state", strategy),
        row["version_id"],
        row["saved_ns"],
    )


__all__ = ["DayClose", "JournalCarry", "KeptRefusal"]
