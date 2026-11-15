"""The live session's journal (08-live-trading sections 6.1 and 7): one SQLite file per
OpenAlgo account, holding what a restart needs.

The order path writes an intent before its HTTP call, then binds it to the broker's order
ID, or marks it unbound when the outcome is unknown (a 502, a timeout, a disconnect) so
that it is resolved against the order book before anything else is sent. Every order
update is kept, in arrival order. The journal also holds each leg's broker-held stop
(D31: one exit owner per leg), the risk rules' state (extremes, trail level, daily
realized P&L) and the strategy's own `self.state`.

Every write is committed, and synced to disk, before its method returns: an intent is
durable before its order is sent. The file uses write-ahead logging and carries its
schema version in `PRAGMA user_version` (`_journal_file`).

A journal is bound to one account and one broker: an exit sent to another broker would
open a new position instead of closing the old one, so opening it with another broker is
refused. It keeps one ledger per mode: a handle reads and writes the ledger of the mode
it was opened in, so a sandbox session never restores, reconciles or exits a live
position, nor the reverse. Intent IDs are unique across both ledgers.

Times are UTC nanoseconds. Every failure raises `JournalError` saying what to do.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ..errors import JournalError
from ._journal_file import (
    JOURNAL_FILE_NAME,
    LIVE_DIR_VARIABLE,
    SCHEMA_VERSION,
    check_name,
    connect,
    default_journal_path,
)
from ._journal_rows import (
    INSERT_INTENT,
    INSERT_UPDATE,
    day_key,
    intent_from,
    intent_values,
    state_json,
    update_from,
    update_values,
)
from ._types import Mode, Now, OrderIntent, OrderUpdate

_UNRESOLVED = "SELECT * FROM intents WHERE ledger = ? AND state IN ('recorded', 'unbound')"
# A terminal update is final: a later one is stale, so the terminal one comes first.
_LAST_UPDATE = (
    "SELECT * FROM updates WHERE ledger = ? AND order_id = ? "
    "ORDER BY terminal DESC, seq DESC LIMIT 1"
)
_OPEN_ORDERS = (
    "SELECT * FROM intents AS i WHERE ledger = ? AND state = 'bound' AND NOT EXISTS "
    "(SELECT 1 FROM updates AS u WHERE u.ledger = i.ledger AND u.order_id = i.order_id "
    "AND u.terminal) ORDER BY seq"
)
_LEG = "ledger = ? AND strategy = ? AND symbol = ? AND exchange = ?"
_SESSION = "SELECT * FROM sessions WHERE ledger = ? AND day = ?"
_START_SESSION = (
    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT (ledger, day) "
    "DO UPDATE SET last_started_ns = excluded.last_started_ns, starts = starts + 1"
)
_RMS_STATE, _USER_STATE = "rms", "user"


@dataclass(frozen=True)
class JournalOrder:
    """A bound order: its intent and the last update it had (None before the first)."""

    order_id: str
    intent: OrderIntent
    last_update: OrderUpdate | None


@dataclass(frozen=True)
class JournalSession:
    """A trading day's session in the journal's mode: the broker it ran on, when it first
    and last started (UTC nanoseconds) and how many times (more than one after a restart)."""

    day: date
    mode: Mode
    broker: str
    first_started_ns: int
    last_started_ns: int
    starts: int


@dataclass(frozen=True)
class JournalSnapshot:
    """What a restart restores, from the journal's mode: the unbound intents and the open
    orders (oldest first), `stops` mapping (strategy, symbol, exchange) to the stop's order
    ID, and the risk and user states by strategy. A copy: changing it changes nothing."""

    unbound: tuple[OrderIntent, ...]
    open_orders: tuple[JournalOrder, ...]
    stops: dict[tuple[str, str, str], str]
    rms_states: dict[str, dict[str, Any]]
    user_states: dict[str, dict[str, Any]]


class Journal:
    """The journal of one account; see the module docstring. Implements `JournalLike`.

    Lookups of something the journal does not hold return None. Every other failure
    raises `JournalError`, with SQLite's own error chained, and says what to do.
    """

    def __init__(
        self, connection: sqlite3.Connection, path: Path, mode: Mode, broker: str, now: Now
    ) -> None:
        """Use `Journal.open`: this wraps a connection it has already checked."""
        self._connection: sqlite3.Connection | None = connection
        self._path = path
        self._mode = mode
        self._ledger = mode.value
        self._broker = broker
        self._now = now

    @classmethod
    def open(
        cls, path: str | os.PathLike[str], *, account: str, broker: str, mode: Mode, now: Now
    ) -> Journal:
        """Create or open the journal at `path` (its folders are created) for `account`,
        bound to `broker`, reading and writing the ledger of `mode`; `now` gives the UTC
        nanoseconds of the times it records.

        Raises `JournalError` when the file cannot be opened, is not a journal, was
        written by a newer version or has an unknown schema version, belongs to another
        account or is bound to another broker.
        """
        if not isinstance(mode, Mode):
            raise JournalError(f"mode must be Mode.SANDBOX or Mode.LIVE, not {mode!r}")
        path = Path(path)
        return cls(connect(path, account, broker), path, mode, broker, now)

    @property
    def path(self) -> Path:
        """The journal file."""
        return self._path

    def close(self) -> None:
        """Close the file. A closed journal raises `JournalError`; closing twice is harmless."""
        connection, self._connection = self._connection, None
        if connection is not None:
            with self._guard("close the journal", connection):
                connection.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def record_intent(self, intent: OrderIntent) -> None:
        """Write `intent` before its HTTP call; it is unbound until `bind` or a mark.
        Raises `JournalError` when its `intent_id` is already in the journal."""
        values = intent_values(intent) | {"ledger": self._ledger, "now": self._now()}
        with self._sql("record the intent") as sql:
            known = "SELECT 1 FROM intents WHERE intent_id = ?"
            if sql.execute(known, (intent.intent_id,)).fetchone():
                raise JournalError(
                    f"intent {intent.intent_id!r} is already in the journal: an intent is "
                    "recorded once, before its order is sent, so give each order a new intent_id"
                )
            sql.execute(INSERT_INTENT, values)

    def bind(self, intent_id: str, order_id: str) -> None:
        """Bind an intent to the broker's order ID (binding the same pair again changes
        nothing). Raises `JournalError` when the intent is unknown, is bound to another
        order, or the order is bound to another intent."""
        check_name("order_id", order_id)
        with self._sql("bind the intent") as sql:
            bound = self._bound_order(sql, intent_id)
            if bound == order_id:
                return
            if bound is not None:
                raise JournalError(
                    f"intent {intent_id!r} is already bound to order {bound!r}, not "
                    f"{order_id!r}: one intent sends one order, so look for a duplicate in "
                    "the order book"
                )
            owner = self._intent_row(sql, "order_id", order_id)
            if owner is not None:
                raise JournalError(
                    f"order {order_id!r} is already bound to intent {owner['intent_id']!r}: "
                    f"an order belongs to one intent, so {intent_id!r} needs its own order"
                )
            sql.execute(
                "UPDATE intents SET state = 'bound', order_id = ?, changed_ns = ? "
                "WHERE intent_id = ?",
                (order_id, self._now(), intent_id),
            )

    def mark_unbound(self, intent_id: str, reason: str) -> None:
        """Note an intent whose outcome is unknown, and why; it stays in `unbound()` until
        resolved. Raises `JournalError` when the intent is unknown or already bound."""
        self._resolve(intent_id, "unbound", reason)

    def mark_not_placed(self, intent_id: str, reason: str) -> None:
        """Note an intent whose order never reached the broker (refused, or unbound and not
        in the order book), and why; it leaves `unbound()`. Raises `JournalError` when the
        intent is unknown or already bound."""
        self._resolve(intent_id, "not_placed", reason)

    def unbound(self) -> list[OrderIntent]:
        """The intents neither bound nor marked not placed, oldest first."""
        with self._sql("read the unbound intents") as sql:
            rows = sql.execute(f"{_UNRESOLVED} ORDER BY seq", (self._ledger,))
            return [intent_from(row) for row in rows]

    def intent(self, intent_id: str) -> OrderIntent | None:
        """The recorded intent, or None."""
        with self._sql("read the intent") as sql:
            row = self._intent_row(sql, "intent_id", intent_id)
            return None if row is None else intent_from(row)

    def order_id_of(self, intent_id: str) -> str | None:
        """The order ID an intent is bound to, or None while it is not bound (or unknown)."""
        with self._sql("read the order ID") as sql:
            row = self._intent_row(sql, "intent_id", intent_id)
            return None if row is None else row["order_id"]

    def intent_of(self, order_id: str) -> OrderIntent | None:
        """The intent an order ID is bound to, or None for an order the engine did not send."""
        with self._sql("read the intent of the order") as sql:
            row = self._intent_row(sql, "order_id", order_id)
            return None if row is None else intent_from(row)

    def record_update(self, update: OrderUpdate) -> None:
        """Keep an order update, after every earlier one. An update may arrive before its
        order is bound, or be of an order the engine did not send: it is kept all the same."""
        values = update_values(update) | {"ledger": self._ledger}
        with self._sql("record the order update") as sql:
            sql.execute(INSERT_UPDATE, values)

    def updates(self, order_id: str) -> list[OrderUpdate]:
        """Every update of an order, in arrival order."""
        with self._sql("read the order updates") as sql:
            rows = sql.execute(
                "SELECT * FROM updates WHERE ledger = ? AND order_id = ? ORDER BY seq",
                (self._ledger, order_id),
            )
            return [update_from(row) for row in rows]

    def last_update(self, order_id: str) -> OrderUpdate | None:
        """An order's latest update to arrive, or None. Once a terminal update (complete,
        rejected, cancelled) has arrived, a later one is stale and ignored."""
        with self._sql("read the last order update") as sql:
            return self._last_update(sql, order_id)

    def open_orders(self) -> list[JournalOrder]:
        """The bound orders with no terminal update (none at all, or not yet terminal),
        oldest first."""
        with self._sql("read the open orders") as sql:
            orders = []
            for row in sql.execute(_OPEN_ORDERS, (self._ledger,)).fetchall():
                order_id = row["order_id"]
                last = self._last_update(sql, order_id)
                orders.append(JournalOrder(order_id, intent_from(row), last))
            return orders

    def set_stop(self, strategy: str, symbol: str, exchange: str, order_id: str) -> None:
        """Record the broker-held stop of a leg (setting the same one again changes nothing).
        Raises `JournalError` when the leg has another stop: one exit owner per leg."""
        with self._sql("record the stop") as sql:
            held = self._stop(sql, strategy, symbol, exchange)
            if held == order_id:
                return
            if held is not None:
                raise JournalError(
                    f"strategy {strategy!r} on {exchange}:{symbol} already has stop order "
                    f"{held!r}: a leg has one exit owner, so cancel that stop, wait for the "
                    f"cancel to be confirmed and clear_stop it before recording {order_id!r}"
                )
            sql.execute(
                "INSERT INTO stops VALUES (?, ?, ?, ?, ?, ?)",
                (self._ledger, strategy, symbol, exchange, order_id, self._now()),
            )

    def stop_of(self, strategy: str, symbol: str, exchange: str) -> str | None:
        """The order ID of a leg's stop, or None."""
        with self._sql("read the stop") as sql:
            return self._stop(sql, strategy, symbol, exchange)

    def clear_stop(self, strategy: str, symbol: str, exchange: str) -> None:
        """Forget a leg's stop (nothing happens when it has none)."""
        leg = (self._ledger, strategy, symbol, exchange)
        with self._sql("clear the stop") as sql:
            sql.execute(f"DELETE FROM stops WHERE {_LEG}", leg)

    def put_rms_state(self, strategy: str, state: dict[str, Any]) -> None:
        """Save the risk rules' state of a strategy (extremes, trail level, daily realized
        P&L), replacing the last. Raises `JournalError` naming a key whose value does not
        survive a JSON round trip."""
        self._put_state(_RMS_STATE, strategy, state, "rms_state")

    def rms_state(self, strategy: str) -> dict[str, Any] | None:
        """The last risk state saved for a strategy, or None."""
        return self._state(_RMS_STATE, strategy)

    def put_user_state(self, strategy: str, state: dict[str, Any]) -> None:
        """Save a strategy's `self.state`, replacing the last. Raises `JournalError` naming
        a key whose value does not survive a JSON round trip."""
        self._put_state(_USER_STATE, strategy, state, "self.state")

    def user_state(self, strategy: str) -> dict[str, Any] | None:
        """The last `self.state` saved for a strategy, or None."""
        return self._state(_USER_STATE, strategy)

    def start_session(self, day: date, mode: Mode, broker: str) -> JournalSession:
        """Record that the session of trading day `day` started now, and return it; a
        restart on the same day counts one more start and keeps the first. Raises
        `JournalError` when `mode` or `broker` is not the one the journal was opened with."""
        if mode is not self._mode or broker != self._broker:
            raise JournalError(
                f"this journal was opened for the {self._ledger} ledger at broker "
                f"{self._broker!r}, not {mode!r} at {broker!r}: open it with the mode and "
                "broker OpenAlgo is in now, then start the session"
            )
        key, now = day_key(day), self._now()
        with self._sql("record the session start") as sql:
            sql.execute(_START_SESSION, (self._ledger, key, broker, now, now))
            return self._session_from(day, sql.execute(_SESSION, (self._ledger, key)).fetchone())

    def session(self, day: date) -> JournalSession | None:
        """The session of trading day `day` in the journal's mode, or None if none started."""
        with self._sql("read the session") as sql:
            row = sql.execute(_SESSION, (self._ledger, day_key(day))).fetchone()
            return None if row is None else self._session_from(day, row)

    def snapshot(self) -> JournalSnapshot:
        """What a restart restores: unbound intents, open orders with their last update,
        stops, and the risk and user states."""
        with self._sql("read the stops") as sql:
            rows = sql.execute("SELECT * FROM stops WHERE ledger = ?", (self._ledger,))
            stops = {
                (row["strategy"], row["symbol"], row["exchange"]): row["order_id"] for row in rows
            }
        return JournalSnapshot(
            unbound=tuple(self.unbound()),
            open_orders=tuple(self.open_orders()),
            stops=stops,
            rms_states=self._states(_RMS_STATE),
            user_states=self._states(_USER_STATE),
        )

    @contextmanager
    def _sql(self, action: str) -> Iterator[sqlite3.Connection]:
        if self._connection is None:
            raise JournalError(
                f"cannot {action}: the live journal at {self._path} is closed; open it again "
                "with Journal.open"
            )
        with self._guard(action, self._connection) as connection:
            yield connection

    @contextmanager
    def _guard(self, action: str, connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        try:
            yield connection
        except sqlite3.Error as error:
            raise JournalError(
                f"cannot {action} in the live journal at {self._path} (SQLite: {error}): check "
                "that the disk has space and no other program holds the file, then start the "
                "session again"
            ) from error

    def _intent_row(self, sql: sqlite3.Connection, column: str, value: str) -> Any:
        # `column` is "intent_id" or "order_id", never input.
        query = f"SELECT * FROM intents WHERE ledger = ? AND {column} = ?"
        return sql.execute(query, (self._ledger, value)).fetchone()

    def _bound_order(self, sql: sqlite3.Connection, intent_id: str) -> str | None:
        row = self._intent_row(sql, "intent_id", intent_id)
        if row is None:
            raise JournalError(
                f"intent {intent_id!r} is not in the {self._ledger} ledger of the journal: "
                "record it with record_intent before its order is sent"
            )
        return row["order_id"]

    def _resolve(self, intent_id: str, state: str, reason: str) -> None:
        with self._sql(f"mark the intent {state}") as sql:
            bound = self._bound_order(sql, intent_id)
            if bound is not None:
                raise JournalError(
                    f"intent {intent_id!r} is bound to order {bound!r}, so it reached the "
                    "broker: resolve it from that order's updates"
                )
            sql.execute(
                "UPDATE intents SET state = ?, reason = ?, changed_ns = ? WHERE intent_id = ?",
                (state, reason, self._now(), intent_id),
            )

    def _stop(self, sql: sqlite3.Connection, strategy: str, symbol: str, exchange: str) -> Any:
        leg = (self._ledger, strategy, symbol, exchange)
        row = sql.execute(f"SELECT order_id FROM stops WHERE {_LEG}", leg).fetchone()
        return None if row is None else row["order_id"]

    def _last_update(self, sql: sqlite3.Connection, order_id: str) -> OrderUpdate | None:
        row = sql.execute(_LAST_UPDATE, (self._ledger, order_id)).fetchone()
        return None if row is None else update_from(row)

    def _session_from(self, day: date, row: sqlite3.Row) -> JournalSession:
        first, last = row["first_started_ns"], row["last_started_ns"]
        return JournalSession(day, self._mode, row["broker"], first, last, row["starts"])

    def _put_state(self, kind: str, strategy: str, state: dict[str, Any], name: str) -> None:
        text = state_json(state, name, strategy)
        with self._sql(f"save {name}") as sql:
            sql.execute(
                "INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)",
                (self._ledger, kind, strategy, text, self._now()),
            )

    def _state(self, kind: str, strategy: str) -> dict[str, Any] | None:
        with self._sql("read the state") as sql:
            row = sql.execute(
                "SELECT state FROM states WHERE ledger = ? AND kind = ? AND strategy = ?",
                (self._ledger, kind, strategy),
            ).fetchone()
            return None if row is None else json.loads(row["state"])

    def _states(self, kind: str) -> dict[str, dict[str, Any]]:
        with self._sql("read the states") as sql:
            query = "SELECT * FROM states WHERE ledger = ? AND kind = ? ORDER BY strategy"
            rows = sql.execute(query, (self._ledger, kind))
            return {row["strategy"]: json.loads(row["state"]) for row in rows}


__all__ = [
    "JOURNAL_FILE_NAME",
    "LIVE_DIR_VARIABLE",
    "SCHEMA_VERSION",
    "Journal",
    "JournalOrder",
    "JournalSession",
    "JournalSnapshot",
    "default_journal_path",
]
